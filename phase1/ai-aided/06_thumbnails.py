"""Small JPEG thumbnails so that the notebook and the HTML view run without the dataset.

- results/thumbs_64.parquet: 64 px thumbnails of the sweep subset (40 per class) plus all
  107 misclassified test images (shared predictions).
- results/thumbs_128.parquet: 128 px thumbnails of the images shown in static figures.
- results/perturb_examples.parquet: one example image under every perturbation level (96 px).
- results/train_examples.parquet: the 8 darkest Tomato late blight training images (128 px),
  used to illustrate where the dark -> late blight shortcut may come from (needs
  results/image_brightness.parquet from 03_input_space.py).
- results/late_blight_sessions.json: Tomato late blight training images per capture session
  (file-name prefix in leaf_id), with how many fall into the darkest 1% of all training images.
Runtime about 1 min.
"""
import io

import json

import numpy as np
import pandas as pd

import perturb
import pv

EXAMPLE_IDX = 9000  # a Tomato spider-mite leaf with a clear background


def jpeg_bytes(img, side, q=80):
    buf = io.BytesIO()
    img.convert("RGB").resize((side, side)).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def main():
    train, test = pv.load_splits()
    labels = np.asarray(test["class_idx"])
    shared = pd.read_csv(pv.SHARED_PRED)
    subset = pv.stratified_subset(labels, 40, seed=0)
    errors = shared.loc[~shared.correct, "sample_idx"].to_numpy()
    ids64 = np.union1d(subset, errors)
    pd.DataFrame({"sample_idx": ids64,
                  "jpeg": [jpeg_bytes(test[int(i)]["image"], 64, 75) for i in ids64]}
                 ).to_parquet(pv.RESULTS / "thumbs_64.parquet", index=False)

    # figure thumbnails: Tomato bacterial spot -> yellow leaf curl (one photo per leaf),
    # random correct bacterial spot, plus all errors for the error gallery
    e = shared[(shared.true_idx == 28) & (shared.pred_idx == 35)].drop_duplicates("leaf_id")
    ok = shared[(shared.true_idx == 28) & shared.correct].sample(7, random_state=0)
    ids128 = np.union1d(np.union1d(e.sample_idx, ok.sample_idx), errors)
    pd.DataFrame({"sample_idx": ids128,
                  "jpeg": [jpeg_bytes(test[int(i)]["image"], 128, 85) for i in ids128]}
                 ).to_parquet(pv.RESULTS / "thumbs_128.parquet", index=False)

    br = pd.read_parquet(pv.RESULTS / "image_brightness.parquet")
    dark = br[(br.split == "train") & (br.class_idx == 30)].nsmallest(8, "luminance")
    pd.DataFrame({"idx_in_split": dark.idx_in_split.to_numpy(), "luminance": dark.luminance.to_numpy(),
                  "leaf_id": [train[int(i)]["leaf_id"] for i in dark.idx_in_split],
                  "jpeg": [jpeg_bytes(train[int(i)]["image"], 128, 85) for i in dark.idx_in_split]}
                 ).to_parquet(pv.RESULTS / "train_examples.parquet", index=False)

    tr = br[br.split == "train"].sort_values("idx_in_split").reset_index(drop=True)
    q = float(tr.luminance.quantile(0.01))
    fn = pd.Series(train["leaf_id"]).str.split("___").str[-1]
    tag = fn.str.extract(r"^(GHLB_PS|GHLB2?|GH_HL|[A-Za-z]+)")[0]
    sessions = {}
    for t in tag[tr.class_idx == 30].value_counts().index:
        m = (tr.class_idx == 30) & (tag == t)
        sessions[t] = {"n_train": int(m.sum()), "n_in_darkest_1pct": int((m & (tr.luminance <= q)).sum()),
                       "median_luminance": round(float(tr.luminance[m].median()), 1)}
    test_fn = pd.Series(test["leaf_id"]).str.split("___").str[-1]
    n_ps_test = int(((np.asarray(test["class_idx"]) == 30) & test_fn.str.startswith("GHLB_PS")).sum())
    (pv.RESULTS / "late_blight_sessions.json").write_text(json.dumps(
        {"darkest_1pct_luminance": round(q, 1), "train_sessions": sessions, "GHLB_PS_in_test": n_ps_test}, indent=1))

    img = test[EXAMPLE_IDX]["image"].convert("RGB")
    rows = [{"family": "none", "rank": 0, "level": 0.0, "jpeg": jpeg_bytes(img, 96, 85)}]
    for fam, rank, level, _ in perturb.conditions():
        rows.append({"family": fam, "rank": rank, "level": float(level),
                     "jpeg": jpeg_bytes(perturb.apply(img, fam, level), 96, 85)})
    pd.DataFrame(rows).to_parquet(pv.RESULTS / "perturb_examples.parquet", index=False)
    print(len(ids64), len(ids128), len(rows))


if __name__ == "__main__":
    main()
