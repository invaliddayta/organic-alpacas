"""Controlled input changes on a stratified test subset (40 images per class, seed 0).

For every condition in perturb.FAMILIES (18 families x 4 strengths) the model is run on
the modified images; outputs are compared with the same images unmodified, run through
the same CPU pipeline ("reference"). Writes results/perturbation_sweep.parquet with one
row per image and condition. Runtime about 50 min on 8 CPU cores (about 40 s per condition).
"""
import time

import numpy as np
import pandas as pd

import perturb
import pv

PER_CLASS = 40
OUT = pv.RESULTS / "perturbation_sweep.parquet"
PARTS = pv.RESULTS / "_sweep_parts"


def rows_for(probs, sample_idx, true_idx, ref_pred, family, rank, level, group):
    n = len(probs)
    s = pv.summarize_probs(probs)
    return pd.DataFrame({
        "sample_idx": sample_idx, "family": family, "rank": rank, "level": float(level),
        "group": group, "pred_idx": s["pred_idx"].astype(np.int16),
        "confidence": s["confidence"].astype(np.float32),
        "second_idx": s["second_idx"].astype(np.int16),
        "p_true": probs[np.arange(n), true_idx].astype(np.float32),
        "p_ref_pred": probs[np.arange(n), ref_pred].astype(np.float32),
    })


def main():
    t0 = time.time()
    _, test = pv.load_splits()
    labels = np.asarray(test["class_idx"])
    idx = pv.stratified_subset(labels, PER_CLASS, seed=0)
    images = [test[int(i)]["image"].convert("RGB") for i in idx]
    true_idx = labels[idx]
    print(f"subset: {len(idx)} images, loaded in {time.time() - t0:.0f}s")

    model = pv.load_model()
    PARTS.mkdir(parents=True, exist_ok=True)

    ref_file = PARTS / "reference.parquet"
    if ref_file.exists():
        ref = pd.read_parquet(ref_file)
    else:
        probs = pv.predict_images(model, images)
        ref = rows_for(probs, idx, true_idx, probs.argmax(1), "none", 0, 0, "none")
        ref.to_parquet(ref_file, index=False)
    ref_pred = ref["pred_idx"].to_numpy()

    for fam, rank, level, group in perturb.conditions():
        f = PARTS / f"{fam.replace(' ', '_')}_{rank}.parquet"
        if f.exists():
            continue
        t = time.time()
        probs = pv.predict_images(model, (perturb.apply(im, fam, level) for im in images))
        rows_for(probs, idx, true_idx, ref_pred, fam, rank, level, group).to_parquet(f, index=False)
        acc = (probs.argmax(1) == true_idx).mean()
        print(f"{fam:14s} {rank} {level:>7} acc={acc:.3f} ({time.time() - t:.0f}s)", flush=True)

    parts = [pd.read_parquet(ref_file)] + [
        pd.read_parquet(PARTS / f"{fam.replace(' ', '_')}_{rank}.parquet")
        for fam, rank, _, _ in perturb.conditions()
    ]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.name}: {len(df)} rows, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
