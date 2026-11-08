"""Input-space descriptors (these describe the IMAGES, not the model).

For all 54,304 images:
- a 32x32 RGB thumbnail, flattened and standardised per image, so that the dot product of
  two thumbnails is their Pearson correlation (a simple pixel-space similarity);
- the median Lab colour and Lab spread of a 6 px border frame (the "background colour").

Outputs
- results/test_nearest_train.parquet: for every test image the most similar training image
  (pixel correlation), its class, and the best similarity within the true class / other classes.
- results/border_colour.parquet: border colour stats for every image (train and test).
- results/border_knn.json: accuracy of a k-NN classifier that only sees the border colour
  (dataset-bias check, cf. Noyan 2022), compared with chance.
Thumbnails are cached in cache/ (gitignored). Runtime about 5 min.
"""
import json
import time

import cv2
import numpy as np
import pandas as pd

import pv

CACHE = pv.HERE / "cache"
CACHE.mkdir(exist_ok=True)
SIDE = 32
B = 6


def describe(img):
    rgb = np.asarray(img.convert("RGB"))
    thumb = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    frame = np.concatenate([lab[:B].reshape(-1, 3), lab[-B:].reshape(-1, 3),
                            lab[:, :B].reshape(-1, 3), lab[:, -B:].reshape(-1, 3)])
    med = np.median(frame, axis=0)
    spread = np.median(np.abs(frame - med), axis=0)
    return thumb, np.concatenate([med, spread])


def load_or_compute(name, ds):
    f = CACHE / f"{name}_desc.npz"
    if f.exists():
        z = np.load(f)
        return z["thumbs"], z["border"]
    thumbs = np.zeros((len(ds), SIDE, SIDE, 3), np.uint8)
    border = np.zeros((len(ds), 6), np.float32)
    for i, row in enumerate(ds):
        thumbs[i], border[i] = describe(row["image"])
        if i % 5000 == 0:
            print(name, i, flush=True)
    np.savez(f, thumbs=thumbs, border=border)
    return thumbs, border


def standardise(thumbs):
    x = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    x -= x.mean(1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    return x


def main():
    t0 = time.time()
    train, test = pv.load_splits()
    tr_thumbs, tr_border = load_or_compute("train", train)
    te_thumbs, te_border = load_or_compute("test", test)
    print(f"descriptors done {time.time() - t0:.0f}s", flush=True)
    y_tr = np.asarray(train["class_idx"])
    y_te = np.asarray(test["class_idx"])

    # ---- nearest training image in pixel space
    Xtr, Xte = standardise(tr_thumbs), standardise(te_thumbs)
    nn_idx = np.zeros(len(Xte), np.int64)
    nn_sim = np.zeros(len(Xte), np.float32)
    best_same = np.zeros(len(Xte), np.float32)
    best_other = np.zeros(len(Xte), np.float32)
    best_other_cls = np.zeros(len(Xte), np.int16)
    for s in range(0, len(Xte), 1000):
        S = Xte[s:s + 1000] @ Xtr.T
        j = S.argmax(1)
        nn_idx[s:s + 1000] = j
        nn_sim[s:s + 1000] = S[np.arange(len(S)), j]
        same = y_tr[None, :] == y_te[s:s + 1000, None]
        best_same[s:s + 1000] = np.where(same, S, -2).max(1)
        So = np.where(same, -2, S)
        jo = So.argmax(1)
        best_other[s:s + 1000] = So[np.arange(len(So)), jo]
        best_other_cls[s:s + 1000] = y_tr[jo]
    nn = pd.DataFrame({
        "sample_idx": np.arange(len(Xte)), "nn_train_idx": nn_idx, "nn_sim": nn_sim,
        "nn_class": y_tr[nn_idx].astype(np.int16), "best_sim_same_class": best_same,
        "best_sim_other_class": best_other, "best_other_class": best_other_cls,
    })
    nn.to_parquet(pv.RESULTS / "test_nearest_train.parquet", index=False)
    print(f"pixel 1-NN accuracy: {(nn.nn_class.to_numpy() == y_te).mean():.3f}", flush=True)

    # ---- border colour only: how much does the background tell about the class?
    bc = pd.DataFrame(np.concatenate([tr_border, te_border]),
                      columns=["L", "a", "b", "L_spread", "a_spread", "b_spread"])
    bc["split"] = ["train"] * len(tr_border) + ["test"] * len(te_border)
    bc["idx_in_split"] = np.concatenate([np.arange(len(tr_border)), np.arange(len(te_border))])
    bc["class_idx"] = np.concatenate([y_tr, y_te])
    bc.to_parquet(pv.RESULTS / "border_colour.parquet", index=False)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(tr_border[:, :3])
    knn = KNeighborsClassifier(n_neighbors=15).fit(sc.transform(tr_border[:, :3]), y_tr)
    pred = knn.predict(sc.transform(te_border[:, :3]))
    hosts = np.array([pv.host_of(n) for n in pv.class_names(test)])
    prior = np.bincount(y_tr, minlength=38) / len(y_tr)
    res = {
        "features": "median Lab colour of a 6 px border frame (3 numbers per image)",
        "classifier": "15-NN on standardised features, fit on the 43,356 training images",
        "test_accuracy": float((pred == y_te).mean()),
        "test_crop_accuracy": float((hosts[pred] == hosts[y_te]).mean()),
        "chance_uniform": 1 / 38,
        "chance_majority_class": float(np.bincount(y_te).max() / len(y_te)),
        "chance_prior_matching": float((prior * np.bincount(y_te, minlength=38) / len(y_te)).sum()),
        "per_class_accuracy": {int(c): float((pred[y_te == c] == c).mean()) for c in range(38)},
    }
    (pv.RESULTS / "border_knn.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class_accuracy"}, indent=2))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
