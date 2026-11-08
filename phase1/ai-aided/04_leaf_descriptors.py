"""Leaf / background colour descriptors for every test image (input description only).

Uses segment.leaf_mask (colour distance to the border + GrabCut). Writes
results/test_leaf_descriptors.parquet with leaf area, mask usability flags and the mean
Lab colour of the leaf and of the background. Lab: L = lightness (0-100), a = green(-)/red(+),
b = blue(-)/yellow(+). Runtime about 8 min with 2 worker processes.
"""
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

import pv
import segment as sg

_test = None


def _init():
    global _test
    cv2.setNumThreads(1)
    _, _test = pv.load_splits()


def describe(i):
    img = _test[int(i)]["image"].convert("RGB")
    m = sg.leaf_mask(img)
    q = sg.mask_quality(m)
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = np.stack([lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128], -1)
    leaf = lab[m] if m.any() else lab.reshape(-1, 3)
    bg = lab[~m] if (~m).any() else np.full((1, 3), np.nan, np.float32)
    hsv = (cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV_FULL)[m] if m.any() else np.zeros((1, 3))).astype(np.float32)
    return {
        "sample_idx": int(i), **q, "usable": sg.usable(q),
        "leaf_L": float(leaf[:, 0].mean()), "leaf_a": float(leaf[:, 1].mean()),
        "leaf_b": float(leaf[:, 2].mean()),
        "leaf_yellow_frac": float(((hsv[:, 0] * 360 / 256 > 40) & (hsv[:, 0] * 360 / 256 < 75)
                                   & (hsv[:, 1] > 60)).mean()),
        "bg_L": float(np.nanmean(bg[:, 0])), "bg_a": float(np.nanmean(bg[:, 1])),
        "bg_b": float(np.nanmean(bg[:, 2])),
    }


if __name__ == "__main__":
    t0 = time.time()
    n = 10948
    with Pool(2, initializer=_init) as pool:
        rows = pool.map(describe, range(n), chunksize=64)
    df = pd.DataFrame(rows)
    df.to_parquet(pv.RESULTS / "test_leaf_descriptors.parquet", index=False)
    print(df.describe().T.to_string())
    print(f"done {time.time() - t0:.0f}s")
