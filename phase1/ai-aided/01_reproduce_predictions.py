"""Milestone 1: rerun the shared checkpoint on the original test split and compare
with shared/outputs/resnet18/test_predictions.csv (same sample_idx, same pred_idx?).

Writes results/reproduction_check.json. Runtime about 3 min on 8 CPU cores.
"""
import json
import time

import numpy as np
import pandas as pd

import pv

t0 = time.time()
_, test = pv.load_splits()
shared = pd.read_csv(pv.SHARED_PRED)

# same rows in the same order?
meta_ok = (
    (np.asarray(test["class_idx"]) == shared["true_idx"].to_numpy()).all()
    and (pd.Series(test["leaf_id"]).fillna("") == shared["leaf_id"].fillna("")).all()
)
print("metadata order matches:", meta_ok)

model = pv.load_model()
probs = pv.predict_images(model, (row["image"] for row in test))
pred = probs.argmax(1)

shared_probs = shared[[f"prob_{i}" for i in range(pv.NUM_CLASSES)]].to_numpy()
res = {
    "n_test": int(len(test)),
    "metadata_order_matches": bool(meta_ok),
    "accuracy_reproduced": float((pred == shared["true_idx"].to_numpy()).mean()),
    "accuracy_shared": float(shared["correct"].mean()),
    "pred_idx_mismatches": int((pred != shared["pred_idx"].to_numpy()).sum()),
    "max_abs_prob_diff": float(np.abs(probs - shared_probs).max()),
    "mean_abs_conf_diff": float(np.abs(probs.max(1) - shared["confidence"].to_numpy()).mean()),
    "seconds": round(time.time() - t0, 1),
}
# explain every mismatch: is it a near-tie between the top-2 classes?
mismatch = []
for i in np.flatnonzero(pred != shared["pred_idx"].to_numpy()):
    top_s, top_r = np.argsort(-shared_probs[i])[:2], np.argsort(-probs[i])[:2]
    mismatch.append({
        "sample_idx": int(i), "true_idx": int(shared["true_idx"][i]),
        "shared_top2": [[int(c), round(float(shared_probs[i, c]), 5)] for c in top_s],
        "reproduced_top2": [[int(c), round(float(probs[i, c]), 5)] for c in top_r],
    })
res["mismatches"] = mismatch
pv.RESULTS.mkdir(exist_ok=True)
print(json.dumps(res, indent=2))
(pv.RESULTS / "reproduction_check.json").write_text(json.dumps(res, indent=2))
