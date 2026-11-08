"""Shared helpers for the AI-aided Phase I analysis (black-box only).

Loads the dataset with the same code path and split logic as shared/baseline.ipynb,
rebuilds the ResNet-18 from the shared checkpoint and runs it on (possibly modified)
PIL images. Only inputs and outputs (softmax probabilities) are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # phase1/ai-aided
REPO = HERE.parents[1]                          # repo root
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CHECKPOINT = REPO / "shared" / "outputs" / "resnet18" / "best_model.pth"
SHARED_PRED = REPO / "shared" / "outputs" / "resnet18" / "test_predictions.csv"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 38


# ----------------------------------------------------------------------------- data
def load_splits():
    """Return (train_dataset, test_dataset) exactly as in the baseline notebook.

    The baseline filters with `lambda x: x["split"] == ...` over full rows; filtering
    on the `split` column only gives the same rows in the same order (checked against
    shared/outputs/resnet18/test_predictions.csv in 01_reproduce_predictions.py).
    The validation split is not needed here: train = original train part (43,356).
    """
    from datasets import load_dataset

    data = load_dataset("geraldmc/plantvillage-full")["train"]
    train = data.filter(lambda s: s == "train", input_columns="split")
    test = data.filter(lambda s: s == "test", input_columns="split")
    return train, test


def class_names(ds) -> list[str]:
    m = (
        pd.DataFrame({"i": ds["class_idx"], "l": ds["class_label"]})
        .drop_duplicates()
        .set_index("i")["l"]
        .sort_index()
    )
    assert len(m) == NUM_CLASSES
    return m.tolist()


def short_name(label: str) -> str:
    """'Corn_(maize)___Northern_Leaf_Blight' -> 'Corn: Northern Leaf Blight'."""
    host, dis = label.split("___")
    host = (host.replace("_(including_sour)", "").replace("_(maize)", "")
            .replace(",_bell", "").replace("_", " "))
    dis = (dis.replace("_", " ").replace("Tomato ", "").replace("(Black Measles)", "")
           .replace("(Isariopsis Leaf Spot)", "").replace("Cercospora leaf spot ", "")
           .replace("Two-spotted spider mite", "").replace("(Citrus greening)", "")
           .replace("Haunglongbing", "Huanglongbing")
           .replace("Yellow Leaf Curl Virus", "Yellow leaf curl").strip())
    return f"{host}: {dis}"


def host_of(label: str) -> str:
    return label.split("___")[0]


def stratified_subset(labels, per_class: int, seed: int = 0) -> np.ndarray:
    """Indices of up to `per_class` random samples per class (sorted)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        out.append(idx if len(idx) <= per_class else rng.choice(idx, per_class, replace=False))
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------------------- model
def load_model(threads: int = 8):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    torch.set_num_threads(threads)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def eval_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def predict_images(model, images, batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities (n, 38) for an iterable of PIL images (black-box call)."""
    import torch

    tf = eval_transform()
    probs, batch = [], []

    def flush():
        with torch.no_grad():
            x = torch.stack(batch)
            probs.append(torch.softmax(model(x), dim=1).numpy())
        batch.clear()

    for im in images:
        batch.append(tf(im.convert("RGB")))
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return np.concatenate(probs).astype(np.float32)


def summarize_probs(probs: np.ndarray, prefix: str = "") -> pd.DataFrame:
    """pred_idx, confidence, runner-up and its probability."""
    order = np.argsort(-probs, axis=1)
    return pd.DataFrame({
        f"{prefix}pred_idx": order[:, 0],
        f"{prefix}confidence": probs[np.arange(len(probs)), order[:, 0]],
        f"{prefix}second_idx": order[:, 1],
        f"{prefix}second_prob": probs[np.arange(len(probs)), order[:, 1]],
    })
