"""Leaf / background separation for PlantVillage images (input preprocessing only).

This describes the INPUT images; it does not use the model. Steps:
1. background colour = median Lab colour of a 6 px border frame;
2. colour distance of every pixel to that background colour (lightness down-weighted,
   so leaf shadows on the paper count as background), Otsu threshold;
3. GrabCut refinement seeded with that mask;
4. keep the largest connected component, fill holes.
Images where the leaf fills the frame (e.g. close-up corn leaves) have no usable
background; `mask_quality` flags them so that background experiments can skip them.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

BORDER = 6


def leaf_mask(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    frame = np.zeros((h, w), bool)
    frame[:BORDER], frame[-BORDER:], frame[:, :BORDER], frame[:, -BORDER:] = True, True, True, True
    bg = np.median(lab[frame], axis=0)
    diff = lab - bg
    dist = np.sqrt(0.25 * diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    thr, fg = cv2.threshold(cv2.GaussianBlur(d8, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gc = np.where(fg > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[d8 < max(thr * 0.4, 8)] = cv2.GC_BGD
    gc[frame & (fg == 0)] = cv2.GC_BGD
    sure_fg = cv2.erode((fg > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    gc[sure_fg > 0] = cv2.GC_FGD
    if (gc == cv2.GC_BGD).any() and ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).any():
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb[..., ::-1].copy(), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            pass
    m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    lab_cc, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab_cc, range(1, n + 1))
        m = lab_cc == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    return m


def mask_quality(mask: np.ndarray) -> dict:
    """Leaf area fraction and share of the image border covered by leaf."""
    h, w = mask.shape
    frame = np.zeros_like(mask)
    frame[:BORDER], frame[-BORDER:], frame[:, :BORDER], frame[:, -BORDER:] = True, True, True, True
    return {"leaf_frac": float(mask.mean()), "border_leaf_frac": float(mask[frame].mean())}


def usable(q: dict) -> bool:
    """Background experiments need a visible background and a leaf of sensible size."""
    return 0.08 <= q["leaf_frac"] <= 0.75 and q["border_leaf_frac"] <= 0.35


# --------------------------------------------------------------------------- edits
def background_only(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Remove the leaf: inpaint the (dilated) leaf region from the surrounding background."""
    rgb = np.asarray(img.convert("RGB"))
    m = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8))
    out = cv2.inpaint(rgb, m, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(out)


def leaf_on_colour(img: Image.Image, mask: np.ndarray, colour) -> Image.Image:
    rgb = np.asarray(img.convert("RGB")).copy()
    soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]
    out = rgb * soft + np.asarray(colour, np.float32)[None, None, :] * (1 - soft)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def leaf_on_background(img: Image.Image, mask: np.ndarray, background: Image.Image) -> Image.Image:
    """Paste the leaf of `img` onto `background` (an image whose own leaf was removed)."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    bg = np.asarray(background.convert("RGB")).astype(np.float32)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]
    return Image.fromarray((rgb * soft + bg * (1 - soft)).clip(0, 255).astype(np.uint8))
