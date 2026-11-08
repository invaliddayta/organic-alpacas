"""Controlled input changes ("capture conditions") applied to the 256x256 RGB image
before the usual eval transform (resize 224, ImageNet normalisation).

Each family has an ordered list of levels; the level value is the parameter shown
in the figures. Rotation by 180 deg equals a horizontal plus vertical flip, which the
model saw as training augmentation, so it serves as a built-in control.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _np(img):
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0


def _pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _border_colour(img):
    a = np.asarray(img.convert("RGB"))
    frame = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3),
                            a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(frame, axis=0))


def brightness(img, f):
    return ImageEnhance.Brightness(img).enhance(f)


def contrast(img, f):
    return ImageEnhance.Contrast(img).enhance(f)


def saturation(img, f):
    return ImageEnhance.Color(img).enhance(f)


def hue_shift(img, deg):
    """Rotate hue by `deg` degrees (negative: green -> yellow)."""
    hsv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2HSV_FULL).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + int(round(deg / 360 * 256))) % 256
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL))


def colour_temp(img, t):
    """Warm (t>0) or cool (t<0) light: red gain 1+t, blue gain 1-t."""
    a = _np(img)
    a[..., 0] *= 1 + t
    a[..., 2] *= 1 - t
    return _pil(a)


def gaussian_blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(sigma))


def motion_blur(img, k):
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    return Image.fromarray(cv2.filter2D(np.asarray(img.convert("RGB")), -1, kern))


def jpeg(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise(img, std, seed=0):
    a = _np(img)
    rng = np.random.default_rng(seed)
    return _pil(a + rng.normal(0, std, a.shape))


def rotate(img, deg):
    if deg % 90 == 0:
        return img.rotate(deg)
    return img.rotate(deg, resample=Image.BILINEAR, fillcolor=_border_colour(img))


def zoom_in(img, frac):
    """Centre crop keeping `frac` of the side length, resized back to 256."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)


def zoom_out(img, frac):
    """Shrink the image to `frac` of the side length and pad with the border colour."""
    w, h = img.size
    small = img.resize((int(w * frac), int(h * frac)), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), _border_colour(img))
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return canvas


def shadow(img, strength):
    """Uneven light: linear darkening from left (x1) to right (x 1-strength)."""
    a = _np(img)
    ramp = np.linspace(1.0, 1.0 - strength, a.shape[1], dtype=np.float32)[None, :, None]
    return _pil(a * ramp)


FAMILIES = {
    # name: (function, levels, group, unit label)
    "darker":        (brightness,    [0.8, 0.6, 0.4, 0.25], "lighting", "brightness factor"),
    "brighter":      (brightness,    [1.25, 1.5, 1.8, 2.2], "lighting", "brightness factor"),
    "low contrast":  (contrast,      [0.8, 0.6, 0.4, 0.25], "lighting", "contrast factor"),
    "high contrast": (contrast,      [1.25, 1.5, 2.0, 2.5], "lighting", "contrast factor"),
    "uneven light":  (shadow,        [0.2, 0.4, 0.6, 0.8], "lighting", "darkening at right edge"),
    "warm light":    (colour_temp,   [0.05, 0.1, 0.2, 0.3], "colour", "red gain - 1"),
    "cool light":    (colour_temp,   [-0.05, -0.1, -0.2, -0.3], "colour", "red gain - 1"),
    "hue to yellow": (hue_shift,     [-5, -10, -20, -30], "colour", "hue shift (deg)"),
    "hue to blue":   (hue_shift,     [5, 10, 20, 30], "colour", "hue shift (deg)"),
    "desaturate":    (saturation,    [0.75, 0.5, 0.25, 0.0], "colour", "saturation factor"),
    "oversaturate":  (saturation,    [1.25, 1.5, 2.0, 3.0], "colour", "saturation factor"),
    "gaussian blur": (gaussian_blur, [0.5, 1, 2, 4], "sharpness", "sigma (px of 256)"),
    "motion blur":   (motion_blur,   [3, 5, 9, 15], "sharpness", "kernel length (px)"),
    "jpeg":          (jpeg,          [50, 20, 10, 5], "sharpness", "JPEG quality"),
    "noise":         (noise,         [0.02, 0.05, 0.1, 0.15], "sharpness", "noise std"),
    "rotate":        (rotate,        [180, 90, 15, 45], "geometry", "degrees"),
    "zoom in":       (zoom_in,       [0.9, 0.75, 0.6, 0.45], "geometry", "crop side fraction"),
    "zoom out":      (zoom_out,      [0.9, 0.75, 0.6, 0.45], "geometry", "scale"),
}

# strength rank 1..4 for plotting (rotation: 180 is the in-training control)
def conditions():
    for fam, (fn, levels, group, unit) in FAMILIES.items():
        for rank, lv in enumerate(levels, start=1):
            yield fam, rank, lv, group


def apply(img, family, level):
    fn = FAMILIES[family][0]
    return fn(img.convert("RGB"), level)
