"""Plot style and small helpers shared by the notebook figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colour-blind safe palette
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}
GROUP_COLOURS = {"lighting": OI["orange"], "colour": OI["purple"],
                 "sharpness": OI["blue"], "geometry": OI["green"], "background": OI["purple"]}


def style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.titlesize": 17, "figure.titleweight": "bold",
    })


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (arrays ok)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
