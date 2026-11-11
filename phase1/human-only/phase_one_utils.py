import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from functools import lru_cache
from PIL import Image, ImageEnhance, ImageFilter

# plot helper
def plot_horizontal_bars(
    data,
    x,
    title,
    xlabel,
    y="Class",
    figsize=(11, 6.5),
    value_format=None,
    percentage_axis=False,
    zero_line=False,
    xlim=None
):
    fig, ax = plt.subplots(figsize=figsize)

    # Blue gradient across categories
    colors = sns.color_palette("Blues_r", n_colors=len(data))

    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=y,
        palette=colors,
        dodge=False,
        legend=False,
        edgecolor="none",
        ax=ax
    )

    # Optional value labels
    if value_format is not None:
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt=value_format,
                padding=5,
                fontsize=10
            )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
        loc="left"
    )

    ax.set_xlabel(xlabel, fontsize=12, labelpad=12)
    ax.set_ylabel("")

    if percentage_axis:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.0f}%")
        )

    if zero_line:
        ax.axvline(0, color="gray", linewidth=0.8)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="both", length=0, labelsize=10)

    plt.tight_layout()
    plt.show()

def plot_robustness_curves(summary, perturbations, baseline_accuracy):
    fig, axes = plt.subplots(1, len(perturbations),
                            figsize=(16, 5), sharey=True)

    for ax, kind in zip(axes, perturbations):
        data = summary[
            summary["Perturbation"] == kind
        ].sort_values("Strength")

        ax.errorbar(
            data["Strength"],
            data["accuracy_mean"],
            yerr=data["accuracy_std"],
            color="steelblue",
            marker="o",
            linewidth=2,
            capsize=3
        )

        ax.axhline(
            baseline_accuracy,
            color="gray",
            linestyle="--",
            label="Baseline"
        )

        ax.set_title(
            kind, fontsize=14, fontweight="bold",
            pad=15, loc="left"
        )
        ax.set_xlabel("Perturbation Strength")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Classification Accuracy (%)")
    axes[0].set_ylim(
        max(0, summary["accuracy_mean"].min() - 5), 101
    )

    fig.suptitle(
        "Model Robustness Under Image Perturbations",
        fontsize=16, fontweight="bold",
        x=0.04, ha="left", y=1.05
    )

    sns.despine(fig=fig, left=True, bottom=True)
    plt.tight_layout()
    plt.show()

def plot_class_sensitivity(sensitivity_results, selected_strengths):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))

    for ax, (kind, data) in zip(axes, sensitivity_results.items()):
        colors = sns.color_palette("Blues_r", n_colors=len(data))

        sns.barplot(
            data=data,
            x="Recall Drop (pp)",
            y="Class",
            hue="Class",
            palette=colors,
            dodge=False,
            legend=False,
            edgecolor="none",
            ax=ax
        )

        ax.set_title(
            f"{kind} (Strength {selected_strengths[kind]})",
            fontsize=14,
            fontweight="bold",
            pad=15,
            loc="left"
        )

        ax.set_xlabel("Recall Drop (Percentage Points)")
        ax.set_ylabel("")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=9)

        sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle(
        "Classes Most Affected by Image Perturbations",
        fontsize=16,
        fontweight="bold",
        x=0.02,
        ha="left"
    )

    plt.tight_layout()
    plt.show()

def plot_segmentation_examples(dataset, indices, image_size, seed=42):
    fig, axes = plt.subplots(
        len(indices), 4,
        figsize=(13, 3 * len(indices)),
        squeeze=False
    )

    titles = ["Original", "Leaf Mask", "Foreground Only", "Background Only"]

    for row, idx in enumerate(indices):
        idx = int(idx)
        original = dataset[idx]["image"].convert("RGB").resize(
            (image_size, image_size)
        )

        images = [
            original,
            get_leaf_mask(idx, dataset, image_size, seed),
            region_transform(original, idx, "foreground", dataset, image_size, seed),
            region_transform(original, idx, "background", dataset, image_size, seed)
        ]

        for col, img in enumerate(images):
            axes[row, col].imshow(img, cmap="gray" if col == 1 else None)
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=12, fontweight="bold")
            axes[row, col].axis("off")

    fig.suptitle(
        "Leaf Segmentation and Background Removal",
        fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.show()

def plot_region_recall(df_top):
    df_plot = df_top.melt(
        id_vars="Class",
        value_vars=["Original", "Foreground Only"],
        var_name="Input",
        value_name="Recall (%)"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=df_plot,
        x="Recall (%)",
        y="Class",
        hue="Input",
        palette=["#326A9C", "#9BC2DE"],
        ax=ax
    )

    ax.set_title(
        "Classes Most Affected by Background Removal",
        fontsize=16, fontweight="bold",
        pad=20, loc="left"
    )
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.legend(title="", frameon=False)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="both", length=0, labelsize=10)

    plt.tight_layout()
    plt.show()

# 3.1 perturbation experiments
def perturb_image(image, idx, kind, strength, repetition=0, seed=42):
    if kind == "Brightness":
        return ImageEnhance.Brightness(image).enhance(strength)

    if kind == "Blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius=strength)
        )

    if kind == "Noise":
        if strength == 0:
            return image

        rng = np.random.default_rng(
            seed + idx + repetition * 100_000
        )

        array = np.asarray(image).astype(np.float32) / 255.0

        noise = rng.normal(
            0, strength, size=array.shape
        )

        array = np.clip(array + noise, 0, 1)

        return Image.fromarray(
            np.round(array * 255).astype(np.uint8)
        )

    raise ValueError(f"Unknown perturbation: {kind}")


# section 3.2 foreground vs background
def segment_leaf(image, image_size, seed=42):
    image = image.resize((image_size, image_size))
    array = np.asarray(image.convert("RGB")).copy()

    height, width = array.shape[:2]

    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))

    rectangle = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.setRNGSeed(seed)

    cv2.grabCut(
        cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
        mask,
        rectangle,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    return np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        1, 0
    ).astype(np.uint8)

def get_leaf_mask(idx, dataset, image_size, seed=42):
    image = dataset[int(idx)]["image"].convert("RGB")
    return segment_leaf(image, image_size, seed)


def region_transform(image, idx, mode, dataset, image_size, seed=42):
    image = image.resize((image_size, image_size))
    array = np.asarray(image.convert("RGB")).copy()

    mask = get_leaf_mask(idx, dataset, image_size, seed).astype(bool)
    replacement = np.array([128, 128, 128], dtype=np.uint8)

    if mode == "foreground":
        array[~mask] = replacement
    elif mode == "background":
        array[mask] = replacement
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Image.fromarray(array)