"""Figure functions for phase1_ai_aided.ipynb. Every function reads cached results only
(no model inference), draws one figure, saves it to figures/ and returns the numbers
that the notebook text quotes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pv
import viz
from viz import OI

viz.style()

CROP_ORDER = ["Tomato", "Potato", "Pepper,_bell", "Apple", "Peach", "Cherry_(including_sour)",
              "Strawberry", "Raspberry", "Blueberry", "Grape", "Orange", "Soybean", "Squash",
              "Corn_(maize)"]


# ----------------------------------------------------------------------------- data
def load_shared():
    sh = pd.read_csv(pv.SHARED_PRED)
    names = sh.drop_duplicates("true_idx").sort_values("true_idx").true_label.tolist()
    return sh, names


def class_meta(names):
    hosts = np.array([pv.host_of(n) for n in names])
    dis = np.array([n.split("___")[1].replace("_", " ").lower().strip() for n in names])
    short = [pv.short_name(n) for n in names]
    order = np.array(sorted(range(len(names)), key=lambda i: (
        CROP_ORDER.index(hosts[i]), dis[i] == "healthy", dis[i])))
    return hosts, dis, short, order


def relation(a, b, hosts, dis):
    if a == b:
        return "same class"
    if hosts[a] == hosts[b]:
        return "same crop"
    if dis[a] == dis[b]:
        return "healthy -> healthy\n(other crop)" if dis[a] == "healthy" else "same disease name\n(other crop)"
    return "unrelated"


REL_ORDER = ["same crop", "same disease name\n(other crop)", "healthy -> healthy\n(other crop)", "unrelated"]


def relation_table(sh, names):
    """Share of (a) wrong predictions, (b) runner-up classes of correct predictions and
    (c) a uniformly random other class, per relation to the true class."""
    hosts, dis, _, _ = class_meta(names)
    P = sh[[f"prob_{i}" for i in range(38)]].to_numpy()
    y, p = sh.true_idx.to_numpy(), sh.pred_idx.to_numpy()
    ok = sh.correct.to_numpy()
    second = np.argsort(-P, 1)[:, 1]
    err = pd.Series([relation(a, b, hosts, dis) for a, b in zip(y[~ok], p[~ok])])
    run = pd.Series([relation(a, b, hosts, dis) for a, b in zip(y[ok], second[ok])])
    rel = np.array([[relation(a, b, hosts, dis) for b in range(38)] for a in range(38)])
    chance = pd.Series({r: np.mean([(rel[a][np.arange(38) != a] == r).mean() for a in y[ok]])
                        for r in REL_ORDER})
    tab = pd.DataFrame({
        "wrong predictions": err.value_counts().reindex(REL_ORDER, fill_value=0),
        "runner-up (correct predictions)": run.value_counts().reindex(REL_ORDER, fill_value=0),
    })
    shares = tab / tab.sum()
    shares["random other class"] = chance
    return tab, shares


# ----------------------------------------------------------------------------- fig 1
def fig_soft_confusion(sh, names, path):
    hosts, dis, short, ordr = class_meta(names)
    P = sh[[f"prob_{i}" for i in range(38)]].to_numpy()
    y = sh.true_idx.to_numpy()
    M = np.stack([P[y == t].mean(0) for t in range(38)])
    np.fill_diagonal(M, np.nan)
    tab, shares = relation_table(sh, names)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(21, 12), gridspec_kw=dict(width_ratios=[2.9, 1], wspace=0.42))
    L = np.log10(M[np.ix_(ordr, ordr)])
    im = ax.imshow(L, cmap="magma_r", vmin=-6, vmax=-1)
    ax.set_xticks(range(38))
    ax.set_xticklabels([short[i] for i in ordr], rotation=90, fontsize=11)
    ax.set_yticks(range(38))
    ax.set_yticklabels([short[i] for i in ordr], fontsize=11)
    hs = [hosts[i] for i in ordr]
    start = 0
    for k in range(1, 39):
        if k == 38 or hs[k] != hs[start]:
            ax.add_patch(plt.Rectangle((start - .5, start - .5), k - start, k - start,
                                       fill=False, ec=OI["blue"], lw=2.2))
            start = k
    for r, a in enumerate(ordr):
        for c, b in enumerate(ordr):
            if hosts[a] != hosts[b] and dis[a] == dis[b] and dis[a] != "healthy":
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=OI["skyblue"], lw=2.6))
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.015)
    cb.set_ticks([-6, -5, -4, -3, -2, -1])
    cb.set_ticklabels(["1e-6", "1e-5", "1e-4", "1e-3", "0.01", "0.1"])
    cb.set_label("mean probability given to the wrong class (log scale)")
    ax.set_xlabel("class that receives the probability")
    ax.set_ylabel("true class")
    ax.set_title("Mean softmax probability per (true class, wrong class), all 10,948 test images\n"
                 "dark-blue boxes = crops; light-blue cells = same disease name in another crop",
                 fontsize=13.5)

    cols = {"wrong predictions": OI["vermillion"], "runner-up (correct predictions)": OI["blue"],
            "random other class": OI["grey"]}
    yy = np.arange(len(REL_ORDER))
    h = 0.26
    for k, (col, c) in enumerate(cols.items()):
        vals = shares[col].to_numpy() * 100
        n = int(tab[col].sum()) if col in tab else None
        lab = f"{col} (n={n:,})" if n else f"{col} (chance)"
        ax2.barh(yy + (k - 1) * h, vals, h, color=c, label=lab)
        for yv, v in zip(yy + (k - 1) * h, vals):
            ax2.text(v + 1, yv, f"{v:.0f}%", va="center", fontsize=11.5)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(REL_ORDER, fontsize=12.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("% of cases")
    ax2.set_title("How is the wrong / second class related\nto the true class?", fontsize=14)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), frameon=False, fontsize=12)
    ax2.set_box_aspect(1.25)
    n_err = int(tab["wrong predictions"].sum())
    fig.suptitle(f"Confusions follow the crop; across crops they keep the disease. "
                 f"{tab['wrong predictions'].iloc[0]} of {n_err} errors stay in the crop (chance 10%),\n"
                 f"{tab['wrong predictions'].iloc[1]} of the {n_err - tab['wrong predictions'].iloc[0]} cross-crop errors "
                 f"are potato <-> tomato blights with the same disease name",
                 x=0.08, y=0.95, ha="left", fontsize=17)
    viz.save(fig, path)
    return tab, shares


# ----------------------------------------------------------------------------- fig 2
def leaf_clustering(sh, n_sim=5000, seed=0):
    """Errors among leaf-grouped images: distinct physical leaves observed vs. expected if
    the same number of errors per class fell on random images of that class."""
    g = sh[sh.leaf_grouped].copy()
    e = g[~g.correct]
    obs = e.leaf_id.nunique()
    rng = np.random.default_rng(seed)
    byc = {c: d.leaf_id.to_numpy() for c, d in g.groupby("true_idx")}
    k = e.groupby("true_idx").size()
    sims = np.array([len(set().union(*[set(rng.choice(byc[c], kc, replace=False)) for c, kc in k.items()]))
                     for _ in range(n_sim)])
    return {"n_grouped_images": len(g), "n_leaves": g.leaf_id.nunique(), "n_errors": len(e),
            "observed_leaves": obs, "null_mean": sims.mean(), "null_lo": np.percentile(sims, 2.5),
            "null_hi": np.percentile(sims, 97.5), "p_value": (np.sum(sims <= obs) + 1) / (n_sim + 1),
            "sims": sims, "errors_per_leaf": e.leaf_id.value_counts()}


def same_leaf_conditional(sh):
    """P(another photo of the same leaf is wrong | this photo is wrong / correct)."""
    g = sh[sh.leaf_grouped]
    le = g.groupby("leaf_id").agg(n=("correct", "size"), err=("correct", lambda s: (~s).sum()))
    le = le[le.n >= 2]
    w = (le.err * (le.err - 1)).sum(), (le.err * (le.n - 1)).sum()
    c = ((le.n - le.err) * le.err).sum(), ((le.n - le.err) * (le.n - 1)).sum()
    return {"p_other_wrong_given_wrong": w[0] / w[1], "pairs_wrong": (int(w[0]), int(w[1])),
            "p_other_wrong_given_correct": c[0] / c[1], "pairs_correct": (int(c[0]), int(c[1])),
            "error_rate_grouped": float((~g.correct).mean())}


def fig_errors_by_leaf(sh, names, desc, thumbs, path):
    """Left: errors concentrate on few physical leaves (vs. random placement).
    Right: the 18 Tomato bacterial spot -> yellow leaf curl errors are pale, yellow leaves."""
    lc = leaf_clustering(sh)
    cond = same_leaf_conditional(sh)
    d = sh[["sample_idx", "true_idx", "pred_idx", "correct", "leaf_id"]].merge(desc, on="sample_idx")
    tbs, ylcv = 28, 35
    a = d[(d.true_idx == tbs) & d.correct]
    b = d[(d.true_idx == tbs) & (d.pred_idx == ylcv)]
    c = d[(d.true_idx == ylcv) & d.correct]

    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.05, 1.05], height_ratios=[1, 0.42], hspace=0.38, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    sims = lc["sims"]
    ax.hist(sims, bins=np.arange(sims.min() - .5, sims.max() + 1.5), color=OI["grey"],
            label="random placement\n(5,000 simulations)")
    ax.axvline(lc["observed_leaves"], color=OI["vermillion"], lw=3, label="observed")
    ax.set_xlim(40, 95)
    ax.set_xlabel("number of distinct leaves among the errors")
    ax.set_ylabel("simulations")
    ax.set_title(f"{lc['n_errors']} errors on leaf-grouped images come from\n"
                 f"{lc['observed_leaves']} leaves; random placement: "
                 f"{lc['null_mean']:.0f} ({lc['null_lo']:.0f}-{lc['null_hi']:.0f})", fontsize=14)
    ax.legend(frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[0, 1])
    vals = [cond["p_other_wrong_given_wrong"] * 100, cond["p_other_wrong_given_correct"] * 100]
    ax2.bar([0, 1], vals, color=[OI["vermillion"], OI["blue"]], width=0.6)
    for x, v, (k, n) in zip([0, 1], vals, [cond["pairs_wrong"], cond["pairs_correct"]]):
        ax2.text(x, v + 1, f"{v:.1f}%\n({k}/{n:,} pairs)", ha="center", fontsize=12.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["this photo\nis wrong", "this photo\nis correct"])
    ax2.set_ylabel("% of other photos of the same leaf\nthat are also wrong")
    ax2.set_ylim(0, 50)
    ax2.set_title("If one photo of a leaf is wrong, other photos\nof the same leaf are often wrong too", fontsize=14)

    ax3 = fig.add_subplot(gs[0, 2])
    for df, col, lab, s, mk in [(c, OI["skyblue"], f"Yellow leaf curl, correct (n={len(c)})", 10, "o"),
                                (a, OI["green"], f"Bacterial spot, correct (n={len(a)})", 12, "o"),
                                (b, OI["vermillion"], f"Bacterial spot -> predicted yellow leaf curl (n={len(b)})", 70, "X")]:
        ax3.scatter(df.leaf_b, df.leaf_L, s=s, c=col, marker=mk, alpha=0.9 if mk == "X" else 0.45,
                    label=lab, edgecolors="black" if mk == "X" else "none", linewidths=0.6)
    ax3.set_xlabel("leaf yellowness (mean Lab b*, from the leaf mask)")
    ax3.set_ylabel("leaf lightness (mean Lab L*)")
    ax3.set_title("Tomato: the 18 bacterial spot leaves called\n'yellow leaf curl' are paler and yellower", fontsize=14)
    ax3.legend(frameon=False, fontsize=10.5, loc="lower right", markerscale=1.3)

    # thumbnails: misclassified (top) vs correctly classified bacterial spot
    axt = fig.add_subplot(gs[1, :])
    axt.imshow(thumbs)
    axt.axis("off")
    axt.set_title("Top row: Tomato bacterial spot predicted as yellow leaf curl (one photo per leaf, 7 leaves).  "
                  "Bottom row: correctly classified bacterial spot (random).", fontsize=13)
    fig.suptitle("Errors are a property of the leaf, not of the photo: they cluster on a few leaves "
                 "with an unusual look", x=0.05, ha="left", y=1.0)
    viz.save(fig, path)
    stats = {"tbs_correct_b": a.leaf_b.median(), "tbs_err_b": b.leaf_b.median(), "ylcv_b": c.leaf_b.median(),
             "tbs_correct_L": a.leaf_L.median(), "tbs_err_L": b.leaf_L.median(), "ylcv_L": c.leaf_L.median(),
             "n_tbs_err": len(b), "n_tbs_err_leaves": b.leaf_id.nunique()}
    return lc, cond, stats
