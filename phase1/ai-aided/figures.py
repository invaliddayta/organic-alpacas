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

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 0.5], hspace=0.32, wspace=0.28)
    ax2 = fig.add_subplot(gs[0, 0])
    vals = [cond["p_other_wrong_given_wrong"] * 100, cond["p_other_wrong_given_correct"] * 100]
    ax2.bar([0, 1], vals, color=[OI["vermillion"], OI["blue"]], width=0.6)
    for x, v, (k, n) in zip([0, 1], vals, [cond["pairs_wrong"], cond["pairs_correct"]]):
        ax2.text(x, v + 1, f"{v:.1f}%\n({k}/{n:,} pairs)", ha="center", fontsize=12.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["this photo\nis wrong", "this photo\nis correct"])
    ax2.set_ylabel("% of the other photos of the same leaf\nthat are also wrong")
    ax2.set_ylim(0, 50)
    ax2.set_title(f"If one photo of a leaf is wrong, other photos of the same\nleaf are often wrong too. "
                  f"{lc['n_errors']} errors fall on {lc['observed_leaves']} leaves\n"
                  f"(random placement: {lc['null_mean']:.0f}, 95% range {lc['null_lo']:.0f}-{lc['null_hi']:.0f})",
                  fontsize=13.5)

    ax3 = fig.add_subplot(gs[0, 1])
    for df, col, lab, s, mk in [(c, OI["orange"], f"Yellow leaf curl, correct (n={len(c)})", 10, "o"),
                                (a, OI["blue"], f"Bacterial spot, correct (n={len(a)})", 12, "o"),
                                (b, OI["black"], f"Bacterial spot predicted as\nyellow leaf curl (n={len(b)}, {b.leaf_id.nunique()} leaves)", 90, "X")]:
        ax3.scatter(df.leaf_b, df.leaf_L, s=s, c=col, marker=mk, alpha=1.0 if mk == "X" else 0.45,
                    label=lab, edgecolors="white" if mk == "X" else "none", linewidths=0.8)
    ax3.set_xlabel("leaf yellowness (mean Lab b* inside the leaf mask)")
    ax3.set_ylabel("leaf lightness (mean Lab L*)")
    ax3.set_title("Largest confusion (Tomato bacterial spot -> yellow leaf curl):\n"
                  "the misclassified leaves are paler and yellower, like yellow-leaf-curl leaves", fontsize=13.5)
    leg = ax3.legend(frameon=True, fontsize=11, loc="upper left", markerscale=1.0)
    for h in leg.legend_handles[:2]:
        h.set_sizes([60]); h.set_alpha(1)

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


def load_thumbs(side=128):
    import io
    from PIL import Image
    th = pd.read_parquet(pv.RESULTS / f"thumbs_{side}.parquet")
    return {int(i): np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for i, b in zip(th.sample_idx, th.jpeg)}


def thumb_strip(rows, thumbs, pad=4):
    """rows: list of lists of sample_idx -> one image with white padding."""
    side = next(iter(thumbs.values())).shape[0]
    ncol = max(len(r) for r in rows)
    out = np.full((len(rows) * (side + pad), ncol * (side + pad), 3), 255, np.uint8)
    for r, ids in enumerate(rows):
        for c, i in enumerate(ids):
            out[r * (side + pad):r * (side + pad) + side, c * (side + pad):c * (side + pad) + side] = thumbs[int(i)]
    return out


def tbs_strip(sh, thumbs):
    e = sh[(sh.true_idx == 28) & (sh.pred_idx == 35)].drop_duplicates("leaf_id")
    ok = sh[(sh.true_idx == 28) & sh.correct].sample(7, random_state=0)
    return thumb_strip([list(e.sample_idx), list(ok.sample_idx)], thumbs)


# ----------------------------------------------------------------------------- fig 3
def load_sweep(sh):
    sw = pd.read_parquet(pv.RESULTS / "perturbation_sweep.parquet")
    sw["true_idx"] = sw.sample_idx.map(sh.set_index("sample_idx").true_idx)
    ref = sw[sw.family == "none"].set_index("sample_idx").pred_idx
    sw["ref_pred"] = sw.sample_idx.map(ref)
    sw["correct"] = sw.pred_idx == sw.true_idx
    sw["flipped"] = sw.pred_idx != sw.ref_pred
    return sw


def sweep_summary(sw, names):
    """Per condition: accuracy (+95% CI), flip rate, top sink class and its share of wrong predictions."""
    import perturb
    _, _, short, _ = class_meta(names)
    rows = []
    for (fam, rank), g in sw.groupby(["family", "rank"]):
        wrong = g[~g.correct]
        vc = wrong.pred_idx.value_counts()
        lo, hi = viz.wilson(g.correct.sum(), len(g))
        rows.append({"family": fam, "rank": rank, "level": g.level.iloc[0],
                     "group": g.group.iloc[0], "n": len(g), "acc": g.correct.mean(),
                     "acc_lo": float(lo), "acc_hi": float(hi), "flip_rate": g.flipped.mean(),
                     "n_wrong": len(wrong),
                     "sink": short[vc.index[0]] if len(vc) else "", "sink_idx": int(vc.index[0]) if len(vc) else -1,
                     "sink_n": int(vc.iloc[0]) if len(vc) else 0,
                     "sink_share": float(vc.iloc[0] / len(wrong)) if len(vc) else np.nan,
                     "median_conf_wrong": float(wrong.confidence.median()) if len(wrong) else np.nan})
    df = pd.DataFrame(rows)
    order = {f: k for k, f in enumerate(perturb.FAMILIES)}
    df["forder"] = df.family.map(order).fillna(-1)
    return df.sort_values(["forder", "rank"]).reset_index(drop=True)


def fig_sweep(summary, path):
    import perturb
    fams = list(perturb.FAMILIES)
    base = summary[summary.family == "none"].iloc[0]
    A = np.full((len(fams), 4), np.nan)
    txt = [["" for _ in range(4)] for _ in fams]
    sink_txt = []
    for r, f in enumerate(fams):
        d = summary[summary.family == f].set_index("rank")
        for k in range(1, 5):
            if k in d.index:
                A[r, k - 1] = d.loc[k, "acc"]
                lv = d.loc[k, "level"]
                lvs = f"{lv:g}"
                txt[r][k - 1] = f"{lvs}\n{100 * d.loc[k, 'acc']:.0f}%"
        if 4 in d.index and d.loc[4, "n_wrong"] >= 20:
            s4 = d.loc[4]
            sink_txt.append(f"{s4['sink']}  {100 * s4['sink_share']:.0f}%  ({s4['sink_n']}/{s4['n_wrong']})")
        else:
            sink_txt.append("")
    fig, ax = plt.subplots(figsize=(15, 12.5))
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for r in range(len(fams)):
        for k in range(4):
            if txt[r][k]:
                ax.text(k, r, txt[r][k], ha="center", va="center", fontsize=11,
                        color="white" if A[r, k] < 0.6 else "black")
        ax.text(3.6, r, sink_txt[r], ha="left", va="center", fontsize=12)
    groups = [perturb.FAMILIES[f][2] for f in fams]
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"{f}  ({perturb.FAMILIES[f][3]})" for f in fams], fontsize=12)
    for t, g in zip(ax.get_yticklabels(), groups):
        t.set_color(viz.GROUP_COLOURS[g])
    for r in range(1, len(fams)):
        if groups[r] != groups[r - 1]:
            ax.axhline(r - 0.5, color="white", lw=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["strength 1", "2", "3", "4"])
    ax.xaxis.tick_top()
    ax.set_xlim(-0.5, 3.5)
    ax.text(3.6, -0.85, "most common wrong class at strength 4\n(share of all wrong predictions)",
            fontsize=12.5, fontweight="bold", va="bottom")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02, aspect=40)
    cb.set_label(f"accuracy on {int(base.n):,} test images (40 per class); unmodified: {100 * base.acc:.1f}%")
    ax.set_title("", pad=40)
    fig.suptitle("Simple capture changes break the 99% model, and the wrong predictions collapse into a few 'sink' classes",
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.955, "cell = parameter value and accuracy; row label colour = group "
             "(orange = lighting, pink = colour, blue = sharpness, green = geometry)", fontsize=12.5, ha="left")
    viz.save(fig, path)


# ----------------------------------------------------------------------------- fig 4
def dark_sink_numbers(sw, bg, bright, names, sink=30):
    """How often is a non-late-blight image predicted as Tomato late blight?"""
    out = {}
    s = sw[sw.true_idx != sink]
    for fam in ["darker", "low contrast"]:
        d = s[s.family.isin([fam, "none"])]
        g = d.groupby(["family", "rank", "level"]).apply(
            lambda x: pd.Series({"n": len(x), "k": int((x.pred_idx == sink).sum())}), include_groups=False).reset_index()
        out[fam] = g
    b = bg[bg.true_idx != sink]
    gb = b.groupby("condition").apply(lambda x: pd.Series({"n": len(x), "k": int((x.pred_idx == sink).sum())}),
                                      include_groups=False)
    out["background"] = gb
    tr = bright[bright.split == "train"]
    q = tr.luminance.quantile(0.01)
    dark = tr[tr.luminance <= q]
    out["train_dark_q"] = float(q)
    out["train_dark_n"] = len(dark)
    out["train_dark_counts"] = dark.class_idx.value_counts()
    out["train_share"] = tr.class_idx.value_counts(normalize=True)
    return out


def fig_dark_sink(nums, names, train_thumbs, sessions, path, sink=30):
    _, _, short, _ = class_meta(names)
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.36], hspace=0.45, wspace=0.42)

    # A: whole-image darkening vs leaf-only / background-only / leaf on black
    ax = fig.add_subplot(gs[0, 0])
    gb = nums["background"]
    conds = [("original", "unmodified", OI["grey"]),
             ("darker x0.25: leaf only", "darker x0.25: leaf only", OI["green"]),
             ("darker x0.25: background only", "darker x0.25: background only", OI["skyblue"]),
             ("darker x0.25: whole image", "darker x0.25: whole image", OI["blue"]),
             ("leaf on grey", "leaf on flat grey", OI["grey"]),
             ("leaf on black", "leaf on flat black", OI["black"]),
             ("background only", "leaf removed (background only)", OI["purple"])]
    vals, los, his, ns = [], [], [], []
    for key, _, _ in conds:
        k, n = gb.loc[key, "k"], gb.loc[key, "n"]
        lo, hi = viz.wilson(k, n)
        vals.append(100 * k / n); los.append(100 * (k / n - lo)); his.append(100 * (hi - k / n))
        ns.append((int(k), int(n)))
    y = np.arange(len(vals))
    ax.barh(y, vals, xerr=[los, his], color=[c[2] for c in conds], capsize=3)
    for yy, v, (k, n) in zip(y, vals, ns):
        ax.text(v + max(his) + 1.5, yy, f"{v:.1f}%  ({k:,}/{n:,})", va="center", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels([c[1] for c in conds], fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of images (true class is NOT Tomato late blight)\npredicted as Tomato late blight (95% CI)")
    ax.set_title("Same leaves, edited regions: a dark leaf alone (5%) or a dark\n"
                 "background alone (11%) rarely triggers it; a dark whole image does", fontsize=14)

    # B: training data
    ax2 = fig.add_subplot(gs[0, 1])
    cnt = nums["train_dark_counts"]
    share = nums["train_share"]
    top = cnt.head(5)
    yy = np.arange(len(top))
    ax2.barh(yy - 0.2, 100 * top.values / nums["train_dark_n"], 0.4, color=OI["black"],
             label=f"darkest 1% of training images (n={nums['train_dark_n']})")
    ax2.barh(yy + 0.2, 100 * share.reindex(top.index).values, 0.4, color=OI["grey"], label="all training images")
    for k, (c, n) in enumerate(top.items()):
        ax2.text(100 * n / nums["train_dark_n"] + 0.6, k - 0.2, f"{n}", va="center", fontsize=11.5)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([short[i] for i in top.index], fontsize=12.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("% of images in the group")
    ax2.legend(frameon=False, loc="lower right", fontsize=11.5)
    ax2.set_title("Training data: Tomato late blight is 8x over-represented\n"
                  "among the darkest images (28% vs 3.5%)", fontsize=14)

    ps = sessions["train_sessions"]["GHLB_PS"]
    axt = fig.add_subplot(gs[1, :])
    axt.imshow(train_thumbs)
    axt.axis("off")
    axt.set_title(f"The 8 darkest Tomato late blight training images. All come from one capture session (file names 'GHLB_PS ...'): "
                  f"{ps['n_in_darkest_1pct']} of its {ps['n_train']} training images\nare in the darkest 1% "
                  f"(median brightness {ps['median_luminance']:.0f} of 255; the other late blight sessions: about "
                  f"{sessions['train_sessions']['GHLB2']['median_luminance']:.0f})", fontsize=12.5)
    fig.suptitle("Darkened images become 'Tomato late blight'; a dark late-blight capture session in the training data "
                 "is a likely source", x=0.02, ha="left", y=1.0)
    viz.save(fig, path)


# ----------------------------------------------------------------------------- fig 5
def border_knn_on(sample_ids, k=15):
    """Border-colour-only k-NN (input statistic, not our model) evaluated on given test images."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    bc = pd.read_parquet(pv.RESULTS / "border_colour.parquet")
    tr, te = bc[bc.split == "train"], bc[bc.split == "test"].set_index("idx_in_split")
    sc = StandardScaler().fit(tr[["L", "a", "b"]])
    knn = KNeighborsClassifier(n_neighbors=k).fit(sc.transform(tr[["L", "a", "b"]]), tr.class_idx)
    x = te.loc[sample_ids]
    pred = knn.predict(sc.transform(x[["L", "a", "b"]]))
    return float((pred == x.class_idx.to_numpy()).mean()), len(x)


def background_summary(bg):
    rows = []
    for cond, d in bg.groupby("condition"):
        n = len(d)
        leaf = int((d.pred_idx == d.true_idx).sum())
        donor = int(((d.pred_idx == d.donor_class) & (d.donor_class != d.true_idx)).sum())
        rows.append({"condition": cond, "n": n, "leaf_class": leaf, "donor_class": donor,
                     "other": n - leaf - donor, "acc": leaf / n})
    return pd.DataFrame(rows).set_index("condition")


def fig_background(bg, names, path):
    _, _, short, _ = class_meta(names)
    S = background_summary(bg)
    ids = bg[bg.condition == "original"].sample_idx.unique()
    knn_acc, knn_n = border_knn_on(ids)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(19, 8.2), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.6))

    # A: background alone
    items = [("border colour only: 15-NN on training images\n(3 numbers per image; input statistic, NOT our model)",
              knn_acc, knn_n, OI["grey"]),
             ("our model, leaf removed\n(inpainted background)", S.loc["background only", "acc"],
              S.loc["background only", "n"], OI["purple"]),
             ("our model, leaf-free corner patches only", S.loc["background corners only", "acc"],
              S.loc["background corners only", "n"], OI["purple"])]
    y = np.arange(len(items))
    for k, (lab, a, n, c) in enumerate(items):
        lo, hi = viz.wilson(a * n, n)
        ax.barh(k, 100 * a, color=c, xerr=[[100 * (a - lo)], [100 * (hi - a)]], capsize=4)
        ax.text(100 * hi + 1.5, k, f"{100 * a:.0f}%  (n={n:,})", va="center", fontsize=12.5)
    ax.axvline(100 / 38, color=OI["vermillion"], ls="--", lw=2)
    ax.text(100 / 38 + 0.8, -0.42, "chance 2.6%", color=OI["vermillion"], fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("% predicted as the image's own class (95% CI)")
    ax.set_title("Without the leaf: the background holds a lot of class\ninformation; our model recovers part of it", fontsize=14)

    # B: swaps and flat backgrounds
    conds = [("original", "unmodified"), ("swap: same class", "background from another image\nof the SAME class (control)"),
             ("swap: other class", "background from an image\nof ANOTHER class"),
             ("leaf on grey", "flat grey background"), ("leaf on white", "flat white background"),
             ("leaf on black", "flat black background")]
    yy = np.arange(len(conds))
    left = np.zeros(len(conds))
    parts = [("leaf_class", "leaf's own class (correct)", OI["blue"]),
             ("donor_class", "class of the background's image", OI["orange"]),
             ("other", "some other class", OI["grey"])]
    for col, lab, c in parts:
        v = np.array([100 * S.loc[k, col] / S.loc[k, "n"] for k, _ in conds])
        ax2.barh(yy, v, left=left, color=c, label=lab)
        for j, (vv, l0) in enumerate(zip(v, left)):
            if vv >= 4:
                ax2.text(l0 + vv / 2, j, f"{vv:.0f}%", ha="center", va="center", fontsize=12,
                         color="white" if c != OI["orange"] else "black")
            elif vv >= 0.5 and col == "donor_class":
                ax2.annotate(f"{vv:.0f}% background's class", (l0 + vv / 2, j - 0.4), (l0 - 12, j - 0.62),
                             fontsize=11, ha="center", va="bottom",
                             arrowprops=dict(arrowstyle="-", color=OI["black"], lw=1))
        left += v
    for j, (k, _) in enumerate(conds):
        ax2.text(101, j, f"n={S.loc[k, 'n']:,}", va="center", fontsize=11)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([c[1] for c in conds], fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% of predictions")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, frameon=False, fontsize=11.5)
    ax2.set_title("Same leaf, new background: a foreign or flat background changes\n"
                  "the prediction; the same-class swap is the control for pasting", fontsize=14)
    a_bo = S.loc["background only", "acc"]
    a_same, a_other = S.loc["swap: same class", "acc"], S.loc["swap: other class", "acc"]
    fig.suptitle(f"The background carries class information and the model uses part of it: a foreign background lowers accuracy "
                 f"from {100 * a_same:.0f}% to {100 * a_other:.0f}%,\nand the background alone gives the right class in "
                 f"{100 * a_bo:.0f}% of cases (chance 2.6%)", x=0.02, ha="left", y=1.06)
    viz.save(fig, path)
    return S, knn_acc


# ----------------------------------------------------------------------------- supporting figures
def fig_class_vulnerability(sw, names, path, rank=3):
    import perturb
    _, _, short, order = class_meta(names)
    fams = list(perturb.FAMILIES)
    d = sw[(sw["rank"] == rank) & sw.family.isin(fams)]
    M = d.groupby(["true_idx", "family"]).flipped.mean().unstack().reindex(index=order, columns=fams)
    base_err = sw[sw.family == "none"].groupby("true_idx").correct.mean().reindex(order)
    fig, ax = plt.subplots(figsize=(17, 14))
    im = ax.imshow(100 * M.to_numpy(), cmap="magma_r", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(38))
    ax.set_yticklabels([short[i] for i in order], fontsize=11.5)
    lv = {f: perturb.FAMILIES[f][1][rank - 1] for f in fams}
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f"{f} ({lv[f]:g})" for f in fams], rotation=50, ha="right", fontsize=12)
    for r in range(38):
        for c in range(len(fams)):
            v = 100 * M.iat[r, c]
            if v >= 50:
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white")
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label("% of the class's images whose prediction changes (n=40 per class)")
    worst = M.mean(1).sort_values(ascending=False)
    ax.set_title(f"Per class and change (strength {rank}): fragility differs a lot between classes. Most fragile on average: "
                 f"{short[worst.index[0]]} ({100 * worst.iloc[0]:.0f}%), {short[worst.index[1]]} ({100 * worst.iloc[1]:.0f}%); "
                 f"least: {short[worst.index[-1]]} ({100 * worst.iloc[-1]:.0f}%)", fontsize=13, loc="left")
    viz.save(fig, path)
    return M, worst


def fig_nearest_train(sh, nn, path):
    d = sh[["sample_idx", "true_idx", "pred_idx", "correct", "confidence"]].merge(nn, on="sample_idx")
    d["pct"] = d.groupby("true_idx").best_sim_same_class.rank(pct=True)
    d["quintile"] = pd.cut(d.pct, [0, .2, .4, .6, .8, 1.0], labels=["1 (least similar)", "2", "3", "4", "5 (most similar)"])
    q = d.groupby("quintile", observed=True).agg(n=("correct", "size"), err=("correct", lambda s: int((~s).sum())))
    d["closer"] = np.where(d.best_sim_other_class > d.best_sim_same_class,
                           "an image of ANOTHER class", "an image of its OWN class")
    c = d.groupby("closer").agg(n=("correct", "size"), err=("correct", lambda s: int((~s).sum())))
    dup = d[d.nn_sim > 0.93]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw=dict(wspace=0.35))
    for a, t, xl in [(ax, q, "within-class quintile of similarity to the nearest\nsame-class training image (32x32 pixels)"),
                     (ax2, c, "the most similar training image (pixel space) is ...")]:
        rate = 100 * t.err / t.n
        lo, hi = viz.wilson(t.err, t.n)
        a.bar(range(len(t)), rate, color=OI["blue"], yerr=[rate - 100 * lo, 100 * hi - rate], capsize=4)
        for k, (e, n) in enumerate(zip(t.err, t.n)):
            a.text(k, 100 * hi[k] * 1.03, f"{e}/{n:,}", ha="center", fontsize=11.5)
        a.set_ylim(0, 100 * hi.max() * 1.22)
        a.set_xticks(range(len(t)))
        a.set_xticklabels(t.index, fontsize=12)
        a.set_xlabel(xl)
        a.set_ylabel("error rate (%) with 95% CI")
    ax.set_title("Similarity to the training set barely changes the error rate", fontsize=14)
    ax2.set_title("Errors are 4x more common when the closest training\nimage belongs to another class", fontsize=14)
    fig.suptitle(f"Near-duplicates do not explain the 99%: only {len(dup)} test images have a near-copy in training "
                 f"(r > 0.93), {int((~dup.correct).sum())} of them wrong", x=0.02, ha="left", y=1.04)
    viz.save(fig, path)
    return q, c, dup


BG_EXCLUDE = {10}   # Corn healthy: close-ups where the leaf fills the frame (green border), so there is no background to test


def load_background():
    """Background test results without classes that have no visible background (visual check, see README)."""
    bg = pd.read_parquet(pv.RESULTS / "background_tests.parquet")
    return bg[~bg.true_idx.isin(BG_EXCLUDE) & ~bg.donor_class.isin(BG_EXCLUDE)].reset_index(drop=True)
