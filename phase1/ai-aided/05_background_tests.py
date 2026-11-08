"""Background tests (outputs only). Same stratified subset as the perturbation sweep
(40 test images per class, seed 0), restricted to images whose leaf mask is usable
(visible background, sensible leaf size; see segment.usable).

Conditions per image
- leaf on grey / black / white: background replaced by a flat colour
  (grey = median border colour of all training images);
- background only: leaf removed (dilated mask inpainted from the surrounding background);
- background corners only: the leaf-free 48 px corner patches tiled to a full image (does not
  depend on the mask being exact inside the image);
- swap (other class) x2: the leaf pasted onto the background of a random usable image
  of another class (whose leaf was removed the same way);
- swap (same class): same, donor from the same class but another image (control for
  pasting artefacts);
- darker / brighter applied to the whole image, to the leaf only or to the background only
  (brightness x0.25 and x2.2, strength 4 of the sweep): which region triggers the sink class?
Writes results/background_tests.parquet (one row per image and condition, with the
probability of the leaf's class and of the donor background's class) and
results/background_masks_meta.parquet. Runtime about 15 min.
"""
import time

import numpy as np
import pandas as pd

import perturb
import pv
import segment as sg

PER_CLASS = 40


def main():
    t0 = time.time()
    _, test = pv.load_splits()
    labels = np.asarray(test["class_idx"])
    idx = pv.stratified_subset(labels, PER_CLASS, seed=0)
    images = {int(i): test[int(i)]["image"].convert("RGB") for i in idx}

    masks, meta = {}, []
    for i in idx:
        m = sg.leaf_mask(images[int(i)])
        q = sg.mask_quality(m)
        masks[int(i)] = m
        meta.append({"sample_idx": int(i), "true_idx": int(labels[i]), **q, "usable": sg.usable(q)})
    meta = pd.DataFrame(meta)
    meta.to_parquet(pv.RESULTS / "background_masks_meta.parquet", index=False)
    use = meta[meta.usable]
    print(f"masks done ({time.time() - t0:.0f}s): {len(use)}/{len(meta)} usable", flush=True)

    bc = pd.read_parquet(pv.RESULTS / "border_colour.parquet")
    tr = bc[bc.split == "train"][["L", "a", "b"]].median().to_numpy()
    import cv2
    grey = cv2.cvtColor(np.array([[[tr[0], tr[1], tr[2]]]], np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
    colours = {"leaf on grey": tuple(int(v) for v in grey), "leaf on black": (0, 0, 0),
               "leaf on white": (255, 255, 255)}
    print("grey background colour (RGB):", colours["leaf on grey"])

    rng = np.random.default_rng(1)
    ids = use.sample_idx.to_numpy()
    cls = use.true_idx.to_numpy()
    bg_only = {int(i): sg.background_only(images[int(i)], masks[int(i)]) for i in ids}
    print(f"inpainting done ({time.time() - t0:.0f}s)", flush=True)

    # jobs hold a recipe, not the image, so that memory stays small (images are built lazily)
    jobs = []  # (sample_idx, condition, donor_idx, make_image)
    for i, c in zip(ids, cls):
        i = int(i)
        im, m = images[i], masks[i]
        jobs.append((i, "original", -1, lambda im=im: im))
        for name, col in colours.items():
            jobs.append((i, name, -1, lambda im=im, m=m, col=col: sg.leaf_on_colour(im, m, col)))
        jobs.append((i, "background only", i, lambda i=i: bg_only[i]))
        tex = sg.corner_texture(im, m)
        if tex is not None:  # leaf-free corner patches only (robust to mask errors)
            jobs.append((i, "background corners only", i, lambda tex=tex: tex))
        other = ids[cls != c]
        for d in rng.choice(other, 2, replace=False):
            jobs.append((i, "swap: other class", int(d),
                         lambda im=im, m=m, d=int(d): sg.leaf_on_background(im, m, bg_only[d])))
        for fam, f in [("darker", 0.25), ("brighter", 2.2)]:
            jobs.append((i, f"{fam} x{f}: whole image", -1, lambda im=im, f=f: perturb.brightness(im, f)))
            jobs.append((i, f"{fam} x{f}: leaf only", -1,
                         lambda im=im, m=m, f=f: sg.leaf_on_background(perturb.brightness(im, f), m, im)))
            jobs.append((i, f"{fam} x{f}: background only", -1,
                         lambda im=im, m=m, f=f: sg.leaf_on_background(im, m, perturb.brightness(im, f))))
        same = ids[(cls == c) & (ids != i)]
        if len(same):
            d = int(rng.choice(same))
            jobs.append((i, "swap: same class", d, lambda im=im, m=m, d=d: sg.leaf_on_background(im, m, bg_only[d])))
    print(f"{len(jobs)} model calls", flush=True)

    model = pv.load_model()
    probs = pv.predict_images(model, (j[3]() for j in jobs))
    s = pv.summarize_probs(probs)
    true = labels[[j[0] for j in jobs]]
    donor = np.array([j[2] for j in jobs])
    donor_cls = np.where(donor >= 0, labels[np.maximum(donor, 0)], -1)
    out = pd.DataFrame({
        "sample_idx": [j[0] for j in jobs], "condition": [j[1] for j in jobs],
        "donor_idx": donor, "true_idx": true, "donor_class": donor_cls,
        "pred_idx": s.pred_idx, "confidence": s.confidence, "second_idx": s.second_idx,
        "p_true": probs[np.arange(len(jobs)), true],
        "p_donor": np.where(donor_cls >= 0, probs[np.arange(len(jobs)), np.maximum(donor_cls, 0)], np.nan),
    })
    out.to_parquet(pv.RESULTS / "background_tests.parquet", index=False)
    print(out.groupby("condition").apply(lambda d: pd.Series({
        "n": len(d), "acc": (d.pred_idx == d.true_idx).mean(),
        "pred=donor": (d.pred_idx == d.donor_class).mean()}), include_groups=False))
    print(f"done {time.time() - t0:.0f}s")

    # small visual check sheet for the report (not committed)
    from PIL import Image
    ex = [j for j in jobs if j[0] in set(ids[:: max(1, len(ids) // 6)][:6])]
    tiles = [np.asarray(j[3]().resize((112, 112))) for j in ex[:6 * 15]]
    rows = [np.concatenate(tiles[r * 15:(r + 1) * 15], 1) for r in range(len(tiles) // 15)]
    (pv.HERE / "cache").mkdir(exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(pv.HERE / "cache" / "background_examples.png")


if __name__ == "__main__":
    main()
