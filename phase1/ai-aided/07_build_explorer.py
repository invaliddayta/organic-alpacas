"""Builds behaviour_explorer.html: an offline, self-contained interactive view
(vanilla JavaScript + SVG, no external libraries, thumbnails embedded as base64).

Linked views:
1. condition grid (perturbation family x strength, background tests, original test set),
   coloured by accuracy; click to select a condition;
2. 38x38 matrix true class x predicted class for that condition (diagonal hidden),
   hover for counts, click a cell to see the images;
3. "where do the wrong predictions go" bar chart (column sums = sink classes), click a bar;
4. thumbnail panel with confidence, plus a preview of the selected perturbation.
Reads only cached results (no dataset, no model).
"""
import base64
import json

import pandas as pd

import figures
import perturb
import pv

OUT = pv.HERE / "behaviour_explorer.html"


def b64(x):
    return base64.b64encode(x).decode("ascii")


def condition_payload(df, true_col="true_idx"):
    """Off-diagonal cells -> [[sample_idx, confidence], ...] and per-class totals."""
    cells = {}
    wrong = df[df.pred_idx != df[true_col]]
    for (t, p), g in wrong.groupby([true_col, "pred_idx"]):
        cells[f"{int(t)},{int(p)}"] = [[int(i), round(float(c), 3)] for i, c in zip(g.sample_idx, g.confidence)]
    n_per_class = df.groupby(true_col).size().reindex(range(38), fill_value=0).astype(int).tolist()
    return {"cells": cells, "n_per_class": n_per_class, "n": int(len(df)),
            "acc": float((df.pred_idx == df[true_col]).mean())}


def main():
    sh, names = figures.load_shared()
    hosts, dis, short, order = figures.class_meta(names)
    true_of = sh.set_index("sample_idx").true_idx

    conds = []
    p = condition_payload(sh[["sample_idx", "true_idx", "pred_idx", "confidence"]])
    conds.append({"id": "test", "label": "Original test set (all 10,948 images)", "group": "test",
                  "family": "none", "rank": 0, "level": "", **p})

    sw = pd.read_parquet(pv.RESULTS / "perturbation_sweep.parquet")
    sw["true_idx"] = sw.sample_idx.map(true_of)
    ref = sw[sw.family == "none"]
    conds.append({"id": "ref", "label": "Sweep subset, unmodified (1,512 images, 40 per class)",
                  "group": "subset", "family": "none", "rank": 0, "level": "", **condition_payload(ref)})
    for fam, rank, level, group in perturb.conditions():
        d = sw[(sw.family == fam) & (sw["rank"] == rank)]
        conds.append({"id": f"{fam}|{rank}", "label": f"{fam}: {perturb.FAMILIES[fam][3]} = {level}",
                      "group": group, "family": fam, "rank": rank, "level": level, **condition_payload(d)})

    bgf = pv.RESULTS / "background_tests.parquet"
    if bgf.exists():
        bg = figures.load_background()  # without Corn healthy (no visible background)
        for k, cond in enumerate(["original", "leaf on grey", "leaf on black", "leaf on white",
                                  "background only", "background corners only", "swap: same class", "swap: other class",
                                  "darker x0.25: whole image", "darker x0.25: leaf only",
                                  "darker x0.25: background only", "brighter x2.2: whole image",
                                  "brighter x2.2: leaf only", "brighter x2.2: background only"]):
            d = bg[bg.condition == cond]
            conds.append({"id": f"bg|{cond}", "label": f"background test: {cond} ({len(d):,} images; masks usable, Corn healthy excluded)",
                          "group": "background", "family": cond, "rank": k, "level": "", **condition_payload(d)})

    th = pd.read_parquet(pv.RESULTS / "thumbs_64.parquet")
    thumbs = {int(i): b64(j) for i, j in zip(th.sample_idx, th.jpeg)}
    ex = pd.read_parquet(pv.RESULTS / "perturb_examples.parquet")
    examples = {}
    for fam, g in ex.groupby("family"):
        examples[fam] = [[float(lv), b64(j)] for lv, j in zip(g.level, g.jpeg)]

    data = {
        "classes": [{"name": names[i], "short": short[i], "host": hosts[i]} for i in range(38)],
        "order": [int(i) for i in order],
        "true_of": {int(k): int(v) for k, v in true_of.items() if int(k) in thumbs},
        "conditions": conds, "thumbs": thumbs, "examples": examples,
        "families": [[f, v[2], v[3], v[1]] for f, v in perturb.FAMILIES.items()],
    }
    html = TEMPLATE.replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB, {len(conds)} conditions, {len(thumbs)} thumbnails")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PlantVillage ResNet-18: behaviour explorer (Phase I, AI-aided pair)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:12px;color:#222;background:#fff}
h1{font-size:20px;margin:0 0 4px} .sub{font-size:13px;color:#555;margin-bottom:10px;max-width:1500px}
#wrap{display:flex;gap:14px;align-items:flex-start}
#left{width:330px;flex:none} #mid{flex:none} #right{flex:1;min-width:300px}
table.grid{border-collapse:collapse;font-size:12px} table.grid td,table.grid th{padding:2px 4px;text-align:center}
table.grid td.c{cursor:pointer;width:40px;border:1px solid #fff;color:#000}
table.grid td.c.sel{outline:3px solid #000}
table.grid td.f{text-align:right;white-space:nowrap}
.btn{display:block;margin:3px 0;padding:4px 6px;border:1px solid #888;background:#f4f4f4;cursor:pointer;font-size:12px;text-align:left;width:100%}
.btn.sel{background:#0072B2;color:#fff}
#tip{position:fixed;pointer-events:none;background:#222;color:#fff;font-size:12px;padding:4px 6px;border-radius:3px;display:none;z-index:9}
#thumbs{display:flex;flex-wrap:wrap;gap:4px;max-height:640px;overflow-y:auto}
.th{font-size:10px;text-align:center;width:70px} .th img{width:64px;height:64px;display:block;margin:auto}
.legend{font-size:11px;color:#444}
#preview img{width:72px;height:72px;margin-right:3px} #preview{font-size:11px;margin-bottom:6px}
</style></head><body>
<h1>How does the model behave? Behaviour explorer for the PlantVillage ResNet-18</h1>
<div class="sub">Black-box view: only inputs and predictions. <b>1.</b> Pick a condition on the left (cell colour = accuracy).
<b>2.</b> The matrix shows true class (rows) vs predicted class (columns) for that condition; the diagonal (correct) is hidden,
so every coloured cell is a confusion. Hover for counts, click a cell to see the images. <b>3.</b> The bars under the matrix show where
wrong predictions go (sink classes); click a bar. Thumbnails show the <i>unmodified</i> image; the strip above them previews the selected change on one example leaf.
Perturbation results use a stratified subset of 40 test images per class (32 for Potato healthy).</div>
<div id="wrap">
 <div id="left"><div id="buttons"></div><table class="grid" id="grid"></table>
  <div class="legend" style="margin-top:6px">cell colour: accuracy (dark = low). number: accuracy in %.</div></div>
 <div id="mid"><div id="ctitle" style="font-weight:600;font-size:14px;margin-bottom:4px"></div><svg id="mat"></svg></div>
 <div id="right"><div id="preview"></div><div id="ttitle" style="font-weight:600;font-size:13px;margin-bottom:4px">Click a matrix cell or a bar.</div><div id="thumbs"></div></div>
</div>
<div id="tip"></div>
<script>
const D = /*DATA*/;
const C = D.classes, O = D.order, N = 38;
const cond = {}; D.conditions.forEach(c => cond[c.id] = c);
let cur = "test";
const tip = document.getElementById("tip");
function showTip(e, html){tip.innerHTML = html; tip.style.display = "block"; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";}
function hideTip(){tip.style.display = "none";}
// sequential colour scale (viridis-like, colour-blind safe), t in [0,1]
const VIR = ["#fde725","#b5de2b","#6ece58","#35b779","#1f9e89","#26828e","#31688e","#3e4989","#482878","#440154"];
function vir(t){t=Math.max(0,Math.min(1,t)); return VIR[Math.round(t*(VIR.length-1))];}
function accColour(a){return vir((1-a)/0.8);}
// ---- condition picker
function buildPicker(){
  const b = document.getElementById("buttons");
  D.conditions.filter(c => c.group==="test" || c.group==="subset").forEach(c => {
    const el = document.createElement("button"); el.className = "btn"; el.textContent = c.label + "  acc " + (100*c.acc).toFixed(1) + "%";
    el.onclick = () => select(c.id); el.dataset.id = c.id; b.appendChild(el);});
  const g = document.getElementById("grid");
  let h = "<tr><th></th><th colspan=4>strength 1 &rarr; 4</th></tr>";
  D.families.forEach(([f, grp, unit, levels]) => {
    h += `<tr><td class="f" title="${unit}">${f}</td>`;
    for (let r = 1; r <= 4; r++){const c = cond[f+"|"+r]; if(!c){h += "<td></td>"; continue;}
      h += `<td class="c" data-id="${c.id}" style="background:${accColour(c.acc)};color:${c.acc<0.55?'#fff':'#000'}" title="${c.label}">${Math.round(100*c.acc)}</td>`;}
    h += "</tr>";});
  const bgc = D.conditions.filter(c => c.group==="background");
  if (bgc.length){h += `<tr><th colspan=5 style="text-align:left;padding-top:8px">background tests (leaf mask needed)</th></tr>`;
    bgc.forEach(c => {h += `<tr><td class="f">${c.family}</td><td class="c" data-id="${c.id}" style="background:${accColour(c.acc)};color:${c.acc<0.55?'#fff':'#000'}" title="${c.label}">${Math.round(100*c.acc)}</td><td colspan=3></td></tr>`;});}
  g.innerHTML = h;
  g.querySelectorAll("td.c").forEach(td => td.onclick = () => select(td.dataset.id));
}
// ---- matrix
const cs = 16, lab = 190, top0 = 200;
function drawMatrix(){
  const c = cond[cur]; const svg = document.getElementById("mat");
  const W = lab + N*cs + 140, H = top0 + N*cs + 150; svg.setAttribute("width", W); svg.setAttribute("height", H);
  let maxv = 1; Object.values(c.cells).forEach(v => maxv = Math.max(maxv, v.length));
  const pos = {}; O.forEach((k, i) => pos[k] = i);
  let s = "";
  // crop separators
  for (let i = 0; i < N; i++){ const k = O[i];
    s += `<text x="${lab-4}" y="${top0+i*cs+cs*0.75}" font-size="10.5" text-anchor="end">${C[k].short}</text>`;
    s += `<text transform="translate(${lab+i*cs+cs*0.75},${top0-4}) rotate(-60)" font-size="10.5">${C[k].short}</text>`;
    s += `<rect x="${lab+i*cs}" y="${top0+i*cs}" width="${cs}" height="${cs}" fill="#e6e6e6"/>`;}
  for (let i = 1; i < N; i++) if (C[O[i]].host !== C[O[i-1]].host){
    s += `<line x1="${lab}" x2="${lab+N*cs}" y1="${top0+i*cs}" y2="${top0+i*cs}" stroke="#0072B2" stroke-width="0.8"/>`;
    s += `<line y1="${top0}" y2="${top0+N*cs}" x1="${lab+i*cs}" x2="${lab+i*cs}" stroke="#0072B2" stroke-width="0.8"/>`;}
  const colsum = new Array(N).fill(0);
  for (const [key, v] of Object.entries(c.cells)){ const [t, p] = key.split(",").map(Number); colsum[p] += v.length;
    const t01 = Math.log(1 + v.length) / Math.log(1 + maxv);
    s += `<rect class="cell" data-k="${key}" x="${lab+pos[p]*cs+0.5}" y="${top0+pos[t]*cs+0.5}" width="${cs-1}" height="${cs-1}" fill="${vir(0.12+0.88*t01)}" style="cursor:pointer"/>`;}
  s += `<rect x="${lab}" y="${top0}" width="${N*cs}" height="${N*cs}" fill="none" stroke="#999"/>`;
  s += `<text x="${lab-4}" y="${top0-150}" font-size="12" text-anchor="end" font-weight="600">rows: true class</text>`;
  s += `<text x="${lab-4}" y="${top0-136}" font-size="12" text-anchor="end" font-weight="600">columns: predicted</text>`;
  // sink bars
  const by = top0 + N*cs + 12, bh = 110; const mx = Math.max(1, ...colsum);
  s += `<text x="${lab-4}" y="${by+12}" font-size="11.5" text-anchor="end" font-weight="600">wrong predictions</text>`;
  s += `<text x="${lab-4}" y="${by+26}" font-size="11.5" text-anchor="end" font-weight="600">per predicted class</text>`;
  s += `<text x="${lab-4}" y="${by+40}" font-size="11" text-anchor="end">(max ${mx})</text>`;
  for (let p = 0; p < N; p++){ const v = colsum[p]; if (!v) continue; const h = bh * v / mx;
    s += `<rect class="bar" data-p="${p}" x="${lab+pos[p]*cs+1}" y="${by}" width="${cs-2}" height="${h}" fill="#D55E00" style="cursor:pointer"/>`;}
  svg.innerHTML = s;
  const nWrong = colsum.reduce((a, b) => a + b, 0);
  document.getElementById("ctitle").textContent = `${c.label}: accuracy ${(100*c.acc).toFixed(1)}%, ${nWrong} wrong of ${c.n}`;
  svg.querySelectorAll("rect.cell").forEach(r => {
    const [t, p] = r.dataset.k.split(",").map(Number); const v = c.cells[r.dataset.k];
    r.onmousemove = e => showTip(e, `true: ${C[t].short}<br>predicted: ${C[p].short}<br><b>${v.length}</b> of ${c.n_per_class[t]} images`);
    r.onmouseleave = hideTip; r.onclick = () => showImages(`${C[t].short} predicted as ${C[p].short}`, v);});
  svg.querySelectorAll("rect.bar").forEach(r => { const p = +r.dataset.p;
    const items = []; for (const [key, v] of Object.entries(c.cells)){ if (+key.split(",")[1] === p) items.push(...v);}
    r.onmousemove = e => showTip(e, `predicted ${C[p].short}<br><b>${items.length}</b> wrong predictions`);
    r.onmouseleave = hideTip; r.onclick = () => showImages(`all wrong predictions of ${C[p].short}`, items);});
}
function showImages(title, items){
  document.getElementById("ttitle").textContent = `${title}: ${items.length} images (number = confidence of the wrong prediction)`;
  const t = document.getElementById("thumbs"); let h = "";
  items.slice().sort((a, b) => b[1]-a[1]).forEach(([i, conf]) => { const src = D.thumbs[i];
    const tr = D.true_of[i] !== undefined ? C[D.true_of[i]].short : "";
    h += `<div class="th" title="sample_idx ${i}, true: ${tr}">${src ? `<img src="data:image/jpeg;base64,${src}">` : "(no thumb)"}#${i}<br>${conf.toFixed(2)}</div>`;});
  t.innerHTML = h;
}
function drawPreview(){
  const c = cond[cur]; const p = document.getElementById("preview"); const ex = D.examples[c.family];
  if (!ex || c.family === "none"){p.innerHTML = ""; return;}
  let h = `<div>Preview of <b>${c.family}</b> on one example leaf (unmodified, then strength 1-4; selected one framed):</div>`;
  ex.forEach(([lv, src], k) => { h += `<img src="data:image/jpeg;base64,${src}" title="${lv}" style="${k===c.rank?'outline:3px solid #D55E00':''}">`;});
  if (D.examples["none"]) h = h.replace("<div>Preview", `<div>Preview`);
  p.innerHTML = h;
}
function select(id){
  cur = id;
  document.querySelectorAll("td.c").forEach(td => td.classList.toggle("sel", td.dataset.id === id));
  document.querySelectorAll(".btn").forEach(b => b.classList.toggle("sel", b.dataset.id === id));
  drawMatrix(); drawPreview(); document.getElementById("thumbs").innerHTML = "";
  document.getElementById("ttitle").textContent = "Click a matrix cell or a bar.";
}
// unmodified example goes first in each preview strip
for (const f in D.examples) if (f !== "none" && D.examples["none"]) D.examples[f] = [D.examples["none"][0], ...D.examples[f]];
buildPicker(); select("test");
</script></body></html>
"""

if __name__ == "__main__":
    main()
