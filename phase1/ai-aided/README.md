# Phase I – AI-aided pair: how does the model behave?

Paul Mairhofer (k12312103), Fabian Matusek (k12111864) · team Organic Alpacas

**How this was made.** An autonomous coding agent (Claude, run through `pi`) did the analysis, wrote the code and the text, and made the figures
from one prompt, with no human input during the run (8 Nov 2026, about 2 h, CPU only). The record of that session is in
[`ai-interactions/phase1/`](../../ai-interactions/phase1/), including the figure log. The human-only comparison is still open.

**Scope.** Black box only: images, metadata, predictions, softmax probabilities, and outputs on images we modified. No activations, gradients,
weights or saliency maps. Leaf masks, pixel thumbnails and colour statistics describe the *input images*, not the model.
Training images were used only to form hypotheses. Nothing was trained or tuned.

## Findings

All numbers come from `results/` and are collected in `results/key_numbers.json` by the notebook.

1. **Confusions follow the crop first and the disease second.** 83 of the 107 test errors stay within the crop, although only 10% of the wrong
   classes are in the same crop. 12 of the 24 cross-crop errors keep the disease name (potato <-> tomato late and early blight). The runner-up
   class of *correct* predictions shows the same structure: same crop 37% (chance 10%), same disease in another crop 3.7% (chance 1.0%). (Fig 1)
2. **Errors belong to leaves, not to photos.** The 90 errors on leaf-grouped images fall on 51 leaves; random placement within the class
   would give 85 (95% range 81-88, p < 0.001). If one photo of a leaf is wrong, 38.3% of its other photos are wrong too; if it is correct, 0.4% are.
   5 leaves are wrong in all 4 photos. The 18 Tomato bacterial spot -> yellow leaf curl errors come from 7 leaves that are much paler and
   yellower than correctly classified ones (median leaf b* 42 vs 13; yellow leaf curl: 28). (Fig 2)
3. **Capture changes break the model, and the errors collapse into sink classes.** On 1,512 test images (40 per class), mild changes keep
   94-99% accuracy and the 180° flip control changes 11 predictions. Strong changes break it: darker x0.25 25%, desaturated 22%, Gaussian blur
   sigma 4 22%, noise 10%, zoom-out 18%. At strengths 3-4, 51% of the 21,164 wrong predictions go to three tomato classes (late blight 27%,
   Septoria 14%, early blight 10%). Dark, low-contrast, blurred and zoomed-out images go to Tomato late blight (73-87% of wrong predictions);
   bright, high-contrast and oversaturated images go to Septoria leaf spot. Confidence does not warn: 31% of wrong predictions on modified images
   have confidence > 0.9, and the AUROC of confidence drops from 0.971 to 0.877. Blur, not colour, reproduces the bacterial spot -> yellow leaf
   curl confusion (25 of 40 at sigma 2). (Fig 3)
4. **Darkness -> "Tomato late blight" is a likely dataset shortcut.** Darkening the whole image sends 66% of other-class images to late blight;
   darkening only the leaf (5%) or only the background (11%) rarely does; a leaf on flat black 26%. With the leaf removed, 41% go there.
   In training, late blight is 28% of the darkest 1% of images (3.5% overall), because one capture session ("GHLB_PS", 118 images) was shot
   dark on black: 117 of its images are in the darkest 1%. (Fig 4)
5. **The background carries class information and the model uses part of it.** The border colour alone (3 numbers, 15-NN on training images)
   gives 51% accuracy on the full test set (40% on the class-balanced subset in Fig 5; chance 2.6%), in line with Noyan (2022), who got 49% from
   8 background pixels. Our model, with the leaf removed, still names the right class 12% of the time. Pasting the same leaf onto a background from another class lowers accuracy from 96% (same-class control) to 84%; flat
   backgrounds: grey 86%, white 75%, black 56%. (Fig 5)
6. **Near-duplicates do not explain the accuracy.** 135 test images have a near-copy in training (pixel correlation > 0.93) and only 1 of them
   is wrong. Errors are 4x more common when the most similar training image belongs to another class (1.8% vs 0.45%). One near-copy pair
   (test 8252, "late blight", vs a "healthy" training photo of the same leaf) looks like a labelling error in the dataset. (table in Section 6)

**Open questions for Phase II.** Where does the model look on dark or blurred images that end up in the sink classes? Are the sink classes
central in the embedding space? Do background-sensitive classes (Tomato healthy, Corn) attend to the border? Are the 51 error leaves outliers
in the embedding space?

**Caveats.** Perturbations and background tests use a stratified subset (40 per class, Potato healthy 32), not the full test set. Leaf masks
are classical segmentation (colour distance to the border + GrabCut), checked visually on 76 random images and per class; 1,427 of 1,512 masks
are usable, and Corn healthy (close-ups without background) is excluded from background tests. Mask errors leave some background in "leaf only"
edits, so background effects are if anything underestimated. Zoom-out and 15°/45° rotation pad with the border colour, which leaves a visible
edge. The shared CSV is used for the natural test set; modified images are compared with the same images run through our CPU pipeline
(10,947/10,948 predictions identical to the CSV; the one difference is a near-tie).

## What is where

| Path | Content |
|---|---|
| `phase1_ai_aided.ipynb` | The analysis: question -> figure -> finding -> follow-up, six questions. Reads only `results/`. |
| `phase1_ai_aided.html` | Static HTML export of the executed notebook. |
| `behaviour_explorer.html` | Offline interactive explorer (open in a browser): condition grid, confusion matrix per condition, sink bars, thumbnails. |
| `figures/` | Static PNGs of the five key figures and an explorer screenshot. |
| `results/` | Cached results (parquet/json, about 9 MB), see the table below. |
| `pv.py` | Paths, dataset loading (same split as the baseline), model loading, batched prediction. |
| `perturb.py` | The 18 capture changes x 4 strengths. |
| `segment.py` | Leaf mask and background edits (input description only). |
| `viz.py`, `figures.py` | Plot style (Okabe-Ito palette) and all figure functions used by the notebook. |
| `01_...py` - `07_...py` | Scripts that compute `results/` and the explorer (below). |

| Script | Writes | CPU time (8 cores) |
|---|---|---|
| `01_reproduce_predictions.py` | `reproduction_check.json` | ~3 min |
| `02_perturbation_sweep.py` | `perturbation_sweep.parquet` (110,376 rows: 1,512 images x 73 conditions) | ~50 min, resumable |
| `03_input_space.py` | `test_nearest_train.parquet`, `border_colour.parquet`, `border_knn.json`, `image_brightness.parquet` | ~5 min |
| `04_leaf_descriptors.py` | `test_leaf_descriptors.parquet` | ~8 min |
| `05_background_tests.py` | `background_tests.parquet`, `background_masks_meta.parquet` | ~15 min |
| `06_thumbnails.py` | `thumbs_64/128.parquet`, `perturb_examples.parquet`, `train_examples.parquet`, `late_blight_sessions.json` | ~1 min |
| `07_build_explorer.py` | `behaviour_explorer.html` | seconds |

## How to rerun

Requirements: the packages of `shared/requirements.txt` plus PyTorch/torchvision, `opencv-python-headless`, `scipy`, `pyarrow` and
`nbconvert` (tested with Python 3.14, torch 2.15 CPU, opencv 5.0, pandas 3.1). The checkpoint `shared/outputs/resnet18/best_model.pth` is not in
git; put it there first. The scripts download the dataset (`geraldmc/plantvillage-full`, about 1.6 GB) through `datasets`.

```bash
cd phase1/ai-aided
# only the notebook (uses the cached results, about 10 s)
jupyter nbconvert --to notebook --execute --inplace phase1_ai_aided.ipynb
jupyter nbconvert --to html phase1_ai_aided.ipynb

# recompute everything (about 1.5 h on 8 CPU cores); 03 must run before 06, 02 and 05 before 07
python 01_reproduce_predictions.py
python 03_input_space.py
python 02_perturbation_sweep.py
python 04_leaf_descriptors.py
python 05_background_tests.py
python 06_thumbnails.py
python 07_build_explorer.py
```

Subsets are fixed by seed (`pv.stratified_subset(labels, 40, seed=0)`), so reruns use the same images. Expect probability differences in the
third decimal between GPU and CPU.
