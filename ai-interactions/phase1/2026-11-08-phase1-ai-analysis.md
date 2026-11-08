# 2026-11-08 – Phase I analysis by an autonomous agent (AI-aided pair)

## Setup

- **Tool:** Claude (Anthropic) as an autonomous coding agent in the `pi` harness, CPU-only laptop (8 cores, torch CPU, Python 3.14 venv).
- **Prompt (one prompt, shortened):** "Produce the AI-aided pair's Phase I deliverables (Vis4AI: how does the model behave?) for team Organic Alpacas:
  black-box only (inputs, outputs, probabilities, outputs on modified inputs; no activations/gradients/weights/Grad-CAM), not a performance report,
  systematic patterns with counts, controls and uncertainty, broader than a human-only pair, stay in the group direction (confusions; lighting,
  blur, background; confusion matrices, heatmaps, clustering). Render, view and critique every candidate figure and keep a log. Notebook with
  3-5 key figures plus an offline interactive view, README, AI-interaction record, reflection. Make all decisions yourself and document them.
  No contact with anyone, no push." Plus hard rules on paths, memory and repo hygiene.
- **Human input during the run:** none. Paul and Fabian have not yet reviewed the result; this record is written by the agent.
- **Run:** 15:52 to about 17:55 (about 2 h wall time; the budget was 4-5 h). The heavy part was model inference on CPU (about 130,000 modified images, roughly 1.5 h, run in the background).

## Key decisions

- **Trust, but check the shared predictions first.**
  - Result: rerunning the checkpoint on CPU gave 10,947/10,948 identical predictions; the one mismatch is a near-tie (0.484 vs 0.482), so GPU vs CPU rounding.
  - Decision: accepted the shared CSV for natural-test analyses. Modified images are compared with the same images through our CPU pipeline, never with the CSV.
  - Effect: perturbation "flips" are not polluted by pipeline differences.
- **Look at the data card and file names before analysing.**
  - Result: the HF dataset card says Tomato Target Spot has no usable leaf grouping; for ungrouped classes `leaf_id` contains the original
    file names, which encode capture sessions (e.g. `GHLB_PS ... Day 16`).
  - Decision: used the session names as a source of hypotheses (allowed: training data, not model internals).
  - Effect: led directly to the dark late-blight session behind the strongest sink (Fig 4).
- **Soft confusion matrix instead of counts.** 107 errors are too few for a count matrix; mean softmax mass to wrong classes uses all 10,948 predictions.
  Classes ordered by crop with potato/pepper next to tomato (same plant family), after a clustering order was tried and rejected.
- **Chance baselines everywhere.** Each "X% of errors are Y" is shown next to what a random wrong class would give (e.g. same crop 78% vs 10%).
- **Leaf-level null model.** Errors per leaf compared with 5,000 random placements within class, instead of a visual impression of "clusters".
- **Controlled edits on a fixed subset.** 40 test images per class (1,512), 18 kinds of change x 4 strengths, plus a 180° rotation as an
  in-training control. Chosen over the full test set to fit CPU time; stratified so every class counts equally.
- **Background tests need a leaf mask.** Classical segmentation (colour distance to the border + GrabCut), checked visually on 76 random images
  and again per class after the run. It is described as an input description, not as part of the model.
- **Region test added after a hypothesis appeared.** When the training data showed a dark late-blight session, the agent added "darken leaf only /
  background only / whole image" and "leaf on black" to separate the two explanations, and ran a 1-image-per-class dry run before the full run.
- **Five key figures plus one interactive view; two candidate figures turned into tables.** See `figure-log.md`.
- **Paused the long perturbation sweep** (resumable from saved parts) to run the background tests first, because they answered the open question.

## Verification

- Starting facts from the brief (107 errors, 83/24 within/cross crop, 18 bacterial spot -> yellow leaf curl, 43 confident errors, 183 unconfident
  correct, 3,549 ungrouped images, class sizes 32-1,156) recomputed from the CSV before use: all matched.
- Literature claim (Noyan 2022, 8 background pixels -> 49% accuracy) checked on the arXiv abstract; a blog copy that refused access was not used.
- Every perturbation and every background edit was rendered on example images and viewed before the full runs.
- Every figure was viewed after its last change (`figure-log.md`); the interactive page was loaded in a headless browser and clicked through.
- Numbers in README, notebook and reflection come from one cached file (`results/key_numbers.json`) written by the notebook.
- The notebook was executed in a fresh kernel with `jupyter nbconvert --execute`.

## Mistakes caught (by the agent itself)

- **Wrong figure title from a too-small sample.** The first Fig 4 was built from the dry run (n = 34) and said "the dark-background cue is enough".
  The full run (n = 1,360) showed a dark background alone gives only 11% late blight vs 66% for a dark whole image. Title and interpretation rewritten.
- **Misremembered number.** An early note said the dark late-blight session was "8% of late blight training images" (122/1,527); that was the
  count of dark images, not the session. Rechecked from file names: the session has 118 images, 117 of them in the darkest 1%.
- **Mask check found a class without background.** Corn healthy photos are close-ups; the "background" the mask left was leaf. Excluded at analysis time.
- **uint8 overflow** in the HSV/Lab leaf descriptors gave nonsense values; fixed by casting before arithmetic.
- **JavaScript name clash**: a global `const top` collides with `window.top`, so the explorer silently showed nothing. Found only by loading the page.
- **Inpainted "background only" images leak leaf colour** where the mask misses parts; added a second, mask-robust condition (leaf-free corner patches).
- **Own process error:** `pkill -f 02_perturbation_sweep.py` also matched the agent's own shell command and killed it (exit 143). No data lost; use the PID.
- **GPU/CPU near-tie** in the reproduction check was explained instead of ignored.

## Effect on the analysis

The agent's breadth (six questions, ~130k model calls, an interactive page) came from automation; the depth came from following anomalies
(sink classes -> training brightness -> file-name sessions -> region test). The open part is the comparison with the human-only pair,
which needs Paul and Fabian (see `ai_reflection_midpoint.md`).
