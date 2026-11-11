# Reflection on GenAI use – Phase I

**AI-aided pair:** Paul Mairhofer (k12312103), Fabian Matusek (k12111864)

> **How the AI-aided analysis was produced.** The whole AI-aided Phase I (code, results, figures, notebook, README and a first draft of this
> reflection) was produced by an autonomous coding agent (Claude, run through `pi`) from **one prompt**, with no human input during the run
> (8 Nov 2026, about 2 h, CPU only). The prompt set the rules (black box only, systematic patterns with controls, render and critique every
> figure, document all decisions) and the group direction; the agent chose the questions, methods and figures itself. The session record is in
> `ai-interactions/phase1/2026-11-08-phase1-ai-analysis.md`, the figure log in `ai-interactions/phase1/figure-log.md`. The comparison with the
> human-only pair was added after both pairs had pushed their work (11 Nov).

## Insights found by one pair but not the other

What the AI-aided analysis found (details and numbers in `phase1/ai-aided/README.md`):

- Confusions follow crop first, disease second (83 of 107 errors within the crop; 12 of 24 cross-crop errors keep the disease name), and the
  same structure appears in the runner-up class of correct predictions.
- Errors are tied to leaves: 90 errors on 51 leaves (random: 85); 38% of the other photos of a wrong leaf are also wrong (0.4% otherwise).
- Under capture changes the model collapses into sink classes (51% of 21,164 wrong predictions go to three tomato classes) and stays confident.
- Darkness -> Tomato late blight traced to one dark training session (117 of 118 images in the darkest 1%), with a region test.
- Background: foreign background lowers accuracy from 96% to 84%; background alone still gives 12% (chance 2.6%).
- Near-duplicates do not explain the accuracy; one likely mislabelled image pair in the dataset (test 8252).

**Comparison with the human-only pair** (Nico, Timo: `phase1/human-only/experiments.ipynb`, `additional_experiments.ipynb`):

- **Found by both:** blur and noise hurt most, brightness much less. Their sweep on the full test set gives similar numbers to ours on the 40-per-class subset
  (noise σ 0.10: 42% vs 34%, σ 0.15: 8% vs 10%; brightness ×0.5: 87%, ours ×0.6 / ×0.4: 91% / 67%). Both found that the model still works on the leaf
  alone (grey background: 80% for them, 86% for us) Both ran into Corn healthy close-ups, where leaf masking fails (0% recall for them, excluded by us).
- **Only human-only:** class size vs error rate (no clear link), per-class recall drop for each perturbation (Potato healthy loses 81 pp when darker,
  Tomato healthy 100 pp under blur), and the plain list of classes with zero errors. A second segmentation method (rembg) next to GrabCut.
- **Only AI-aided:** where the errors go (sink classes), errors clustered by leaf, the dark capture session behind the late-blight sink, background swaps
  with a same-class control, border-colour k-NN, near-duplicates.
- **Disagreement:** background only. Their GrabCut version gives 35.6%, their rembg version 9%, ours 12%. GrabCut masks leave the stem and leaf edge in
  the image (visible in their segmentation figure), so the 35.6% is probably too high; the rembg and our number agree.
- **Check needed:** in `additional_experiments.ipynb`, `get_acc` stops after ~100 images without shuffling, so each point is the first 128 test
  images (123 Apple scab, 5 Apple black rot). The curves there describe Apple only, not the whole model.
- **Agent scale vs human looking:** the 130k modified images, null model and region tests were only realistic for the agent. The human-only pair
  looked at the segmentation masks directly, which is how the GrabCut leak can be seen at all.

## Visualization choices

- **Soft confusion matrix instead of a count matrix.** 107 errors in 1,406 off-diagonal cells give an almost empty count matrix (rendered and
  rejected). Averaging softmax mass over all 10,948 predictions shows the similarity structure; classes are ordered by crop with crop boxes.
  A clustering order was tried and rejected because it broke the crop blocks.
- **Every claim next to its baseline.** Bars against "random other class" (Fig 1), a simulated null for leaf clustering (Fig 2), the 180° flip
  control (Fig 3), a same-class background swap as control for pasting (Fig 5), chance lines and Wilson 95% intervals.
- **Heatmap with parameter values in the cells for the sweep (Fig 3)**, because strength ranks are not comparable across families; a line plot
  with 18 lines was rendered and rejected as unreadable. A sink column turns "accuracy drops" into the systematic part: where the errors go.
- **Input statistics clearly separated from the model.** The border-colour k-NN bar is labelled "input statistic, NOT our model".
- **Interactive explorer** (offline HTML) for details that would need too many static figures: any condition, any confusion cell, and the images behind it.
- **Process.** Every candidate was rendered and looked at by the agent; 5 key figures kept, 2 turned into tables, several layouts reworked
  (log of all candidates in `figure-log.md`). Colour-blind safe palettes (Okabe-Ito, viridis); the colour-group colour was changed from vermillion
  to pink to keep it apart from orange.

- The human-only pair used a standard normalised 38×38 confusion matrix (the one the agent rejected as almost empty), horizontal bar charts for
  per-class error and recall drop, and one line plot per perturbation with the baseline as a dashed line.
- Their charts are quicker to read for "how much does accuracy drop"; ours answer "where do the errors go", but need more reading (longer titles,
  more panels, a sink column).

## Breadth and depth of the investigation

- **Breadth:** six questions (confusion structure, leaf clustering, 18 capture changes x 4 strengths, the dark sink, background, near-duplicates),
  about 130,000 model calls on modified images, three input descriptions (pixel thumbnails, border colour, leaf colour), the dataset card and
  file-name metadata, and one literature check (Noyan 2022).
- **Depth came from following anomalies:** sink classes in the sweep -> brightness of training images -> file names show one dark capture session
  -> region test (leaf only / background only / whole image / leaf on black) to separate explanations. Similarly: largest confusion -> 7 leaves ->
  leaf colour -> which change reproduces it (blur, not colour).
- **Not done:** no field images, no test of other architectures, no causal test of the shortcut (that would need retraining, which Phase I excludes),
  perturbations on a subset of 40 images per class only.

- Human-only: 3 perturbations × 5 strengths on the full test set, foreground/background test with two masking methods, plus a 6-corruption sweep.
  Fewer questions, but full test set and per-class views.
- AI-aided: 6 questions, 18 changes × 4 strengths, but on a 40-per-class subset. The agent asked data questions (capture sessions, file names,
  near-duplicates) the human pair did not; the human pair asked about class size and per-class sensitivity, which the agent did not.

## Speed vs. extra work to verify AI suggestions

- **Speed:** the full pipeline, 130k model calls, five figures with several drafts each, the explorer, the notebook and the texts took about
  2 h of wall time; about 1.5 h of CPU inference ran in the background while the agent worked on figures and text.
- **Verification the agent did itself:** recomputed every starting fact before use; reran the checkpoint to check the shared predictions
  (10,947/10,948 identical, the one difference explained); viewed every perturbation, every background edit and the masks of every class before
  trusting them; viewed every figure after its last change; loaded the explorer in a headless browser and clicked through it; kept one cached
  file of key numbers so README, notebook and reflection agree; executed the notebook in a fresh kernel.
- **Cost of verification:** several self-caught mistakes (below) would have gone into the report without these checks. The verification is
  still done by the same agent that made the claims, so it is not independent.

- Time: Paul about 4 h in total (prompt, setup, running and checking the output); Fabian reviewed the result for about 45 min and found no errors.
- The agent's results were easier to check than to produce: numbers are in one file (`key_numbers.json`) and every figure has a script.

## Confidence in the conclusions

- **High:** the descriptive results on the natural test set (confusion structure, leaf clustering, near-duplicates), because they are counts on
  the full test set with chance baselines and a null model; the sweep results (1,512 images per condition, narrow intervals, a flip control).
- **Medium:** the background results, because they depend on leaf masks (checked visually but imperfect; Corn healthy had to be excluded) and
  pasting may add artefacts (the same-class control limits this to about 2 points).
- **Hypothesis only:** the cause of the late-blight sink (dark training session) and the cause of the bacterial spot -> yellow leaf curl confusion
  (pale leaves, spots hidden). Both are consistent with several observations, but causality needs Phase II or retraining.
- **Evidence of a different kind:** input statistics (border-colour k-NN, pixel similarity, leaf colour) describe the data, not the model, and
  are labelled that way.

- Where both pairs measured the same thing, the numbers agree within a few points (noise, blur, brightness, leaf-only), which raises our confidence
  in those results. The background-only number differs, and our 12% (supported by their rembg 9%) is the more reliable one.
- We trust the AI-aided conclusions after review, but the dark-session shortcut is still a hypothesis until Phase II.

## Failure patterns of AI assistance

Observed in this run (all caught by the agent itself, see the session record):

- **Overclaiming from a small sample.** A figure title ("the dark background is enough") was written from a 1-image-per-class dry run (n = 34)
  and turned out wrong at n = 1,360 (dark background alone: 11% vs 66% for a dark whole image).
- **Misremembering its own numbers.** A note confused "122 dark late-blight images" with the size of the capture session (118 images, 117 dark).
- **Silent technical failures:** a uint8 overflow produced plausible-looking but wrong colour values; a JavaScript name clash (`const top`)
  made the explorer show nothing without any error in the build.
- **Methods that fail on part of the data:** the leaf mask treated corn close-ups as having a background; only a per-class visual check found it.
- **Tool side effects:** `pkill -f` matched the agent's own shell command and killed it.
- **General risks of an autonomous run** (not observed as errors here, but structural): the agent checks its own work; it decides alone which
  question is interesting; it writes fluent text that can sound more certain than the evidence; and no human saw the images it reasoned about.

- Fabian's review did not find errors the agent missed.
- The comparison found one point where the agent was weaker: it only used a 40-per-class subset, where the human pair used the full test set.

## Best practices

- Recompute any number given in the prompt or in earlier notes before using it; keep all reported numbers in one generated file.
- Pair every claim with a baseline or control: chance, a null simulation, an in-training control transformation, a same-class swap.
- Do a small dry run to check edits visually, but never write conclusions from the dry run.
- Look at the images behind every method (masks per class, edits per condition, the images in a confusion cell).
- Render and look at every figure; keep a log of rejected candidates and why.
- Label input statistics as input statistics; keep black-box claims about the model separate from claims about the data.
- Test interactive outputs in a real browser.
- Use the data card and file metadata early: they turned a vague "dark images fail" into a concrete, checkable dataset shortcut.

- AI-aided should adopt: full test set where it is cheap, simple per-class bar charts next to the more complex figures.
- Human-only could adopt: controls (same-class swap, flip), checking where errors go, and fixed sampling with a shuffle and stated n.
