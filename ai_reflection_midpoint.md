# Reflection on GenAI use – Phase I

**AI-aided pair:** Paul Mairhofer (k12312103), Fabian Matusek (k12111864)

> **How the AI-aided analysis was produced.** The whole AI-aided Phase I (code, results, figures, notebook, README and a first draft of this
> reflection) was produced by an autonomous coding agent (Claude, run through `pi`) from **one prompt**, with no human input during the run
> (8 Nov 2026, about 2 h, CPU only). The prompt set the rules (black box only, systematic patterns with controls, render and critique every
> figure, document all decisions) and the group direction; the agent chose the questions, methods and figures itself. The session record is in
> `ai-interactions/phase1/2026-11-08-phase1-ai-analysis.md`, the figure log in `ai-interactions/phase1/figure-log.md`. The parts below that
> depend on the human-only pair are marked *to complete after comparing with the human-only pair*.

## Insights found by one pair but not the other

What the AI-aided analysis found (details and numbers in `phase1/ai-aided/README.md`):

- Confusions follow crop first, disease second (83 of 107 errors within the crop; 12 of 24 cross-crop errors keep the disease name), and the
  same structure appears in the runner-up class of correct predictions.
- Errors are tied to leaves: 90 errors on 51 leaves (random: 85); 38% of the other photos of a wrong leaf are also wrong (0.4% otherwise).
- Under capture changes the model collapses into sink classes (51% of 21,164 wrong predictions go to three tomato classes) and stays confident.
- Darkness -> Tomato late blight traced to one dark training session (117 of 118 images in the darkest 1%), with a region test.
- Background: foreign background lowers accuracy from 96% to 84%; background alone still gives 12% (chance 2.6%).
- Near-duplicates do not explain the accuracy; one likely mislabelled image pair in the dataset (test 8252).

*To complete after comparing with the human-only pair:*

- [ ] Which of these did the human-only pair also find? With which method and figure?
- [ ] What did the human-only pair find that the agent missed (e.g. insights from looking at many images by hand, domain knowledge about the diseases)?
- [ ] Do the two pairs disagree anywhere (e.g. on how strong the background effect is, or on lighting vs blur)? Who is right, and why?
- [ ] Which findings are only possible at the agent's scale (130k modified images, region tests, null models), and which only by careful human looking?

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

*To complete after comparing with the human-only pair:*

- [ ] Which visualisations did the human-only pair choose for the same questions? Which were easier to read?
- [ ] Did the agent's figures need too much reading (long titles, many panels) compared with the human-made ones?

## Breadth and depth of the investigation

- **Breadth:** six questions (confusion structure, leaf clustering, 18 capture changes x 4 strengths, the dark sink, background, near-duplicates),
  about 130,000 model calls on modified images, three input descriptions (pixel thumbnails, border colour, leaf colour), the dataset card and
  file-name metadata, and one literature check (Noyan 2022).
- **Depth came from following anomalies:** sink classes in the sweep -> brightness of training images -> file names show one dark capture session
  -> region test (leaf only / background only / whole image / leaf on black) to separate explanations. Similarly: largest confusion -> 7 leaves ->
  leaf colour -> which change reproduces it (blur, not colour).
- **Not done:** no field images, no test of other architectures, no causal test of the shortcut (that would need retraining, which Phase I excludes),
  perturbations on a subset of 40 images per class only.

*To complete after comparing with the human-only pair:*

- [ ] How does the number of questions and conditions compare with the human-only pair? Did they go deeper on fewer questions?
- [ ] Which of the agent's questions would the human pair not have asked, and vice versa?

## Speed vs. extra work to verify AI suggestions

- **Speed:** the full pipeline, 130k model calls, five figures with several drafts each, the explorer, the notebook and the texts took about
  2 h of wall time; about 1.5 h of CPU inference ran in the background while the agent worked on figures and text.
- **Verification the agent did itself:** recomputed every starting fact before use; reran the checkpoint to check the shared predictions
  (10,947/10,948 identical, the one difference explained); viewed every perturbation, every background edit and the masks of every class before
  trusting them; viewed every figure after its last change; loaded the explorer in a headless browser and clicked through it; kept one cached
  file of key numbers so README, notebook and reflection agree; executed the notebook in a fresh kernel.
- **Cost of verification:** several self-caught mistakes (below) would have gone into the report without these checks. The verification is
  still done by the same agent that made the claims, so it is not independent.

*To complete after comparing with the human-only pair:*

- [ ] How long did Paul and Fabian need to review and check the agent's output (rerun, spot-check images, check numbers)? Did they find errors?
- [ ] Time spent by the human-only pair for comparable results.

## Confidence in the conclusions

- **High:** the descriptive results on the natural test set (confusion structure, leaf clustering, near-duplicates), because they are counts on
  the full test set with chance baselines and a null model; the sweep results (1,512 images per condition, narrow intervals, a flip control).
- **Medium:** the background results, because they depend on leaf masks (checked visually but imperfect; Corn healthy had to be excluded) and
  pasting may add artefacts (the same-class control limits this to about 2 points).
- **Hypothesis only:** the cause of the late-blight sink (dark training session) and the cause of the bacterial spot -> yellow leaf curl confusion
  (pale leaves, spots hidden). Both are consistent with several observations, but causality needs Phase II or retraining.
- **Evidence of a different kind:** input statistics (border-colour k-NN, pixel similarity, leaf colour) describe the data, not the model, and
  are labelled that way.

*To complete after comparing with the human-only pair:*

- [ ] Do the human-only pair's results agree? Where they differ, which evidence is stronger?
- [ ] After review: how confident are Paul and Fabian in conclusions they did not produce themselves?

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

*To complete after comparing with the human-only pair:*

- [ ] Errors found by Paul and Fabian during review that the agent did not catch.
- [ ] Places where the agent's framing or wording does not match what the figures show.

## Best practices

- Recompute any number given in the prompt or in earlier notes before using it; keep all reported numbers in one generated file.
- Pair every claim with a baseline or control: chance, a null simulation, an in-training control transformation, a same-class swap.
- Do a small dry run to check edits visually, but never write conclusions from the dry run.
- Look at the images behind every method (masks per class, edits per condition, the images in a confusion cell).
- Render and look at every figure; keep a log of rejected candidates and why.
- Label input statistics as input statistics; keep black-box claims about the model separate from claims about the data.
- Test interactive outputs in a real browser.
- Use the data card and file metadata early: they turned a vague "dark images fail" into a concrete, checkable dataset shortcut.

*To complete after comparing with the human-only pair:*

- [ ] Which practices of the human-only pair should the AI-aided workflow adopt (and the other way round)?
