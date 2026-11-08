# Organic Alpacas – AI and Visualization Lab (JKU, WS 2026)

**Task:** plant disease classification from leaf images (healthy vs. disease, per crop species).

**Model:** ResNet-18, pretrained on ImageNet and fine-tuned on PlantVillage (test accuracy 99.0 %, macro F1 98.8 %). Training and checkpoint: [`shared/`](shared/README.md).

**Dataset:** [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), Hugging Face version [geraldmc/plantvillage-full](https://huggingface.co/datasets/geraldmc/plantvillage-full) – 54,304 leaf images, 38 classes over 14 crop species, original train/test split 43,356 / 10,948.

**Questions:** Phase I – how does the model behave (confusions, effect of lighting, blur, background)? Phase II – what did it learn (Grad-CAM, Integrated Gradients, embeddings), does it look at the symptoms or at the background?

## Team

| Name | Student ID | Phase I | Phase II |
| --- | --- | --- | --- |
| Nicolas Madella-Mella | k12021216 | human-only | AI-aided |
| Timo Wendner | k11934121 | human-only | AI-aided |
| Paul Mairhofer | k12312103 | AI-aided | human-only |
| Fabian Matusek | k12111864 | AI-aided | human-only |

## Repository structure

```
shared/                 baseline notebook, predictions, checkpoint link, requirements – used by both pairs
phase1/ai-aided/        Phase I, Paul + Fabian (with GenAI)
phase1/human-only/      Phase I, Nico + Timo (without GenAI)
ai-interactions/        record of GenAI use
ai_reflection_midpoint.md
team-info.json, workload.json
```

Model weights are not in git (`*.pth` is ignored), the notebook downloads the dataset from Hugging Face. See `shared/README.md`.
