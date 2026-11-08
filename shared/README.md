# PlantVillage ResNet18 Baseline

Shared setup for the **VIS4AI / AI and Visualization** project. The goal is not to optimize model accuracy exhaustively, but to establish a reproducible plant-disease classifier that both project subgroups can analyze.

## Task and dataset

We use a ResNet18 initialized with ImageNet-pretrained weights and fine-tune it to predict **38 plant crop–disease/healthy categories** from leaf photographs. The dataset is [geraldmc/plantvillage-full](https://huggingface.co/datasets/geraldmc/plantvillage-full), a PlantVillage distribution with **54,304 images** and metadata including `class_idx`, `class_label`, `host`, `disease`, `leaf_id`, and `leaf_grouped`.

The Hugging Face dataset exposes one `train` dataset; its `split` column identifies the supplied original train/test partition (43,356 / 10,948 images). We retain the original test subset and form a validation subset from the original training subset using `GroupShuffleSplit` (12.5% of groups, seed 42). Grouping uses `(host, disease, leaf_id)` when available to reduce leakage across training and validation. For images without leaf-group metadata, independence cannot be guaranteed; group splitting also does not enforce exact class stratification.

The PlantVillage images are comparatively controlled; performance on these images does **not** establish robustness to field conditions.

## Shared baseline notebook

[`baseline.ipynb`](baseline.ipynb) follows the original notebook structure:

1. **Imports & setup:** dependencies, random seed, and device.
2. **Dataset preparation:** load the dataset, preserve original test split, create group-aware validation split, inspect group overlap and class counts. Optional class-distribution plots are disabled by default.
3. **Model setup & preprocessing:** resize RGB images to 224×224, ImageNet normalization, random flips for training only, and a 38-output ImageNet-pretrained ResNet18.
4. **PyTorch datasets & DataLoaders:** batch size 32, evaluation loaders without shuffling.
5. **Training & exports:** full-network fine-tuning with Adam (`1e-4`), maximum 10 epochs, early stopping on validation loss (patience 3, minimum improvement 0.001), save best/last checkpoints, export predictions, and print test accuracy and macro F1.

Reusable helpers live in [`baseline_utils.py`](baseline_utils.py): `PlantVillageDataset`, `get_leaf_groups`, `plot_class_distributions`, `train_one_epoch`, `validate`, `train_model`, `plot_training_history`, and `export_test_predictions`. The notebook imports only the helpers it calls. The utilities manage their own imports, avoiding duplicate notebook imports for implementation details.

## Setup and running

Install Python 3.10+ and the dependencies:

```bash
pip install -r requirements.txt
```
Pytorch has to be added manually depending on your hardware.

Open `baseline.ipynb` from **this directory** and run cells from top to bottom. A GPU is recommended; CPU training is substantially slower. The first run downloads the Hugging Face dataset and pretrained ResNet18 weights. Training runs when the training cell is executed; to skip retraining, skip that cell and load an existing best checkpoint in the following cell.

## Outputs

The notebook writes to `outputs/resnet18/`:

| File | Contents |
| --- | --- |
| `best_model.pth` | Checkpoint with lowest validation loss, selected for analysis |
| `last_model.pth` | Final training state, including optimizer state |
| `training_history.csv` | Training and validation loss/accuracy by epoch |
| `test_predictions.csv` | One row per test image: sample index, true/predicted class, confidence, metadata, and `prob_0`–`prob_37` |

`sample_idx` indexes `test_dataset` in the notebook's deterministic original-test ordering. Keep the dataset version and splitting code consistent when sharing results. Checkpoints contain model weights, not the architecture definition; recreate ResNet18 with 38 outputs before loading. Do not use test results to select hyperparameters or checkpoints.

`test_predictions.csv` and `training_history.csv` are in the repo. The checkpoint is not (134 MB, too big for git): download `best_model.pth` from [Google Drive](https://drive.google.com/drive/folders/1Iu6XkYwWm-vF9WOIyMR8yKHrIZDDOUxY?usp=sharing) into `outputs/resnet18/`.

## How the two subgroups use the baseline

**Phase I — Model Behavior & Robustness:** use the shared predictions and model to inspect class-wise errors, confusion patterns, sensitivity to lighting/blur/background changes, and visualize the results using confusion matrices, heatmaps, or clustering.

**Phase II — Model Explainability:** load the same `best_model.pth` and inspect correct/incorrect cases using Grad-CAM and Integrated Gradients, including whether predictions rely on disease symptoms or backgrounds. Saliency is evidence of model sensitivity, not proof of biological understanding.

Both subgroups should use the same selected checkpoint and preserve the untouched test data for comparable analyses.