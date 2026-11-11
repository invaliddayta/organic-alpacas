import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader

class PlantVillageDataset(Dataset):
    def __init__(self, dataset, transform=None, image_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = int(sample["class_idx"])

        # Optional perturbation or masking before preprocessing
        if self.image_transform is not None:
            image = self.image_transform(image, idx)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


@torch.inference_mode()
def evaluate(
    model,
    dataset,
    device,
    transform,
    image_transform=None,
    batch_size=32,
    description="Evaluating"
):
    loader = DataLoader(
        PlantVillageDataset(
            dataset,
            transform=transform,
            image_transform=image_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model.eval()
    true_labels = []
    pred_labels = []

    for images, labels in tqdm(loader, desc=description, unit="batch"):
        predictions = model(
            images.to(device)
        ).argmax(dim=1)

        true_labels.extend(labels.numpy())
        pred_labels.extend(predictions.cpu().numpy())

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    return {
        "accuracy": 100 * np.mean(true_labels == pred_labels),
        "true": true_labels,
        "pred": pred_labels
    }

def class_recall(y_true, y_pred, num_classes):
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(num_classes)
    )
    totals = cm.sum(axis=1)

    return np.divide(
        np.diag(cm),
        totals,
        out=np.full(num_classes, np.nan),
        where=totals > 0
    ) * 100