import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# dataset checks
def get_leaf_groups(ds):
    return {
        (row["host"], row["disease"], row["leaf_id"])
        for row in ds
        if row["leaf_grouped"] and row["leaf_id"] is not None
    }

def plot_class_distributions(
    class_counts,
    barplot=True,
    heatmap=False,
    difference_heatmap=True
):
    """Visualize class distributions across train, validation and test splits."""

    sns.set_theme(style="whitegrid", context="notebook")

    splits = ["Train", "Validation", "Test"]

    # Calculate relative class frequencies (%)
    class_pct = (
        class_counts[splits]
        .div(class_counts[splits].sum(axis=0), axis=1)
        * 100
    )

    # Sort classes by total frequency
    class_order = class_counts[splits].sum(axis=1).sort_values(
        ascending=False
    ).index

    # 1. Horizontal grouped bar chart
    if barplot:
        plot_data = (
            class_pct.loc[class_order]
            .rename_axis("Class")
            .reset_index()
            .melt(id_vars="Class", var_name="Split", value_name="Percentage")
        )

        plt.figure(figsize=(12, 15))
        sns.barplot(
            data=plot_data,
            x="Percentage",
            y="Class",
            hue="Split",
            order=class_order,
            palette="Set2"
        )
        plt.title("Class Distribution Across Dataset Splits")
        plt.xlabel("Percentage of Split (%)")
        plt.ylabel("")
        sns.despine()
        plt.tight_layout()
        plt.show()

# training procedure
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total
    
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    patience=3,
    min_delta=0.001,
    output_dir="outputs/resnet18"
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = output_dir / "best_model.pth"
    last_checkpoint = output_dir / "last_model.pth"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs), desc="Training"):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Record metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_checkpoint)

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Save final training state
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, last_checkpoint)

    # Export training history
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return history

def plot_training_history(history):
    sns.set_theme(style="whitegrid")

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set(
        title="Training and Validation Loss",
        xlabel="Epoch",
        ylabel="Cross-Entropy Loss"
    )
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Validation")
    axes[1].set(
        title="Training and Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy"
    )
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def export_test_predictions(model, loader, dataset, device, output_path):
    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Test inference"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            predictions.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    probabilities = np.asarray(probabilities)

    # Build prediction DataFrame using original metadata
    df = pd.DataFrame({
        "sample_idx": range(len(dataset)),
        "true_idx": dataset["class_idx"],
        "true_label": dataset["class_label"],
        "pred_idx": predictions,
        "confidence": probabilities.max(axis=1),
        "host": dataset["host"],
        "disease": dataset["disease"],
        "leaf_id": dataset["leaf_id"],
        "leaf_grouped": dataset["leaf_grouped"]
    })

    # Mark correct / incorrect predictions
    df["correct"] = df["true_idx"] == df["pred_idx"]

    # Save all class probabilities for later analysis
    for i in range(probabilities.shape[1]):
        df[f"prob_{i}"] = probabilities[:, i]

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} predictions to {output_path}")
    print(f"Test accuracy: {df['correct'].mean():.2%}")

    return df