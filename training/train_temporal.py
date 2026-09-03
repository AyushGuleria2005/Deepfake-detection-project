import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from training.temporal_dataloader import (
    TemporalDeepfakeDataset,
    temporal_transform,
    eval_transform,
)

from models.temporal_resnet_lstm import TemporalResNetLSTM


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"

CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_CSV = EXPERIMENTS_DIR / "train_temporal.csv"
VAL_CSV = EXPERIMENTS_DIR / "validation_temporal.csv"

BATCH_SIZE = 4
EPOCHS = 5

LEARNING_RATE = 1e-4

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# DATASETS
# =========================================================

train_dataset = TemporalDeepfakeDataset(
    TRAIN_CSV,
    transform=temporal_transform
)

val_dataset = TemporalDeepfakeDataset(
    VAL_CSV,
    transform=eval_transform
)


# =========================================================
# WEIGHTED SAMPLER
#
# Train distribution:
# Real = 800
# Fake = 4000
#
# We compensate for the 1:5 imbalance.
# =========================================================

class_counts = (
    train_dataset.data["label"]
    .value_counts()
    .sort_index()
)

print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(class_counts)


class_weights = {
    label: 1.0 / count
    for label, count in class_counts.items()
}


sample_weights = (
    train_dataset.data["label"]
    .map(class_weights)
    .to_numpy()
)


sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(
        sample_weights
    ),
    num_samples=len(train_dataset),
    replacement=True
)


# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# =========================================================
# MODEL
# =========================================================

model = TemporalResNetLSTM().to(device)


# =========================================================
# LOSS
# =========================================================

criterion = nn.BCEWithLogitsLoss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# =========================================================
# TRAINING
# =========================================================

best_auc = -1.0

best_checkpoint = (
    CHECKPOINT_DIR
    / "temporal_resnet_lstm_best.pth"
)


print()
print("=" * 70)
print("TEMPORAL RESNET18 + LSTM TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)

print()


for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_labels = []

    for frames, labels in train_loader:

        frames = frames.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(frames)

        outputs = outputs.squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * frames.size(0)
        )

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    train_loss = (
        running_loss
        / len(train_dataset)
    )

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    val_probabilities = []
    val_labels = []

    with torch.no_grad():

        for frames, labels in val_loader:

            frames = frames.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(frames)

            outputs = outputs.squeeze(1)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * frames.size(0)
            )

            probabilities = torch.sigmoid(
                outputs
            )

            val_probabilities.extend(
                probabilities.cpu()
                .numpy()
            )

            val_labels.extend(
                labels.cpu()
                .numpy()
            )


    val_loss = (
        val_loss_total
        / len(val_dataset)
    )

    val_predictions = (
        np.array(val_probabilities)
        >= 0.5
    ).astype(int)

    val_labels_array = np.array(
        val_labels
    )

    val_accuracy = accuracy_score(
        val_labels_array,
        val_predictions
    )

    val_balanced_accuracy = (
        balanced_accuracy_score(
            val_labels_array,
            val_predictions
        )
    )

    val_auc = roc_auc_score(
        val_labels_array,
        val_probabilities
    )


    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"Val Balanced Acc: "
        f"{val_balanced_accuracy * 100:.2f}% | "
        f"Val ROC-AUC: {val_auc:.4f}"
    )


    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy":
                    val_balanced_accuracy,
            },
            best_checkpoint
        )

        print(
            "✓ Best checkpoint saved"
        )

    print()


# =========================================================
# FINISHED
# =========================================================

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best Validation ROC-AUC:",
    f"{best_auc:.4f}"
)

print()
print(
    "Best checkpoint:",
    best_checkpoint
)

print("=" * 70)