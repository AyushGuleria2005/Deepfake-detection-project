import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from models.resnet_baseline import ResNetBaseline
from training.dataloader import DeepfakeDataset, eval_transform


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "checkpoints"
    / "resnet18_seen_nt_control_best.pth"
)

TEST_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "test_seen_neuraltextures_control.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8

# Same threshold used in our previous analysis
THRESHOLD = 0.55

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("SEEN NEURALTEXTURES CONTROL - TEST")
print("=" * 75)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Checkpoint:")
print(CHECKPOINT)

print()
print("Test CSV:")
print(TEST_CSV)

print()
print("Threshold:", THRESHOLD)


# ============================================================
# DATASET
# ============================================================

dataset = DeepfakeDataset(
    TEST_CSV,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = ResNetBaseline().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()


# ============================================================
# INFERENCE
# ============================================================

all_labels = []
all_probs = []

total_loss = 0.0

criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(
            images
        ).squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probs = torch.sigmoid(
            outputs
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )


# ============================================================
# ARRAYS
# ============================================================

labels = torch.tensor(
    all_labels
).numpy()

probs = torch.tensor(
    all_probs
).numpy()

predictions = (
    probs >= THRESHOLD
).astype(int)

avg_loss = (
    total_loss / len(dataset)
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    labels,
    probs
)

cm = confusion_matrix(
    labels,
    predictions
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print("=" * 75)
print("OVERALL RESULTS")
print("=" * 75)

print("Samples:", len(dataset))

print(
    f"Loss: {avg_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy * 100:.2f}%"
)

print(
    f"Fake Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Fake Recall: "
    f"{recall * 100:.2f}%"
)

print(
    f"Fake F1: "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC: "
    f"{auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")
print(
    "[[TN  FP]"
)
print(
    " [FN  TP]]"
)
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Original / Real",
            "NeuralTextures / Fake"
        ],
        digits=4
    )
)


# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

real_probs = probs[
    labels == 0
]

nt_probs = probs[
    labels == 1
]

print()
print("=" * 75)
print("PROBABILITY ANALYSIS")
print("=" * 75)

print(
    f"Original mean probability: "
    f"{real_probs.mean():.4f}"
)

print(
    f"NeuralTextures mean probability: "
    f"{nt_probs.mean():.4f}"
)

print()
print("Prediction Distribution:")

print(
    "Predicted Real:",
    (predictions == 0).sum()
)

print(
    "Predicted Fake:",
    (predictions == 1).sum()
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("CONTROL TEST COMPLETE")
print("=" * 75)