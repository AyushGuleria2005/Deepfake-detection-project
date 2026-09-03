import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from torch.utils.data import DataLoader
from models.resnet_baseline import ResNetBaseline
from training.dataloader import DeepfakeDataset, eval_transform

BASE_DIR = Path(r"D:\DeepFake-Learning")

CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "resnet18_face_crop_best.pth"
CSV_FILE = BASE_DIR / "research" / "experiments" / "validation_face.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("RESNET18 FACE-CROP DIAGNOSTIC")
print("=" * 60)
print("Device:", device)

# Dataset
dataset = DeepfakeDataset(
    CSV_FILE,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Model
model = ResNetBaseline().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

criterion = nn.BCEWithLogitsLoss()

all_labels = []
all_probs = []
total_loss = 0.0

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images).squeeze(1)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert
labels = torch.tensor(all_labels)
probs = torch.tensor(all_probs)

predictions = (probs >= 0.5).long()

avg_loss = total_loss / len(dataset)

# Metrics
accuracy = (predictions == labels.long()).float().mean().item()

cm = confusion_matrix(
    labels.numpy(),
    predictions.numpy()
)

auc = roc_auc_score(
    labels.numpy(),
    probs.numpy()
)

print()
print("Samples:", len(dataset))
print(f"Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC: {auc:.4f}")

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")
print(
    classification_report(
        labels.numpy(),
        predictions.numpy(),
        target_names=["Real", "Fake"],
        digits=4
    )
)

# Probability analysis
real_probs = probs[labels == 0]
fake_probs = probs[labels == 1]

print()
print("Probability Analysis:")
print(f"Real probability mean: {real_probs.mean():.4f}")
print(f"Fake probability mean: {fake_probs.mean():.4f}")

print()
print("Prediction Distribution:")
print("Predicted Real:", (predictions == 0).sum().item())
print("Predicted Fake:", (predictions == 1).sum().item())

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)