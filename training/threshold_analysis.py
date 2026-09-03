import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

from models.resnet_baseline import ResNetBaseline
from training.dataloader import DeepfakeDataset, eval_transform

BASE_DIR = Path(r"D:\DeepFake-Learning")

CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "resnet18_face_crop_best.pth"
CSV_FILE = BASE_DIR / "research" / "experiments" / "validation_face.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in loader:

        images = images.to(device, non_blocking=True)

        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

labels = torch.tensor(all_labels).numpy()
probs = torch.tensor(all_probs).numpy()

print("=" * 75)
print("THRESHOLD ANALYSIS - RESNET18 FACE CROP")
print("=" * 75)

results = []

for threshold in [i / 100 for i in range(10, 91, 5)]:

    predictions = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)

    precision = precision_score(
        labels, predictions, zero_division=0
    )

    recall = recall_score(
        labels, predictions, zero_division=0
    )

    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    cm = confusion_matrix(labels, predictions)

    tn, fp, fn, tp = cm.ravel()

    real_recall = tn / (tn + fp)
    fake_recall = tp / (tp + fn)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "real_recall": real_recall,
        "fake_recall": fake_recall
    })

print()
print(
    f"{'Threshold':<10}"
    f"{'Accuracy':<11}"
    f"{'Bal.Acc':<11}"
    f"{'Fake Prec':<11}"
    f"{'Fake Rec':<11}"
    f"{'Fake F1':<10}"
    f"{'Real Rec':<10}"
)
print("-" * 75)

for r in results:
    print(
        f"{r['threshold']:<10.2f}"
        f"{r['accuracy']*100:<11.2f}"
        f"{r['balanced_accuracy']*100:<11.2f}"
        f"{r['precision_fake']*100:<11.2f}"
        f"{r['recall_fake']*100:<11.2f}"
        f"{r['f1_fake']:.4f}    "
        f"{r['real_recall']*100:<10.2f}"
    )

best = max(
    results,
    key=lambda x: x["balanced_accuracy"]
)

print()
print("=" * 75)
print("BEST THRESHOLD BY BALANCED ACCURACY")
print("=" * 75)

print(f"Threshold: {best['threshold']:.2f}")
print(f"Accuracy: {best['accuracy']*100:.2f}%")
print(f"Balanced Accuracy: {best['balanced_accuracy']*100:.2f}%")
print(f"Fake Precision: {best['precision_fake']*100:.2f}%")
print(f"Fake Recall: {best['recall_fake']*100:.2f}%")
print(f"Fake F1: {best['f1_fake']:.4f}")
print(f"Real Recall: {best['real_recall']*100:.2f}%")

print()
print("=" * 75)