from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.fusion_baseline import FusionBaseline


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"
CKPT_DIR = ROOT / "outputs" / "checkpoints"

INPUT = EXP / "fakeavceleb_fusion_dataset.csv"
OUTPUT = CKPT_DIR / "fusion_baseline_best.pth"

SEED = 42
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FusionDataset(Dataset):

    def __init__(self, df):
        self.visual = torch.tensor(
            df["visual_fake_probability"].values,
            dtype=torch.float32
        ).unsqueeze(1)

        self.audio = torch.tensor(
            df["audio_fake_probability"].values,
            dtype=torch.float32
        ).unsqueeze(1)

        self.labels = torch.tensor(
            df["video_label"].values,
            dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.visual[idx],
            self.audio[idx],
            self.labels[idx]
        )


print("=" * 70)
print("TRAINING MULTIMODAL FUSION BASELINE")
print("=" * 70)

seed_everything(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

df = pd.read_csv(INPUT)

print("Total samples:", len(df))

# ---------------------------------------------------------
# IMPORTANT:
# Split by SOURCE, not by individual video.
# This prevents source leakage.
# ---------------------------------------------------------

sources = df["source"].drop_duplicates().sample(
    frac=1,
    random_state=SEED
).tolist()

n_sources = len(sources)

train_sources = set(
    sources[:int(0.8 * n_sources)]
)

val_sources = set(
    sources[int(0.8 * n_sources):int(0.9 * n_sources)]
)

test_sources = set(
    sources[int(0.9 * n_sources):]
)

train_df = df[
    df["source"].isin(train_sources)
].copy()

val_df = df[
    df["source"].isin(val_sources)
].copy()

test_df = df[
    df["source"].isin(test_sources)
].copy()

print("\nSource split:")
print("Train sources:", len(train_sources))
print("Validation sources:", len(val_sources))
print("Test sources:", len(test_sources))

print("\nSample split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ---------------------------------------------------------
# Check source overlap
# ---------------------------------------------------------

if train_sources & val_sources:
    raise RuntimeError("Train/validation source overlap!")

if train_sources & test_sources:
    raise RuntimeError("Train/test source overlap!")

if val_sources & test_sources:
    raise RuntimeError("Validation/test source overlap!")

print("Source overlap: 0")

# ---------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------

train_dataset = FusionDataset(train_df)
val_dataset = FusionDataset(val_df)
test_dataset = FusionDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = FusionBaseline().to(device)

print("\nParameters:", sum(
    p.numel() for p in model.parameters()
))

# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------

train_labels = train_df["video_label"].values

real_count = (train_labels == 0).sum()
fake_count = (train_labels == 1).sum()

pos_weight = torch.tensor(
    [real_count / fake_count],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

print("Train real:", real_count)
print("Train fake:", fake_count)
print("Pos weight:", pos_weight.item())

# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate(loader):

    model.eval()

    all_labels = []
    all_probs = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for visual, audio, labels in loader:

            visual = visual.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            logits = model(
                visual,
                audio
            )

            loss = criterion(
                logits,
                labels
            )

            probs = torch.sigmoid(logits)

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_samples += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy().ravel()
            )

            all_probs.extend(
                probs.cpu().numpy().ravel()
            )

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    preds = (
        all_probs >= 0.5
    ).astype(int)

    bal_acc = balanced_accuracy_score(
        all_labels,
        preds
    )

    return (
        total_loss / total_samples,
        auc,
        bal_acc
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_auc = -1.0

CKPT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    total = 0

    for visual, audio, labels in train_loader:

        visual = visual.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(
            visual,
            audio
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        total += labels.size(0)

    train_loss = running_loss / total

    val_loss, val_auc, val_bal_acc = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val AUC: {val_auc:.4f} | "
        f"Val Bal.Acc: {val_bal_acc:.4f}"
    )

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_balanced_accuracy": val_bal_acc
            },
            OUTPUT
        )

        print(
            f"  -> Best checkpoint saved "
            f"(AUC={val_auc:.4f})"
        )

# ---------------------------------------------------------
# Final test
# ---------------------------------------------------------

checkpoint = torch.load(
    OUTPUT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_auc, test_bal_acc = evaluate(
    test_loader
)

print("\n" + "=" * 70)
print("FINAL FUSION BASELINE TEST")
print("=" * 70)

print("Best epoch:", checkpoint["epoch"])
print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
print(f"Test AUC:       {test_auc:.4f}")
print(f"Test Bal.Acc:   {test_bal_acc:.4f}")

print("\nCheckpoint:")
print(OUTPUT)

print("=" * 70)
print("FUSION TRAINING COMPLETE")
print("=" * 70)