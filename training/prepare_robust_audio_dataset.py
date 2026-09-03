import os
import sys

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from models.audio_cnn import AudioCNN


BASE_DIR = r"D:\DeepFake-Learning"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_generator_holdout_train.csv"
)

CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "audio_cnn_robust_best.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class RobustAudioDataset(Dataset):

    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature = np.load(row["feature_path"]).astype(np.float32)

        # Expected shape: (64, 512)
        if feature.shape != (64, 512):
            raise ValueError(
                f"Unexpected feature shape: {feature.shape}"
            )

        # -------------------------------------------------
        # Robust augmentation
        # -------------------------------------------------

        if self.augment:

            # Random feature-level gain shift
            if random.random() < 0.50:
                gain = random.uniform(-2.0, 2.0)
                feature = feature + gain

            # Time masking
            if random.random() < 0.40:
                width = random.randint(8, 40)
                start = random.randint(
                    0,
                    512 - width
                )
                feature[:, start:start + width] = 0

            # Frequency masking
            if random.random() < 0.30:
                width = random.randint(2, 8)
                start = random.randint(
                    0,
                    64 - width
                )
                feature[start:start + width, :] = 0

            # Small Gaussian noise
            if random.random() < 0.30:
                noise = np.random.normal(
                    0,
                    0.05,
                    feature.shape
                ).astype(np.float32)

                feature = feature + noise

        feature = np.clip(feature, -20, 10)

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.float32
        )

        return feature, label


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("=" * 60)
print("ROBUST AUDIO TRAINING")
print("=" * 60)

df = pd.read_csv(TRAIN_CSV)

print("Total samples:", len(df))

# Safety checks
if "rtvc" in df["method"].astype(str).str.lower().unique():
    raise RuntimeError(
        "RTVC found in training data!"
    )

if df["actual_path"].duplicated().any():
    raise RuntimeError(
        "Duplicate physical audio/video paths found!"
    )

missing = df[
    ~df["feature_path"].apply(os.path.exists)
]

print("Missing features:", len(missing))

if len(missing) > 0:
    raise RuntimeError(
        "Some feature files are missing."
    )


# ---------------------------------------------------------
# Source-level train/validation split
# ---------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_idx, val_idx = next(
    splitter.split(
        df,
        groups=df["source"]
    )
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

train_sources = set(train_df["source"])
val_sources = set(val_df["source"])

print()
print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))

print(
    "Train sources:",
    len(train_sources)
)

print(
    "Validation sources:",
    len(val_sources)
)

print(
    "Source overlap:",
    len(train_sources & val_sources)
)

if train_sources & val_sources:
    raise RuntimeError(
        "Source leakage detected!"
    )


# ---------------------------------------------------------
# Class distribution
# ---------------------------------------------------------

print()
print("Train labels:")
print(train_df["audio_label"].value_counts())

print()
print("Validation labels:")
print(val_df["audio_label"].value_counts())


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

train_dataset = RobustAudioDataset(
    train_df,
    augment=True
)

val_dataset = RobustAudioDataset(
    val_df,
    augment=False
)


# ---------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------

class_counts = train_df[
    "audio_label"
].value_counts()

weights = train_df[
    "audio_label"
].map(
    {
        0: 1.0 / class_counts.get(0, 1),
        1: 1.0 / class_counts.get(1, 1)
    }
).values

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(weights),
    num_samples=len(train_df),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = AudioCNN().to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

EPOCHS = 12

best_auc = -1

print()
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size: 64")
print()


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for features, labels in train_loader:

        features = features.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        ).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(features)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * features.size(0)
        )

    train_loss = (
        running_loss /
        len(train_loader.dataset)
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(features)

            probs = torch.sigmoid(
                logits
            ).squeeze(1)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    val_auc = roc_auc_score(
        all_labels,
        all_probs
    )

    scheduler.step(val_auc)

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.6f} | "
        f"Val AUC: {val_auc:.4f} | "
        f"LR: {lr:.2e}"
    )


    # -----------------------------------------------------
    # Save best
    # -----------------------------------------------------

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_auc":
                    val_auc
            },
            CHECKPOINT
        )

        print(
            f"  ✓ Best checkpoint saved "
            f"(AUC={val_auc:.4f})"
        )


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation AUC: {best_auc:.4f}"
)

print(
    "Checkpoint:",
    CHECKPOINT
)