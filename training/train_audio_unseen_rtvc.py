import os
import sys

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

from models.audio_cnn import AudioCNN


# ============================================================
# PATHS
# ============================================================

BASE = PROJECT_ROOT

EXP = os.path.join(
    BASE,
    "research",
    "experiments"
)

CKPT_DIR = os.path.join(
    BASE,
    "outputs",
    "checkpoints"
)

TRAIN_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_unseen_rtvc_train.csv"
)

VAL_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_validation.csv"
)

CHECKPOINT = os.path.join(
    CKPT_DIR,
    "audio_cnn_unseen_rtvc_best.pth"
)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature_path = str(
            row["feature_path"]
        )

        audio_label = int(
            row["audio_label"]
        )

        feature = np.load(
            feature_path
        ).astype(
            np.float32
        )

        # [64, 512]
        # -> [1, 64, 512]

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            audio_label,
            dtype=torch.float32
        )

        return feature, label


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("UNSEEN AUDIO METHOD TRAINING")
print("=" * 70)

print(f"Device: {device}")

if device.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD UNSEEN TRAIN SPLIT
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

print(
    f"\nUnseen training CSV samples: "
    f"{len(train_df)}"
)


# ============================================================
# LOAD EXISTING FEATURE MAPPINGS
# ============================================================

print(
    "\nLoading existing feature mappings..."
)

feature_maps = []

for split in [
    "train",
    "validation",
    "test"
]:

    meta_path = os.path.join(
        EXP,
        f"fakeavceleb_audio_{split}.csv"
    )

    df = pd.read_csv(
        meta_path
    )

    print(
        f"{split}: {len(df)} rows"
    )

    feature_maps.append(
        df[
            [
                "actual_path",
                "feature_path"
            ]
        ]
    )


feature_map = pd.concat(
    feature_maps,
    ignore_index=True
)


# Remove duplicate mappings

feature_map = feature_map.drop_duplicates(
    subset=["actual_path"],
    keep="first"
)


print(
    f"Unique feature mappings: "
    f"{len(feature_map)}"
)


# ============================================================
# ATTACH FEATURE PATHS
# ============================================================

train_df = train_df.merge(
    feature_map,
    on="actual_path",
    how="left"
)


# ============================================================
# SAFETY CHECK
# ============================================================

missing_features = train_df[
    "feature_path"
].isna().sum()


print(
    f"Missing training features: "
    f"{missing_features}"
)


if missing_features > 0:

    raise RuntimeError(
        "Some training samples do not have "
        "preprocessed audio features."
    )


# ============================================================
# SAVE MERGED TRAIN CSV
# ============================================================

TEMP_TRAIN_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_unseen_rtvc_train_with_features.csv"
)

train_df.to_csv(
    TEMP_TRAIN_CSV,
    index=False
)


print(
    "\nMerged training CSV saved:"
)

print(
    TEMP_TRAIN_CSV
)


# ============================================================
# DATASETS
# ============================================================

train_ds = AudioDataset(
    TEMP_TRAIN_CSV
)

val_ds = AudioDataset(
    VAL_CSV
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = AudioCNN().to(
    device
)


total_params = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"\nTrain samples: "
    f"{len(train_ds)}"
)

print(
    f"Validation samples: "
    f"{len(val_ds)}"
)

print(
    f"Parameters: "
    f"{total_params:,}"
)


# ============================================================
# CLASS BALANCE
# ============================================================

train_labels = (
    train_ds.df[
        "audio_label"
    ].values
)


real_count = np.sum(
    train_labels == 0
)

fake_count = np.sum(
    train_labels == 1
)


pos_weight = (
    real_count /
    fake_count
)


print(
    f"Real audio: "
    f"{real_count}"
)

print(
    f"Fake audio: "
    f"{fake_count}"
)

print(
    f"Positive weight: "
    f"{pos_weight:.4f}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=device
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

epochs = 10

best_auc = 0.0

os.makedirs(
    CKPT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    epochs + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    samples_seen = 0


    for features, labels in train_loader:

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        ).unsqueeze(1)


        optimizer.zero_grad()


        logits = model(
            features
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        batch_size = (
            features.size(0)
        )


        running_loss += (
            loss.item()
            * batch_size
        )

        samples_seen += (
            batch_size
        )


    train_loss = (
        running_loss /
        samples_seen
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_probs = []

    val_labels = []


    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(
                device,
                non_blocking=True
            )


            logits = model(
                features
            )


            probs = torch.sigmoid(
                logits
            ).squeeze(1)


            val_probs.extend(
                probs.cpu().numpy()
            )

            val_labels.extend(
                labels.numpy()
            )


    val_auc = roc_auc_score(
        val_labels,
        val_probs
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_auc > best_auc:

        best_auc = val_auc


        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "val_auc": val_auc
            },
            CHECKPOINT
        )


        print(
            f"  --> BEST CHECKPOINT SAVED "
            f"(AUC={val_auc:.4f})"
        )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Best validation AUC: "
    f"{best_auc:.4f}"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT}"
)

print(
    "=" * 70
)