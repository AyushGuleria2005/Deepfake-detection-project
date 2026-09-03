from pathlib import Path
import sys
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

INPUT = EXP / "fakeavceleb_fusion_dataset.csv"
OUTPUT = EXP / "fakeavceleb_audio_true_oof.csv"

sys.path.append(str(ROOT))

from models.audio_cnn import AudioCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
N_SPLITS = 5

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3


torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class AudioDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature = np.load(
            row["feature_path"]
        ).astype(np.float32)

        if feature.shape != (64, 512):
            raise RuntimeError(
                f"Unexpected feature shape: {feature.shape}"
            )

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            row["audio_label"],
            dtype=torch.float32
        )

        return feature, label


def train_one_fold(train_df, val_df, fold):

    print()
    print("=" * 70)
    print(f"TRAINING AUDIO MODEL — FOLD {fold}")
    print("=" * 70)

    print("Train samples:", len(train_df))
    print("Held-out samples:", len(val_df))

    train_dataset = AudioDataset(train_df)
    val_dataset = AudioDataset(val_df)

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

    model = AudioCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    real_count = (
        train_df["audio_label"] == 0
    ).sum()

    fake_count = (
        train_df["audio_label"] == 1
    ).sum()

    pos_weight = (
        real_count / max(fake_count, 1)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=DEVICE
        )
    )

    best_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_samples = 0

        for features, labels in tqdm(
            train_loader,
            desc=f"Fold {fold} Epoch {epoch}/{EPOCHS}"
        ):

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

            optimizer.step()

            batch_size = features.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss /
            max(total_samples, 1)
        )

        # Validation on completely held-out sources.
        model.eval()

        probabilities = []
        targets = []

        with torch.no_grad():

            for features, labels in val_loader:

                features = features.to(
                    DEVICE,
                    non_blocking=True
                )

                logits = model(features)

                probs = torch.sigmoid(
                    logits
                ).flatten()

                probabilities.extend(
                    probs.cpu().numpy()
                )

                targets.extend(
                    labels.numpy()
                )

        val_auc = roc_auc_score(
            targets,
            probabilities
        )

        print(
            f"Epoch {epoch}: "
            f"loss={train_loss:.4f} "
            f"held-out AUC={val_auc:.4f}"
        )

        if val_auc > best_auc:

            best_auc = val_auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

    # Restore best fold model.
    model.load_state_dict(best_state)

    model.eval()

    predictions = []

    with torch.no_grad():

        for features, _ in tqdm(
            val_loader,
            desc=f"Fold {fold} OOF prediction"
        ):

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(features)

            probs = torch.sigmoid(
                logits
            ).flatten()

            predictions.extend(
                probs.cpu().numpy()
            )

    return np.asarray(
        predictions,
        dtype=np.float32
    ), best_auc


print("=" * 70)
print("TRUE SOURCE-LEVEL AUDIO OOF")
print("=" * 70)

print("Device:", DEVICE)

df = pd.read_csv(INPUT)

print("Total samples:", len(df))
print("Unique sources:", df["source"].nunique())

if df["actual_path"].duplicated().any():
    raise RuntimeError(
        "Duplicate actual_path detected!"
    )

if df["source"].isna().any():
    raise RuntimeError(
        "Missing source IDs detected!"
    )

if df["audio_label"].nunique() != 2:
    raise RuntimeError(
        "Audio labels must contain both classes!"
    )


groups = df["source"].values
labels = df["audio_label"].values

# GroupKFold guarantees that a source never
# appears in both training and held-out fold.
gkf = GroupKFold(
    n_splits=N_SPLITS
)

oof = np.full(
    len(df),
    np.nan,
    dtype=np.float32
)

fold_auc = []


for fold, (train_idx, val_idx) in enumerate(
    gkf.split(
        df,
        labels,
        groups
    ),
    start=1
):

    train_df = df.iloc[
        train_idx
    ].copy()

    val_df = df.iloc[
        val_idx
    ].copy()

    train_sources = set(
        train_df["source"]
    )

    val_sources = set(
        val_df["source"]
    )

    overlap = (
        train_sources &
        val_sources
    )

    print()
    print(
        f"Source overlap in fold {fold}:",
        len(overlap)
    )

    if overlap:
        raise RuntimeError(
            f"Source leakage detected in fold {fold}!"
        )

    predictions, auc = train_one_fold(
        train_df,
        val_df,
        fold
    )

    if len(predictions) != len(val_idx):
        raise RuntimeError(
            f"Prediction count mismatch in fold {fold}"
        )

    oof[val_idx] = predictions

    fold_auc.append(auc)


if np.isnan(oof).any():

    missing = np.isnan(oof).sum()

    raise RuntimeError(
        f"Missing OOF predictions: {missing}"
    )


df["audio_true_oof_fake_probability"] = oof


overall_auc = roc_auc_score(
    df["audio_label"],
    df["audio_true_oof_fake_probability"]
)

print()
print("=" * 70)
print("FINAL OOF RESULTS")
print("=" * 70)

for i, auc in enumerate(
    fold_auc,
    start=1
):
    print(
        f"Fold {i} AUC: {auc:.4f}"
    )

print(
    f"Overall OOF AUC: {overall_auc:.4f}"
)

print(
    "Missing OOF predictions:",
    np.isnan(oof).sum()
)

print()
print("OOF predictions by audio label:")

print(
    df.groupby("audio_label")[
        "audio_true_oof_fake_probability"
    ].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)

print("=" * 70)
print("TRUE SOURCE-LEVEL OOF COMPLETE")
print("=" * 70)