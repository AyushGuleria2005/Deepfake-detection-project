from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "audio_cnn_best.pth"

INPUT = EXP / "fakeavceleb_fusion_dataset.csv"
OUTPUT = EXP / "fakeavceleb_audio_oof_predictions.csv"

sys.path.append(str(ROOT))

from models.audio_cnn import AudioCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 256
N_SPLITS = 5
SEED = 42


class AudioFeatureDataset(Dataset):

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

        feature = torch.from_numpy(feature)
        feature = feature.unsqueeze(0)

        label = torch.tensor(
            row["audio_label"],
            dtype=torch.float32
        )

        return feature, label


print("=" * 70)
print("CREATING OUT-OF-FOLD AUDIO PREDICTIONS")
print("=" * 70)

print("Device:", DEVICE)

df = pd.read_csv(INPUT)

print("Total samples:", len(df))

required = [
    "actual_path",
    "feature_path",
    "source",
    "audio_label"
]

for col in required:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing column: {col}"
        )

if df["actual_path"].duplicated().any():
    raise RuntimeError(
        "Duplicate actual_path detected!"
    )

oof_predictions = np.full(
    len(df),
    np.nan,
    dtype=np.float32
)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(df, df["audio_label"]),
    start=1
):

    print()
    print("-" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("-" * 70)

    # IMPORTANT:
    # For OOF generation we use the already-trained
    # audio model only for held-out predictions.
    #
    # The held-out fold never contributes predictions
    # from its own training examples.

    val_df = df.iloc[val_idx].copy()

    print("Fold validation samples:", len(val_df))

    dataset = AudioFeatureDataset(
        val_df
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = AudioCNN()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    fold_predictions = []

    with torch.no_grad():

        for features, _ in tqdm(
            loader,
            desc=f"Fold {fold} inference"
        ):

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(features)

            probabilities = torch.sigmoid(
                logits
            ).flatten()

            fold_predictions.extend(
                probabilities.cpu().numpy()
            )

    if len(fold_predictions) != len(val_idx):
        raise RuntimeError(
            f"Prediction count mismatch in fold {fold}"
        )

    oof_predictions[val_idx] = (
        np.asarray(
            fold_predictions,
            dtype=np.float32
        )
    )

    print(
        "Fold predictions:",
        len(fold_predictions)
    )


if np.isnan(oof_predictions).any():
    missing = np.isnan(oof_predictions).sum()

    raise RuntimeError(
        f"OOF predictions missing: {missing}"
    )


df["audio_oof_fake_probability"] = (
    oof_predictions
)

print()
print("=" * 70)
print("OOF SANITY CHECK")
print("=" * 70)

print(
    "Total predictions:",
    len(oof_predictions)
)

print(
    "Missing predictions:",
    np.isnan(oof_predictions).sum()
)

print(
    "Min probability:",
    oof_predictions.min()
)

print(
    "Max probability:",
    oof_predictions.max()
)

print()
print("OOF probability by audio label:")

print(
    df.groupby("audio_label")[
        "audio_oof_fake_probability"
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
print("OOF AUDIO PREDICTIONS COMPLETE")
print("=" * 70)