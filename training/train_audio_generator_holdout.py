from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\DeepFake-Learning")

sys.path.insert(0, str(PROJECT_ROOT))

from models.audio_cnn import AudioCNN


TRAIN_CSV = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "audio_generator_holdout_train.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "audio_cnn_generator_holdout_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 0
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class AudioFeatureDataset(Dataset):

    def __init__(self, df):

        self.df = df.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature_path = row["feature_path"]

        feature = np.load(
            feature_path
        ).astype(np.float32)

        if feature.shape != (64, 512):

            raise RuntimeError(
                f"Unexpected feature shape "
                f"{feature.shape} for:\n"
                f"{feature_path}"
            )

        # [64, 512] -> [1, 64, 512]

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.float32
        )

        return feature, label


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    print("=" * 70)
    print("LOADING GENERATOR-HOLDOUT TRAINING DATA")
    print("=" * 70)

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Training sources:",
        train_df["source"].nunique()
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "actual_path",
        "feature_path",
        "source",
        "method",
        "audio_label"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in train_df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f"Missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Missing feature paths
    # --------------------------------------------------------

    missing_features = (
        train_df[
            "feature_path"
        ].isna().sum()
    )

    print(
        "Missing feature paths:",
        missing_features
    )

    if missing_features > 0:

        raise RuntimeError(
            f"Missing feature paths: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Physical feature-file existence
    # --------------------------------------------------------

    nonexistent = []

    for path in train_df[
        "feature_path"
    ]:

        if not Path(path).exists():

            nonexistent.append(
                path
            )

            if len(nonexistent) >= 10:
                break

    if nonexistent:

        print()
        print(
            "Example missing feature files:"
        )

        for path in nonexistent:
            print(path)

        raise RuntimeError(
            "Audio feature files are missing "
            "from disk."
        )

    # --------------------------------------------------------
    # Duplicate physical samples
    # --------------------------------------------------------

    duplicate_paths = (
        train_df[
            "actual_path"
        ].duplicated().sum()
    )

    print(
        "Duplicate physical paths:",
        duplicate_paths
    )

    if duplicate_paths > 0:

        raise RuntimeError(
            "Duplicate physical samples "
            "detected!"
        )

    # --------------------------------------------------------
    # Verify RTVC is absent
    # --------------------------------------------------------

    rtvc_count = (
        train_df[
            "method"
        ]
        .astype(str)
        .str.lower()
        .eq("rtvc")
        .sum()
    )

    print(
        "RTVC samples in training:",
        rtvc_count
    )

    if rtvc_count != 0:

        raise RuntimeError(
            "RTVC LEAKAGE DETECTED!"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print()
    print(
        "Audio labels:"
    )

    print(
        train_df[
            "audio_label"
        ].value_counts().sort_index()
    )

    print()
    print(
        "Training methods:"
    )

    print(
        train_df[
            "method"
        ].value_counts()
    )

    if train_df[
        "audio_label"
    ].nunique() != 2:

        raise RuntimeError(
            "Training set must contain "
            "both audio classes."
        )

    return train_df


# ============================================================
# TRAINING
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AUDIO CNN — GENERATOR HOLDOUT TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = load_training_data()

    # --------------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------------

    dataset = AudioFeatureDataset(
        train_df
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    print()
    print(
        "Batches per epoch:",
        len(loader)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AudioCNN().to(
        device
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Model parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    real_count = (
        train_df[
            "audio_label"
        ] == 0
    ).sum()

    fake_count = (
        train_df[
            "audio_label"
        ] == 1
    ).sum()

    pos_weight_value = (
        real_count /
        max(fake_count, 1)
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device
    )

    print()
    print(
        "Real audio:",
        real_count
    )

    print(
        "Fake audio:",
        fake_count
    )

    print(
        "pos_weight:",
        f"{pos_weight_value:.4f}"
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    best_loss = float(
        "inf"
    )

    best_epoch = -1

    print()
    print("=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        samples_seen = 0

        progress = tqdm(
            loader,
            desc=(
                f"Epoch "
                f"{epoch}/{EPOCHS}"
            )
        )

        for features, labels in progress:

            features = features.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                features
            ).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            samples_seen += (
                batch_size
            )

            current_loss = (
                running_loss /
                samples_seen
            )

            progress.set_postfix(
                loss=f"{current_loss:.4f}"
            )

        epoch_loss = (
            running_loss /
            samples_seen
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} "
            f"| train_loss="
            f"{epoch_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best training-loss checkpoint
        # ----------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = (
                epoch_loss
            )

            best_epoch = (
                epoch
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "train_loss":
                        epoch_loss,
                    "holdout_method":
                        "rtvc",
                    "train_samples":
                        len(train_df),
                    "train_sources":
                        train_df[
                            "source"
                        ].nunique(),
                    "seed":
                        SEED,
                },
                CHECKPOINT
            )

            print(
                "  -> Best checkpoint "
                "saved"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best train loss:",
        f"{best_loss:.6f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "RTVC was never used "
        "during this training."
    )

    print(
        "Final evaluation must use "
        "audio_generator_holdout_test.csv."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()