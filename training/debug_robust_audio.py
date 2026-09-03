import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from models.audio_cnn_gn import AudioCNNGroupNorm


CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "research",
    "experiments",
    "audio_generator_holdout_train.csv"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
    "audio_cnn_gn_best.pth"
)

SEED = 42
BATCH_SIZE = 64


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
            raise ValueError(
                f"Bad feature shape: {feature.shape}"
            )

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.float32
        )

        return feature, label


def main():

    print("=" * 70)
    print("EXACT GROUPSHUFFLESPLIT CHECK")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    df = pd.read_csv(CSV_PATH)

    # -----------------------------------------------------
    # EXACT SAME SPLIT AS train_audio_gn.py
    # -----------------------------------------------------

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

    print("\nSPLIT")
    print("-" * 50)

    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))

    print(
        "Train sources:",
        train_df["source"].nunique()
    )

    print(
        "Validation sources:",
        val_df["source"].nunique()
    )

    overlap = (
        set(train_df["source"])
        &
        set(val_df["source"])
    )

    print(
        "Source overlap:",
        len(overlap)
    )

    print(
        "Validation labels:",
        val_df["audio_label"]
        .value_counts()
        .to_dict()
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    val_dataset = AudioFeatureDataset(
        val_df
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = AudioCNNGroupNorm().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nCheckpoint epoch:",
          checkpoint.get("epoch"))

    print(
        "Recorded Val AUC:",
        checkpoint.get("val_auc")
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    probabilities = []
    labels = []

    with torch.no_grad():

        for features, batch_labels in val_loader:

            features = features.to(device)

            logits = model(features)

            probs = torch.sigmoid(
                logits
            ).squeeze(1)

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    auc = roc_auc_score(
        labels,
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions
        )
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Recorded training AUC : "
        f"{checkpoint['val_auc']:.6f}"
    )

    print(
        f"Reproduced AUC        : "
        f"{auc:.6f}"
    )

    print(
        f"Difference             : "
        f"{auc - checkpoint['val_auc']:+.6f}"
    )

    print(
        f"Accuracy               : "
        f"{accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy      : "
        f"{balanced_accuracy:.6f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------------------------------
    # Probability statistics
    # -----------------------------------------------------

    real_probs = probabilities[
        labels == 0
    ]

    fake_probs = probabilities[
        labels == 1
    ]

    print("\nProbability statistics")

    print(
        f"Real mean:  {real_probs.mean():.6f}"
    )

    print(
        f"Fake mean:  {fake_probs.mean():.6f}"
    )

    print(
        f"Real median: {np.median(real_probs):.6f}"
    )

    print(
        f"Fake median: {np.median(fake_probs):.6f}"
    )


if __name__ == "__main__":
    main()