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


# =========================================================
# Configuration
# =========================================================

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

BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42


# =========================================================
# Dataset
# =========================================================

class AudioFeatureDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature = np.load(
            row["feature_path"]
        ).astype(np.float32)

        if feature.shape != (64, 512):
            raise ValueError(
                f"Unexpected feature shape "
                f"{feature.shape} for "
                f"{row['feature_path']}"
            )

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.float32
        )

        return feature, label


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("GROUPNORM CHECKPOINT - EXACT TRAINING SPLIT DEBUG")
    print("=" * 70)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    df = pd.read_csv(CSV_PATH)

    print(
        f"\nTotal samples: {len(df)}"
    )

    # -----------------------------------------------------
    # Safety: RTVC must not be present
    # -----------------------------------------------------

    methods = (
        df["method"]
        .astype(str)
        .str.lower()
    )

    if "rtvc" in methods.unique():

        raise RuntimeError(
            "RTVC found in training dataset!"
        )

    # -----------------------------------------------------
    # Safety: duplicate physical files
    # -----------------------------------------------------

    duplicates = df[
        "actual_path"
    ].duplicated().sum()

    print(
        f"Duplicate physical paths: {duplicates}"
    )

    if duplicates > 0:

        raise RuntimeError(
            "Duplicate physical files detected!"
        )

    # -----------------------------------------------------
    # Verify features
    # -----------------------------------------------------

    missing = 0

    for path in df["feature_path"]:

        if not os.path.exists(path):
            missing += 1

    print(
        f"Missing features: {missing}"
    )

    if missing > 0:

        raise RuntimeError(
            "Missing feature files!"
        )

    # =====================================================
    # EXACT SAME SPLIT AS TRAINING
    # =====================================================

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

    train_df = df.iloc[
        train_idx
    ].copy()

    val_df = df.iloc[
        val_idx
    ].copy()

    train_sources = set(
        train_df["source"].astype(str)
    )

    val_sources = set(
        val_df["source"].astype(str)
    )

    print("\n" + "=" * 70)
    print("EXACT TRAINING SPLIT")
    print("=" * 70)

    print(
        f"Train samples:       {len(train_df)}"
    )

    print(
        f"Validation samples:  {len(val_df)}"
    )

    print(
        f"Train sources:       {len(train_sources)}"
    )

    print(
        f"Validation sources:  {len(val_sources)}"
    )

    overlap = (
        train_sources &
        val_sources
    )

    print(
        f"Source overlap:      {len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Source leakage detected!"
        )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    print("\nTrain labels:")

    print(
        train_df[
            "audio_label"
        ].value_counts().to_dict()
    )

    print("\nValidation labels:")

    print(
        val_df[
            "audio_label"
        ].value_counts().to_dict()
    )

    # -----------------------------------------------------
    # Methods
    # -----------------------------------------------------

    print("\nValidation methods:")

    print(
        val_df[
            "method"
        ].value_counts().sort_index()
    )

    # =====================================================
    # Dataset
    # =====================================================

    val_dataset = AudioFeatureDataset(
        val_df
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # =====================================================
    # Model
    # =====================================================

    print("\nLoading GroupNorm model...")

    model = AudioCNNGroupNorm()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    # Remove DataParallel prefix if present

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[key] = value

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    print(
        f"Missing model keys: "
        f"{len(missing_keys)}"
    )

    print(
        f"Unexpected model keys: "
        f"{len(unexpected_keys)}"
    )

    if missing_keys:
        print(
            "Missing keys:",
            missing_keys
        )

    if unexpected_keys:
        print(
            "Unexpected keys:",
            unexpected_keys
        )

    model.to(device)

    model.eval()

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # -----------------------------------------------------
    # Checkpoint metadata
    # -----------------------------------------------------

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_auc" in checkpoint:

            print(
                f"Checkpoint recorded Val AUC: "
                f"{checkpoint['val_auc']:.6f}"
            )

    # =====================================================
    # Inference
    # =====================================================

    probabilities = []
    labels_all = []

    print("\nRunning validation inference...")

    with torch.no_grad():

        for batch_idx, (
            features,
            labels
        ) in enumerate(val_loader):

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

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

            if batch_idx % 10 == 0:

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(val_loader)}"
                )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    labels_all = np.asarray(
        labels_all,
        dtype=np.int32
    )

    # =====================================================
    # Metrics
    # =====================================================

    auc = roc_auc_score(
        labels_all,
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int32)

    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels_all,
            predictions
        )
    )

    cm = confusion_matrix(
        labels_all,
        predictions
    )

    # =====================================================
    # Statistics
    # =====================================================

    real_probs = probabilities[
        labels_all == 0
    ]

    fake_probs = probabilities[
        labels_all == 1
    ]

    print("\n" + "=" * 70)
    print("EXACT TRAINING VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Samples:             {len(labels_all)}"
    )

    print(
        f"ROC-AUC:             {auc:.6f}"
    )

    print(
        f"Accuracy:            {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy:   "
        f"{balanced_accuracy:.6f}"
    )

    print("\nProbability statistics:")

    print(
        f"Overall mean:        "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Overall median:      "
        f"{np.median(probabilities):.6f}"
    )

    print(
        f"Real mean:           "
        f"{real_probs.mean():.6f}"
    )

    print(
        f"Real median:         "
        f"{np.median(real_probs):.6f}"
    )

    print(
        f"Fake mean:           "
        f"{fake_probs.mean():.6f}"
    )

    print(
        f"Fake median:         "
        f"{np.median(fake_probs):.6f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    # =====================================================
    # Compare
    # =====================================================

    recorded_auc = None

    if isinstance(checkpoint, dict):

        recorded_auc = checkpoint.get(
            "val_auc"
        )

    if recorded_auc is not None:

        difference = (
            auc -
            float(recorded_auc)
        )

        print("\n" + "=" * 70)
        print("CHECKPOINT AUC COMPARISON")
        print("=" * 70)

        print(
            f"Training-recorded AUC: "
            f"{float(recorded_auc):.6f}"
        )

        print(
            f"Reproduced AUC:        "
            f"{auc:.6f}"
        )

        print(
            f"Difference:            "
            f"{difference:+.6f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()