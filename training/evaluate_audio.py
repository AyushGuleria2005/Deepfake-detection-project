import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from models.audio_cnn import AudioCNN


# ============================================================
# CONFIG
# ============================================================

ROOT = PROJECT_ROOT

TEST_CSV = (
    ROOT
    / "research"
    / "experiments"
    / "fakeavceleb_audio_test.csv"
)

CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "audio_cnn_best.pth"
)

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        feature = np.load(
            row["feature_path"]
        ).astype(np.float32)

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.long
        )

        return feature, label


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO CNN TEST EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    dataset = AudioDataset(TEST_CSV)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(
        "Test samples:",
        len(dataset)
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = AudioCNN().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation AUC:",
        f'{checkpoint["val_auc"]:.4f}'
    )

    print()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(features)

            probabilities = torch.sigmoid(
                logits.squeeze(1)
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = 0.5

    predictions = (
        all_probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            predictions
        )
    )

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        predictions
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC             : {auc:.4f}"
    )

    print(
        f"Accuracy            : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy   : {balanced_accuracy:.4f}"
    )

    print(
        f"Precision           : {precision:.4f}"
    )

    print(
        f"Recall              : {recall:.4f}"
    )

    print(
        f"F1 Score            : {f1:.4f}"
    )

    print()

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "Rows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(cm)

    print()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            all_labels,
            predictions,
            target_names=[
                "Real Audio",
                "Fake Audio"
            ],
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # PROBABILITY STATISTICS
    # --------------------------------------------------------

    real_probs = all_probabilities[
        all_labels == 0
    ]

    fake_probs = all_probabilities[
        all_labels == 1
    ]

    print("=" * 70)
    print("PROBABILITY STATISTICS")
    print("=" * 70)

    print(
        f"Real mean probability : "
        f"{real_probs.mean():.4f}"
    )

    print(
        f"Fake mean probability : "
        f"{fake_probs.mean():.4f}"
    )

    print(
        f"Real min/max          : "
        f"{real_probs.min():.4f} / "
        f"{real_probs.max():.4f}"
    )

    print(
        f"Fake min/max          : "
        f"{fake_probs.min():.4f} / "
        f"{fake_probs.max():.4f}"
    )

    print()

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()