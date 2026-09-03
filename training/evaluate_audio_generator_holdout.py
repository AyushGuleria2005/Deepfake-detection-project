from pathlib import Path
import sys

import numpy as np
import pandas as pd
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
)

PROJECT_ROOT = Path(r"D:\DeepFake-Learning")
sys.path.insert(0, str(PROJECT_ROOT))

from models.audio_cnn import AudioCNN


TEST_CSV = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "audio_generator_holdout_test.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "audio_cnn_generator_holdout_best.pth"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "audio_generator_holdout_predictions.csv"
)

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
            raise RuntimeError(
                f"Unexpected feature shape "
                f"{feature.shape}\n"
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


def main():

    print("=" * 70)
    print("UNSEEN RTVC — FINAL AUDIO EVALUATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load test set
    # --------------------------------------------------------

    test_df = pd.read_csv(TEST_CSV)

    required = [
        "actual_path",
        "feature_path",
        "source",
        "method",
        "audio_label",
    ]

    missing = [
        c for c in required
        if c not in test_df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    print(
        f"Test samples: {len(test_df)}"
    )

    print(
        f"Test sources: "
        f"{test_df['source'].nunique()}"
    )

    # --------------------------------------------------------
    # Verify test composition
    # --------------------------------------------------------

    print("\nTest labels:")
    print(
        test_df[
            "audio_label"
        ].value_counts().sort_index()
    )

    print("\nTest methods:")
    print(
        test_df[
            "method"
        ].value_counts()
    )

    # RTVC must be the only fake-audio method
    fake_methods = set(
        test_df.loc[
            test_df["audio_label"] == 1,
            "method"
        ].astype(str).str.lower()
    )

    if fake_methods != {"rtvc"}:
        raise RuntimeError(
            f"Unexpected fake methods: "
            f"{fake_methods}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = AudioCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"\nCheckpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Training loss: "
        f"{checkpoint.get('train_loss')}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = AudioFeatureDataset(
        test_df
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = []
    labels = []

    with torch.no_grad():

        for features, batch_labels in loader:

            features = features.to(
                device,
                non_blocking=True
            )

            logits = model(
                features
            ).squeeze(1)

            probs = torch.sigmoid(
                logits
            )

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
    ).astype(int)

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    auc = roc_auc_score(
        labels,
        probabilities
    )

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

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    # --------------------------------------------------------
    # Per-class probability statistics
    # --------------------------------------------------------

    real_probs = probabilities[
        labels == 0
    ]

    rtvc_probs = probabilities[
        labels == 1
    ]

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC:           {auc:.4f}"
    )

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )

    print(
        f"Precision:         {precision:.4f}"
    )

    print(
        f"Recall:            {recall:.4f}"
    )

    print(
        f"F1 Score:          {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nProbability statistics:")

    print(
        f"Real audio:"
        f" mean={real_probs.mean():.6f}"
        f" median={np.median(real_probs):.6f}"
        f" min={real_probs.min():.6f}"
        f" max={real_probs.max():.6f}"
    )

    print(
        f"RTVC fake audio:"
        f" mean={rtvc_probs.mean():.6f}"
        f" median={np.median(rtvc_probs):.6f}"
        f" min={rtvc_probs.min():.6f}"
        f" max={rtvc_probs.max():.6f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results = test_df.copy()

    results[
        "audio_fake_probability"
    ] = probabilities

    results[
        "prediction"
    ] = predictions

    results[
        "correct"
    ] = (
        predictions ==
        labels
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nPredictions saved:"
    )

    print(
        OUTPUT_CSV
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()