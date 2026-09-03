from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from training.dataloader import DeepfakeDataset, eval_transform
from models.resnet_baseline import ResNetBaseline


BASE_DIR = Path(r"D:\DeepFake-Learning")

EXPERIMENTS_DIR = (
    BASE_DIR / "research" / "experiments"
)

CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "checkpoints"
    / "resnet18_baseline_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("=" * 60)
    print("RESNET18 VALIDATION DIAGNOSTIC")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------

    validation_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / "validation.csv",
        transform=eval_transform
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = ResNetBaseline().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("Checkpoint loaded successfully!")

    print(
        "Checkpoint validation accuracy:",
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                images
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                labels.long().cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    # --------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------

    labels = torch.tensor(
        all_labels
    )

    predictions = torch.tensor(
        all_predictions
    )

    print()
    print(
        "Validation samples:",
        len(labels)
    )

    print()
    print("Actual distribution:")

    print(
        "Real:",
        (labels == 0).sum().item()
    )

    print(
        "Fake:",
        (labels == 1).sum().item()
    )

    print()
    print("Predicted distribution:")

    print(
        "Real:",
        (predictions == 0).sum().item()
    )

    print(
        "Fake:",
        (predictions == 1).sum().item()
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    print()
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 Real  Fake")

    print(
        f"Actual Real     "
        f"{cm[0][0]:5d} "
        f"{cm[0][1]:5d}"
    )

    print(
        f"Actual Fake     "
        f"{cm[1][0]:5d} "
        f"{cm[1][1]:5d}"
    )

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    print()
    print("Classification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Real",
                "Fake"
            ],
            zero_division=0,
            digits=4
        )
    )

    # --------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------

    try:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

        print(
            f"ROC-AUC: {auc:.4f}"
        )

    except ValueError:

        print(
            "ROC-AUC could not be calculated."
        )

    # --------------------------------------------------
    # MANIPULATION-WISE RESULTS
    # --------------------------------------------------

    validation_df = pd.read_csv(
        EXPERIMENTS_DIR
        / "validation.csv"
    )

    validation_df["prediction"] = (
        all_predictions
    )

    validation_df["probability_fake"] = (
        all_probabilities
    )

    print()
    print("=" * 60)
    print("MANIPULATION-WISE RESULTS")
    print("=" * 60)

    for manipulation, group in (
        validation_df.groupby("manipulation")
    ):

        accuracy = (
            group["prediction"]
            == group["label"]
        ).mean()

        print(
            f"{manipulation:15s}: "
            f"{accuracy * 100:.2f}% "
            f"({len(group)} images)"
        )

    # --------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------

    output_file = (
        EXPERIMENTS_DIR
        / "resnet_validation_predictions.csv"
    )

    validation_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        "Prediction results saved to:"
    )

    print(output_file)

    print()
    print("=" * 60)
    print("RESNET DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()