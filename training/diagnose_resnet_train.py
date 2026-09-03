from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

from training.dataloader import DeepfakeDataset, eval_transform
from models.resnet_baseline import ResNetBaseline


BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"
CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "resnet18_baseline_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def evaluate_training_set():

    dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / "train.csv",
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Training samples: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Load model only once
    model = ResNetBaseline().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = []
    predictions = []
    probabilities = []

    print("Running inference...")
    print()

    with torch.no_grad():

        progress = tqdm(
            loader,
            total=len(loader),
            desc="Training-set diagnostic"
        )

        for images, batch_labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(images).squeeze(1)

            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).long()

            labels.extend(
                batch_labels.long().tolist()
            )

            predictions.extend(
                preds.cpu().tolist()
            )

            probabilities.extend(
                probs.cpu().tolist()
            )

    return labels, predictions, probabilities


def print_results(labels, predictions, probabilities):

    print()
    print("=" * 60)
    print("TRAINING SET RESULTS")
    print("=" * 60)

    print("Total samples:", len(labels))

    print()
    print("Actual:")
    print("Real:", labels.count(0))
    print("Fake:", labels.count(1))

    print()
    print("Predicted:")
    print("Real:", predictions.count(0))
    print("Fake:", predictions.count(1))

    # Confusion matrix
    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    print()
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 Real  Fake")
    print(
        f"Actual Real     {cm[0][0]:5d} {cm[0][1]:5d}"
    )
    print(
        f"Actual Fake     {cm[1][0]:5d} {cm[1][1]:5d}"
    )

    # Classification report
    print()
    print("Classification Report:")

    print(
        classification_report(
            labels,
            predictions,
            target_names=["Real", "Fake"],
            zero_division=0,
            digits=4
        )
    )

    # ROC-AUC
    auc = roc_auc_score(
        labels,
        probabilities
    )

    print(f"ROC-AUC: {auc:.4f}")

    # Probability distribution
    real_probs = [
        p for p, y in zip(probabilities, labels)
        if y == 0
    ]

    fake_probs = [
        p for p, y in zip(probabilities, labels)
        if y == 1
    ]

    print()
    print("=" * 60)
    print("PROBABILITY DISTRIBUTION")
    print("=" * 60)

    print(
        f"Real probability mean: "
        f"{sum(real_probs) / len(real_probs):.4f}"
    )

    print(
        f"Fake probability mean: "
        f"{sum(fake_probs) / len(fake_probs):.4f}"
    )

    print(
        f"Real probability min: "
        f"{min(real_probs):.4f}"
    )

    print(
        f"Real probability max: "
        f"{max(real_probs):.4f}"
    )

    print(
        f"Fake probability min: "
        f"{min(fake_probs):.4f}"
    )

    print(
        f"Fake probability max: "
        f"{max(fake_probs):.4f}"
    )


def main():

    print("=" * 60)
    print("RESNET18 TRAINING-SET DIAGNOSTIC")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    labels, predictions, probabilities = evaluate_training_set()

    print_results(
        labels,
        predictions,
        probabilities
    )

    print()
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()