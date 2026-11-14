import argparse
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from models.resnet_baseline import ResNetBaseline
from training.dataloader import DeepfakeDataset, eval_transform

BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"
DEFAULT_CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "resnet18_face_crop_best.pth"


def evaluate_dataset(split="test", checkpoint_path=None, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_name = f"{split}_face.csv"
    csv_path = EXPERIMENTS_DIR / csv_name
    if not csv_path.exists():
        csv_path = EXPERIMENTS_DIR / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV dataset for split '{split}' at {csv_path}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print("=" * 65)
    print("DEEPFAKE IMAGE MODEL EVALUATION")
    print("=" * 65)
    print(f"Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Dataset:    {csv_path.name}")

    dataset = DeepfakeDataset(csv_path, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = ResNetBaseline().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    print(f"Total samples to evaluate: {len(dataset)}")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    predictions = (all_probs >= 0.5).long()

    # Metrics
    avg_loss = total_loss / len(dataset)
    acc = accuracy_score(all_labels, predictions)
    b_acc = balanced_accuracy_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, predictions)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Average Loss:            {avg_loss:.4f}")
    print(f"Accuracy:                {acc * 100:.2f}%")
    print(f"Balanced Accuracy:       {b_acc * 100:.2f}%")
    print(f"ROC-AUC Score:           {auc:.4f}")
    print("\nConfusion Matrix:")
    print("                 Predicted Real    Predicted Fake")
    print(f"  Actual Real         {cm[0][0]:<14}   {cm[0][1]:<14}")
    print(f"  Actual Fake         {cm[1][0]:<14}   {cm[1][1]:<14}")

    print("\nClassification Report:")
    print(classification_report(all_labels, predictions, target_names=["Real (0)", "Fake (1)"], digits=4))

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "roc_auc": auc,
        "loss": avg_loss,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Deepfake Model Accuracy")
    parser.add_argument("--split", choices=["test", "validation", "train_balanced"], default="test", help="Which dataset split to test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluate_dataset(split=args.split, checkpoint_path=args.checkpoint, batch_size=args.batch_size)
