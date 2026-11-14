from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm

from models.resnet_baseline import ResNetBaseline
from training.dataloader import DeepfakeDataset, eval_transform

BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5

# Enhanced Augmentation Transform for fine-tuning
finetune_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast():
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    predictions = (all_probs >= 0.5).long()

    acc = accuracy_score(all_labels, predictions)
    b_acc = balanced_accuracy_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, b_acc, auc, all_labels, predictions


def main():
    print("=" * 65)
    print("FINE-TUNING RESNET18 IMAGE MODEL (FACE-CROP)")
    print("=" * 65)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | Mixed Precision: AMP")

    # Datasets
    train_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / "train_balanced_face.csv",
        transform=finetune_transform
    )
    val_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / "validation_face.csv",
        transform=eval_transform
    )
    test_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / "test_face.csv",
        transform=eval_transform
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    # Balanced Sampler for Training
    class_counts = train_dataset.data["label"].value_counts().sort_index()
    class_weights = {lbl: 1.0 / count for lbl, count in class_counts.items()}
    sample_weights = train_dataset.data["label"].map(class_weights).to_numpy(copy=True)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load Model with Existing Checkpoint
    model = ResNetBaseline().to(DEVICE)
    prev_ckpt_path = CHECKPOINT_DIR / "resnet18_face_crop_prev.pth"
    if prev_ckpt_path.exists():
        print(f"Loading checkpoint from: {prev_ckpt_path}")
        ckpt = torch.load(prev_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Starting from ImageNet pretrained weights.")

    # Differential Layer Unfreezing
    # Freeze stem, layer1, layer2
    for param in model.model.conv1.parameters():
        param.requires_grad = False
    for param in model.model.bn1.parameters():
        param.requires_grad = False
    for param in model.model.layer1.parameters():
        param.requires_grad = False
    for param in model.model.layer2.parameters():
        param.requires_grad = False

    # Unfreeze layer3, layer4, fc
    for param in model.model.layer3.parameters():
        param.requires_grad = True
    for param in model.model.layer4.parameters():
        param.requires_grad = True
    for param in model.model.fc.parameters():
        param.requires_grad = True

    print("\nTrainable Layers:")
    print("  - layer3 (lr=5e-6)")
    print("  - layer4 (lr=2e-5)")
    print("  - fc     (lr=1e-4)")

    optimizer = AdamW([
        {"params": model.model.layer3.parameters(), "lr": 5e-6},
        {"params": model.model.layer4.parameters(), "lr": 2e-5},
        {"params": model.model.fc.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_checkpoint_path = CHECKPOINT_DIR / "resnet18_face_crop_best.pth"

    start_train_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        correct_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train_samples += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_train_loss / total_train_samples
        train_acc = correct_train / total_train_samples

        # Validation
        val_loss, val_acc, val_b_acc, val_auc, _, _ = evaluate(model, val_loader, criterion)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Balanced Acc: {val_b_acc*100:.2f}% | Val ROC-AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_b_acc,
                "val_auc": val_auc,
                "val_loss": val_loss,
            }, best_checkpoint_path)
            print(f"  >>> Best model saved (Val ROC-AUC: {val_auc:.4f}) to {best_checkpoint_path.name}")

    total_time = time.time() - start_train_time
    print(f"\nFine-tuning completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation ROC-AUC: {best_val_auc:.4f}")

    # Final Evaluation on Unseen Test Set
    print("\n" + "=" * 65)
    print("FINAL EVALUATION ON TEST DATASET (test_face.csv)")
    print("=" * 65)
    best_ckpt = torch.load(best_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_b_acc, test_auc, y_true, y_pred = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_b_acc*100:.2f}%")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))


if __name__ == "__main__":
    main()
