import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from models.resnet_baseline import ResNetBaseline
from training.dataloader import create_dataloaders


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

CHECKPOINT_DIR = (
    BASE_DIR / "outputs" / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "resnet18_unseen_nt_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
EPOCHS = 5

BACKBONE_LR = 1e-5
FC_LR = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("UNSEEN NEURALTEXTURES EXPERIMENT")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Training CSV:")
print("train_without_neuraltextures.csv")

print()
print("Validation CSV:")
print("validation_face.csv")

print()
print("Unseen NeuralTextures will NOT be used during training.")


# ============================================================
# DATA
# ============================================================

train_loader, validation_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE,
    train_csv="train_without_neuraltextures.csv",
    validation_csv="validation_face.csv",
    test_csv="test_unseen_neuraltextures.csv"
)


# ============================================================
# MODEL
# ============================================================

model = ResNetBaseline().to(device)


# Freeze entire backbone
for parameter in model.model.parameters():
    parameter.requires_grad = False


# Unfreeze layer4
for parameter in model.model.layer4.parameters():
    parameter.requires_grad = True


# Unfreeze final classifier
for parameter in model.model.fc.parameters():
    parameter.requires_grad = True


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    [
        {
            "params": model.model.layer4.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.model.fc.parameters(),
            "lr": FC_LR
        }
    ]
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        ).squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total

    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            ).squeeze(1)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                torch.sigmoid(outputs) >= 0.5
            ).float()

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "val_loss": val_loss
            },
            BEST_CHECKPOINT
        )

        print()
        print(
            "✓ Best checkpoint saved!"
        )

        print(
            BEST_CHECKPOINT
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print()
print("Checkpoint:")
print(BEST_CHECKPOINT)

print("=" * 70)