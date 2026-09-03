from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from training.dataloader import create_dataloaders
from models.resnet_baseline import ResNetBaseline


BASE_DIR = Path(r"D:\DeepFake-Learning")

CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 5

BACKBONE_LR = 0.00001
FC_LR = 0.0001


def calculate_accuracy(logits, labels):

    predictions = (
        torch.sigmoid(logits) >= 0.5
    ).float()

    return (
        predictions == labels
    ).sum().item()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc="Training"
    )

    for images, labels in progress_bar:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(
            images
        ).squeeze(1)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() *
            images.size(0)
        )

        total_correct += (
            calculate_accuracy(
                logits,
                labels
            )
        )

        total_samples += images.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        progress_bar = tqdm(
            loader,
            desc="Validation"
        )

        for images, labels in progress_bar:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                images
            ).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            total_correct += (
                calculate_accuracy(
                    logits,
                    labels
                )
            )

            total_samples += images.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def main():

    print("=" * 60)
    print("RESNET18 FACE-CROP DEEPFAKE DETECTION")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Backbone LR:",
        BACKBONE_LR
    )

    print(
        "FC LR:",
        FC_LR
    )

    print()

    # --------------------------------------------------
    # FACE-CROP DATASETS
    # --------------------------------------------------

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(

        batch_size=BATCH_SIZE,

        train_csv="train_balanced_face.csv",

        validation_csv="validation_face.csv",

        test_csv="test_face.csv"
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = ResNetBaseline().to(
        DEVICE
    )

    # Freeze everything
    for parameter in model.model.parameters():

        parameter.requires_grad = False

    # Train final convolutional block
    for parameter in model.model.layer4.parameters():

        parameter.requires_grad = True

    # Train classifier
    for parameter in model.model.fc.parameters():

        parameter.requires_grad = True

    print()

    print("Trainable layers:")
    print("  ResNet layer4")
    print("  ResNet FC")

    print()

    # --------------------------------------------------
    # LOSS
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------

    optimizer = Adam(

        [
            {
                "params":
                model.model.layer4.parameters(),

                "lr":
                BACKBONE_LR
            },

            {
                "params":
                model.model.fc.parameters(),

                "lr":
                FC_LR
            }
        ]
    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "-" * 60
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_accuracy = (
            validate(
                model,
                validation_loader,
                criterion
            )
        )

        print()

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

        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint_path = (
                CHECKPOINT_DIR /
                "resnet18_face_crop_best.pth"
            )

            torch.save(
                {
                    "epoch":
                    epoch + 1,

                    "model_state_dict":
                    model.state_dict(),

                    "optimizer_state_dict":
                    optimizer.state_dict(),

                    "val_accuracy":
                    val_accuracy,

                    "val_loss":
                    val_loss
                },
                checkpoint_path
            )

            print()
            print(
                "Best face-crop model saved:"
            )

            print(
                checkpoint_path
            )

    print()
    print("=" * 60)
    print("FACE-CROP TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":

    main()