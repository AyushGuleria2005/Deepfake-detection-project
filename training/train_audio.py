import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.audio_cnn import AudioCNN


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = PROJECT_ROOT

TRAIN_CSV = (
    ROOT
    / "research"
    / "experiments"
    / "fakeavceleb_audio_train.csv"
)

VAL_CSV = (
    ROOT
    / "research"
    / "experiments"
    / "fakeavceleb_audio_validation.csv"
)

CHECKPOINT_DIR = (
    ROOT
    / "outputs"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "audio_cnn_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


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

        feature_path = row["feature_path"]

        feature = np.load(
            feature_path
        ).astype(np.float32)

        # Original:
        # [64, 512]

        # Add channel dimension:
        # [1, 64, 512]

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.float32
        )

        return feature, label


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            # AudioCNN returns logits only
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

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    return auc


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO CNN TRAINING")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("Loading datasets...")

    train_dataset = AudioDataset(
        TRAIN_CSV
    )

    validation_dataset = AudioDataset(
        VAL_CSV
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    print()

    # ========================================================
    # MODEL
    # ========================================================

    model = AudioCNN().to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print()

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    train_labels = (
        train_dataset.df["audio_label"]
        .values
    )

    real_count = np.sum(
        train_labels == 0
    )

    fake_count = np.sum(
        train_labels == 1
    )

    positive_weight = (
        real_count / fake_count
    )

    print(
        "Real audio:",
        real_count
    )

    print(
        "Fake audio:",
        fake_count
    )

    print(
        "Positive class weight:",
        round(positive_weight, 4)
    )

    print()

    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=DEVICE
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_validation_auc = 0.0

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}"
        )

        # ====================================================
        # BATCH LOOP
        # ====================================================

        for features, labels in progress_bar:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            ).unsqueeze(1)

            # Clear gradients
            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward pass
            logits = model(
                features
            )

            # Calculate loss
            loss = criterion(
                logits,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update model
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # ====================================================
        # TRAINING LOSS
        # ====================================================

        average_train_loss = (
            running_loss
            / len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_auc = evaluate(
            model,
            validation_loader
        )

        print()

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: {average_train_loss:.4f}"
        )

        print(
            f"Validation ROC-AUC: {validation_auc:.4f}"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_auc > best_validation_auc:

            best_validation_auc = validation_auc

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": validation_auc
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

            print(
                "BEST MODEL SAVED:"
            )

            print(
                CHECKPOINT_PATH
            )

        else:

            print(
                "Best model unchanged."
            )

        print()
        print("-" * 70)

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("AUDIO CNN TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best Validation ROC-AUC:",
        f"{best_validation_auc:.4f}"
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()