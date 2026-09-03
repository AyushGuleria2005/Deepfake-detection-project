from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(r"D:\DeepFake-Learning")

INPUT_CSV = (
    PROJECT_ROOT
    / "research"
    / "experiments"
    / "fakeavceleb_fusion_dataset.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "fusion_best.pth"
)

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
VAL_SOURCE_FRACTION = 0.20
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        visual = float(
            row["visual_fake_probability"]
        )

        audio = float(
            row["audio_fake_probability"]
        )

        x = torch.tensor(
            [visual, audio],
            dtype=torch.float32
        )

        # Final multimodal target:
        # video fake OR audio fake => manipulated
        label = float(
            max(
                int(row["video_label"]),
                int(row["audio_label"])
            )
        )

        y = torch.tensor(
            label,
            dtype=torch.float32
        )

        return x, y


# ============================================================
# MODEL
# ============================================================

class FusionMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# SOURCE-LEVEL SPLIT
# ============================================================

def source_split(df):

    sources = sorted(
        df["source"].astype(str).unique()
    )

    rng = np.random.default_rng(SEED)

    rng.shuffle(sources)

    n_val = max(
        1,
        int(
            len(sources)
            * VAL_SOURCE_FRACTION
        )
    )

    val_sources = set(
        sources[:n_val]
    )

    train_sources = set(
        sources[n_val:]
    )

    train_df = df[
        df["source"].isin(train_sources)
    ].copy()

    val_df = df[
        df["source"].isin(val_sources)
    ].copy()

    overlap = (
        set(train_df["source"])
        &
        set(val_df["source"])
    )

    if overlap:
        raise RuntimeError(
            f"Source leakage: "
            f"{len(overlap)} sources overlap."
        )

    return train_df, val_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTIMODAL FUSION TRAINING")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    required = [
        "source",
        "visual_fake_probability",
        "audio_fake_probability",
        "video_label",
        "audio_label",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    print(
        f"Total samples: {len(df)}"
    )

    print(
        f"Unique sources: "
        f"{df['source'].nunique()}"
    )

    # --------------------------------------------------------
    # Remove invalid probabilities
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "visual_fake_probability",
            "audio_fake_probability"
        ]
    ).copy()

    df[
        "visual_fake_probability"
    ] = df[
        "visual_fake_probability"
    ].clip(0, 1)

    df[
        "audio_fake_probability"
    ] = df[
        "audio_fake_probability"
    ].clip(0, 1)

    # --------------------------------------------------------
    # Source-level split
    # --------------------------------------------------------

    train_df, val_df = source_split(
        df
    )

    print()
    print("SOURCE-LEVEL SPLIT")
    print("-" * 50)

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Train sources: "
        f"{train_df['source'].nunique()}"
    )

    print(
        f"Validation sources: "
        f"{val_df['source'].nunique()}"
    )

    print(
        "Source overlap:",
        len(
            set(train_df["source"])
            &
            set(val_df["source"])
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = FusionDataset(
        train_df
    )

    val_dataset = FusionDataset(
        val_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FusionMLP().to(
        device
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Fusion parameters: {params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    train_targets = np.maximum(
        train_df["video_label"].astype(int),
        train_df["audio_label"].astype(int)
    )

    negatives = (
        train_targets == 0
    ).sum()

    positives = (
        train_targets == 1
    ).sum()

    pos_weight = torch.tensor(
        [negatives / max(positives, 1)],
        dtype=torch.float32,
        device=device
    )

    print(
        f"Train real: {negatives}"
    )

    print(
        f"Train manipulated: {positives}"
    )

    print(
        f"pos_weight: "
        f"{pos_weight.item():.4f}"
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_auc = -1.0

    print()
    print("=" * 70)
    print("TRAINING START")
    print("=" * 70)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        samples_seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}"
        )

        for x, y in progress:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x
            ).squeeze(1)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            n = y.size(0)

            running_loss += (
                loss.item() * n
            )

            samples_seen += n

            progress.set_postfix(
                loss=(
                    running_loss
                    / samples_seen
                )
            )

        train_loss = (
            running_loss
            / samples_seen
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_probs = []
        val_labels = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(
                    device,
                    non_blocking=True
                )

                logits = model(
                    x
                ).squeeze(1)

                probs = torch.sigmoid(
                    logits
                )

                val_probs.extend(
                    probs.cpu().numpy()
                )

                val_labels.extend(
                    y.numpy()
                )

        val_probs = np.asarray(
            val_probs
        )

        val_labels = np.asarray(
            val_labels
        )

        val_auc = roc_auc_score(
            val_labels,
            val_probs
        )

        val_predictions = (
            val_probs >= 0.5
        ).astype(int)

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| train_loss={train_loss:.6f} "
            f"| val_auc={val_auc:.4f} "
            f"| val_acc={val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_auc > best_auc:

            best_auc = val_auc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "train_loss":
                        train_loss,
                    "val_auc":
                        val_auc,
                    "val_accuracy":
                        val_accuracy,
                    "train_sources":
                        train_df[
                            "source"
                        ].nunique(),
                    "val_sources":
                        val_df[
                            "source"
                        ].nunique(),
                },
                CHECKPOINT
            )

            print(
                "  -> Best fusion checkpoint saved"
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FUSION TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation AUC: "
        f"{best_auc:.4f}"
    )

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )


if __name__ == "__main__":
    main()