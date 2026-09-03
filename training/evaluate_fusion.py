from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

DATASET = (
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

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "fusion_test_predictions.csv"
)


SEED = 42


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


def source_split(df):

    sources = sorted(
        df["source"].astype(str).unique()
    )

    rng = np.random.default_rng(SEED)
    rng.shuffle(sources)

    n_val = int(len(sources) * 0.20)

    val_sources = set(
        sources[:n_val]
    )

    train_sources = set(
        sources[n_val:]
    )

    train_df = df[
        df["source"].astype(str).isin(train_sources)
    ].copy()

    val_df = df[
        df["source"].astype(str).isin(val_sources)
    ].copy()

    return train_df, val_df


def main():

    print("=" * 70)
    print("FINAL MULTIMODAL FUSION TEST")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    df = pd.read_csv(DATASET)

    # Recreate the exact source split used during training
    _, val_df = source_split(df)

    # The remaining 20% source-disjoint portion is NOT the
    # original final test set, so we will instead use the
    # dedicated FakeAVCeleb source-disjoint test partition
    # if available.

    test_candidates = [
        PROJECT_ROOT
        / "research"
        / "experiments"
        / "audio_generator_holdout_test.csv",
    ]

    holdout_path = test_candidates[0]

    if holdout_path.exists():

        holdout = pd.read_csv(
            holdout_path
        )

        test_sources = set(
            holdout["source"].astype(str)
        )

        test_df = df[
            df["source"].astype(str).isin(test_sources)
        ].copy()

        print(
            f"Using generator-holdout sources: "
            f"{len(test_sources)}"
        )

    else:

        raise FileNotFoundError(
            "Generator holdout test file not found."
        )

    print(
        f"Test samples: {len(test_df)}"
    )

    print(
        f"Test sources: "
        f"{test_df['source'].nunique()}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = FusionMLP().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint validation AUC: "
        f"{checkpoint.get('val_auc', 'unknown')}"
    )

    # --------------------------------------------------------
    # Prepare inputs
    # --------------------------------------------------------

    visual = test_df[
        "visual_fake_probability"
    ].astype(float).clip(0, 1).values

    audio = test_df[
        "audio_fake_probability"
    ].astype(float).clip(0, 1).values

    x = torch.tensor(
        np.column_stack(
            [visual, audio]
        ),
        dtype=torch.float32
    ).to(device)

    # Final manipulated target:
    # either video or audio is fake
    y = np.maximum(
        test_df["video_label"].astype(int).values,
        test_df["audio_label"].astype(int).values
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(x).squeeze(1)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    auc = roc_auc_score(
        y,
        probabilities
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC:              {auc:.4f}"
    )

    print(
        f"Accuracy:             {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy:    {balanced_accuracy:.4f}"
    )

    print(
        f"Precision:             {precision:.4f}"
    )

    print(
        f"Recall:                {recall:.4f}"
    )

    print(
        f"F1 Score:             {f1:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Modality combinations
    # --------------------------------------------------------

    result_df = test_df.copy()

    result_df[
        "fusion_fake_probability"
    ] = probabilities

    result_df[
        "fusion_prediction"
    ] = predictions

    print()
    print("Modality Combination Performance")
    print("-" * 70)

    for combination, group in result_df.groupby(
        "modality_combination"
    ):

        if len(
            group["audio_label"].unique()
        ) < 2 and len(
            group["video_label"].unique()
        ) < 2:
            group_auc = float("nan")
        else:

            group_target = np.maximum(
                group["video_label"].astype(int),
                group["audio_label"].astype(int)
            )

            if len(np.unique(group_target)) < 2:
                group_auc = float("nan")
            else:
                group_auc = roc_auc_score(
                    group_target,
                    group["fusion_fake_probability"]
                )

        print(
            f"{combination:30s} "
            f"n={len(group):4d} "
            f"AUC={group_auc:.4f}"
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print(
        f"Predictions saved:"
    )

    print(
        OUTPUT
    )

    print()
    print("=" * 70)
    print("FINAL FUSION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()