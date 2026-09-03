import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.audio_cnn import AudioCNN


BASE_DIR = r"D:\DeepFake-Learning"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_generator_holdout_train.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_generator_holdout_test.csv"
)

CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "audio_cnn_robust_best.pth"
)

RESULT_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_robust_final_predictions.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42


# =========================================================
# Dataset
# =========================================================

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
            raise ValueError(
                f"Unexpected feature shape: {feature.shape}"
            )

        feature = np.array(
            feature,
            dtype=np.float32,
            copy=True
        )

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = torch.tensor(
            float(row["audio_label"]),
            dtype=torch.float32
        )

        return feature, label


# =========================================================
# Prediction function
# =========================================================

def predict(model, df):

    dataset = AudioFeatureDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for features, batch_labels in loader:

            features = features.to(
                DEVICE
            )

            logits = model(features)

            probs = torch.sigmoid(
                logits
            ).squeeze(1)

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    return (
        np.array(labels),
        np.array(probabilities)
    )


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            )
    }


# =========================================================
# Start
# =========================================================

print("=" * 65)
print("ROBUST AUDIO FINAL EVALUATION")
print("=" * 65)

print(
    "Device:",
    DEVICE
)

# ---------------------------------------------------------
# Load training CSV
# ---------------------------------------------------------

train_df = pd.read_csv(
    TRAIN_CSV
)

# Safety check
if "rtvc" in (
    train_df["method"]
    .astype(str)
    .str.lower()
    .unique()
):

    raise RuntimeError(
        "ERROR: RTVC found in training data!"
    )


# ---------------------------------------------------------
# Recreate EXACT validation split
# ---------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_idx, val_idx = next(
    splitter.split(
        train_df,
        groups=train_df["source"]
    )
)

val_df = train_df.iloc[
    val_idx
].copy()


# ---------------------------------------------------------
# Load untouched RTVC test
# ---------------------------------------------------------

test_df = pd.read_csv(
    TEST_CSV
)

print()
print("Validation samples:", len(val_df))
print("Validation sources:", val_df["source"].nunique())

print()
print("Final test samples:", len(test_df))
print("Final test sources:", test_df["source"].nunique())

print()
print("Test methods:")
print(
    test_df["method"]
    .value_counts()
)


# ---------------------------------------------------------
# Safety checks
# ---------------------------------------------------------

train_sources = set(
    train_df.iloc[train_idx]["source"]
)

val_sources = set(
    val_df["source"]
)

test_sources = set(
    test_df["source"]
)

print()
print(
    "Train/Val source overlap:",
    len(train_sources & val_sources)
)

print(
    "Train/Test source overlap:",
    len(train_sources & test_sources)
)

print(
    "Val/Test source overlap:",
    len(val_sources & test_sources)
)

if train_sources & test_sources:
    raise RuntimeError(
        "TEST SOURCE LEAKAGE!"
    )

if val_sources & test_sources:
    raise RuntimeError(
        "VALIDATION/TEST SOURCE LEAKAGE!"
    )


# ---------------------------------------------------------
# Verify RTVC is unseen
# ---------------------------------------------------------

test_methods = (
    test_df["method"]
    .astype(str)
    .str.lower()
)

rtvc_count = (
    test_methods == "rtvc"
).sum()

print()
print(
    "RTVC test samples:",
    rtvc_count
)

if rtvc_count == 0:
    raise RuntimeError(
        "No RTVC samples found in test!"
    )


# =========================================================
# Load model
# =========================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model = AudioCNN().to(DEVICE)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    checkpoint_auc = checkpoint.get(
        "val_auc",
        "unknown"
    )

else:

    model.load_state_dict(
        checkpoint
    )

    checkpoint_epoch = "unknown"
    checkpoint_auc = "unknown"


print()
print("Checkpoint epoch:", checkpoint_epoch)
print("Checkpoint Val AUC:", checkpoint_auc)


# =========================================================
# VALIDATION
# =========================================================

print()
print("=" * 65)
print("STEP 1: VALIDATION THRESHOLD SELECTION")
print("=" * 65)

val_labels, val_probs = predict(
    model,
    val_df
)

val_auc = roc_auc_score(
    val_labels,
    val_probs
)

print(
    f"Validation ROC-AUC: {val_auc:.4f}"
)


# ---------------------------------------------------------
# Search threshold ONLY on validation
# ---------------------------------------------------------

thresholds = np.arange(
    0.05,
    0.951,
    0.01
)

threshold_results = []

for threshold in thresholds:

    metrics = calculate_metrics(
        val_labels,
        val_probs,
        threshold
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "balanced_accuracy":
                metrics["balanced_accuracy"],
            "accuracy":
                metrics["accuracy"],
            "precision":
                metrics["precision"],
            "recall":
                metrics["recall"],
            "f1":
                metrics["f1"]
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

# Select threshold using validation
best_row = threshold_df.loc[
    threshold_df["balanced_accuracy"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

print()
print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation Balanced Accuracy: "
    f"{best_row['balanced_accuracy']:.4f}"
)

print(
    f"Validation F1: "
    f"{best_row['f1']:.4f}"
)


# =========================================================
# FINAL UNTOUCHED TEST
# =========================================================

print()
print("=" * 65)
print("STEP 2: FINAL UNSEEN RTVC TEST")
print("=" * 65)

test_labels, test_probs = predict(
    model,
    test_df
)

test_auc = roc_auc_score(
    test_labels,
    test_probs
)

test_metrics = calculate_metrics(
    test_labels,
    test_probs,
    best_threshold
)

print()
print(
    f"Test ROC-AUC: "
    f"{test_auc:.4f}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall: "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1: "
    f"{test_metrics['f1']:.4f}"
)

print()
print("Confusion Matrix:")
print(
    test_metrics["confusion_matrix"]
)


# =========================================================
# Method-wise analysis
# =========================================================

print()
print("=" * 65)
print("METHOD-WISE TEST RESULTS")
print("=" * 65)

test_output = test_df.copy()

test_output[
    "predicted_fake_probability"
] = test_probs

test_output[
    "prediction"
] = (
    test_probs >= best_threshold
).astype(int)


for method, group in test_output.groupby(
    "method"
):

    labels = group[
        "audio_label"
    ].values

    probs = group[
        "predicted_fake_probability"
    ].values

    preds = group[
        "prediction"
    ].values

    print()
    print(
        f"Method: {method}"
    )

    print(
        "Samples:",
        len(group)
    )

    if len(np.unique(labels)) > 1:

        print(
            "AUC:",
            f"{roc_auc_score(labels, probs):.4f}"
        )

    print(
        "Accuracy:",
        f"{accuracy_score(labels, preds):.4f}"
    )

    print(
        "Mean fake probability:",
        f"{probs.mean():.4f}"
    )


# =========================================================
# Save predictions
# =========================================================

test_output.to_csv(
    RESULT_CSV,
    index=False
)

print()
print(
    "Predictions saved:"
)

print(
    RESULT_CSV
)

print()
print("=" * 65)
print("FINAL EVALUATION COMPLETE")
print("=" * 65)