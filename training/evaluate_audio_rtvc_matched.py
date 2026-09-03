import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from models.audio_cnn import AudioCNN


# ============================================================
# PATHS
# ============================================================

BASE = PROJECT_ROOT

EXP = os.path.join(
    BASE,
    "research",
    "experiments"
)

CKPT = os.path.join(
    BASE,
    "outputs",
    "checkpoints",
    "audio_cnn_unseen_rtvc_best.pth"
)

TEST_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_rtvc_matched_test.csv"
)

OUTPUT_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_rtvc_matched_predictions.csv"
)


# ============================================================
# LOAD FEATURE MAPPING
# ============================================================

feature_maps = []

for split in [
    "train",
    "validation",
    "test"
]:

    meta_path = os.path.join(
        EXP,
        f"fakeavceleb_audio_{split}.csv"
    )

    df = pd.read_csv(
        meta_path
    )

    feature_maps.append(
        df[
            [
                "actual_path",
                "feature_path"
            ]
        ]
    )


feature_map = pd.concat(
    feature_maps,
    ignore_index=True
)

feature_map = feature_map.drop_duplicates(
    subset=["actual_path"],
    keep="first"
)


# ============================================================
# LOAD MATCHED TEST SET
# ============================================================

test_df = pd.read_csv(
    TEST_CSV
)

test_df = test_df.merge(
    feature_map,
    on="actual_path",
    how="left"
)


missing = test_df[
    "feature_path"
].isna().sum()

if missing > 0:

    raise RuntimeError(
        f"Missing feature files: {missing}"
    )


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, df):

        self.df = df.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature = np.load(
            row["feature_path"]
        ).astype(
            np.float32
        )

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        label = int(
            row["eval_label"]
        )

        return (
            feature,
            label
        )


dataset = AudioDataset(
    test_df
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("MATCHED UNSEEN RTVC AUDIO EVALUATION")
print("=" * 70)

print(
    f"Device: {device}"
)

if device.type == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Total samples: "
    f"{len(dataset)}"
)

print(
    f"Real samples: "
    f"{(test_df['eval_label'] == 0).sum()}"
)

print(
    f"RTVC fake samples: "
    f"{(test_df['eval_label'] == 1).sum()}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = AudioCNN().to(
    device
)

checkpoint = torch.load(
    CKPT,
    map_location=device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    f"\nCheckpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation AUC: "
    f"{checkpoint['val_auc']:.4f}"
)


# ============================================================
# PREDICTION
# ============================================================

all_probs = []
all_labels = []


with torch.no_grad():

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True
        )

        logits = model(
            features
        )

        probs = torch.sigmoid(
            logits
        ).squeeze(1)

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_probs = np.array(
    all_probs
)

all_labels = np.array(
    all_labels
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

threshold = 0.50

predictions = (
    all_probs >= threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    predictions
)

precision = precision_score(
    all_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    all_labels,
    all_probs
)

cm = confusion_matrix(
    all_labels,
    predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MATCHED RTVC RESULTS"
)

print(
    "=" * 70
)

print(
    f"ROC-AUC           : {auc:.4f}"
)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)


print(
    "\nConfusion Matrix"
)

print(
    "[[TN FP]"
)

print(
    " [FN TP]]"
)

print(cm)


# ============================================================
# CLASS-WISE PROBABILITY
# ============================================================

real_probs = all_probs[
    all_labels == 0
]

fake_probs = all_probs[
    all_labels == 1
]


print(
    "\n" + "=" * 70
)

print(
    "CLASS-WISE PROBABILITY"
)

print(
    "=" * 70
)

print(
    f"Real mean probability : "
    f"{real_probs.mean():.4f}"
)

print(
    f"Real median           : "
    f"{np.median(real_probs):.4f}"
)

print(
    f"Real maximum          : "
    f"{real_probs.max():.4f}"
)

print(
    f"RTVC mean probability : "
    f"{fake_probs.mean():.4f}"
)

print(
    f"RTVC median           : "
    f"{np.median(fake_probs):.4f}"
)

print(
    f"RTVC minimum          : "
    f"{fake_probs.min():.4f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_df[
    "fake_probability"
] = all_probs

test_df[
    "predicted_label"
] = predictions

test_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EVALUATION COMPLETE"
)

print(
    "=" * 70
)

print(
    "Predictions saved:"
)

print(
    OUTPUT_CSV
)

print(
    "=" * 70
)