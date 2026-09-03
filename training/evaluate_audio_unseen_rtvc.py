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

from models.audio_cnn import AudioCNN


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
    "fakeavceleb_audio_unseen_rtvc_test.csv"
)

OUTPUT_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_unseen_rtvc_predictions.csv"
)


# ============================================================
# LOAD FEATURE MAPPING
# ============================================================

feature_maps = []

for split in ["train", "validation", "test"]:

    meta_path = os.path.join(
        EXP,
        f"fakeavceleb_audio_{split}.csv"
    )

    df = pd.read_csv(meta_path)

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
# LOAD RTVC TEST SPLIT
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
        f"Missing features: {missing}"
    )


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, df):

        self.df = df.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature = np.load(
            row["feature_path"]
        ).astype(np.float32)

        feature = torch.from_numpy(
            feature
        ).unsqueeze(0)

        return feature


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
print("UNSEEN RTVC AUDIO EVALUATION")
print("=" * 70)

print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"RTVC samples: {len(dataset)}"
)


# ============================================================
# MODEL
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
    f"Checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation AUC: "
    f"{checkpoint['val_auc']:.4f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

all_probs = []

with torch.no_grad():

    for features in loader:

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


all_probs = np.array(
    all_probs
)


# ============================================================
# RTVC DETECTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("RTVC PROBABILITY STATISTICS")
print("=" * 70)

print(
    f"Mean fake probability : "
    f"{all_probs.mean():.4f}"
)

print(
    f"Median probability    : "
    f"{np.median(all_probs):.4f}"
)

print(
    f"Minimum probability   : "
    f"{all_probs.min():.4f}"
)

print(
    f"Maximum probability   : "
    f"{all_probs.max():.4f}"
)

print(
    f"Std deviation         : "
    f"{all_probs.std():.4f}"
)


# ============================================================
# THRESHOLD DETECTION RATES
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN RTVC DETECTION RATE")
print("=" * 70)

thresholds = [
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90
]

for threshold in thresholds:

    detected = (
        all_probs >= threshold
    ).sum()

    rate = (
        detected /
        len(all_probs)
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Detected {detected:3d}/{len(all_probs)} | "
        f"Recall {rate:.4f} "
        f"({rate * 100:.2f}%)"
    )


# ============================================================
# QUANTILES
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY QUANTILES")
print("=" * 70)

for q in [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99
]:

    value = np.quantile(
        all_probs,
        q
    )

    print(
        f"{int(q * 100):02d}th percentile: "
        f"{value:.4f}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_df[
    "fake_probability"
] = all_probs

test_df[
    "predicted_label_0_5"
] = (
    all_probs >= 0.5
).astype(int)

test_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Predictions saved:"
)

print(
    OUTPUT_CSV
)

print("=" * 70)