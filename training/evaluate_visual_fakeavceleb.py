from pathlib import Path
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

INPUT = EXP / "fakeavceleb_visual_predictions.csv"

print("=" * 70)
print("VISUAL MODEL — FAKEAVCELEB SANITY CHECK")
print("=" * 70)

df = pd.read_csv(INPUT)

print("Samples:", len(df))

# ---------------------------------------------------------
# Overall video-label evaluation
# ---------------------------------------------------------

y_true = df["video_label"].astype(int).values
y_prob = df["visual_fake_probability"].astype(float).values

y_pred = (y_prob >= 0.50).astype(int)

auc = roc_auc_score(y_true, y_prob)
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nOVERALL VIDEO DETECTION")
print("-" * 50)
print(f"ROC-AUC:           {auc:.4f}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1:                {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ---------------------------------------------------------
# Modality combinations
# ---------------------------------------------------------

print("\nMODALITY COMBINATIONS")
print("-" * 50)

groups = (
    df.groupby(["video_label", "audio_label"])
    .agg(
        samples=("actual_path", "size"),
        mean_visual_prob=("visual_fake_probability", "mean"),
        median_visual_prob=("visual_fake_probability", "median"),
    )
    .reset_index()
)

for _, row in groups.iterrows():

    video = int(row["video_label"])
    audio = int(row["audio_label"])

    if video == 0 and audio == 0:
        name = "Real Video + Real Audio"
    elif video == 0 and audio == 1:
        name = "Real Video + Fake Audio"
    elif video == 1 and audio == 0:
        name = "Fake Video + Real Audio"
    else:
        name = "Fake Video + Fake Audio"

    print(
        f"{name:28s} | "
        f"n={int(row['samples']):5d} | "
        f"mean={row['mean_visual_prob']:.4f} | "
        f"median={row['median_visual_prob']:.4f}"
    )

# ---------------------------------------------------------
# Method-wise visual performance
# ---------------------------------------------------------

print("\nMETHOD-WISE VIDEO DETECTION")
print("-" * 50)

for method, g in df.groupby("method"):

    if g["video_label"].nunique() < 2:
        print(
            f"{method:20s} | "
            f"n={len(g):5d} | "
            f"only one video class"
        )
        continue

    y = g["video_label"].astype(int)
    p = g["visual_fake_probability"].astype(float)
    pred = (p >= 0.50).astype(int)

    method_auc = roc_auc_score(y, p)
    method_bal = balanced_accuracy_score(y, pred)

    print(
        f"{method:20s} | "
        f"n={len(g):5d} | "
        f"AUC={method_auc:.4f} | "
        f"Bal.Acc={method_bal:.4f}"
    )

print("\nVisual probability statistics:")
print(f"Real video mean: {y_prob[y_true == 0].mean():.4f}")
print(f"Fake video mean: {y_prob[y_true == 1].mean():.4f}")

print("=" * 70)
print("VISUAL SANITY CHECK COMPLETE")
print("=" * 70)