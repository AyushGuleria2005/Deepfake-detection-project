from pathlib import Path

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

INPUT = EXP / "fakeavceleb_audio_true_oof.csv"


print("=" * 70)
print("TRUE OOF AUDIO AUDIT")
print("=" * 70)

df = pd.read_csv(INPUT)

PROB = "audio_true_oof_fake_probability"

required = [
    "actual_path",
    "source",
    "method",
    "type",
    "video_label",
    "audio_label",
    PROB,
]

for col in required:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing column: {col}"
        )

print("Samples:", len(df))
print("Sources:", df["source"].nunique())

if df[PROB].isna().any():
    raise RuntimeError("Missing OOF probabilities!")

if df["actual_path"].duplicated().any():
    raise RuntimeError("Duplicate physical paths detected!")


# ------------------------------------------------------------
# OVERALL
# ------------------------------------------------------------

y_true = df["audio_label"].values
y_prob = df[PROB].values

y_pred = (
    y_prob >= 0.5
).astype(int)

auc = roc_auc_score(
    y_true,
    y_prob
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("=" * 70)
print("OVERALL OOF PERFORMANCE")
print("=" * 70)

print(f"ROC-AUC:           {auc:.4f}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1:                {f1:.4f}")

print()
print("Confusion Matrix:")
print(cm)


# ------------------------------------------------------------
# PROBABILITY BY LABEL
# ------------------------------------------------------------

print()
print("=" * 70)
print("PROBABILITY BY AUDIO LABEL")
print("=" * 70)

print(
    df.groupby("audio_label")[PROB].agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
)


# ------------------------------------------------------------
# METHOD-WISE
# ------------------------------------------------------------

print()
print("=" * 70)
print("METHOD-WISE OOF AUDIT")
print("=" * 70)

for method, group in df.groupby("method"):

    labels = group["audio_label"].values
    probs = group[PROB].values

    print()
    print(f"Method: {method}")
    print("-" * 70)

    print("Samples:", len(group))
    print(
        "Audio labels:",
        group["audio_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "Mean probability:",
        round(probs.mean(), 6)
    )

    # AUC only if both classes exist.
    if len(np.unique(labels)) == 2:

        method_auc = roc_auc_score(
            labels,
            probs
        )

        print(
            "ROC-AUC:",
            round(method_auc, 4)
        )

    else:

        predicted = (
            probs >= 0.5
        ).astype(int)

        print(
            "Single-class method; "
            "AUC not applicable"
        )

        print(
            "Predicted fake rate:",
            round(predicted.mean(), 4)
        )


# ------------------------------------------------------------
# TYPE-WISE
# ------------------------------------------------------------

print()
print("=" * 70)
print("DATASET TYPE-WISE AUDIT")
print("=" * 70)

for sample_type, group in df.groupby("type"):

    print()
    print(f"Type: {sample_type}")
    print("-" * 70)

    print("Samples:", len(group))

    print(
        "Audio labels:",
        group["audio_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "Mean OOF probability:",
        round(
            group[PROB].mean(),
            6
        )
    )


# ------------------------------------------------------------
# VIDEO/AUDIO MODALITY COMBINATIONS
# ------------------------------------------------------------

print()
print("=" * 70)
print("CROSS-MODALITY AUDIT")
print("=" * 70)

df["modality_combination"] = (
    "video_" +
    df["video_label"].map({
        0: "real",
        1: "fake"
    }) +
    "_audio_" +
    df["audio_label"].map({
        0: "real",
        1: "fake"
    })
)

for combo, group in df.groupby(
    "modality_combination"
):

    print()
    print(combo)
    print("-" * 70)

    print("Samples:", len(group))

    print(
        "Mean audio probability:",
        round(
            group[PROB].mean(),
            6
        )
    )

    print(
        "Median audio probability:",
        round(
            group[PROB].median(),
            6
        )
    )

    print(
        "Predicted fake rate:",
        round(
            (group[PROB] >= 0.5).mean(),
            4
        )
    )


# ------------------------------------------------------------
# SOURCE-LEVEL CHECK
# ------------------------------------------------------------

print()
print("=" * 70)
print("SOURCE-LEVEL AUDIT")
print("=" * 70)

source_stats = (
    df.groupby("source")
    .agg(
        samples=("actual_path", "count"),
        mean_probability=(PROB, "mean"),
        min_probability=(PROB, "min"),
        max_probability=(PROB, "max"),
        audio_real_count=(
            "audio_label",
            lambda x: (x == 0).sum()
        ),
        audio_fake_count=(
            "audio_label",
            lambda x: (x == 1).sum()
        ),
    )
)

print(
    source_stats[
        [
            "samples",
            "mean_probability",
            "min_probability",
            "max_probability",
            "audio_real_count",
            "audio_fake_count",
        ]
    ].describe()
)


# ------------------------------------------------------------
# EXTREME ERRORS
# ------------------------------------------------------------

print()
print("=" * 70)
print("MOST CONFIDENT WRONG PREDICTIONS")
print("=" * 70)

df["error"] = (
    df["audio_label"] !=
    (df[PROB] >= 0.5).astype(int)
)

errors = df[df["error"]].copy()

print(
    "Total errors:",
    len(errors)
)

if len(errors) > 0:

    errors["confidence"] = np.where(
        errors["audio_label"] == 1,
        errors[PROB],
        1 - errors[PROB]
    )

    print()
    print(
        errors.sort_values(
            "confidence"
        ).tail(
            20
        )[
            [
                "source",
                "method",
                "type",
                "audio_label",
                PROB,
                "actual_path",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("=" * 70)
print("OOF AUDIO AUDIT COMPLETE")
print("=" * 70)