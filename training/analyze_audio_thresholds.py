import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


BASE_DIR = r"D:\DeepFake-Learning"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_generator_holdout_raw_rms_predictions.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_rms_threshold_analysis.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_CSV)

y_true = df["audio_label"].astype(int).values

y_prob = df[
    "rms_normalized_fake_probability"
].values


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for threshold in np.arange(
    0.05,
    1.00,
    0.05
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    results.append({

        "threshold": round(
            threshold,
            2
        ),

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT ALL THRESHOLDS
# ============================================================

print()
print("=" * 90)
print("AUDIO RMS-NORMALIZED THRESHOLD ANALYSIS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# BEST BALANCED ACCURACY
# ============================================================

best_balanced = (
    results_df
    .sort_values(
        "balanced_accuracy",
        ascending=False
    )
    .iloc[0]
)

print()
print("=" * 90)
print("BEST BALANCED ACCURACY")
print("=" * 90)

print(
    f"Threshold:          "
    f"{best_balanced['threshold']:.2f}"
)

print(
    f"Balanced Accuracy:  "
    f"{best_balanced['balanced_accuracy']:.4f}"
)

print(
    f"Accuracy:           "
    f"{best_balanced['accuracy']:.4f}"
)

print(
    f"Precision:          "
    f"{best_balanced['precision']:.4f}"
)

print(
    f"Recall:             "
    f"{best_balanced['recall']:.4f}"
)

print(
    f"F1:                 "
    f"{best_balanced['f1']:.4f}"
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = (
    results_df
    .sort_values(
        "f1",
        ascending=False
    )
    .iloc[0]
)

print()
print("=" * 90)
print("BEST F1")
print("=" * 90)

print(
    f"Threshold:          "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"F1:                 "
    f"{best_f1['f1']:.4f}"
)

print(
    f"Precision:          "
    f"{best_f1['precision']:.4f}"
)

print(
    f"Recall:             "
    f"{best_f1['recall']:.4f}"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Saved:")
print(OUTPUT_CSV)

print("=" * 90)