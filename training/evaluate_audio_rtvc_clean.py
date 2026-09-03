from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
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
sys.path.insert(0, str(ROOT))

from models.audio_cnn import AudioCNN


TEST_CSV = (
    ROOT / "research" / "experiments" /
    "fakeavceleb_audio_rtvc_clean_test.csv"
)

FEATURE_CSV = (
    ROOT / "research" / "experiments" /
    "fakeavceleb_audio_all_features.csv"
)

PRED_CSV = (
    ROOT / "research" / "experiments" /
    "fakeavceleb_audio_rtvc_clean_predictions.csv"
)

CHECKPOINT = (
    ROOT / "outputs" / "checkpoints" /
    "audio_cnn_unseen_rtvc_best.pth"
)


print("=" * 70)
print("CLEAN UNSEEN RTVC AUDIO EVALUATION")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Load clean test
# ------------------------------------------------------------

test_df = pd.read_csv(TEST_CSV)

print("\nClean test samples:", len(test_df))
print("Real samples:", int((test_df["eval_label"] == 0).sum()))
print("RTVC fake samples:", int((test_df["eval_label"] == 1).sum()))


# ------------------------------------------------------------
# Load feature mapping
# ------------------------------------------------------------

feature_df = pd.read_csv(FEATURE_CSV)

print("\nFeature mappings:", len(feature_df))

required = ["actual_path", "feature_path"]

for col in required:
    if col not in feature_df.columns:
        raise RuntimeError(
            f"Required column missing from feature CSV: {col}"
        )


# ------------------------------------------------------------
# Merge clean test with feature paths
# ------------------------------------------------------------

df = test_df.merge(
    feature_df[["actual_path", "feature_path"]],
    on="actual_path",
    how="left"
)

missing = df["feature_path"].isna().sum()

print("Missing feature mappings:", missing)

if missing > 0:
    print(df[df["feature_path"].isna()][["actual_path"]])
    raise RuntimeError("Some clean test samples have no feature mapping.")


print("Feature mapping verification: PASS")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = AudioCNN().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nCheckpoint epoch:", checkpoint.get("epoch"))
print("Validation AUC:", f'{checkpoint.get("val_auc", 0):.4f}')


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

probabilities = []
labels = []

with torch.no_grad():

    for _, row in df.iterrows():

        feature_path = Path(row["feature_path"])

        if not feature_path.is_absolute():
            feature_path = ROOT / feature_path

        feature = np.load(feature_path).astype(np.float32)

        x = torch.from_numpy(feature)
        x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(device)

        logits = model(x)

        probability = torch.sigmoid(logits).item()

        probabilities.append(probability)
        labels.append(int(row["eval_label"]))


y_true = np.array(labels)
y_prob = np.array(probabilities)

y_pred = (y_prob >= 0.50).astype(int)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

auc = roc_auc_score(y_true, y_prob)
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

cm = confusion_matrix(y_true, y_pred)


print("\n" + "=" * 70)
print("CLEAN RTVC RESULTS")
print("=" * 70)

print(f"ROC-AUC           : {auc:.4f}")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")

print("\nConfusion Matrix")
print("[[TN FP]")
print(" [FN TP]]")
print(cm)


# ------------------------------------------------------------
# Probability analysis
# ------------------------------------------------------------

real_probs = y_prob[y_true == 0]
rtvc_probs = y_prob[y_true == 1]

print("\n" + "=" * 70)
print("CLASS-WISE PROBABILITY")
print("=" * 70)

print(f"Real mean probability : {real_probs.mean():.4f}")
print(f"Real median           : {np.median(real_probs):.4f}")
print(f"Real maximum          : {real_probs.max():.4f}")

print(f"RTVC mean probability : {rtvc_probs.mean():.4f}")
print(f"RTVC median           : {np.median(rtvc_probs):.4f}")
print(f"RTVC minimum          : {rtvc_probs.min():.4f}")


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

df["predicted_probability"] = y_prob
df["predicted_label"] = y_pred

df.to_csv(PRED_CSV, index=False)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("Predictions saved:")
print(PRED_CSV)

print("=" * 70)