from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

PRED_CSV = EXP / "fakeavceleb_audio_rtvc_clean_predictions.csv"
OUTPUT = EXP / "audio_method_diagnosis.csv"

print("=" * 70)
print("AUDIO METHOD-WISE DIAGNOSIS")
print("=" * 70)

df = pd.read_csv(PRED_CSV)

print("\nPrediction samples:", len(df))

required = [
    "actual_path",
    "source",
    "method",
    "eval_label",
    "predicted_probability",
    "predicted_label",
]

for col in required:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing required column '{col}' in prediction CSV."
        )

print("\nMethod distribution:")
print(df["method"].value_counts())

print("\n" + "=" * 70)
print("METHOD-WISE PROBABILITY")
print("=" * 70)

summary = (
    df.groupby(["method", "eval_label"])["predicted_probability"]
    .agg(
        samples="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max",
    )
    .reset_index()
)

print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("METHOD-WISE ACCURACY")
print("=" * 70)

df["correct"] = (
    df["predicted_label"] == df["eval_label"]
)

accuracy = (
    df.groupby(["method", "eval_label"])["correct"]
    .mean()
    .reset_index(name="accuracy")
)

print(accuracy.to_string(index=False))

summary.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)