from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

pred_path = ROOT / "research" / "experiments" / "fakeavceleb_audio_rtvc_matched_predictions.csv"
test_path = ROOT / "research" / "experiments" / "fakeavceleb_audio_rtvc_matched_test.csv"

pred = pd.read_csv(pred_path)
test = pd.read_csv(test_path)

print("=" * 70)
print("AUDIO SOURCE-LEVEL DIAGNOSIS")
print("=" * 70)

print("\nPrediction columns:")
print(pred.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())

# Keep only prediction information needed for the merge
pred_small = pred[
    ["actual_path", "fake_probability", "predicted_label"]
].copy()

# Merge predictions into the test metadata
df = test.merge(
    pred_small,
    on="actual_path",
    how="inner"
)

print("\nMerged samples:", len(df))

# ------------------------------------------------------------
# BASIC CHECKS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC CHECKS")
print("=" * 70)

print("\nEvaluation labels:")
print(df["eval_label"].value_counts().sort_index())

print("\nMethods:")
print(df["method"].value_counts())

print("\nUnique sources:")
print(df["source"].nunique())

print("\nSource counts:")
print(df.groupby("eval_label")["source"].nunique())

# ------------------------------------------------------------
# CLASS STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS STATISTICS")
print("=" * 70)

for label, group in df.groupby("eval_label"):
    name = "REAL" if label == 0 else "RTVC FAKE"

    p = group["fake_probability"].astype(float)

    print(f"\n{name}")
    print("Samples :", len(group))
    print("Mean    :", f"{p.mean():.6f}")
    print("Median  :", f"{p.median():.6f}")
    print("Std     :", f"{p.std():.6f}")
    print("Min     :", f"{p.min():.6f}")
    print("Max     :", f"{p.max():.6f}")

# ------------------------------------------------------------
# SOURCE-LEVEL PAIRING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SOURCE-LEVEL PAIRING")
print("=" * 70)

pair_counts = (
    df.groupby(["source", "eval_label"])
    .size()
    .unstack(fill_value=0)
)

real_count = pair_counts[0] if 0 in pair_counts.columns else 0
rtvc_count = pair_counts[1] if 1 in pair_counts.columns else 0

paired_sources = pair_counts[
    (real_count > 0) &
    (rtvc_count > 0)
].index

print("\nSources with both REAL and RTVC:")
print("Count:", len(paired_sources))

paired = df[df["source"].isin(paired_sources)].copy()

# ------------------------------------------------------------
# SOURCE-LEVEL PROBABILITIES
# ------------------------------------------------------------

source_stats = (
    paired
    .groupby(["source", "eval_label"])["fake_probability"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)

print("\nFirst 20 source-level results:")
print(source_stats.head(20).to_string(index=False))

# ------------------------------------------------------------
# REAL vs RTVC DIRECT COMPARISON
# ------------------------------------------------------------

pivot = (
    paired
    .pivot_table(
        index="source",
        columns="eval_label",
        values="fake_probability",
        aggfunc="mean"
    )
)

pivot = pivot.rename(
    columns={
        0: "real_probability",
        1: "rtvc_probability"
    }
)

pivot["probability_gap"] = (
    pivot["rtvc_probability"] -
    pivot["real_probability"]
)

print("\n" + "=" * 70)
print("REAL vs RTVC PROBABILITY GAP")
print("=" * 70)

print("\nMean gap:",
      f"{pivot['probability_gap'].mean():.6f}")

print("Median gap:",
      f"{pivot['probability_gap'].median():.6f}")

print("Minimum gap:",
      f"{pivot['probability_gap'].min():.6f}")

print("Maximum gap:",
      f"{pivot['probability_gap'].max():.6f}")

# ------------------------------------------------------------
# WORST SEPARATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10 WORST SOURCE-LEVEL SEPARATIONS")
print("=" * 70)

print(
    pivot
    .sort_values("probability_gap")
    .head(10)
    .to_string()
)

# ------------------------------------------------------------
# BEST SEPARATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10 BEST SOURCE-LEVEL SEPARATIONS")
print("=" * 70)

print(
    pivot
    .sort_values("probability_gap", ascending=False)
    .head(10)
    .to_string()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

out_path = (
    ROOT /
    "research" /
    "experiments" /
    "audio_source_diagnosis.csv"
)

pivot.to_csv(out_path)

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

print("Saved:")
print(out_path)
print("=" * 70)