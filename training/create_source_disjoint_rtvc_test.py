from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

INDEX = EXP / "fakeavceleb_index_clean.csv"
OUTPUT_TRAIN = EXP / "fakeavceleb_audio_source_disjoint_train.csv"
OUTPUT_TEST = EXP / "fakeavceleb_audio_source_disjoint_rtvc_test.csv"

RANDOM_SEED = 42
TEST_SOURCES = 50

print("=" * 70)
print("SOURCE-DISJOINT RTVC SPLIT")
print("=" * 70)

df = pd.read_csv(INDEX)

print("Total samples:", len(df))
print("Unique sources:", df["source"].nunique())

# Only RTVC samples for the unseen test
rtvc = df[df["method"].str.lower() == "rtvc"].copy()

if len(rtvc) == 0:
    raise RuntimeError("No RTVC samples found.")

rtvc_sources = sorted(rtvc["source"].unique())

print("RTVC samples:", len(rtvc))
print("RTVC sources:", len(rtvc_sources))

# Select 50 completely unseen source IDs
rng = np.random.default_rng(RANDOM_SEED)

if len(rtvc_sources) < TEST_SOURCES:
    raise RuntimeError("Not enough RTVC source IDs.")

test_sources = sorted(
    rng.choice(
        rtvc_sources,
        size=TEST_SOURCES,
        replace=False
    )
)

test_sources = set(test_sources)

print("\nSelected unseen RTVC sources:", len(test_sources))

# Test = ONLY RTVC from unseen sources
test_df = rtvc[
    rtvc["source"].isin(test_sources)
].copy()

# Training = everything except selected sources
train_df = df[
    ~df["source"].isin(test_sources)
].copy()

# Safety checks
train_sources = set(train_df["source"])
test_source_set = set(test_df["source"])

overlap = train_sources & test_source_set

print("\n" + "=" * 70)
print("SAFETY CHECKS")
print("=" * 70)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))
print("Train sources:", len(train_sources))
print("Test sources:", len(test_source_set))
print("Source overlap:", len(overlap))

if len(overlap) != 0:
    raise RuntimeError("SOURCE LEAKAGE DETECTED!")

print("\nTest methods:")
print(test_df["method"].value_counts())

print("\nTest labels:")
print(test_df["audio_label"].value_counts())

print("\nTrain methods:")
print(train_df["method"].value_counts())

print("\nTrain audio labels:")
print(train_df["audio_label"].value_counts())

print("\nSelected test sources:")
print(sorted(test_sources))

train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

print("\nSaved:")
print(OUTPUT_TRAIN)
print(OUTPUT_TEST)

print("=" * 70)
print("SOURCE-DISJOINT SPLIT COMPLETE")
print("=" * 70)