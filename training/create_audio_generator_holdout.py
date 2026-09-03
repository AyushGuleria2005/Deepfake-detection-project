from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\DeepFake-Learning")

INPUT_CSV = PROJECT_ROOT / "research" / "experiments" / "fakeavceleb_fusion_dataset.csv"
OUTPUT_TRAIN = PROJECT_ROOT / "research" / "experiments" / "audio_generator_holdout_train.csv"
OUTPUT_TEST = PROJECT_ROOT / "research" / "experiments" / "audio_generator_holdout_test.csv"

TEST_SOURCES = 100
SEED = 42

df = pd.read_csv(INPUT_CSV)

required = {"actual_path", "source", "method", "audio_label"}
missing = required - set(df.columns)

if missing:
    raise RuntimeError(f"Missing columns: {missing}")

df["source"] = df["source"].astype(str)
df["method"] = df["method"].astype(str)
df["audio_label"] = df["audio_label"].astype(int)

print(f"Total samples: {len(df)}")
print(f"Unique sources: {df['source'].nunique()}")

# ---------------------------------------------------------
# 1. Select RTVC fake-audio sources for completely unseen test
# ---------------------------------------------------------

rtvc_fake = df[
    (df["method"].str.lower() == "rtvc") &
    (df["audio_label"] == 1)
].copy()

rtvc_sources = sorted(rtvc_fake["source"].unique())

print(f"RTVC fake samples: {len(rtvc_fake)}")
print(f"RTVC sources: {len(rtvc_sources)}")

if len(rtvc_sources) < TEST_SOURCES:
    raise RuntimeError("Not enough RTVC sources for test.")

# Deterministic selection
selected_test_sources = rtvc_sources[:TEST_SOURCES]

print(f"Selected test sources: {len(selected_test_sources)}")
print(f"First test sources: {selected_test_sources[:10]}")

# ---------------------------------------------------------
# 2. Select ONE RTVC fake sample per test source
# ---------------------------------------------------------

rtvc_test = (
    rtvc_fake[
        rtvc_fake["source"].isin(selected_test_sources)
    ]
    .sort_values(["source", "actual_path"])
    .groupby("source", as_index=False)
    .first()
)

if len(rtvc_test) != TEST_SOURCES:
    raise RuntimeError(
        f"Expected {TEST_SOURCES} RTVC test samples, "
        f"got {len(rtvc_test)}"
    )

# ---------------------------------------------------------
# 3. Find ONE REAL-AUDIO sample for every same source
#    IMPORTANT: RTVC is excluded from real candidates
# ---------------------------------------------------------

real_candidates = df[
    (df["audio_label"] == 0) &
    (df["method"].str.lower() != "rtvc") &
    (df["source"].isin(selected_test_sources))
].copy()

real_test = (
    real_candidates
    .sort_values(["source", "actual_path"])
    .groupby("source", as_index=False)
    .first()
)

if len(real_test) != TEST_SOURCES:
    missing_sources = sorted(
        set(selected_test_sources) -
        set(real_test["source"])
    )
    raise RuntimeError(
        f"Could not find real-audio match for sources: "
        f"{missing_sources}"
    )

# ---------------------------------------------------------
# 4. Combine matched test set
# ---------------------------------------------------------

test_df = pd.concat(
    [real_test, rtvc_test],
    ignore_index=True
)

test_df = test_df.sort_values(
    ["source", "audio_label", "actual_path"]
).reset_index(drop=True)

# ---------------------------------------------------------
# 5. Training data
#
# RTVC completely excluded.
# ALL samples from test sources excluded.
# ---------------------------------------------------------

train_df = df[
    (df["method"].str.lower() != "rtvc") &
    (~df["source"].isin(selected_test_sources))
].copy()

# ---------------------------------------------------------
# 6. Safety checks
# ---------------------------------------------------------

train_sources = set(train_df["source"])
test_sources = set(test_df["source"])

overlap = train_sources & test_sources

if overlap:
    raise RuntimeError(
        f"SOURCE LEAKAGE DETECTED: {len(overlap)} overlapping sources"
    )

if "rtvc" in set(train_df["method"].str.lower()):
    raise RuntimeError("RTVC FOUND IN TRAINING!")

if len(test_df) != TEST_SOURCES * 2:
    raise RuntimeError(
        f"Expected {TEST_SOURCES * 2} test samples, got {len(test_df)}"
    )

if test_df["audio_label"].value_counts().to_dict() != {0: TEST_SOURCES, 1: TEST_SOURCES}:
    raise RuntimeError("Test labels are not perfectly balanced.")

if train_df["actual_path"].duplicated().any():
    raise RuntimeError("Duplicate physical files in training.")

if test_df["actual_path"].duplicated().any():
    raise RuntimeError("Duplicate physical files in test.")

if set(train_df["actual_path"]) & set(test_df["actual_path"]):
    raise RuntimeError("Physical-file leakage detected!")

# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

# ---------------------------------------------------------
# 8. Report
# ---------------------------------------------------------

print("\n========== FINAL HOLDOUT ==========")

print(f"Train samples: {len(train_df)}")
print(f"Test samples:  {len(test_df)}")

print(f"Train sources: {len(train_sources)}")
print(f"Test sources:  {len(test_sources)}")

print(f"Source overlap: {len(overlap)}")

print("\nTrain audio labels:")
print(train_df["audio_label"].value_counts().sort_index())

print("\nTest audio labels:")
print(test_df["audio_label"].value_counts().sort_index())

print("\nTrain methods:")
print(train_df["method"].value_counts())

print("\nTest methods:")
print(test_df["method"].value_counts())

print("\nTest source examples:")
print(test_df[["source", "method", "audio_label", "actual_path"]].head(10).to_string(index=False))

print("\n===================================")
print("AUDIT PASSED")
print("RTVC completely excluded from training.")
print("Test sources completely excluded from training.")
print("Source overlap = 0")
print("Physical-file overlap = 0")
print(f"\nSaved:")
print(OUTPUT_TRAIN)
print(OUTPUT_TEST)