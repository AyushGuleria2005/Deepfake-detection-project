import pandas as pd
from pathlib import Path
import random

# ============================================================
# PATHS
# ============================================================

INDEX = Path(
    r"D:\DeepFake-Learning\research\experiments\fakeavceleb_index_clean.csv"
)

OUTPUT = Path(
    r"D:\DeepFake-Learning\research\experiments"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CLEAN INDEX
# ============================================================

if not INDEX.exists():
    raise FileNotFoundError(
        f"Clean index not found:\n{INDEX}"
    )

df = pd.read_csv(INDEX)

print("Loaded clean index rows:", len(df))


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "actual_path",
    "source",
    "video_label",
    "audio_label"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# Make sure every physical video appears only once
duplicate_count = df["actual_path"].duplicated().sum()

print("Duplicate actual paths:", duplicate_count)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate actual_path values found in clean index."
    )


# ============================================================
# SOURCE-LEVEL SPLIT
# ============================================================

sources = sorted(df["source"].unique())

print("Total source IDs:", len(sources))

if len(sources) != 500:
    raise ValueError(
        f"Expected 500 source IDs, found {len(sources)}"
    )


# Fixed seed for reproducibility
random.seed(42)

random.shuffle(sources)


# 80% Train = 400 sources
# 10% Validation = 50 sources
# 10% Test = 50 sources

train_sources = set(sources[:400])
val_sources = set(sources[400:450])
test_sources = set(sources[450:500])


# ============================================================
# CREATE SPLITS
# ============================================================

train_df = df[
    df["source"].isin(train_sources)
].copy()

val_df = df[
    df["source"].isin(val_sources)
].copy()

test_df = df[
    df["source"].isin(test_sources)
].copy()


# ============================================================
# SAVE SPLITS
# ============================================================

train_path = OUTPUT / "fakeavceleb_train.csv"
val_path = OUTPUT / "fakeavceleb_validation.csv"
test_path = OUTPUT / "fakeavceleb_test.csv"

train_df.to_csv(
    train_path,
    index=False
)

val_df.to_csv(
    val_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ============================================================
# VERIFICATION
# ============================================================

print("\nRows:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

total = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("Total:", total)

print("\nExpected clean total:", len(df))


# ============================================================
# SOURCE COUNTS
# ============================================================

print("\nUnique sources:")

print(
    "Train:",
    train_df["source"].nunique()
)

print(
    "Validation:",
    val_df["source"].nunique()
)

print(
    "Test:",
    test_df["source"].nunique()
)


# ============================================================
# SOURCE LEAKAGE CHECK
# ============================================================

train_set = set(train_df["source"])
val_set = set(val_df["source"])
test_set = set(test_df["source"])

print("\nSource overlap:")

print(
    "Train ∩ Validation:",
    len(train_set & val_set)
)

print(
    "Train ∩ Test:",
    len(train_set & test_set)
)

print(
    "Validation ∩ Test:",
    len(val_set & test_set)
)


# ============================================================
# VIDEO DUPLICATE CHECK ACROSS SPLITS
# ============================================================

train_paths = set(train_df["actual_path"])
val_paths = set(val_df["actual_path"])
test_paths = set(test_df["actual_path"])

print("\nVideo path overlap:")

print(
    "Train ∩ Validation:",
    len(train_paths & val_paths)
)

print(
    "Train ∩ Test:",
    len(train_paths & test_paths)
)

print(
    "Validation ∩ Test:",
    len(val_paths & test_paths)
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nVideo label distribution:")

print(
    "Train:",
    train_df["video_label"].value_counts().sort_index().to_dict()
)

print(
    "Validation:",
    val_df["video_label"].value_counts().sort_index().to_dict()
)

print(
    "Test:",
    test_df["video_label"].value_counts().sort_index().to_dict()
)


print("\nAudio label distribution:")

print(
    "Train:",
    train_df["audio_label"].value_counts().sort_index().to_dict()
)

print(
    "Validation:",
    val_df["audio_label"].value_counts().sort_index().to_dict()
)

print(
    "Test:",
    test_df["audio_label"].value_counts().sort_index().to_dict()
)


# ============================================================
# FINAL CHECKS
# ============================================================

if total != len(df):
    raise ValueError(
        "Split total does not match clean index total."
    )

if len(train_set & val_set) != 0:
    raise ValueError(
        "Source leakage detected: Train/Validation"
    )

if len(train_set & test_set) != 0:
    raise ValueError(
        "Source leakage detected: Train/Test"
    )

if len(val_set & test_set) != 0:
    raise ValueError(
        "Source leakage detected: Validation/Test"
    )

if len(train_paths & val_paths) != 0:
    raise ValueError(
        "Video leakage detected: Train/Validation"
    )

if len(train_paths & test_paths) != 0:
    raise ValueError(
        "Video leakage detected: Train/Test"
    )

if len(val_paths & test_paths) != 0:
    raise ValueError(
        "Video leakage detected: Validation/Test"
    )


# ============================================================
# DONE
# ============================================================

print("\nSaved:")
print(train_path)
print(val_path)
print(test_path)

print("\nCLEAN SOURCE-LEVEL SPLIT PASSED.")