import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(r"D:\DeepFake-Learning")

INPUT_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "train_raw_temporal_sequences.csv"
)

OUTPUT_DIR = BASE_DIR / "research" / "experiments"

TRAIN_CSV = OUTPUT_DIR / "train_temporal.csv"
VAL_CSV = OUTPUT_DIR / "validation_temporal.csv"
TEST_CSV = OUTPUT_DIR / "test_temporal.csv"


df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("VIDEO-LEVEL TEMPORAL DATASET SPLIT")
print("=" * 70)

print("Total sequences:", len(df))
print("Unique videos:", df["video_id"].nunique())


# ---------------------------------------------------------
# IMPORTANT:
# Split ONLY by source video ID.
#
# The same source video must NEVER appear in
# train + validation + test.
# ---------------------------------------------------------

video_ids = sorted(df["video_id"].unique())

train_videos, temp_videos = train_test_split(
    video_ids,
    test_size=0.20,
    random_state=42
)

val_videos, test_videos = train_test_split(
    temp_videos,
    test_size=0.50,
    random_state=42
)


train_videos = set(train_videos)
val_videos = set(val_videos)
test_videos = set(test_videos)


train_df = df[df["video_id"].isin(train_videos)].copy()
val_df = df[df["video_id"].isin(val_videos)].copy()
test_df = df[df["video_id"].isin(test_videos)].copy()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

train_set = set(train_df["video_id"])
val_set = set(val_df["video_id"])
test_set = set(test_df["video_id"])


print()
print("=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print("Train videos:", len(train_set))
print("Validation videos:", len(val_set))
print("Test videos:", len(test_set))

print()

print("Train sequences:", len(train_df))
print("Validation sequences:", len(val_df))
print("Test sequences:", len(test_df))


print()
print("=" * 70)
print("CATEGORY DISTRIBUTION")
print("=" * 70)

print()
print("TRAIN:")
print(train_df["category"].value_counts().sort_index())

print()
print("VALIDATION:")
print(val_df["category"].value_counts().sort_index())

print()
print("TEST:")
print(test_df["category"].value_counts().sort_index())


print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print()
print("TRAIN:")
print(train_df["label"].value_counts().sort_index())

print()
print("VALIDATION:")
print(val_df["label"].value_counts().sort_index())

print()
print("TEST:")
print(test_df["label"].value_counts().sort_index())


print()
print("=" * 70)
print("VIDEO OVERLAP CHECK")
print("=" * 70)

print("Train ∩ Validation:", len(train_set & val_set))
print("Train ∩ Test:", len(train_set & test_set))
print("Validation ∩ Test:", len(val_set & test_set))


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)

print()
print("Temporal dataset split complete!")
print("=" * 70)