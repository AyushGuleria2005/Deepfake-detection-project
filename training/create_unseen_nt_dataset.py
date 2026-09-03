import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\DeepFake-Learning")

SOURCE_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "train_balanced_face.csv"
)

VAL_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "validation_face.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "research"
    / "experiments"
)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(SOURCE_CSV)
val_df = pd.read_csv(VAL_CSV)


# ============================================================
# IDENTIFY CATEGORY FROM FILEPATH
# ============================================================

def get_category(path):

    path = str(path).replace("\\", "/")

    if "/Deepfakes/" in path:
        return "Deepfakes"

    elif "/Face2Face/" in path:
        return "Face2Face"

    elif "/FaceShifter/" in path:
        return "FaceShifter"

    elif "/FaceSwap/" in path:
        return "FaceSwap"

    elif "/NeuralTextures/" in path:
        return "NeuralTextures"

    elif "/Original/" in path:
        return "Original"

    return "Unknown"


train_df["category"] = train_df["filepath"].apply(
    get_category
)

val_df["category"] = val_df["filepath"].apply(
    get_category
)


# ============================================================
# TRAINING DATA
# REMOVE NEURALTEXTURES COMPLETELY
# ============================================================

train_without_nt = train_df[
    train_df["category"] != "NeuralTextures"
].copy()


# ============================================================
# UNSEEN TEST DATA
# ONLY NEURALTEXTURES + ORIGINAL
# ============================================================

test_unseen_nt = val_df[
    val_df["category"].isin(
        ["NeuralTextures", "Original"]
    )
].copy()


# ============================================================
# REMOVE HELPER COLUMN
# ============================================================

train_without_nt = train_without_nt.drop(
    columns=["category"]
)

test_unseen_nt = test_unseen_nt.drop(
    columns=["category"]
)


# ============================================================
# SAVE
# ============================================================

train_output = (
    OUTPUT_DIR
    / "train_without_neuraltextures.csv"
)

test_output = (
    OUTPUT_DIR
    / "test_unseen_neuraltextures.csv"
)

train_without_nt.to_csv(
    train_output,
    index=False
)

test_unseen_nt.to_csv(
    test_output,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 70)
print("UNSEEN MANIPULATION DATASET")
print("=" * 70)

print()
print("Training WITHOUT NeuralTextures")
print("-" * 40)

print(
    "Total:",
    len(train_without_nt)
)

print(
    "Real:",
    (train_without_nt["label"] == 0).sum()
)

print(
    "Fake:",
    (train_without_nt["label"] == 1).sum()
)

print()
print("Training categories:")

# Recalculate only for display
display_train = train_df[
    train_df["category"] != "NeuralTextures"
]

print(
    display_train["category"]
    .value_counts()
)


print()
print("=" * 70)
print("UNSEEN TEST")
print("=" * 70)

print()
print(
    "Total:",
    len(test_unseen_nt)
)

print(
    "Real / Original:",
    (test_unseen_nt["label"] == 0).sum()
)

print(
    "NeuralTextures:",
    (test_unseen_nt["label"] == 1).sum()
)

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(train_output)
print(test_output)

print()
print("Dataset creation complete!")
print("=" * 70)