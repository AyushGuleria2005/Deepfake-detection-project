import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\DeepFake-Learning")

TRAIN_CSV = (
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
# LOAD
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)


# ============================================================
# CATEGORY
# ============================================================

def get_category(path):

    path = str(path).replace("\\", "/")

    categories = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
        "Original"
    ]

    for category in categories:
        if f"/{category}/" in path:
            return category

    return "Unknown"


train_df["category"] = train_df["filepath"].apply(get_category)
val_df["category"] = val_df["filepath"].apply(get_category)


# ============================================================
# CONTROLLED TRAINING SET
# ============================================================
#
# Use the SAME number of samples as Experiment A
# for each category wherever possible.
#
# Experiment A:
# Original       3978
# Deepfakes       790
# Face2Face       794
# FaceShifter     792
# FaceSwap        792
# NeuralTextures  0
#
# Experiment B:
# Include NeuralTextures with 790 samples.
#
# This keeps the training size controlled.
# ============================================================

target_counts = {
    "Original": 3978,
    "Deepfakes": 790,
    "Face2Face": 794,
    "FaceShifter": 792,
    "FaceSwap": 792,
    "NeuralTextures": 790
}

training_parts = []

for category, count in target_counts.items():

    category_df = train_df[
        train_df["category"] == category
    ].copy()

    if len(category_df) < count:
        print(
            f"WARNING: {category} has only "
            f"{len(category_df)} samples."
        )

        selected = category_df

    else:
        # Fixed random seed for reproducibility
        selected = category_df.sample(
            n=count,
            random_state=42
        )

    training_parts.append(selected)


control_train = pd.concat(
    training_parts,
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

control_train = control_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# REMOVE HELPER COLUMN
# ============================================================

control_train = control_train.drop(
    columns=["category"]
)


# ============================================================
# TEST SET
# ============================================================
#
# IMPORTANT:
# We use validation_face.csv's held-out videos.
#
# Test contains:
# Original + NeuralTextures
#
# These videos were NOT used in training.
# ============================================================

control_test = val_df[
    val_df["category"].isin(
        ["Original", "NeuralTextures"]
    )
].copy()


control_test = control_test.drop(
    columns=["category"]
)


# ============================================================
# SAVE
# ============================================================

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train_seen_neuraltextures_control.csv"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "test_seen_neuraltextures_control.csv"
)

control_train.to_csv(
    TRAIN_OUTPUT,
    index=False
)

control_test.to_csv(
    TEST_OUTPUT,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 70)
print("CONTROLLED NEURALTEXTURES EXPERIMENT")
print("=" * 70)

print()
print("TRAINING SET")
print("-" * 40)

print(
    "Total:",
    len(control_train)
)

print()
print(
    "Category distribution:"
)

train_report = train_df[
    train_df["category"].isin(
        target_counts.keys()
    )
]

# Report from created dataset
report_categories = []

for path in control_train["filepath"]:
    report_categories.append(
        get_category(path)
    )

print(
    pd.Series(report_categories)
    .value_counts()
)


print()
print("=" * 70)
print("TEST SET")
print("=" * 70)

print(
    "Total:",
    len(control_test)
)

test_categories = []

for path in control_test["filepath"]:
    test_categories.append(
        get_category(path)
    )

print()
print(
    pd.Series(test_categories)
    .value_counts()
)


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)

print()
print("Control dataset creation complete!")
print("=" * 70)