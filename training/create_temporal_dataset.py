import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

INPUT_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "train_balanced_face.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "train_temporal_sequences.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("TEMPORAL DATASET CREATION - USING EXISTING VIDEO_ID")
print("=" * 70)

print()
print("Input samples:", len(df))

print()
print("Columns:")
print(df.columns.tolist())


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


df["category"] = df["filepath"].apply(
    get_category
)


# ============================================================
# CHECK VIDEO IDs
# ============================================================

print()
print("=" * 70)
print("VIDEO ID CHECK")
print("=" * 70)

print(
    "Unique video IDs:",
    df["video_id"].nunique()
)

print(
    "Missing video IDs:",
    df["video_id"].isna().sum()
)


# ============================================================
# GROUP SIZE ANALYSIS
# ============================================================

group_sizes = (
    df.groupby("video_id")
    .size()
)

print()
print("=" * 70)
print("GROUP SIZE ANALYSIS")
print("=" * 70)

print(
    "Groups with 5 samples:",
    (group_sizes == 5).sum()
)

print(
    "Groups with <5 samples:",
    (group_sizes < 5).sum()
)

print(
    "Groups with >5 samples:",
    (group_sizes > 5).sum()
)


# ============================================================
# SHOW EXAMPLE VIDEO IDS
# ============================================================

print()
print("=" * 70)
print("EXAMPLE VIDEO GROUPS")
print("=" * 70)

for video_id in group_sizes.head(10).index:

    group = df[
        df["video_id"] == video_id
    ]

    print()
    print(
        "Video ID:",
        video_id
    )

    print(
        "Samples:",
        len(group)
    )

    print(
        "Category:",
        group["category"].unique().tolist()
    )

    print(
        "Labels:",
        group["label"].unique().tolist()
    )

    print(
        "Files:"
    )

    for path in group["filepath"].tolist():
        print(
            " ",
            Path(path).name
        )


# ============================================================
# CREATE SEQUENCES
# ============================================================

sequence_rows = []

SEQUENCE_LENGTH = 5


for video_id, group in df.groupby(
    "video_id",
    sort=False
):

    group = group.reset_index(
        drop=True
    )

    # Need exactly 5 face crops
    if len(group) != SEQUENCE_LENGTH:
        continue


    # --------------------------------------------------------
    # Same label
    # --------------------------------------------------------

    labels = group[
        "label"
    ].tolist()

    if len(set(labels)) != 1:
        continue


    # --------------------------------------------------------
    # Same manipulation category
    # --------------------------------------------------------

    categories = group[
        "category"
    ].tolist()

    if len(set(categories)) != 1:
        continue


    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    paths = group[
        "filepath"
    ].tolist()

    if not all(
        Path(path).exists()
        for path in paths
    ):
        continue


    # --------------------------------------------------------
    # Store sequence
    # --------------------------------------------------------

    sequence_rows.append(
        {
            "video_id": video_id,

            "frame_0": paths[0],
            "frame_1": paths[1],
            "frame_2": paths[2],
            "frame_3": paths[3],
            "frame_4": paths[4],

            "label": labels[0],
            "category": categories[0]
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

sequence_df = pd.DataFrame(
    sequence_rows
)


# ============================================================
# SAVE
# ============================================================

sequence_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("TEMPORAL SEQUENCE RESULTS")
print("=" * 70)

print()
print(
    "Valid 5-frame sequences:",
    len(sequence_df)
)

print()
print("Category distribution:")

if len(sequence_df) > 0:

    print(
        sequence_df[
            "category"
        ].value_counts()
    )

else:

    print(
        "No valid sequences found."
    )


print()
print("Label distribution:")

if len(sequence_df) > 0:

    print(
        sequence_df[
            "label"
        ].value_counts()
    )

else:

    print(
        "No valid sequences found."
    )


# ============================================================
# FIRST VALID SEQUENCE
# ============================================================

if len(sequence_df) > 0:

    print()
    print("=" * 70)
    print("FIRST VALID SEQUENCE")
    print("=" * 70)

    first = sequence_df.iloc[0]

    print(
        "Video ID:",
        first["video_id"]
    )

    print(
        "Category:",
        first["category"]
    )

    print(
        "Label:",
        first["label"]
    )

    print()

    for i in range(5):

        print(
            f"Frame {i}:",
            first[f"frame_{i}"]
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("FILE CREATED")
print("=" * 70)

print(
    OUTPUT_CSV
)

print()
print(
    "Temporal dataset creation complete!"
)

print("=" * 70)