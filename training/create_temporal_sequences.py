import pandas as pd
from pathlib import Path
import re

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


df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("TEMPORAL SEQUENCE RECONSTRUCTION")
print("=" * 70)

print("Input samples:", len(df))


# ---------------------------------------------------------
# CATEGORY
# ---------------------------------------------------------

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


df["category"] = df["filepath"].apply(get_category)


# ---------------------------------------------------------
# ORIGINAL FILENAME
# ---------------------------------------------------------

df["original_filename"] = (
    df["original_filepath"]
    .apply(lambda x: Path(str(x)).name)
)


# ---------------------------------------------------------
# PARSE FILENAME
#
# Original:
#   602_f1.jpg
#
# Fake:
#   579_701_f4.jpg
#
# We create:
#
# Original:
#   sequence_key = 602
#
# Fake:
#   sequence_key = 579_701
#
# frame_number = 0...4
# ---------------------------------------------------------

def parse_filename(filename, category):

    filename = str(filename)

    match = re.match(
        r"^(\d+)(?:_(\d+))?_f(\d+)\.jpg$",
        filename,
        re.IGNORECASE
    )

    if not match:
        return None, None

    first_id = match.group(1)
    second_id = match.group(2)
    frame = int(match.group(3))

    if category == "Original":

        sequence_key = first_id

    else:

        if second_id is None:
            return None, None

        sequence_key = f"{first_id}_{second_id}"

    return sequence_key, frame


parsed = df.apply(
    lambda row: parse_filename(
        row["original_filename"],
        row["category"]
    ),
    axis=1
)

df["sequence_key"] = parsed.apply(lambda x: x[0])
df["frame_number"] = parsed.apply(lambda x: x[1])


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

print()
print("=" * 70)
print("PARSING VALIDATION")
print("=" * 70)

print(
    "Invalid filenames:",
    df["sequence_key"].isna().sum()
)

print(
    "Missing frame numbers:",
    df["frame_number"].isna().sum()
)

print()
print("Samples by category:")
print(df["category"].value_counts().sort_index())


# ---------------------------------------------------------
# BUILD GROUPS
#
# IMPORTANT:
#
# sequence_key + category
#
# This prevents different manipulation types
# from being mixed together.
# ---------------------------------------------------------

df = df.dropna(
    subset=["sequence_key", "frame_number"]
).copy()

groups = df.groupby(
    ["category", "sequence_key"],
    sort=True
)


sequences = []


for (category, sequence_key), group in groups:

    group = group.drop_duplicates(
        subset=["frame_number"],
        keep="first"
    )

    frame_map = {
        int(row["frame_number"]): row["filepath"]
        for _, row in group.iterrows()
    }

    required_frames = [0, 1, 2, 3, 4]

    if all(frame in frame_map for frame in required_frames):

        labels = group["label"].unique()

        # A sequence must have one consistent label.
        if len(labels) != 1:
            continue

        sequences.append({

            "category": category,

            "sequence_key": sequence_key,

            "video_id": group.iloc[0]["video_id"],

            "label": int(labels[0]),

            "frame_0": frame_map[0],

            "frame_1": frame_map[1],

            "frame_2": frame_map[2],

            "frame_3": frame_map[3],

            "frame_4": frame_map[4],
        })


sequence_df = pd.DataFrame(sequences)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

sequence_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("=" * 70)
print("FINAL TEMPORAL DATASET")
print("=" * 70)

print(
    "Valid 5-frame sequences:",
    len(sequence_df)
)


if len(sequence_df) > 0:

    print()
    print("Category distribution:")

    print(
        sequence_df["category"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Label distribution:")

    print(
        sequence_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Sequences per category:")

    print(
        sequence_df
        .groupby(["category", "label"])
        .size()
    )

    print()
    print("=" * 70)
    print("FIRST 10 SEQUENCES")
    print("=" * 70)

    for i, row in sequence_df.head(10).iterrows():

        print()
        print("Sequence:", i)
        print("Category:", row["category"])
        print("Sequence key:", row["sequence_key"])
        print("Video ID:", row["video_id"])
        print("Label:", row["label"])

        print("Frame 0:", row["frame_0"])
        print("Frame 1:", row["frame_1"])
        print("Frame 2:", row["frame_2"])
        print("Frame 3:", row["frame_3"])
        print("Frame 4:", row["frame_4"])


print()
print("=" * 70)
print("FILE CREATED")
print("=" * 70)

print(OUTPUT_CSV)

print()
print("Temporal reconstruction complete!")
print("=" * 70)