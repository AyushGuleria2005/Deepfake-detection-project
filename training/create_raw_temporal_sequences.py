import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(r"D:\DeepFake-Learning")

RAW_DIR = BASE_DIR / "data" / "Raw" / "FF++C32-Frames"

OUTPUT_CSV = (
    BASE_DIR
    / "research"
    / "experiments"
    / "train_raw_temporal_sequences.csv"
)

CATEGORIES = [
    "Original",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]


def parse_filename(filename, category):

    filename = filename.lower()

    if category == "Original":
        match = re.match(r"^(\d+)_f(\d+)\.jpg$", filename)

        if not match:
            return None

        video_id = int(match.group(1))
        frame = int(match.group(2))

        return str(video_id), frame

    else:
        match = re.match(
            r"^(\d+)_(\d+)_f(\d+)\.jpg$",
            filename
        )

        if not match:
            return None

        source_id = int(match.group(1))
        target_id = int(match.group(2))
        frame = int(match.group(3))

        sequence_key = f"{source_id}_{target_id}"

        return sequence_key, frame


print("=" * 70)
print("RAW FF++ TEMPORAL SEQUENCE CREATION")
print("=" * 70)

all_sequences = []

for category in CATEGORIES:

    category_dir = RAW_DIR / category

    files = sorted(category_dir.glob("*.jpg"))

    print()
    print("-" * 70)
    print("Category:", category)
    print("Raw frames:", len(files))

    groups = {}

    for file_path in files:

        parsed = parse_filename(
            file_path.name,
            category
        )

        if parsed is None:
            continue

        sequence_key, frame = parsed

        if sequence_key not in groups:
            groups[sequence_key] = {}

        groups[sequence_key][frame] = str(file_path)


    valid_count = 0

    for sequence_key, frame_map in groups.items():

        required = [0, 1, 2, 3, 4]

        if all(frame in frame_map for frame in required):

            if category == "Original":
                video_id = int(sequence_key)
            else:
                video_id = int(sequence_key.split("_")[0])

            label = 0 if category == "Original" else 1

            all_sequences.append({
                "video_id": video_id,
                "sequence_key": sequence_key,
                "category": category,
                "label": label,
                "frame_0": frame_map[0],
                "frame_1": frame_map[1],
                "frame_2": frame_map[2],
                "frame_3": frame_map[3],
                "frame_4": frame_map[4],
            })

            valid_count += 1

    print("Unique sequence groups:", len(groups))
    print("Valid 5-frame sequences:", valid_count)


sequence_df = pd.DataFrame(all_sequences)

sequence_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("Total temporal sequences:", len(sequence_df))

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
print("Sequences by category:")
print(
    sequence_df
    .groupby(["category", "label"])
    .size()
)

print()
print("=" * 70)
print("FIRST 6 SEQUENCES")
print("=" * 70)

for i, row in sequence_df.head(6).iterrows():

    print()
    print("Sequence:", i)
    print("Category:", row["category"])
    print("Sequence key:", row["sequence_key"])
    print("Label:", row["label"])

    print("Frame 0:", Path(row["frame_0"]).name)
    print("Frame 1:", Path(row["frame_1"]).name)
    print("Frame 2:", Path(row["frame_2"]).name)
    print("Frame 3:", Path(row["frame_3"]).name)
    print("Frame 4:", Path(row["frame_4"]).name)


print()
print("=" * 70)
print("FILE CREATED")
print("=" * 70)

print(OUTPUT_CSV)