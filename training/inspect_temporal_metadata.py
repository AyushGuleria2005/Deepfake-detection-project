import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\DeepFake-Learning")

CSV_FILE = (
    BASE_DIR
    / "research"
    / "experiments"
    / "train_balanced_face.csv"
)

df = pd.read_csv(CSV_FILE)

print("=" * 70)
print("TEMPORAL METADATA INSPECTION")
print("=" * 70)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("=" * 70)
print("FIRST 30 ORIGINAL PATHS")
print("=" * 70)

for i, row in df.head(30).iterrows():

    print()
    print("Index:", i)
    print("Video ID:", row["video_id"])
    print("Label:", row["label"])
    print("Face crop:", row["filepath"])
    print("Original:", row["original_filepath"])

    print(
        "Original filename:",
        Path(str(row["original_filepath"])).name
    )

    print(
        "Original stem:",
        Path(str(row["original_filepath"])).stem
    )

print()
print("=" * 70)
print("CATEGORY + VIDEO ID SAMPLE")
print("=" * 70)

def get_category(path):

    path = str(path).replace("\\", "/")

    for category in [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
        "Original"
    ]:
        if f"/{category}/" in path:
            return category

    return "Unknown"


df["category"] = df["filepath"].apply(
    get_category
)

print(
    df[
        ["video_id", "category", "label", "original_filepath"]
    ].head(30).to_string(index=False)
)

print()
print("=" * 70)