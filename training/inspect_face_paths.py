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
print("FACE-CROP PATH INSPECTION")
print("=" * 70)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Total rows:", len(df))

print()
print("=" * 70)
print("FIRST 20 FILEPATHS")
print("=" * 70)

for i, path in enumerate(df["filepath"].head(20)):
    print(f"{i}: {path}")

print()
print("=" * 70)
print("PATH ANALYSIS")
print("=" * 70)

existing = 0
missing = 0

for path in df["filepath"]:
    if Path(str(path)).exists():
        existing += 1
    else:
        missing += 1

print("Existing paths:", existing)
print("Missing paths:", missing)

print()
print("=" * 70)
print("FIRST 10 FILENAMES")
print("=" * 70)

for path in df["filepath"].head(10):

    print(
        "Full:",
        path
    )

    print(
        "Name:",
        Path(str(path)).name
    )

    print(
        "Stem:",
        Path(str(path)).stem
    )

    print(
        "Parts:",
        Path(str(path)).stem.split("_")
    )

    print("-" * 50)

print("=" * 70)