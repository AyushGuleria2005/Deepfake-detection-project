from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

FEATURES = EXP / "fakeavceleb_audio_all_features.csv"
INDEX = EXP / "fakeavceleb_index_clean.csv"
OUTPUT = EXP / "fakeavceleb_multimodal_index.csv"

print("=" * 70)
print("MULTIMODAL AUDIO-VISUAL INDEX")
print("=" * 70)

features = pd.read_csv(FEATURES)
index = pd.read_csv(INDEX)

print("Audio feature rows:", len(features))
print("Clean index rows:", len(index))

# Keep only required metadata
metadata = index[
    [
        "actual_path",
        "source",
        "method",
        "category",
        "type",
        "race",
        "gender",
        "video_label",
        "audio_label",
    ]
].copy()

# Merge exact physical video path
df = features.merge(
    metadata,
    on="actual_path",
    how="inner"
)

print("Matched multimodal samples:", len(df))

if len(df) != len(features):
    raise RuntimeError(
        "Some audio features could not be matched with metadata!"
    )

if df["actual_path"].duplicated().any():
    raise RuntimeError("Duplicate actual_path detected!")

missing = df.isna().sum()

print("\nMissing values:")
print(missing[missing > 0])

if missing.any():
    raise RuntimeError("Missing metadata detected!")

print("\nAudio label distribution:")
print(df["audio_label"].value_counts())

print("\nVideo label distribution:")
print(df["video_label"].value_counts())

df["label_match"] = (
    df["audio_label"] == df["video_label"]
)

print("\nAudio/video label agreement:")
print(df["label_match"].value_counts())

print(
    "\nLabel agreement percentage:",
    df["label_match"].mean()
)

print("\nMethod distribution:")
print(df["method"].value_counts())

df.drop(columns=["label_match"]).to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("=" * 70)
print("MULTIMODAL INDEX COMPLETE")
print("=" * 70)