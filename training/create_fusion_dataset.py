from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

VISUAL_FILE = EXP / "fakeavceleb_visual_predictions.csv"
AUDIO_FILE = EXP / "fakeavceleb_audio_predictions.csv"
OUTPUT_FILE = EXP / "fakeavceleb_fusion_dataset.csv"

print("=" * 70)
print("CREATING FINAL MULTIMODAL FUSION DATASET")
print("=" * 70)

visual = pd.read_csv(VISUAL_FILE)
audio = pd.read_csv(AUDIO_FILE)

print("Visual samples:", len(visual))
print("Audio samples:", len(audio))

if "actual_path" not in visual.columns:
    raise RuntimeError("actual_path missing from visual predictions")

if "actual_path" not in audio.columns:
    raise RuntimeError("actual_path missing from audio predictions")

if "visual_fake_probability" not in visual.columns:
    raise RuntimeError("visual_fake_probability missing")

if "audio_fake_probability" not in audio.columns:
    raise RuntimeError("audio_fake_probability missing")

# Keep only the audio probability needed for fusion.
audio_small = audio[
    ["actual_path", "audio_fake_probability"]
].copy()

# Check uniqueness before merging.
if audio_small["actual_path"].duplicated().any():
    raise RuntimeError(
        "Duplicate actual_path found in audio predictions"
    )

if visual["actual_path"].duplicated().any():
    raise RuntimeError(
        "Duplicate actual_path found in visual predictions"
    )

df = visual.merge(
    audio_small,
    on="actual_path",
    how="inner",
    validate="one_to_one"
)

print("Merged samples:", len(df))

if len(df) != len(visual):
    raise RuntimeError(
        "Merge lost samples!"
    )

if df["audio_fake_probability"].isna().any():
    raise RuntimeError(
        "Missing audio probabilities!"
    )

# Modality combination
df["modality_combination"] = (
    "video_" +
    df["video_label"].map({
        0: "real",
        1: "fake"
    }) +
    "_audio_" +
    df["audio_label"].map({
        0: "real",
        1: "fake"
    })
)

print()
print("Modality combinations:")
print(
    df["modality_combination"].value_counts()
)

print()
print("Video labels:")
print(
    df["video_label"].value_counts()
)

print()
print("Audio labels:")
print(
    df["audio_label"].value_counts()
)

print()
print("Probability sanity check:")

print(
    "Visual:",
    df["visual_fake_probability"].min(),
    "to",
    df["visual_fake_probability"].max()
)

print(
    "Audio:",
    df["audio_fake_probability"].min(),
    "to",
    df["audio_fake_probability"].max()
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Final columns:")
print(df.columns.tolist())

print("=" * 70)
print("FINAL FUSION DATASET COMPLETE")
print("=" * 70)