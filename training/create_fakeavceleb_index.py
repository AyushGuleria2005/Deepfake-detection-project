import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\DeepFake-Learning\data\Raw\FakeAVCeleb\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
META = ROOT / "meta_data.csv"
OUTPUT = Path(r"D:\DeepFake-Learning\research\experiments")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META)

# Remove accidental CSV index column if present
df = df.drop(columns=["Unnamed: 9"], errors="ignore")

# Scan every actual MP4
print("Scanning MP4 files...")
files = list(ROOT.rglob("*.mp4"))
print("Actual MP4 files:", len(files))

# Build lookup by:
# (type, race, gender, source, filename)
file_lookup = {}

for f in files:
    rel = f.relative_to(ROOT)
    parts = rel.parts

    # Expected:
    # type / race / gender / source / filename
    if len(parts) >= 5:
        key = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[-1],
        )
        file_lookup.setdefault(key, []).append(str(f))

print("Building metadata index...")

rows = []
unresolved = []

for _, r in df.iterrows():
    data_type = str(r["type"])
    race = str(r["race"])
    gender = str(r["gender"])
    source = str(r["source"])
    metadata_path = str(r["path"])

    # First try exact metadata filename
    key = (data_type, race, gender, source, metadata_path)
    matches = file_lookup.get(key, [])

    # For generated FakeVideo-FakeAudio / FakeVideo-RealAudio
    # the downloaded filename may contain target IDs or suffixes.
    if not matches:
        source_dir = ROOT / data_type / race / gender / source

        if source_dir.exists():
            candidates = list(source_dir.glob("*.mp4"))

            stem = Path(metadata_path).stem

            # Candidate filenames beginning with the metadata stem
            prefix_matches = [
                str(f) for f in candidates
                if f.stem.startswith(stem + "_")
                or f.stem == stem
            ]

            if len(prefix_matches) == 1:
                matches = prefix_matches
            elif len(prefix_matches) > 1:
                # Keep unresolved for now rather than guessing
                matches = []

    if len(matches) == 1:
        actual_path = matches[0]
    else:
        actual_path = ""

        unresolved.append({
            "source": source,
            "type": data_type,
            "race": race,
            "gender": gender,
            "metadata_path": metadata_path,
            "candidate_count": len(matches),
        })

    # Binary labels
    video_label = 0 if data_type.startswith("RealVideo") else 1
    audio_label = 0 if data_type.endswith("RealAudio") else 1

    rows.append({
        "actual_path": actual_path,
        "source": source,
        "target1": r["target1"],
        "target2": r["target2"],
        "method": r["method"],
        "category": r["category"],
        "type": data_type,
        "race": race,
        "gender": gender,
        "metadata_path": metadata_path,
        "video_label": video_label,
        "audio_label": audio_label,
    })

index_df = pd.DataFrame(rows)
unresolved_df = pd.DataFrame(unresolved)

index_path = OUTPUT / "fakeavceleb_index.csv"
unresolved_path = OUTPUT / "fakeavceleb_unresolved.csv"

index_df.to_csv(index_path, index=False)
unresolved_df.to_csv(unresolved_path, index=False)

print("\n========== RESULT ==========")
print("Metadata rows:", len(df))
print("Indexed rows:", len(index_df))
print("Resolved:", (index_df["actual_path"] != "").sum())
print("Unresolved:", (index_df["actual_path"] == "").sum())

print("\nVideo labels:")
print(index_df["video_label"].value_counts().sort_index())

print("\nAudio labels:")
print(index_df["audio_label"].value_counts().sort_index())

print("\nSaved:")
print(index_path)
print(unresolved_path)