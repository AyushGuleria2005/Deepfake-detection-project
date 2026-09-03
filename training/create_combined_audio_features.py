from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EXP = ROOT / "research" / "experiments"

FILES = [
    EXP / "fakeavceleb_audio_train.csv",
    EXP / "fakeavceleb_audio_validation.csv",
    EXP / "fakeavceleb_audio_test.csv",
]

OUTPUT = EXP / "fakeavceleb_audio_all_features.csv"

print("=" * 70)
print("CREATE COMBINED AUDIO FEATURE MAPPING")
print("=" * 70)

dfs = []

for path in FILES:
    print("\nLoading:")
    print(path)

    df = pd.read_csv(path)

    print("Rows:", len(df))

    required = ["actual_path", "feature_path"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing required column '{col}' in {path.name}"
            )

    dfs.append(df[required].copy())


combined = pd.concat(dfs, ignore_index=True)

print("\nTotal rows before duplicate removal:", len(combined))

duplicate_paths = combined["actual_path"].duplicated().sum()

print("Duplicate actual paths:", duplicate_paths)

combined = combined.drop_duplicates(
    subset=["actual_path"],
    keep="first"
).reset_index(drop=True)

print("Rows after duplicate removal:", len(combined))

missing_features = combined["feature_path"].isna().sum()

print("Missing feature paths:", missing_features)

if missing_features > 0:
    raise RuntimeError("Some samples have missing feature paths.")


combined.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("\nFinal unique feature mappings:", len(combined))

print("=" * 70)
print("COMBINED FEATURE MAPPING CREATED")
print("=" * 70)