from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "research" / "experiments" /
    "fakeavceleb_audio_rtvc_matched_test.csv"
)

AUDIT_CSV = (
    ROOT / "research" / "experiments" /
    "rtvc_audio_identity_audit.csv"
)

OUTPUT_CSV = (
    ROOT / "research" / "experiments" /
    "fakeavceleb_audio_rtvc_clean_test.csv"
)

print("=" * 70)
print("CREATE CLEAN UNSEEN RTVC TEST")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)
audit = pd.read_csv(AUDIT_CSV)

# Remove every pair whose raw audio was found to be identical.
bad_sources = audit.loc[
    audit["identical"] == True,
    "source"
].tolist()

print("\nInvalid/identical audio sources:", bad_sources)

clean = df[~df["source"].isin(bad_sources)].copy()

print("\nOriginal samples:", len(df))
print("Removed samples :", len(df) - len(clean))
print("Clean samples   :", len(clean))

print("\nClean labels:")
print(clean["eval_label"].value_counts().sort_index())

print("\nClean methods:")
print(clean["method"].value_counts())

print("\nUnique sources:", clean["source"].nunique())

print("\nDuplicate physical files:",
      clean["actual_path"].duplicated().sum())

# Every remaining source should have exactly one real + one RTVC.
pair_check = (
    clean.groupby(["source", "eval_label"])
    .size()
    .unstack(fill_value=0)
)

valid_pairs = (
    (pair_check.get(0, 0) == 1) &
    (pair_check.get(1, 0) == 1)
)

print("Complete real/RTVC pairs:", int(valid_pairs.sum()))

clean.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:")
print(OUTPUT_CSV)

print("=" * 70)
print("CLEAN TEST CREATED")
print("=" * 70)