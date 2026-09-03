import os
import pandas as pd

BASE = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EXP = os.path.join(
    BASE,
    "research",
    "experiments"
)

INDEX_CSV = os.path.join(
    EXP,
    "fakeavceleb_index_clean.csv"
)

RTVC_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_unseen_rtvc_test.csv"
)

OUTPUT_CSV = os.path.join(
    EXP,
    "fakeavceleb_audio_rtvc_matched_test.csv"
)

print("=" * 70)
print("CREATE MATCHED RTVC TEST SET")
print("=" * 70)

index = pd.read_csv(INDEX_CSV)
rtvc = pd.read_csv(RTVC_CSV)

print(f"\nTotal clean index rows: {len(index)}")
print(f"RTVC fake samples: {len(rtvc)}")

# ------------------------------------------------------------
# RTVC source IDs
# ------------------------------------------------------------

rtvc_sources = set(
    rtvc["source"].astype(str)
)

print(
    f"Unique RTVC source IDs: "
    f"{len(rtvc_sources)}"
)

# ------------------------------------------------------------
# Find real-audio samples from same source IDs
# ------------------------------------------------------------

real_candidates = index[
    (index["method"].astype(str) == "real") &
    (index["source"].astype(str).isin(rtvc_sources))
].copy()

print(
    f"Matched real-audio candidates: "
    f"{len(real_candidates)}"
)

# One real sample per source
real_candidates = real_candidates.drop_duplicates(
    subset=["source"],
    keep="first"
)

print(
    f"Unique matched real sources: "
    f"{len(real_candidates)}"
)

# ------------------------------------------------------------
# Select exactly one RTVC sample per source
# ------------------------------------------------------------

rtvc_one = rtvc.drop_duplicates(
    subset=["source"],
    keep="first"
).copy()

print(
    f"Unique RTVC sources selected: "
    f"{len(rtvc_one)}"
)

# ------------------------------------------------------------
# Keep only sources having both RTVC and real audio
# ------------------------------------------------------------

common_sources = set(
    rtvc_one["source"].astype(str)
) & set(
    real_candidates["source"].astype(str)
)

print(
    f"Common sources: "
    f"{len(common_sources)}"
)

rtvc_one = rtvc_one[
    rtvc_one["source"].astype(str).isin(common_sources)
].copy()

real_candidates = real_candidates[
    real_candidates["source"].astype(str).isin(common_sources)
].copy()

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

rtvc_one["eval_label"] = 1
real_candidates["eval_label"] = 0

# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

matched = pd.concat(
    [
        rtvc_one,
        real_candidates
    ],
    ignore_index=True
)

# Shuffle deterministically
matched = matched.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAFETY CHECKS")
print("=" * 70)

print(
    "\nEvaluation labels:"
)

print(
    matched["eval_label"].value_counts()
)

# Source overlap inside pairs is intentional,
# but physical files must be different.

duplicates = (
    matched["actual_path"]
    .astype(str)
    .duplicated()
    .sum()
)

print(
    f"\nDuplicate physical files: "
    f"{duplicates}"
)

# Check methods
print("\nMethods:")
print(
    matched["method"].value_counts()
)

# Check source counts
print(
    f"\nTotal matched samples: "
    f"{len(matched)}"
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

matched.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 70)
print("MATCHED TEST SET CREATED")
print("=" * 70)

print(
    OUTPUT_CSV
)

print("=" * 70)