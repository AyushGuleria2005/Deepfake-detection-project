from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "research" / "experiments" / "fakeavceleb_fusion_dataset.csv"

N_SPLITS = 5


df = pd.read_csv(INPUT)

gkf = GroupKFold(n_splits=N_SPLITS)

print("=" * 70)
print("SOURCE-LEVEL OOF FOLD AUDIT")
print("=" * 70)

print("Total samples:", len(df))
print("Unique sources:", df["source"].nunique())

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(
        df,
        df["audio_label"],
        groups=df["source"]
    ),
    start=1
):

    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    train_sources = set(train["source"])
    val_sources = set(val["source"])

    overlap = train_sources & val_sources

    print()
    print(f"FOLD {fold}")
    print("-" * 70)

    print("Train samples:", len(train))
    print("Held-out samples:", len(val))

    print("Train sources:", len(train_sources))
    print("Held-out sources:", len(val_sources))

    print("Source overlap:", len(overlap))

    print()
    print("Train audio labels:")
    print(train["audio_label"].value_counts().sort_index().to_dict())

    print("Held-out audio labels:")
    print(val["audio_label"].value_counts().sort_index().to_dict())

    if overlap:
        raise RuntimeError(
            f"LEAKAGE DETECTED in fold {fold}"
        )

    if train["audio_label"].nunique() != 2:
        raise RuntimeError(
            f"Training fold {fold} contains only one audio class"
        )

    if val["audio_label"].nunique() != 2:
        raise RuntimeError(
            f"Validation fold {fold} contains only one audio class"
        )

print()
print("=" * 70)
print("AUDIT PASSED")
print("=" * 70)