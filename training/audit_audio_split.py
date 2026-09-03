import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "research",
    "experiments",
    "audio_generator_holdout_train.csv"
)


def main():

    print("=" * 70)
    print("AUDIO VALIDATION SPLIT AUDIT")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    # Same filtering as training/debug
    df = df[
        df["method"].str.lower() != "rtvc"
    ].copy()

    sources = sorted(
        df["source"].astype(str).unique()
    )

    split_index = int(len(sources) * 0.80)

    train_sources = set(
        sources[:split_index]
    )

    val_sources = set(
        sources[split_index:]
    )

    train_df = df[
        df["source"].astype(str).isin(train_sources)
    ].copy()

    val_df = df[
        df["source"].astype(str).isin(val_sources)
    ].copy()

    print(f"\nTotal samples: {len(df)}")
    print(f"Total sources: {len(sources)}")

    print("\nTRAIN")
    print("-" * 40)
    print(f"Samples: {len(train_df)}")
    print(f"Sources: {train_df['source'].nunique()}")
    print(
        "Labels:",
        train_df["audio_label"].value_counts().to_dict()
    )

    print("\nVALIDATION")
    print("-" * 40)
    print(f"Samples: {len(val_df)}")
    print(f"Sources: {val_df['source'].nunique()}")
    print(
        "Labels:",
        val_df["audio_label"].value_counts().to_dict()
    )

    # --------------------------------------------------
    # Source overlap
    # --------------------------------------------------
    overlap = (
        set(train_df["source"].astype(str))
        &
        set(val_df["source"].astype(str))
    )

    print("\nSOURCE OVERLAP")
    print("-" * 40)
    print(f"Overlap: {len(overlap)}")

    if overlap:
        print("WARNING: source leakage detected!")

    # --------------------------------------------------
    # Physical file overlap
    # --------------------------------------------------
    train_paths = set(
        train_df["actual_path"].astype(str)
    )

    val_paths = set(
        val_df["actual_path"].astype(str)
    )

    path_overlap = train_paths & val_paths

    print("\nPHYSICAL FILE OVERLAP")
    print("-" * 40)
    print(f"Overlap: {len(path_overlap)}")

    # --------------------------------------------------
    # Feature overlap
    # --------------------------------------------------
    train_features = set(
        train_df["feature_path"].astype(str)
    )

    val_features = set(
        val_df["feature_path"].astype(str)
    )

    feature_overlap = (
        train_features & val_features
    )

    print("\nFEATURE FILE OVERLAP")
    print("-" * 40)
    print(f"Overlap: {len(feature_overlap)}")

    # --------------------------------------------------
    # Method distribution
    # --------------------------------------------------
    print("\nTRAIN METHODS")
    print("-" * 40)
    print(
        train_df["method"]
        .value_counts()
        .sort_index()
    )

    print("\nVALIDATION METHODS")
    print("-" * 40)
    print(
        val_df["method"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Save exact split
    # --------------------------------------------------
    out_path = os.path.join(
        PROJECT_ROOT,
        "research",
        "experiments",
        "debug_recreated_validation.csv"
    )

    val_df.to_csv(
        out_path,
        index=False
    )

    print("\nSaved:")
    print(out_path)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()