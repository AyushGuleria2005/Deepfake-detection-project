import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\DeepFake-Learning")

INPUT_CSV = (
    ROOT
    / "research"
    / "experiments"
    / "fakeavceleb_index_clean.csv"
)

OUTPUT_DIR = (
    ROOT
    / "research"
    / "experiments"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "fakeavceleb_audio_unseen_rtvc_train.csv"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "fakeavceleb_audio_unseen_rtvc_test.csv"
)


# ============================================================
# CONFIG
# ============================================================

UNSEEN_METHOD = "rtvc"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UNSEEN AUDIO METHOD SPLIT")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)

    print(
        "Total rows:",
        len(df)
    )

    print()
    print("Audio generation methods:")
    print()

    print(
        df["method"]
        .value_counts()
        .to_string()
    )

    print()

    # --------------------------------------------------------
    # Check RTVC exists
    # --------------------------------------------------------

    if UNSEEN_METHOD not in df["method"].unique():

        raise ValueError(
            f"Method '{UNSEEN_METHOD}' not found."
        )

    # --------------------------------------------------------
    # TRAIN = everything except RTVC
    # TEST = RTVC only
    # --------------------------------------------------------

    train_df = df[
        df["method"] != UNSEEN_METHOD
    ].copy()

    test_df = df[
        df["method"] == UNSEEN_METHOD
    ].copy()

    # --------------------------------------------------------
    # Remove duplicate physical files
    # --------------------------------------------------------

    train_df = train_df.drop_duplicates(
        subset=["actual_path"]
    )

    test_df = test_df.drop_duplicates(
        subset=["actual_path"]
    )

    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    train_paths = set(
        train_df["actual_path"]
    )

    test_paths = set(
        test_df["actual_path"]
    )

    overlap = train_paths & test_paths

    print(
        "Train samples:",
        len(train_df)
    )

    print(
        "Unseen RTVC test samples:",
        len(test_df)
    )

    print(
        "Physical-file overlap:",
        len(overlap)
    )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    print()
    print("TRAIN VIDEO LABELS:")
    print(
        train_df["video_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("TRAIN AUDIO LABELS:")
    print(
        train_df["audio_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("TEST VIDEO LABELS:")
    print(
        test_df["video_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("TEST AUDIO LABELS:")
    print(
        test_df["audio_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Verify RTVC is ONLY in test
    # --------------------------------------------------------

    print()
    print(
        "Train methods:"
    )

    print(
        train_df["method"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Test methods:"
    )

    print(
        test_df["method"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test_df.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print()
    print("=" * 70)
    print("SPLIT CREATED")
    print("=" * 70)

    print(
        "Train:",
        TRAIN_OUTPUT
    )

    print(
        "Test:",
        TEST_OUTPUT
    )

    print("=" * 70)


if __name__ == "__main__":
    main()