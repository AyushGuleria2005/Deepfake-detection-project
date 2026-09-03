from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


INDEX_FILE = Path(
    r"D:\DeepFake-Learning\research\experiments\dataset_index.csv"
)

OUTPUT_DIR = Path(
    r"D:\DeepFake-Learning\research\experiments"
)


def create_splits():
    df = pd.read_csv(INDEX_FILE)

    # First: 80% train, 20% temporary
    # Groups = video_id, so frames from the same video
    # can never appear in different splits.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, temp_idx = next(
        splitter.split(
            df,
            groups=df["video_id"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Second: split temporary 20% into
    # 10% validation and 10% test.
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_idx, test_idx = next(
        splitter2.split(
            temp_df,
            groups=temp_df["video_id"]
        )
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Save
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("Dataset split complete!")
    print()
    print(f"Train images:      {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images:       {len(test_df)}")
    print()
    print(f"Train videos:      {train_df['video_id'].nunique()}")
    print(f"Validation videos: {val_df['video_id'].nunique()}")
    print(f"Test videos:       {test_df['video_id'].nunique()}")
    print()
    print("Files saved:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "validation.csv")
    print(OUTPUT_DIR / "test.csv")


if __name__ == "__main__":
    create_splits()