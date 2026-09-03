from pathlib import Path
import pandas as pd


BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"


def get_category(path):
    path = str(path)

    categories = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
        "Original"
    ]

    for category in categories:
        if category in path:
            return category

    return "Unknown"


def check_split(csv_name):

    csv_path = EXPERIMENTS_DIR / csv_name

    df = pd.read_csv(csv_path)

    df["category"] = df["original_filepath"].apply(
        get_category
    )

    print()
    print("=" * 60)
    print(csv_name)
    print("=" * 60)

    print("Total:", len(df))

    print()
    print("Labels:")
    print("Real:", (df["label"] == 0).sum())
    print("Fake:", (df["label"] == 1).sum())

    print()
    print("Categories:")
    print(
        df["category"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Unknown:", (df["category"] == "Unknown").sum())


def main():

    print("=" * 60)
    print("FACE DATASET VERIFICATION")
    print("=" * 60)

    check_split(
        "train_balanced_face.csv"
    )

    check_split(
        "validation_face.csv"
    )

    check_split(
        "test_face.csv"
    )

    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()