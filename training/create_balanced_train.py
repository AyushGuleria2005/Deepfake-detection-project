from pathlib import Path
import pandas as pd


BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"

INPUT_CSV = EXPERIMENTS_DIR / "train.csv"
OUTPUT_CSV = EXPERIMENTS_DIR / "train_balanced.csv"


def main():

    print("=" * 60)
    print("CREATING BALANCED TRAINING SET")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)

    # -----------------------------
    # REAL
    # -----------------------------
    real = df[df["label"] == 0].copy()

    # -----------------------------
    # FAKE
    # -----------------------------
    fake = df[df["label"] == 1].copy()

    print()
    print("Original training set:")
    print("Real:", len(real))
    print("Fake:", len(fake))

    # Extract manipulation type from filepath
    def get_category(path):
        path = str(path)

        for category in [
            "Deepfakes",
            "Face2Face",
            "FaceShifter",
            "FaceSwap",
            "NeuralTextures"
        ]:
            if category in path:
                return category

        return "Unknown"

    fake["category"] = fake["filepath"].apply(get_category)

    print()
    print("Fake categories before sampling:")

    print(
        fake["category"].value_counts()
    )

    # Exactly 800 from each manipulation type
    balanced_fake_parts = []

    categories = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]

    for category in categories:

        category_data = fake[
            fake["category"] == category
        ]

        if len(category_data) < 800:
            raise ValueError(
                f"{category} has only "
                f"{len(category_data)} samples."
            )

        sampled = category_data.sample(
            n=800,
            random_state=42
        )

        balanced_fake_parts.append(sampled)

    balanced_fake = pd.concat(
        balanced_fake_parts,
        ignore_index=True
    )

    # Exactly 4000 Real
    balanced_real = real.sample(
        n=4000,
        random_state=42
    )

    # Combine
    balanced = pd.concat(
        [
            balanced_real,
            balanced_fake
        ],
        ignore_index=True
    )

    # Remove helper column
    balanced = balanced.drop(
        columns=["category"]
    )

    # Shuffle entire dataset
    balanced = balanced.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save
    balanced.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 60)
    print("BALANCED DATASET CREATED")
    print("=" * 60)

    print("Output:")
    print(OUTPUT_CSV)

    print()
    print("Total:", len(balanced))
    print("Real:", (balanced["label"] == 0).sum())
    print("Fake:", (balanced["label"] == 1).sum())

    print()
    print("Fake manipulation distribution:")

    # Recalculate categories for verification
    verification = balanced[
        balanced["label"] == 1
    ].copy()

    verification["category"] = (
        verification["filepath"].apply(get_category)
    )

    print(
        verification["category"].value_counts()
        .sort_index()
    )

    print()
    print("Done!")


if __name__ == "__main__":
    main()