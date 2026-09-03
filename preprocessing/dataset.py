from pathlib import Path
import pandas as pd


DATASET_DIR = Path(
    r"D:\DeepFake-Learning\data\Raw\FF++C32-Frames"
)

OUTPUT_FILE = Path(
    r"D:\DeepFake-Learning\research\experiments\dataset_index.csv"
)


LABEL_MAP = {
    "Original": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceShifter": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
}


def build_dataset_index():
    records = []

    for category, label in LABEL_MAP.items():

        category_dir = DATASET_DIR / category

        if not category_dir.exists():
            print(f"Missing folder: {category}")
            continue

        image_files = sorted(category_dir.glob("*.jpg"))

        for image_path in image_files:

            # Example: 000_f0.jpg
            filename = image_path.name
            video_id = filename.split("_")[0]

            records.append({
                "filepath": str(image_path),
                "filename": filename,
                "video_id": video_id,
                "manipulation": category,
                "label": label
            })

        print(
            f"{category}: {len(image_files)} images indexed"
        )

    df = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset index created!")
    print(f"Total images: {len(df)}")
    print(f"Real images: {(df['label'] == 0).sum()}")
    print(f"Fake images: {(df['label'] == 1).sum()}")
    print(f"Unique videos: {df['video_id'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset_index()