from pathlib import Path
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(r"D:\DeepFake-Learning")

EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "face_crops"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# FACE DETECTOR
# ---------------------------------------------------------

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


# ---------------------------------------------------------
# PROCESS ONE IMAGE
# ---------------------------------------------------------

def extract_largest_face(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # Select largest detected face
    largest_face = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    x, y, w, h = largest_face

    # Add padding around face
    padding = int(
        0.25 * max(w, h)
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image.shape[1],
        x + w + padding
    )

    y2 = min(
        image.shape[0],
        y + h + padding
    )

    face = image[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    # Resize to model input size
    face = cv2.resize(
        face,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    return face


# ---------------------------------------------------------
# PROCESS SPLIT
# ---------------------------------------------------------

def process_split(
    csv_name,
    output_csv_name
):

    csv_path = (
        EXPERIMENTS_DIR /
        csv_name
    )

    output_csv = (
        EXPERIMENTS_DIR /
        output_csv_name
    )

    df = pd.read_csv(
        csv_path
    )

    records = []

    successful = 0
    failed = 0

    print()
    print("=" * 60)
    print(f"PROCESSING {csv_name}")
    print("=" * 60)

    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=csv_name
    ):

        image_path = Path(
            row["filepath"]
        )

        face = extract_largest_face(
            image_path
        )

        if face is None:

            failed += 1

            continue

        # Keep original folder/category
        category = image_path.parent.name

        output_category_dir = (
            OUTPUT_DIR /
            csv_name.replace(
                ".csv",
                ""
            ) /
            category
        )

        output_category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_filename = (
            f"{index:06d}.jpg"
        )

        output_path = (
            output_category_dir /
            output_filename
        )

        cv2.imwrite(
            str(output_path),
            face
        )

        records.append(
            {
                "filepath": str(output_path),
                "label": int(row["label"]),
                "video_id": row["video_id"],
                "original_filepath": str(image_path)
            }
        )

        successful += 1

    output_df = pd.DataFrame(
        records
    )

    output_df.to_csv(
        output_csv,
        index=False
    )

    print()
    print("Successful:", successful)
    print("Failed:", failed)
    print("Output CSV:", output_csv)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("FACE CROP DATASET CREATION")
    print("=" * 60)

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    # Process only the 8k balanced training set
    process_split(
        "train_balanced.csv",
        "train_balanced_face.csv"
    )

    # Validation remains the original source-level split
    process_split(
        "validation.csv",
        "validation_face.csv"
    )

    # Test remains the original source-level split
    process_split(
        "test.csv",
        "test_face.csv"
    )

    print()
    print("=" * 60)
    print("FACE CROP DATASET COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()