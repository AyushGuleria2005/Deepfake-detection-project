import cv2
from pathlib import Path


def extract_faces(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # OpenCV's built-in Haar Cascade face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    image_files = sorted(input_dir.glob("*.jpg"))

    total_images = 0
    total_faces = 0

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        total_images += 1

        for face_number, (x, y, w, h) in enumerate(faces):
            # Add a small margin around the detected face
            margin = int(0.15 * max(w, h))

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))

            output_name = (
                f"{image_path.stem}_face_{face_number}.jpg"
            )

            output_path = output_dir / output_name

            cv2.imwrite(
                str(output_path),
                face,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            total_faces += 1

    print("\nFace extraction complete!")
    print(f"Images processed: {total_images}")
    print(f"Faces detected: {total_faces}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    input_dir = r"D:\DeepFake-Learning\data\processed\frames"
    output_dir = r"D:\DeepFake-Learning\data\processed\faces"

    extract_faces(input_dir, output_dir)