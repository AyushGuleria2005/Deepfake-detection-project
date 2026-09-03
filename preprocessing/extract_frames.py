import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, num_frames=20):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Error: No frames found in {video_path}")
        cap.release()
        return

    # Select evenly spaced frames
    frame_indices = [
        int(i * (total_frames - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    saved = 0

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if not success:
            continue

        frame_name = f"frame_{index:06d}.jpg"
        frame_path = output_dir / frame_name

        cv2.imwrite(
            str(frame_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        saved += 1

    cap.release()

    print(f"Video: {video_path.name}")
    print(f"Total frames: {total_frames}")
    print(f"Frames saved: {saved}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    video_path = r"D:\DeepFake-Learning\data\raw\face_test.mp4"
    output_dir = r"D:\DeepFake-Learning\data\processed\frames"

    extract_frames(
        video_path,
        output_dir,
        num_frames=20
    )