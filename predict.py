import argparse
import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from models.resnet_baseline import ResNetBaseline
from models.temporal_resnet_lstm import TemporalResNetLSTM

BASE_DIR = Path(r"D:\DeepFake-Learning")
IMAGE_CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "resnet18_face_crop_best.pth"
TEMPORAL_CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "temporal_lstm_best.pth"

# Standard evaluation transform
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return None
    return cv2.CascadeClassifier(cascade_path)


def crop_face(cv2_image, face_detector):
    """
    Detects and crops the primary face from a BGR image.
    If no face is detected, returns the center crop.
    """
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    ) if face_detector else []

    h, w, _ = cv2_image.shape

    if len(faces) > 0:
        # Take largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]

        # Add 15% margin around the face
        margin_x = int(fw * 0.15)
        margin_y = int(fh * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        face_crop = cv2_image[y1:y2, x1:x2]
        return face_crop, True
    else:
        # Fallback to center crop
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        return cv2_image[start_y:start_y + min_dim, start_x:start_x + min_dim], False


def predict_image(image_path, model, face_detector, device):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image at {image_path}")

    face, face_detected = crop_face(bgr, face_detector)
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_face)

    tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        fake_prob = torch.sigmoid(logit).item()

    prediction = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = fake_prob if prediction == "FAKE" else (1.0 - fake_prob)

    return {
        "type": "image",
        "file": image_path.name,
        "face_detected": face_detected,
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": 1.0 - fake_prob
    }


def predict_video(video_path, image_model, temporal_model, face_detector, device, max_frames=10):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_indices = np.linspace(0, max(0, total_frames - 1), num=min(max_frames, max(1, total_frames)), dtype=int)

    sampled_face_tensors = []
    frame_scores = []
    faces_detected_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        face, face_detected = crop_face(frame, face_detector)
        if face_detected:
            faces_detected_count += 1

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = eval_transform(Image.fromarray(rgb))
        sampled_face_tensors.append(tensor)

        with torch.no_grad():
            score = torch.sigmoid(image_model(tensor.unsqueeze(0).to(device)).squeeze()).item()
            frame_scores.append(score)

    cap.release()

    if len(sampled_face_tensors) == 0:
        raise ValueError("Could not extract any valid frames from video.")

    avg_visual_prob = float(np.mean(frame_scores))

    # Temporal Sequence Evaluation (requires 5 frames)
    temporal_prob = None
    if temporal_model is not None and len(sampled_face_tensors) >= 5:
        # Take first 5 sampled face frames
        seq_tensor = torch.stack(sampled_face_tensors[:5]).unsqueeze(0).to(device)
        with torch.no_grad():
            t_logit = temporal_model(seq_tensor).squeeze()
            temporal_prob = torch.sigmoid(t_logit).item()

    if temporal_prob is not None:
        overall_fake_prob = 0.5 * avg_visual_prob + 0.5 * temporal_prob
    else:
        overall_fake_prob = avg_visual_prob

    prediction = "FAKE" if overall_fake_prob >= 0.5 else "REAL"
    confidence = overall_fake_prob if prediction == "FAKE" else (1.0 - overall_fake_prob)

    return {
        "type": "video",
        "file": video_path.name,
        "total_frames": total_frames,
        "sampled_frames": len(frame_scores),
        "faces_detected": f"{faces_detected_count}/{len(frame_scores)}",
        "fps": round(fps, 2),
        "prediction": prediction,
        "confidence": confidence,
        "overall_fake_prob": overall_fake_prob,
        "visual_fake_prob": avg_visual_prob,
        "temporal_fake_prob": temporal_prob,
        "frame_scores": [round(s, 4) for s in frame_scores]
    }


def main():
    parser = argparse.ArgumentParser(description="Manual Deepfake Inference CLI (Image & Video)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda' or 'cpu'")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom image model checkpoint")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else IMAGE_CHECKPOINT

    print("=" * 60)
    print("MANUAL DEEPFAKE TEST")
    print("=" * 60)
    print(f"Target File: {input_path.name}")
    print(f"Device:      {device} ({torch.cuda.get_device_name(0) if device_str == 'cuda' else 'CPU'})")
    print(f"Checkpoint:  {ckpt_path.name}")

    # Load image model
    image_model = ResNetBaseline().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    image_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    image_model.eval()

    # Load temporal model if available
    temporal_model = None
    if TEMPORAL_CHECKPOINT.exists():
        try:
            temporal_model = TemporalResNetLSTM().to(device)
            t_ckpt = torch.load(TEMPORAL_CHECKPOINT, map_location=device)
            temporal_model.load_state_dict(t_ckpt.get("model_state_dict", t_ckpt))
            temporal_model.eval()
        except Exception:
            temporal_model = None

    face_detector = load_face_detector()

    # Check extension
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    ext = input_path.suffix.lower()

    if ext in image_exts:
        result = predict_image(input_path, image_model, face_detector, device)
        print("\n--- RESULT ---")
        print(f"Type:             Image")
        print(f"Face Detected:    {'Yes' if result['face_detected'] else 'No (Center crop used)'}")
        print(f"Prediction:       {result['prediction']}")
        print(f"Confidence:       {result['confidence'] * 100:.2f}%")
        print(f"Fake Probability: {result['fake_probability'] * 100:.2f}%")
        print(f"Real Probability: {result['real_probability'] * 100:.2f}%")

    elif ext in video_exts:
        result = predict_video(input_path, image_model, temporal_model, face_detector, device)
        print("\n--- RESULT ---")
        print(f"Type:             Video ({result['total_frames']} frames, {result['fps']} fps)")
        print(f"Faces Detected:   {result['faces_detected']} frames")
        print(f"Prediction:       {result['prediction']}")
        print(f"Confidence:       {result['confidence'] * 100:.2f}%")
        print(f"Visual Fake Prob: {result['visual_fake_prob'] * 100:.2f}%")
        if result['temporal_fake_prob'] is not None:
            print(f"Temporal Fake Prob: {result['temporal_fake_prob'] * 100:.2f}%")
        print(f"Per-Frame Scores: {result['frame_scores']}")
    else:
        print(f"Unsupported file format: {ext}. Supported formats: {list(image_exts | video_exts)}")


if __name__ == "__main__":
    main()
