from pathlib import Path
import sys
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "resnet18_face_crop_best.pth"

INPUT_CSV = EXP / "fakeavceleb_multimodal_index.csv"
OUTPUT_CSV = EXP / "fakeavceleb_visual_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("GENERATING VISUAL PREDICTIONS")
print("=" * 70)

print("Device:", DEVICE)

df = pd.read_csv(INPUT_CSV)

print("Total multimodal samples:", len(df))

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    1
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["model_state_dict"].items()
    }
)
model = model.to(DEVICE)
model.eval()

print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint val AUC:", checkpoint.get("val_auc"))

# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

visual_probs = []
valid_paths = []

with torch.no_grad():

    for path in tqdm(
        df["actual_path"],
        desc="Visual inference"
    ):

        try:
            cap = cv2.VideoCapture(str(path))

        except Exception:
            cap = None

        # This model needs a face image.
        # For FakeAVCeleb videos, extract the middle frame.
        if cap is None or not cap.isOpened():
            visual_probs.append(float("nan"))
            valid_paths.append(False)
            continue

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            visual_probs.append(float("nan"))
            valid_paths.append(False)
            continue

        middle = total_frames // 2

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            middle
        )

        success, frame = cap.read()
        cap.release()

        if not success:
            visual_probs.append(float("nan"))
            valid_paths.append(False)
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        image = transform(image).unsqueeze(0)
        image = image.to(DEVICE)

        logits = model(image)

        prob = torch.sigmoid(logits).item()

        visual_probs.append(prob)
        valid_paths.append(True)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df["visual_fake_probability"] = visual_probs
df["visual_valid"] = valid_paths

print()
print(
    "Successful visual predictions:",
    sum(valid_paths),
    "/",
    len(valid_paths)
)

missing = df["visual_fake_probability"].isna().sum()

print("Failed predictions:", missing)

if missing > 0:
    raise RuntimeError(
        "Some visual predictions failed!"
    )

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Saved:")
print(OUTPUT_CSV)

print("=" * 70)
print("VISUAL PREDICTIONS COMPLETE")
print("=" * 70)