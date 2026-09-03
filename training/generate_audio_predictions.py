from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "audio_cnn_best.pth"

INPUT = EXP / "fakeavceleb_fusion_dataset.csv"
OUTPUT = EXP / "fakeavceleb_audio_predictions.csv"

sys.path.append(str(ROOT))

from models.audio_cnn import AudioCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("GENERATING AUDIO PREDICTIONS")
print("=" * 70)

print("Device:", DEVICE)

df = pd.read_csv(INPUT)

print("Total samples:", len(df))

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = AudioCNN()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

state_dict = checkpoint["model_state_dict"]

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint val AUC:", checkpoint.get("val_auc"))

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

audio_probs = []

with torch.no_grad():

    for feature_path in tqdm(
        df["feature_path"],
        desc="Audio inference"
    ):

        feature = np.load(feature_path).astype(
            np.float32
        )

        if feature.shape != (64, 512):
            raise RuntimeError(
                f"Unexpected feature shape: "
                f"{feature.shape}"
            )

        tensor = torch.from_numpy(
            feature
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(DEVICE)

        logits = model(tensor)

        prob = torch.sigmoid(
            logits
        ).item()

        audio_probs.append(prob)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df["audio_fake_probability"] = audio_probs

print()
print(
    "Successful audio predictions:",
    len(audio_probs),
    "/",
    len(df)
)

if df["audio_fake_probability"].isna().any():
    raise RuntimeError(
        "Missing audio predictions!"
    )

df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)

print("=" * 70)
print("AUDIO PREDICTIONS COMPLETE")
print("=" * 70)