from pathlib import Path
import sys

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import subprocess
import tempfile
import os
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.audio_cnn import AudioCNN

CSV_PATH = ROOT / "research" / "experiments" / "fakeavceleb_audio_rtvc_clean_test.csv"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "audio_cnn_best.pth"
OUTPUT = ROOT / "research" / "experiments" / "audio_rtvc_rms_normalized_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
TARGET_FRAMES = 512
TARGET_RMS = 0.05


def extract_audio(path):
    wav_path = tempfile.mktemp(suffix=".wav")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-c:a",
                "pcm_s16le",
                wav_path,
            ],
            check=True,
        )

        audio, sr = sf.read(wav_path)
        audio = audio.astype(np.float32)

        if sr != SR:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SR
            )

        return audio

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def make_normalized_logmel(audio):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms > 1e-8:
        audio = audio * (TARGET_RMS / rms)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    if logmel.shape[1] > TARGET_FRAMES:
        start = (logmel.shape[1] - TARGET_FRAMES) // 2
        logmel = logmel[:, start:start + TARGET_FRAMES]

    elif logmel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - logmel.shape[1]
        logmel = np.pad(
            logmel,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=-80.0,
        )

    return logmel.astype(np.float32)


print("=" * 70)
print("RMS-NORMALIZED AUDIO EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

df = pd.read_csv(CSV_PATH)

print("Samples:", len(df))
print("Real:", (df["eval_label"] == 0).sum())
print("RTVC:", (df["eval_label"] == 1).sum())

model = AudioCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint val AUC:", checkpoint.get("val_auc"))

predictions = []

with torch.no_grad():

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Normalized evaluation",
        unit="file"
    ):

        try:
            audio = extract_audio(row["actual_path"])

            logmel = make_normalized_logmel(audio)

            x = torch.from_numpy(
                logmel
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)

            logits = model(x)

            probability = torch.sigmoid(logits).item()

            predictions.append({
                "actual_path": row["actual_path"],
                "source": row["source"],
                "method": row["method"],
                "eval_label": int(row["eval_label"]),
                "predicted_probability": probability,
                "predicted_label": int(probability >= 0.5),
            })

        except Exception as e:
            print("\nERROR:", row["actual_path"])
            print(e)


result = pd.DataFrame(predictions)

result.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("NORMALIZED RESULT")
print("=" * 70)

print(
    result.groupby("method")["predicted_probability"]
    .agg(["count", "mean", "median", "min", "max"])
    .to_string()
)

print("\nSaved:")
print(OUTPUT)

print("=" * 70)
print("NORMALIZED EVALUATION COMPLETE")
print("=" * 70)