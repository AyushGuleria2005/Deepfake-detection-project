import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.audio_cnn import AudioCNN


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\DeepFake-Learning"

TEST_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_generator_holdout_test.csv"
)

CHECKPOINT = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "audio_cnn_generator_holdout_best.pth"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "audio_generator_holdout_raw_rms_predictions.csv"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RAW WAVEFORM RMS NORMALIZED AUDIO TEST")
print("=" * 70)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

model = AudioCNN().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("Audio model loaded.")


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(
    TEST_CSV
)

print()
print("Test samples:", len(df))
print(
    "Sources:",
    df["source"].nunique()
)

print(
    "Labels:",
    df["audio_label"].value_counts().to_dict()
)


# ============================================================
# AUDIO FEATURE FUNCTION
# ============================================================

def prepare_rms_normalized_feature(
    audio_path,
    target_rms=0.05
):

    # --------------------------------------------------------
    # Load raw waveform
    # --------------------------------------------------------

    waveform, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    waveform = waveform.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    waveform = waveform - np.mean(
        waveform
    )

    # --------------------------------------------------------
    # Calculate RMS
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            waveform ** 2
        )
        + 1e-8
    )

    # --------------------------------------------------------
    # Normalize waveform energy
    # --------------------------------------------------------

    if rms > 0:

        waveform = (
            waveform
            * (target_rms / rms)
        )

    # --------------------------------------------------------
    # Log-Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
        fmin=20,
        fmax=8000
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Force exactly 512 time frames
    # --------------------------------------------------------

    if mel.shape[1] < 512:

        mel = np.pad(
            mel,
            (
                (0, 0),
                (0, 512 - mel.shape[1])
            ),
            mode="constant",
            constant_values=mel.min()
        )

    else:

        mel = mel[
            :,
            :512
        ]

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    feature = torch.tensor(
        mel,
        dtype=torch.float32
    )

    feature = feature.unsqueeze(0)
    feature = feature.unsqueeze(0)

    return feature


# ============================================================
# EVALUATION
# ============================================================

predictions = []
labels = []

failed = 0

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="RMS-normalized audio"
):

    audio_path = row["feature_path"]

    # --------------------------------------------------------
    # IMPORTANT:
    # feature_path points to saved log-mel features,
    # so we need the original video/audio path.
    # --------------------------------------------------------

    actual_path = row["actual_path"]

    if not os.path.exists(
        actual_path
    ):

        failed += 1
        continue

    # --------------------------------------------------------
    # Extract audio using FFmpeg
    # --------------------------------------------------------

    import subprocess
    import tempfile

    wav_path = tempfile.mktemp(
        suffix=".wav"
    )

    try:

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                actual_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        feature = (
            prepare_rms_normalized_feature(
                wav_path
            )
        )

        feature = feature.to(
            device
        )

        with torch.no_grad():

            logit = model(
                feature
            )

            probability = torch.sigmoid(
                logit
            ).item()

        predictions.append(
            probability
        )

        labels.append(
            int(row["audio_label"])
        )

    except Exception as e:

        failed += 1

    finally:

        if os.path.exists(
            wav_path
        ):

            os.remove(
                wav_path
            )


# ============================================================
# METRICS
# ============================================================

predictions = np.array(
    predictions
)

labels = np.array(
    labels
)

binary_predictions = (
    predictions >= 0.5
).astype(int)

auc = roc_auc_score(
    labels,
    predictions
)

accuracy = accuracy_score(
    labels,
    binary_predictions
)

balanced_accuracy = (
    balanced_accuracy_score(
        labels,
        binary_predictions
    )
)

precision = precision_score(
    labels,
    binary_predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    binary_predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    binary_predictions,
    zero_division=0
)

cm = confusion_matrix(
    labels,
    binary_predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Evaluated samples: {len(labels)}"
)

print(
    f"Failed samples:    {failed}"
)

print(
    f"ROC-AUC:            {auc:.4f}"
)

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Balanced Accuracy:  {balanced_accuracy:.4f}"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall:             {recall:.4f}"
)

print(
    f"F1:                 {f1:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)


# ============================================================
# GROUP STATISTICS
# ============================================================

result_df = df.iloc[
    :len(predictions)
].copy()

result_df[
    "rms_normalized_fake_probability"
] = predictions

print()
print("=" * 70)
print("PROBABILITY BY METHOD")
print("=" * 70)

print(
    result_df.groupby(
        "method"
    )[
        "rms_normalized_fake_probability"
    ].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Saved:")
print(
    OUTPUT_CSV
)

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)