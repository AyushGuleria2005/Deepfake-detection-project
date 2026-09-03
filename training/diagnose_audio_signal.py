from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments"

INPUT_CSV = EXP / "fakeavceleb_audio_rtvc_clean_test.csv"
OUTPUT_CSV = EXP / "audio_signal_diagnosis.csv"

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("AUDIO SIGNAL SHORTCUT DIAGNOSIS")
print("=" * 70)
print(f"Total files: {len(df)}")
print()

results = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Analyzing audio",
    unit="file"
):
    wav_path = tempfile.mktemp(suffix=".wav")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                row["actual_path"],
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                wav_path,
            ],
            check=True,
        )

        audio, sr = sf.read(wav_path)

        audio = audio.astype(np.float32)

        duration = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio ** 2)))
        abs_mean = float(np.mean(np.abs(audio)))
        peak = float(np.max(np.abs(audio)))

        results.append({
            "method": row["method"],
            "source": row["source"],
            "duration": duration,
            "rms": rms,
            "abs_mean": abs_mean,
            "peak": peak,
        })

    except Exception as e:
        print(f"\nERROR: {row['actual_path']}")
        print(e)

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

result = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SIGNAL STATISTICS")
print("=" * 70)

print(
    result.groupby("method")[
        ["duration", "rms", "abs_mean", "peak"]
    ].agg(
        ["mean", "median", "std", "min", "max"]
    ).to_string()
)

result.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:")
print(OUTPUT_CSV)

print("=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)