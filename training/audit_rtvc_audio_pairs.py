from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "research" / "experiments" / "fakeavceleb_audio_rtvc_matched_test.csv"
OUT_CSV = ROOT / "research" / "experiments" / "rtvc_audio_identity_audit.csv"

df = pd.read_csv(TEST_CSV)

real_df = df[df["eval_label"] == 0].copy()
rtvc_df = df[df["eval_label"] == 1].copy()

real_by_source = dict(zip(real_df["source"], real_df["actual_path"]))
rtvc_by_source = dict(zip(rtvc_df["source"], rtvc_df["actual_path"]))


def extract_audio(video_path, wav_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.returncode == 0


def audio_hash(wav_path):
    audio, sr = sf.read(wav_path)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return hashlib.sha256(audio.tobytes()).hexdigest(), audio, sr


print("=" * 70)
print("RTVC AUDIO IDENTITY AUDIT")
print("=" * 70)

print("RTVC pairs:", len(rtvc_by_source))
print("Real pairs:", len(real_by_source))

results = []

with tempfile.TemporaryDirectory() as tmp:

    tmp = Path(tmp)

    for i, source in enumerate(sorted(rtvc_by_source.keys()), start=1):

        real_video = Path(real_by_source[source])
        rtvc_video = Path(rtvc_by_source[source])

        real_wav = tmp / f"{i}_real.wav"
        rtvc_wav = tmp / f"{i}_rtvc.wav"

        real_ok = extract_audio(real_video, real_wav)
        rtvc_ok = extract_audio(rtvc_video, rtvc_wav)

        if not real_ok or not rtvc_ok:
            results.append({
                "source": source,
                "real_audio_extracted": real_ok,
                "rtvc_audio_extracted": rtvc_ok,
                "identical": False,
                "max_abs_difference": np.nan,
                "mean_abs_difference": np.nan,
            })
            continue

        try:
            real_hash, real_audio, real_sr = audio_hash(real_wav)
            rtvc_hash, rtvc_audio, rtvc_sr = audio_hash(rtvc_wav)

            same_shape = real_audio.shape == rtvc_audio.shape

            if same_shape:
                max_diff = float(np.max(np.abs(real_audio - rtvc_audio)))
                mean_diff = float(np.mean(np.abs(real_audio - rtvc_audio)))
                identical = bool(np.array_equal(real_audio, rtvc_audio))
            else:
                max_diff = np.nan
                mean_diff = np.nan
                identical = False

            results.append({
                "source": source,
                "real_audio_extracted": True,
                "rtvc_audio_extracted": True,
                "real_sample_rate": real_sr,
                "rtvc_sample_rate": rtvc_sr,
                "real_samples": len(real_audio),
                "rtvc_samples": len(rtvc_audio),
                "real_hash": real_hash,
                "rtvc_hash": rtvc_hash,
                "identical": identical,
                "max_abs_difference": max_diff,
                "mean_abs_difference": mean_diff,
            })

        except Exception as e:
            results.append({
                "source": source,
                "real_audio_extracted": True,
                "rtvc_audio_extracted": True,
                "identical": False,
                "max_abs_difference": np.nan,
                "mean_abs_difference": np.nan,
                "error": str(e),
            })

        if i % 50 == 0:
            print(f"Processed {i}/{len(rtvc_by_source)}")


results_df = pd.DataFrame(results)

results_df.to_csv(OUT_CSV, index=False)

print("\n" + "=" * 70)
print("AUDIT RESULTS")
print("=" * 70)

print("\nTotal pairs:", len(results_df))

print(
    "Identical audio pairs:",
    int(results_df["identical"].sum())
)

print(
    "Different audio pairs:",
    int((~results_df["identical"]).sum())
)

print(
    "Extraction failures:",
    int(
        (
            ~results_df["real_audio_extracted"].astype(bool)
            | ~results_df["rtvc_audio_extracted"].astype(bool)
        ).sum()
    )
)

if len(results_df) > 0:
    identical_pct = (
        results_df["identical"].mean() * 100
    )

    print(
        f"Identical percentage: {identical_pct:.2f}%"
    )

print("\nIdentical sources:")

identical_sources = results_df[
    results_df["identical"] == True
]["source"]

print(identical_sources.tolist())

print("\nSaved:")
print(OUT_CSV)

print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)