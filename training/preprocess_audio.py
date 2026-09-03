from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\DeepFake-Learning")

EXPERIMENTS_DIR = (
    BASE
    / "research"
    / "experiments"
)

OUTPUT_DIR = (
    BASE
    / "data"
    / "processed"
    / "audio_features"
)

TEMP_DIR = (
    BASE
    / "data"
    / "processed"
    / "audio_temp"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SPLIT CSV FILES
# ============================================================

SPLITS = {
    "train": EXPERIMENTS_DIR / "fakeavceleb_train.csv",
    "validation": EXPERIMENTS_DIR / "fakeavceleb_validation.csv",
    "test": EXPERIMENTS_DIR / "fakeavceleb_test.csv",
}


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 1024

HOP_LENGTH = 256

N_MELS = 64

TARGET_FRAMES = 512


# ============================================================
# EXTRACT AUDIO USING FFMPEG
# ============================================================

def extract_audio(
    video_path,
    wav_path
):

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]

    result = subprocess.run(
        command
    )

    return (
        result.returncode == 0
        and wav_path.exists()
    )


# ============================================================
# CREATE LOG-MEL
# ============================================================

def make_logmel(
    wav_path
):

    audio, sr = sf.read(
        wav_path
    )

    audio = audio.astype(
        np.float32
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return logmel.astype(
        np.float32
    )


# ============================================================
# FIX LOG-MEL TO 64 x 512
# ============================================================

def fix_logmel_size(
    logmel
):

    if logmel.ndim != 2:

        raise ValueError(
            f"Expected 2D Log-Mel, "
            f"got {logmel.shape}"
        )

    mel_bins, time_frames = (
        logmel.shape
    )

    if mel_bins != N_MELS:

        raise ValueError(
            f"Expected {N_MELS} Mel bins, "
            f"got {mel_bins}"
        )

    # --------------------------------------------------------
    # Exact size
    # --------------------------------------------------------

    if time_frames == TARGET_FRAMES:

        return logmel

    # --------------------------------------------------------
    # Center crop
    # --------------------------------------------------------

    if time_frames > TARGET_FRAMES:

        start = (
            time_frames
            - TARGET_FRAMES
        ) // 2

        end = (
            start
            + TARGET_FRAMES
        )

        return logmel[
            :,
            start:end
        ]

    # --------------------------------------------------------
    # Zero padding
    # --------------------------------------------------------

    output = np.zeros(
        (
            N_MELS,
            TARGET_FRAMES
        ),
        dtype=np.float32
    )

    start = (
        TARGET_FRAMES
        - time_frames
    ) // 2

    end = (
        start
        + time_frames
    )

    output[
        :,
        start:end
    ] = logmel

    return output


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    csv_path
):

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING AUDIO: {split_name.upper()}"
    )
    print("=" * 70)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    required_columns = [
        "actual_path",
        "source",
        "video_label",
        "audio_label",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(
        "Videos:",
        len(df)
    )

    # --------------------------------------------------------
    # OUTPUT FOLDER
    # --------------------------------------------------------

    split_output = (
        OUTPUT_DIR
        / split_name
    )

    split_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata_rows = []

    failed_rows = []

    success = 0

    skipped = 0

    failed = 0

    # --------------------------------------------------------
    # PROCESS VIDEOS
    # --------------------------------------------------------

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name
    ):

        video_path = Path(
            row["actual_path"]
        )

        feature_path = (
            split_output
            / f"{idx:06d}.npy"
        )

        wav_path = (
            TEMP_DIR
            / f"{split_name}_{idx:06d}.wav"
        )

        # ----------------------------------------------------
        # SKIP EXISTING FEATURE
        # ----------------------------------------------------

        if feature_path.exists():

            try:

                existing = np.load(
                    feature_path,
                    mmap_mode="r"
                )

                if existing.shape == (
                    N_MELS,
                    TARGET_FRAMES
                ):

                    metadata_rows.append(
                        {
                            "feature_path":
                                str(feature_path),

                            "actual_path":
                                str(video_path),

                            "source":
                                row["source"],

                            "video_label":
                                int(row["video_label"]),

                            "audio_label":
                                int(row["audio_label"]),
                        }
                    )

                    skipped += 1

                    continue

            except Exception:

                pass

        try:

            # ------------------------------------------------
            # CHECK VIDEO
            # ------------------------------------------------

            if not video_path.exists():

                raise FileNotFoundError(
                    f"Video not found: "
                    f"{video_path}"
                )

            # ------------------------------------------------
            # MP4 -> WAV
            # ------------------------------------------------

            audio_ok = extract_audio(
                video_path,
                wav_path
            )

            if not audio_ok:

                raise RuntimeError(
                    "FFmpeg audio extraction failed."
                )

            # ------------------------------------------------
            # WAV -> LOG-MEL
            # ------------------------------------------------

            logmel = make_logmel(
                wav_path
            )

            # ------------------------------------------------
            # FIX SIZE
            # ------------------------------------------------

            fixed_logmel = (
                fix_logmel_size(
                    logmel
                )
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            np.save(
                feature_path,
                fixed_logmel
            )

            # ------------------------------------------------
            # SAVE METADATA
            # ------------------------------------------------

            metadata_rows.append(
                {
                    "feature_path":
                        str(feature_path),

                    "actual_path":
                        str(video_path),

                    "source":
                        row["source"],

                    "video_label":
                        int(row["video_label"]),

                    "audio_label":
                        int(row["audio_label"]),
                }
            )

            success += 1

        except Exception as e:

            failed += 1

            failed_rows.append(
                {
                    "actual_path":
                        str(video_path),

                    "source":
                        row["source"],

                    "video_label":
                        int(row["video_label"]),

                    "audio_label":
                        int(row["audio_label"]),

                    "error":
                        str(e),
                }
            )

        finally:

            # ------------------------------------------------
            # DELETE TEMP WAV
            # ------------------------------------------------

            if wav_path.exists():

                try:

                    wav_path.unlink()

                except Exception:

                    pass

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = (
        EXPERIMENTS_DIR
        / f"fakeavceleb_audio_{split_name}.csv"
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    # ========================================================
    # SAVE FAILED FILES
    # ========================================================

    failed_path = (
        EXPERIMENTS_DIR
        / f"fakeavceleb_audio_{split_name}_failed.csv"
    )

    failed_df = pd.DataFrame(
        failed_rows
    )

    failed_df.to_csv(
        failed_path,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    feature_files = list(
        split_output.glob("*.npy")
    )

    print("\n")
    print("-" * 70)
    print(
        f"{split_name.upper()} COMPLETE"
    )
    print("-" * 70)

    print(
        "Successful:",
        success
    )

    print(
        "Skipped:",
        skipped
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Total expected:",
        len(df)
    )

    print(
        "Feature files:",
        len(feature_files)
    )

    print(
        "Metadata rows:",
        len(metadata_df)
    )

    print(
        "Failed records:",
        len(failed_df)
    )

    print(
        "Metadata saved:",
        metadata_path
    )

    print(
        "Failed list saved:",
        failed_path
    )

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    if len(feature_files) > 0:

        sample_files = feature_files[
            :min(20, len(feature_files))
        ]

        shapes = set()

        for file in sample_files:

            arr = np.load(
                file,
                mmap_mode="r"
            )

            shapes.add(
                arr.shape
            )

        print(
            "Sample feature shapes:",
            shapes
        )

        if shapes != {
            (N_MELS, TARGET_FRAMES)
        }:

            raise ValueError(
                "Feature shape validation failed."
            )

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FakeAVCeleb FULL AUDIO PREPROCESSING")
    print("=" * 70)

    print(
        "Sample rate:",
        SAMPLE_RATE
    )

    print(
        "Mel bins:",
        N_MELS
    )

    print(
        "Target frames:",
        TARGET_FRAMES
    )

    print(
        "Feature shape:",
        f"({N_MELS}, {TARGET_FRAMES})"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    process_split(
        "train",
        SPLITS["train"]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    process_split(
        "validation",
        SPLITS["validation"]
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    process_split(
        "test",
        SPLITS["test"]
    )

    print("\n")
    print("=" * 70)
    print("ALL AUDIO PREPROCESSING COMPLETE")
    print("=" * 70)