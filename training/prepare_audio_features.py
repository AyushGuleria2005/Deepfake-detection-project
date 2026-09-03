from pathlib import Path
import numpy as np
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\DeepFake-Learning")

INPUT_DIR = (
    BASE
    / "data"
    / "processed"
    / "audio_features"
)

OUTPUT_DIR = (
    BASE
    / "data"
    / "processed"
    / "audio_features_fixed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# AUDIO FEATURE SETTINGS
# ============================================================

N_MELS = 64
TARGET_FRAMES = 512


# ============================================================
# FIX LOG-MEL SIZE
# ============================================================

def fix_logmel_size(logmel):

    if logmel.ndim != 2:

        raise ValueError(
            f"Expected 2D Log-Mel, got shape {logmel.shape}"
        )

    mel_bins, time_frames = logmel.shape

    if mel_bins != N_MELS:

        raise ValueError(
            f"Expected {N_MELS} Mel bins, "
            f"got {mel_bins}"
        )

    # --------------------------------------------------------
    # Already correct size
    # --------------------------------------------------------

    if time_frames == TARGET_FRAMES:

        return logmel

    # --------------------------------------------------------
    # Longer than target -> center crop
    # --------------------------------------------------------

    if time_frames > TARGET_FRAMES:

        start = (
            time_frames - TARGET_FRAMES
        ) // 2

        end = (
            start + TARGET_FRAMES
        )

        return logmel[
            :,
            start:end
        ]

    # --------------------------------------------------------
    # Shorter than target -> zero padding
    # --------------------------------------------------------

    output = np.zeros(
        (
            N_MELS,
            TARGET_FRAMES
        ),
        dtype=np.float32
    )

    start = (
        TARGET_FRAMES - time_frames
    ) // 2

    end = (
        start + time_frames
    )

    output[
        :,
        start:end
    ] = logmel

    return output


# ============================================================
# PROCESS FILES
# ============================================================

def process_test_files():

    files = sorted(
        INPUT_DIR.glob("test_*.npy")
    )

    print("=" * 60)
    print("Fixed-Size Audio Feature Test")
    print("=" * 60)

    print(
        "Input files:",
        len(files)
    )

    if len(files) == 0:

        raise FileNotFoundError(
            f"No test features found in:\n{INPUT_DIR}"
        )

    success = 0

    for input_file in tqdm(
        files,
        desc="Preparing audio features"
    ):

        try:

            # ------------------------------------------------
            # Load Log-Mel
            # ------------------------------------------------

            logmel = np.load(
                input_file
            )

            print(
                f"\n{input_file.name}: "
                f"{logmel.shape} -> ",
                end=""
            )

            # ------------------------------------------------
            # Fix size
            # ------------------------------------------------

            fixed = fix_logmel_size(
                logmel
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_file = (
                OUTPUT_DIR
                / input_file.name
            )

            np.save(
                output_file,
                fixed
            )

            print(
                fixed.shape
            )

            success += 1

        except Exception as e:

            print(
                f"\nFailed: {input_file.name}"
            )

            print(
                "Error:",
                e
            )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FIXED-SIZE TEST COMPLETE")
    print("=" * 60)

    print(
        "Successful:",
        success
    )

    print(
        "Expected:",
        len(files)
    )

    output_files = sorted(
        OUTPUT_DIR.glob("test_*.npy")
    )

    print(
        "Output files:",
        len(output_files)
    )

    if output_files:

        shapes = []

        for file in output_files:

            arr = np.load(file)

            shapes.append(
                arr.shape
            )

        unique_shapes = set(
            shapes
        )

        print(
            "Unique output shapes:",
            unique_shapes
        )

        if unique_shapes == {
            (N_MELS, TARGET_FRAMES)
        }:

            print(
                "\nPASS: All features have shape "
                "(64, 512)"
            )

        else:

            print(
                "\nWARNING: Shape mismatch detected."
            )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_test_files()