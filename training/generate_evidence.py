from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"D:\DeepFake-Learning")

INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "fusion_test_predictions.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "fusion_evidence.csv"
)


def evidence(row):

    visual = float(row["visual_fake_probability"])
    audio = float(row["audio_fake_probability"])
    fusion = float(row["fusion_fake_probability"])

    video_fake = int(row["video_label"])
    audio_fake = int(row["audio_label"])

    # --------------------------------------------------------
    # Modality states
    # --------------------------------------------------------

    if visual >= 0.5:
        visual_state = "Fake"
    else:
        visual_state = "Real"

    if audio >= 0.5:
        audio_state = "Fake"
    else:
        audio_state = "Real"

    # --------------------------------------------------------
    # Evidence strength
    # --------------------------------------------------------

    visual_strength = abs(visual - 0.5) * 2
    audio_strength = abs(audio - 0.5) * 2

    if visual >= 0.5 and audio >= 0.5:
        evidence_type = "Cross-modal agreement"
        explanation = (
            "Both visual and audio evidence indicate manipulation."
        )

    elif visual >= 0.5 and audio < 0.5:
        evidence_type = "Visual-dominant evidence"
        explanation = (
            "Visual evidence indicates manipulation, "
            "while audio evidence does not."
        )

    elif visual < 0.5 and audio >= 0.5:
        evidence_type = "Audio-dominant evidence"
        explanation = (
            "Audio evidence indicates manipulation, "
            "while visual evidence does not."
        )

    else:
        evidence_type = "Cross-modal real evidence"
        explanation = (
            "Neither visual nor audio evidence strongly "
            "indicates manipulation."
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if fusion >= 0.5:
        final_label = "FAKE"
    else:
        final_label = "REAL"

    confidence = (
        fusion if fusion >= 0.5
        else 1.0 - fusion
    )

    return pd.Series({
        "visual_state": visual_state,
        "audio_state": audio_state,
        "visual_evidence_strength": round(
            visual_strength, 4
        ),
        "audio_evidence_strength": round(
            audio_strength, 4
        ),
        "evidence_type": evidence_type,
        "evidence_explanation": explanation,
        "final_decision": final_label,
        "final_confidence": round(
            confidence, 4
        ),
    })


def main():

    print("=" * 70)
    print("GENERATING MULTIMODAL EVIDENCE")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    print(
        f"Input samples: {len(df)}"
    )

    evidence_df = df.apply(
        evidence,
        axis=1
    )

    result = pd.concat(
        [df, evidence_df],
        axis=1
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("Evidence distribution:")
    print(
        result["evidence_type"]
        .value_counts()
    )

    print()
    print("Final decisions:")
    print(
        result["final_decision"]
        .value_counts()
    )

    print()
    print("Sample evidence:")
    print(
        result[
            [
                "visual_fake_probability",
                "audio_fake_probability",
                "fusion_fake_probability",
                "evidence_type",
                "evidence_explanation",
                "final_decision",
                "final_confidence",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(OUTPUT)

    print()
    print("=" * 70)
    print("EVIDENCE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()