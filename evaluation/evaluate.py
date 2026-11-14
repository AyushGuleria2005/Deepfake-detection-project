
import os
import random
import time
import requests
import pandas as pd
from pathlib import Path
from collections import Counter

# ---------------- CONFIGURATION ----------------

DATASET_ROOT = Path(
    r"D:\DeepFake-Learning\data\Raw\FakeAVCeleb\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
)

API_URL = "http://127.0.0.1:8000/analyze"

VIDEOS_PER_CATEGORY = 50
RANDOM_SEED = 42

OUTPUT_CSV = Path("evaluation/evaluation_results.csv")

CATEGORIES = {
    "FakeVideo-FakeAudio": "FAKE",
    "FakeVideo-RealAudio": "FAKE",
    "RealVideo-FakeAudio": "FAKE",
    "RealVideo-RealAudio": "REAL"
}

# ---------------- COLLECT VIDEOS ----------------

random.seed(RANDOM_SEED)
test_videos = []

print("\nScanning dataset...\n")

for category, expected_label in CATEGORIES.items():

    folder = DATASET_ROOT / category

    videos = list(folder.rglob("*.mp4"))

    if len(videos) < VIDEOS_PER_CATEGORY:
        raise ValueError(
            f"Not enough videos in {category}: {len(videos)}"
        )

    selected = random.sample(videos, VIDEOS_PER_CATEGORY)

    for video in selected:
        test_videos.append({
            "path": video,
            "category": category,
            "actual": expected_label
        })

    print(f"{category}: {len(selected)} selected")

print(f"\nTotal selected: {len(test_videos)}")

# ---------------- API TESTING ----------------

results = []

correct = 0
failed = 0

start_time = time.time()

for index, item in enumerate(test_videos, start=1):

    video_path = item["path"]

    print(
        f"\n[{index}/{len(test_videos)}] "
        f"Testing: {video_path.name}"
    )

    row = {
        "video": video_path.name,
        "category": item["category"],
        "actual": item["actual"],
        "predicted": None,
        "confidence": None,
        "visual_fake_probability": None,
        "audio_fake_probability": None,
        "fusion_fake_probability": None,
        "evidence": None,
        "correct": None,
        "status": "ERROR",
        "error": None
    }

    try:
        with open(video_path, "rb") as file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        video_path.name,
                        file,
                        "video/mp4"
                    )
                },
                timeout=300
            )

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError("API returned success=false")

        predicted = data["prediction"].upper()

        row["predicted"] = predicted
        row["confidence"] = data.get("confidence")

        row["visual_fake_probability"] = (
            data.get("visual", {}).get("fake_probability")
        )

        row["audio_fake_probability"] = (
            data.get("audio", {}).get("fake_probability")
        )

        row["fusion_fake_probability"] = (
            data.get("fusion", {}).get("fake_probability")
        )

        row["evidence"] = (
            data.get("evidence", {}).get("type")
        )

        row["correct"] = predicted == item["actual"]
        row["status"] = "SUCCESS"

        if row["correct"]:
            correct += 1

        print(
            f"Actual: {item['actual']} | "
            f"Predicted: {predicted} | "
            f"Correct: {row['correct']}"
        )

    except Exception as error:

        failed += 1
        row["error"] = str(error)

        print("ERROR:", error)

    results.append(row)

    # Save progress after every video
    pd.DataFrame(results).to_csv(
        OUTPUT_CSV,
        index=False
    )

# ---------------- FINAL METRICS ----------------

df = pd.DataFrame(results)

successful = df[df["status"] == "SUCCESS"]

if len(successful) > 0:

    actual = successful["actual"]
    predicted = successful["predicted"]

    tp = ((actual == "FAKE") & (predicted == "FAKE")).sum()
    tn = ((actual == "REAL") & (predicted == "REAL")).sum()
    fp = ((actual == "REAL") & (predicted == "FAKE")).sum()
    fn = ((actual == "FAKE") & (predicted == "REAL")).sum()

    accuracy = (tp + tn) / len(successful)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    print("\n========== FINAL RESULTS ==========")

    print("Successful:", len(successful))
    print("Failed:", failed)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print("TP:", tp, "| TN:", tn)
    print("FP:", fp, "| FN:", fn)

    print("\nCategory-wise results:")

    for category, group in successful.groupby("category"):

        category_accuracy = (
            group["actual"] == group["predicted"]
        ).mean()

        print(
            f"{category}: "
            f"{category_accuracy:.2%} "
            f"({len(group)} videos)"
        )

elapsed = time.time() - start_time

print(f"\nTime taken: {elapsed / 60:.2f} minutes")
print("Results saved to:", OUTPUT_CSV)
print("\nEvaluation completed!")
