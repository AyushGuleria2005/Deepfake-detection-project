from pathlib import Path
import sys
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.audio_cnn import AudioCNN

EXP = ROOT / "research" / "experiments"

TEST_CSV = EXP / "fakeavceleb_audio_source_disjoint_rtvc_test.csv"
FEATURE_CSV = EXP / "fakeavceleb_audio_all_features.csv"
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "audio_cnn_best.pth"
OUTPUT = EXP / "audio_source_disjoint_rtvc_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("SOURCE-DISJOINT RTVC AUDIO EVALUATION")
print("=" * 70)

test = pd.read_csv(TEST_CSV)
features = pd.read_csv(FEATURE_CSV)

print("Test samples:", len(test))
print("Test sources:", test["source"].nunique())
print("Feature mappings:", len(features))

df = test.merge(
    features[["actual_path", "feature_path"]],
    on="actual_path",
    how="left"
)

missing = df["feature_path"].isna().sum()

print("Missing features:", missing)

if missing:
    raise RuntimeError("Missing feature mappings.")

print("Source overlap is guaranteed by split construction.")

model = AudioCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint val AUC:", checkpoint.get("val_auc"))

predictions = []

with torch.no_grad():

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluating unseen sources",
        unit="file"
    ):

        feature = torch.from_numpy(
            __import__("numpy").load(row["feature_path"])
        ).float()

        x = feature.unsqueeze(0).unsqueeze(0).to(DEVICE)

        logits = model(x)
        probability = torch.sigmoid(logits).item()

        predictions.append({
            "actual_path": row["actual_path"],
            "source": row["source"],
            "method": row["method"],
            "eval_label": int(row["audio_label"]),
            "predicted_probability": probability,
            "predicted_label": int(probability >= 0.5),
        })

result = pd.DataFrame(predictions)

result.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("SOURCE-DISJOINT RESULT")
print("=" * 70)

print("Samples:", len(result))
print(
    "Accuracy:",
    (result["predicted_label"] == result["eval_label"]).mean()
)

print(
    "Mean fake probability:",
    result["predicted_probability"].mean()
)

print(
    "Median fake probability:",
    result["predicted_probability"].median()
)

print(
    "Minimum fake probability:",
    result["predicted_probability"].min()
)

print(
    "Maximum fake probability:",
    result["predicted_probability"].max()
)

print("\nPredicted labels:")
print(result["predicted_label"].value_counts().sort_index())

print("\nSaved:")
print(OUTPUT)

print("=" * 70)
print("SOURCE-DISJOINT EVALUATION COMPLETE")
print("=" * 70)