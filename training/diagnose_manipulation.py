import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score

from models.resnet_baseline import ResNetBaseline
from training.dataloader import DeepfakeDataset, eval_transform


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "checkpoints"
    / "resnet18_face_crop_best.pth"
)

CSV_FILE = (
    BASE_DIR
    / "research"
    / "experiments"
    / "validation_face.csv"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("MANIPULATION-WISE DIAGNOSTIC")
print("=" * 75)
print("Device:", device)
print("Checkpoint:", CHECKPOINT)
print("CSV:", CSV_FILE)


# ============================================================
# DATASET
# ============================================================

dataset = DeepfakeDataset(
    CSV_FILE,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = ResNetBaseline().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()


# ============================================================
# RUN MODEL
# ============================================================

all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images).squeeze(1)

        probs = torch.sigmoid(outputs)

        all_labels.extend(
            labels.numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

labels = torch.tensor(
    all_labels
).numpy()

probs = torch.tensor(
    all_probs
).numpy()


# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = 0.55

predictions = (
    probs >= THRESHOLD
).astype(int)


# ============================================================
# DETECT MANIPULATION CATEGORY
# ============================================================

categories = []

for path in df["filepath"]:

    path_str = str(path).replace(
        "\\",
        "/"
    )

    if "/Deepfakes/" in path_str:
        categories.append("Deepfakes")

    elif "/Face2Face/" in path_str:
        categories.append("Face2Face")

    elif "/FaceShifter/" in path_str:
        categories.append("FaceShifter")

    elif "/FaceSwap/" in path_str:
        categories.append("FaceSwap")

    elif "/NeuralTextures/" in path_str:
        categories.append("NeuralTextures")

    elif "/Original/" in path_str:
        categories.append("Original")

    else:
        categories.append("Unknown")


# ============================================================
# CHECK CATEGORY COUNTS
# ============================================================

print()
print("Category counts:")
print("-" * 40)

category_counts = pd.Series(
    categories
).value_counts()

for category, count in category_counts.items():

    print(
        f"{category:<20} {count}"
    )


# ============================================================
# MANIPULATION-WISE RESULTS
# ============================================================

fake_categories = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]


print()
print("=" * 75)
print("MANIPULATION RESULTS")
print("=" * 75)

for category in fake_categories:

    indices = [
        i
        for i, c in enumerate(categories)
        if c == category
    ]

    if len(indices) == 0:

        print()
        print(
            f"{category}: NO SAMPLES FOUND"
        )

        continue


    category_labels = labels[indices]

    category_predictions = predictions[
        indices
    ]

    category_probs = probs[
        indices
    ]


    accuracy = accuracy_score(
        category_labels,
        category_predictions
    )


    recall = recall_score(
        category_labels,
        category_predictions,
        zero_division=0
    )


    f1 = f1_score(
        category_labels,
        category_predictions,
        zero_division=0
    )


    mean_probability = (
        category_probs.mean()
    )


    correct = (
        category_predictions
        == category_labels
    ).sum()


    print()
    print(category)
    print("-" * 40)

    print(
        "Samples:",
        len(indices)
    )

    print(
        "Correct:",
        correct
    )

    print(
        "Incorrect:",
        len(indices) - correct
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Fake Recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Fake F1: "
        f"{f1:.4f}"
    )

    print(
        f"Mean Fake Probability: "
        f"{mean_probability:.4f}"
    )


# ============================================================
# ORIGINAL / REAL PERFORMANCE
# ============================================================

original_indices = [
    i
    for i, c in enumerate(categories)
    if c == "Original"
]

if len(original_indices) > 0:

    original_labels = labels[
        original_indices
    ]

    original_predictions = predictions[
        original_indices
    ]

    original_probs = probs[
        original_indices
    ]

    original_accuracy = accuracy_score(
        original_labels,
        original_predictions
    )

    original_recall = recall_score(
        original_labels,
        original_predictions,
        pos_label=0,
        zero_division=0
    )

    print()
    print("=" * 75)
    print("REAL / ORIGINAL PERFORMANCE")
    print("=" * 75)

    print(
        "Samples:",
        len(original_indices)
    )

    print(
        f"Real Accuracy: "
        f"{original_accuracy * 100:.2f}%"
    )

    print(
        f"Real Recall: "
        f"{original_recall * 100:.2f}%"
    )

    print(
        f"Mean Fake Probability: "
        f"{original_probs.mean():.4f}"
    )


# ============================================================
# UNKNOWN CHECK
# ============================================================

unknown_indices = [
    i
    for i, c in enumerate(categories)
    if c == "Unknown"
]

print()
print("=" * 75)
print("FINAL CHECK")
print("=" * 75)

print(
    "Total samples:",
    len(categories)
)

print(
    "Known categories:",
    len(categories) - len(unknown_indices)
)

print(
    "Unknown categories:",
    len(unknown_indices)
)

print()
print("=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)