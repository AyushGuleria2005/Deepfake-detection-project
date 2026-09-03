import torch
import torch.nn as nn
import pandas as pd

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(r"D:\DeepFake-Learning")

EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"

FEATURE_DIR = BASE_DIR / "data" / "processed" / "temporal_features"

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TEMPORAL FEATURE PRECOMPUTATION")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# DATASET
# =========================================================

class TemporalFeatureDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.frame_columns = [
            "frame_0",
            "frame_1",
            "frame_2",
            "frame_3",
            "frame_4"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        frames = []

        for column in self.frame_columns:

            image = Image.open(
                row[column]
            ).convert("RGB")

            image = transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        return frames, label


# =========================================================
# RESNET18 FEATURE EXTRACTOR
# =========================================================

resnet = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

resnet.fc = nn.Identity()

resnet = resnet.to(device)

resnet.eval()

for parameter in resnet.parameters():
    parameter.requires_grad = False


# =========================================================
# PROCESS FUNCTION
# =========================================================

def process_split(
    csv_name,
    output_name,
    batch_size=4
):

    csv_path = EXPERIMENTS_DIR / csv_name

    dataset = TemporalFeatureDataset(
        csv_path
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    all_features = []
    all_labels = []

    print()
    print("-" * 70)
    print("Processing:", csv_name)
    print("Samples:", len(dataset))
    print("-" * 70)

    with torch.no_grad():

        for frames, labels in tqdm(
            loader,
            desc=csv_name
        ):

            batch_size_actual = frames.shape[0]

            sequence_length = frames.shape[1]

            # [B, 5, 3, 224, 224]
            frames = frames.to(
                device,
                non_blocking=True
            )

            # Flatten temporal dimension
            # [B*5, 3, 224, 224]

            frames = frames.view(
                batch_size_actual * sequence_length,
                3,
                224,
                224
            )

            # ResNet
            features = resnet(
                frames
            )

            # [B*5, 512]
            features = features.view(
                batch_size_actual,
                sequence_length,
                512
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels
            )

    all_features = torch.cat(
        all_features,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    output_path = FEATURE_DIR / output_name

    torch.save(
        {
            "features": all_features,
            "labels": all_labels
        },
        output_path
    )

    print()
    print("Saved:", output_path)

    print(
        "Feature shape:",
        all_features.shape
    )

    print(
        "Label shape:",
        all_labels.shape
    )

    print(
        "Feature size:",
        f"{all_features.numel() * 4 / (1024**2):.2f} MB"
    )


# =========================================================
# RUN ALL SPLITS
# =========================================================

process_split(
    "train_temporal.csv",
    "train_temporal_features.pt"
)

process_split(
    "validation_temporal.csv",
    "validation_temporal_features.pt"
)

process_split(
    "test_temporal.csv",
    "test_temporal_features.pt"
)


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 70)
print("FEATURE PRECOMPUTATION COMPLETE")
print("=" * 70)

print("Feature directory:")
print(FEATURE_DIR)

print()
print("Created:")
print("train_temporal_features.pt")
print("validation_temporal_features.pt")
print("test_temporal_features.pt")

print("=" * 70)