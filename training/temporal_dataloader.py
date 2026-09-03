import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision import transforms


BASE_DIR = Path(r"D:\DeepFake-Learning")

EXPERIMENTS_DIR = (
    BASE_DIR
    / "research"
    / "experiments"
)


class TemporalDeepfakeDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.frame_columns = [
            "frame_0",
            "frame_1",
            "frame_2",
            "frame_3",
            "frame_4",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        frames = []

        for column in self.frame_columns:

            image_path = row[column]

            image = Image.open(
                image_path
            ).convert("RGB")

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        return frames, label


# ---------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------

temporal_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


def create_temporal_dataloaders(
    batch_size=4
):

    train_dataset = TemporalDeepfakeDataset(
        EXPERIMENTS_DIR / "train_temporal.csv",
        transform=temporal_transform
    )

    validation_dataset = TemporalDeepfakeDataset(
        EXPERIMENTS_DIR / "validation_temporal.csv",
        transform=eval_transform
    )

    test_dataset = TemporalDeepfakeDataset(
        EXPERIMENTS_DIR / "test_temporal.csv",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TEMPORAL DATALOADER TEST")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_loader, validation_loader, test_loader = (
        create_temporal_dataloaders(
            batch_size=4
        )
    )

    frames, labels = next(
        iter(train_loader)
    )

    print()
    print("Train batch loaded successfully!")

    print(
        "Frame tensor shape:",
        frames.shape
    )

    print(
        "Label tensor shape:",
        labels.shape
    )

    print(
        "Expected frame shape:",
        "[4, 5, 3, 224, 224]"
    )

    print(
        "Expected label shape:",
        "[4]"
    )

    print()
    print("Batch labels:")
    print(labels.tolist())

    print()
    print("Sequence 0 frame shapes:")

    for i in range(5):

        print(
            f"Frame {i}:",
            frames[0, i].shape
        )

    print()
    print("=" * 70)
    print("TEMPORAL DATALOADER TEST COMPLETE")
    print("=" * 70)