from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms


BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"


class DeepfakeDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["filepath"]

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# TRAIN TRANSFORM
# ---------------------------------------------------------

train_transform = transforms.Compose([

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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# VALIDATION / TEST TRANSFORM
# ---------------------------------------------------------

eval_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# CREATE DATALOADERS
# ---------------------------------------------------------

def create_dataloaders(
    batch_size=32,
    train_csv="train.csv",
    validation_csv="validation.csv",
    test_csv="test.csv"
):

    train_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / train_csv,
        transform=train_transform
    )

    validation_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / validation_csv,
        transform=eval_transform
    )

    test_dataset = DeepfakeDataset(
        EXPERIMENTS_DIR / test_csv,
        transform=eval_transform
    )


    # -----------------------------------------------------
    # CLASS DISTRIBUTION
    # -----------------------------------------------------

    class_counts = (
        train_dataset.data["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Training class distribution:")

    for label, count in class_counts.items():

        class_name = (
            "Real"
            if label == 0
            else "Fake"
        )

        print(
            f"{class_name}: {count}"
        )


    # -----------------------------------------------------
    # WEIGHTED SAMPLER
    # -----------------------------------------------------

    class_weights = {
        label: 1.0 / count
        for label, count in class_counts.items()
    }

    sample_weights = (
        train_dataset.data["label"]
        .map(class_weights)
        .to_numpy(copy=True)
    )

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(train_dataset),
        replacement=True
    )


    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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
# QUICK TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, validation_loader, test_loader = (
        create_dataloaders(
            batch_size=32,
            train_csv="train_balanced_face.csv",
            validation_csv="validation_face.csv",
            test_csv="test_face.csv"
        )
    )

    images, labels = next(
        iter(train_loader)
    )

    print()
    print("Face DataLoader test successful!")

    print(
        "Device:",
        device
    )

    print(
        "Batch image shape:",
        images.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    print(
        "Batch Real:",
        (labels == 0).sum().item()
    )

    print(
        "Batch Fake:",
        (labels == 1).sum().item()
    )

    print(
        "First 10 labels:",
        labels[:10].tolist()
    )

    images = images.to(
        device,
        non_blocking=True
    )

    labels = labels.to(
        device,
        non_blocking=True
    )

    print(
        "Images moved to:",
        images.device
    )

    print(
        "Labels moved to:",
        labels.device
    )