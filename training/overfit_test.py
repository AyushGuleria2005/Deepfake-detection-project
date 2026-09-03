from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from training.dataloader import DeepfakeDataset, eval_transform
from models.cnn_baseline import CNNBaseline


BASE_DIR = Path(r"D:\DeepFake-Learning")
EXPERIMENTS_DIR = BASE_DIR / "research" / "experiments"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Use only a tiny fixed subset
SAMPLES = 32
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def main():

    print("=" * 60)
    print("CNN SANITY / OVERFIT TEST")
    print("=" * 60)

    print("Device:", DEVICE)

    df = __import__("pandas").read_csv(
        EXPERIMENTS_DIR / "train.csv"
    )

    # Take 16 real + 16 fake
    real_df = df[df["label"] == 0].head(16)
    fake_df = df[df["label"] == 1].head(16)

    test_df = __import__("pandas").concat(
        [real_df, fake_df],
        ignore_index=True
    )

    temp_csv = EXPERIMENTS_DIR / "overfit_test.csv"
    test_df.to_csv(temp_csv, index=False)

    dataset = DeepfakeDataset(
    temp_csv,
    transform=eval_transform
)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = CNNBaseline().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        total_correct = 0
        total_samples = 0
        total_loss = 0

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()
            optimizer.step()

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            total_loss += (
                loss.item() * labels.size(0)
            )

        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    print()
    print("=" * 60)
    print("OVERFIT TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()