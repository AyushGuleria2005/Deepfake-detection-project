import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataloader import DeepfakeDataset, eval_transform
from models.cnn_baseline import CNNBaseline


BASE_DIR = r"D:\DeepFake-Learning"

CSV_FILE = (
    BASE_DIR
    + r"\research\experiments\train.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD EXACTLY 2 IMAGES
# ============================================================

df = pd.read_csv(CSV_FILE)

real = df[df["label"] == 0].iloc[[0]]
fake = df[df["label"] == 1].iloc[[0]]

test_df = pd.concat(
    [real, fake],
    ignore_index=True
)

TEST_CSV = (
    BASE_DIR
    + r"\research\experiments\two_image_test.csv"
)

test_df.to_csv(
    TEST_CSV,
    index=False
)


dataset = DeepfakeDataset(
    TEST_CSV,
    transform=eval_transform
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE MODEL
# ============================================================

model = CNNBaseline().to(DEVICE)

# Turn Dropout OFF for this controlled test
model.classifier[3].p = 0.0

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("2-IMAGE OVERFIT TEST")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Images:")

print(
    "1:",
    test_df.iloc[0]["filename"],
    "| Label:",
    test_df.iloc[0]["label"],
    "| Real"
)

print(
    "2:",
    test_df.iloc[1]["filename"],
    "| Label:",
    test_df.iloc[1]["label"],
    "| Fake"
)


# ============================================================
# TRAIN FOR 100 EPOCHS
# ============================================================

for epoch in range(100):

    model.train()

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


    # --------------------------------------------------------
    # CHECK EVERY 10 EPOCHS
    # --------------------------------------------------------

    if (epoch + 1) % 10 == 0:

        model.eval()

        with torch.no_grad():

            logits = model(images).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            accuracy = (
                predictions == labels.long()
            ).float().mean().item()

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {loss.item():.6f} | "
            f"Accuracy: {accuracy * 100:.2f}% | "
            f"Probabilities: "
            f"{[round(x, 6) for x in probabilities.cpu().tolist()]}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

model.eval()

with torch.no_grad():

    logits = model(images).squeeze(1)

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).long()

    final_accuracy = (
        predictions == labels.long()
    ).float().mean().item()


print()
print("=" * 60)
print("2-IMAGE TEST COMPLETE")
print("=" * 60)

print(
    f"Final Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    "Final Probabilities:",
    [round(x, 6) for x in probabilities.cpu().tolist()]
)

print(
    "Final Predictions:",
    predictions.cpu().tolist()
)

print(
    "Actual Labels:",
    labels.long().cpu().tolist()
)