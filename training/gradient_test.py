import torch
import torch.nn as nn

from training.dataloader import DeepfakeDataset, eval_transform
from models.cnn_baseline import CNNBaseline

import pandas as pd


BASE_DIR = r"D:\DeepFake-Learning"
CSV_FILE = (
    BASE_DIR
    + r"\research\experiments\train.csv"
)


df = pd.read_csv(CSV_FILE)

# 4 real + 4 fake
real_df = df[df["label"] == 0].head(4)
fake_df = df[df["label"] == 1].head(4)

test_df = pd.concat(
    [real_df, fake_df],
    ignore_index=True
)

temp_csv = (
    BASE_DIR
    + r"\research\experiments\gradient_test.csv"
)

test_df.to_csv(temp_csv, index=False)


dataset = DeepfakeDataset(
    temp_csv,
    transform=eval_transform
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNNBaseline().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


images, labels = next(iter(loader))

images = images.to(device)
labels = labels.to(device)


# Initial prediction
model.eval()

with torch.no_grad():
    initial_logits = model(images).squeeze(1)

print("Initial logits:")
print(initial_logits.detach().cpu().tolist())

print()
print("Initial probabilities:")
print(torch.sigmoid(initial_logits).detach().cpu().tolist())


# Training step
model.train()

optimizer.zero_grad()

logits = model(images).squeeze(1)

loss = criterion(logits, labels)

print()
print("Loss before backward:", loss.item())

loss.backward()


# Check gradients
print()
print("Gradient check:")

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        print(
            f"{name:40s} "
            f"mean_grad={parameter.grad.abs().mean().item():.10f}"
        )

    else:

        print(
            f"{name:40s} "
            f"GRADIENT = NONE"
        )


optimizer.step()


# Prediction after one update
model.eval()

with torch.no_grad():
    updated_logits = model(images).squeeze(1)

print()
print("Updated logits:")
print(updated_logits.detach().cpu().tolist())

print()
print("Updated probabilities:")
print(torch.sigmoid(updated_logits).detach().cpu().tolist())

print()
print("Gradient test complete.")