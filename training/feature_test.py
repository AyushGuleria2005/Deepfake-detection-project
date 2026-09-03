import pandas as pd
import torch

from training.dataloader import DeepfakeDataset, eval_transform
from models.cnn_baseline import CNNBaseline


BASE_DIR = r"D:\DeepFake-Learning"

CSV_FILE = (
    BASE_DIR
    + r"\research\experiments\train.csv"
)


# --------------------------------------------------
# LOAD ONE REAL + ONE FAKE
# --------------------------------------------------

df = pd.read_csv(CSV_FILE)

real = df[df["label"] == 0].iloc[[0]]
fake = df[df["label"] == 1].iloc[[0]]

test_df = pd.concat(
    [real, fake],
    ignore_index=True
)

TEST_CSV = (
    BASE_DIR
    + r"\research\experiments\feature_test.csv"
)

test_df.to_csv(
    TEST_CSV,
    index=False
)


dataset = DeepfakeDataset(
    TEST_CSV,
    transform=eval_transform
)


real_image, real_label = dataset[0]
fake_image, fake_label = dataset[1]


images = torch.stack([
    real_image,
    fake_image
])


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = CNNBaseline().to(device)

model.eval()

images = images.to(device)


# --------------------------------------------------
# INSPECT EACH CNN LAYER
# --------------------------------------------------

x = images

print("=" * 60)
print("CNN FEATURE DIAGNOSTIC")
print("=" * 60)

print()

print("Input difference:")
print(
    torch.abs(x[0] - x[1]).mean().item()
)


for i, layer in enumerate(model.features):

    x = layer(x)

    difference = torch.abs(
        x[0] - x[1]
    ).mean().item()

    print(
        f"Layer {i}: "
        f"{layer.__class__.__name__:20s} "
        f"Shape={str(tuple(x.shape)):25s} "
        f"Difference={difference:.10f}"
    )


# --------------------------------------------------
# FINAL FEATURES
# --------------------------------------------------

print()
print("Final feature difference:")
print(
    torch.abs(x[0] - x[1]).mean().item()
)


# --------------------------------------------------
# CLASSIFIER OUTPUT
# --------------------------------------------------

output = model.classifier(x).squeeze(1)

print()

print("Final logits:")
print(
    output.detach().cpu().tolist()
)

print()

print("Final probabilities:")

print(
    torch.sigmoid(output)
    .detach()
    .cpu()
    .tolist()
)


print()
print("=" * 60)
print("FEATURE TEST COMPLETE")
print("=" * 60)