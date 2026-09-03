import pandas as pd
import torch

from training.dataloader import DeepfakeDataset, eval_transform


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
    + r"\research\experiments\input_test.csv"
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


print("=" * 60)
print("INPUT TENSOR DIAGNOSTIC")
print("=" * 60)

print()

print("Real:")
print("Label:", real_label.item())
print("Shape:", real_image.shape)
print("Min:", real_image.min().item())
print("Max:", real_image.max().item())
print("Mean:", real_image.mean().item())
print("Std:", real_image.std().item())

print()

print("Fake:")
print("Label:", fake_label.item())
print("Shape:", fake_image.shape)
print("Min:", fake_image.min().item())
print("Max:", fake_image.max().item())
print("Mean:", fake_image.mean().item())
print("Std:", fake_image.std().item())


# --------------------------------------------------
# COMPARE IMAGES
# --------------------------------------------------

difference = torch.abs(
    real_image - fake_image
)

print()

print("Difference between Real and Fake:")
print("Mean absolute difference:", difference.mean().item())
print("Maximum difference:", difference.max().item())
print(
    "Number of different values:",
    (difference > 0).sum().item()
)
print(
    "Total values:",
    difference.numel()
)


# --------------------------------------------------
# CHECK WHETHER TENSORS ARE IDENTICAL
# --------------------------------------------------

print()

print(
    "Are tensors identical:",
    torch.equal(real_image, fake_image)
)

print(
    "Are tensors approximately identical:",
    torch.allclose(
        real_image,
        fake_image
    )
)


print()
print("=" * 60)
print("INPUT TEST COMPLETE")
print("=" * 60)