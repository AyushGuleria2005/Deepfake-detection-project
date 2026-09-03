from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR = Path(r"D:\DeepFake-Learning")
CSV_FILE = BASE_DIR / "research" / "experiments" / "train.csv"

df = pd.read_csv(CSV_FILE)

real_samples = df[df["label"] == 0].head(4)
fake_samples = df[df["label"] == 1].head(4)

samples = pd.concat(
    [real_samples, fake_samples],
    ignore_index=True
)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, (_, row) in zip(axes.flat, samples.iterrows()):

    image = Image.open(row["filepath"]).convert("RGB")

    ax.imshow(image)
    ax.set_title(
        f"{row['manipulation']} | label={row['label']}\n"
        f"{row['filename']}"
    )
    ax.axis("off")

plt.tight_layout()

output = BASE_DIR / "outputs" / "plots" / "sample_inspection.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150)
plt.show()

print("Saved:", output)