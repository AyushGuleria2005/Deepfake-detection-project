from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature = np.load(row["feature_path"]).astype(np.float32)

        # [64, 512] -> [1, 64, 512]
        feature = torch.from_numpy(feature).unsqueeze(0)

        label = torch.tensor(
            row["audio_label"],
            dtype=torch.float32
        )

        return feature, label


if __name__ == "__main__":

    ROOT = Path(r"D:\DeepFake-Learning")

    csv_path = (
        ROOT
        / "research"
        / "experiments"
        / "fakeavceleb_audio_train.csv"
    )

    dataset = AudioDataset(csv_path)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    features, labels = next(iter(loader))

    print("=" * 60)
    print("AUDIO DATALOADER SANITY CHECK")
    print("=" * 60)

    print("Dataset size :", len(dataset))
    print("Batch shape  :", features.shape)
    print("Labels shape :", labels.shape)
    print("Labels       :", labels.tolist())
    print("Min feature  :", features.min().item())
    print("Max feature  :", features.max().item())

    print("=" * 60)

    assert features.shape == (4, 1, 64, 512)
    assert labels.shape == (4,)

    print("PASS: Audio DataLoader is working correctly.")