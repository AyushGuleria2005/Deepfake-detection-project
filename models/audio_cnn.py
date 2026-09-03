import torch
import torch.nn as nn


# ============================================================
# AUDIO CNN
# Input:
#   [batch, 1, 64, 512]
#
# Output:
#   logit [batch, 1]
#   embedding [batch, 256]
# ============================================================

class AudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # CNN FEATURE EXTRACTOR
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # 1 -> 32
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 32 -> 64
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 64 -> 128
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 128 -> 256
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        self.embedding = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3)
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            256,
            1
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        return_embedding=False
    ):

        x = self.features(x)

        embedding = self.embedding(x)

        logit = self.classifier(
            embedding
        )

        if return_embedding:

            return logit, embedding

        return logit


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AudioCNN().to(device)

    # Fake Log-Mel batch
    x = torch.randn(
        4,
        1,
        64,
        512
    ).to(device)

    with torch.no_grad():

        logits, embeddings = model(
            x,
            return_embedding=True
        )

    print("=" * 60)
    print("Audio CNN Test")
    print("=" * 60)

    print(
        "Device:",
        device
    )

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Logit shape:",
        logits.shape
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print("=" * 60)

    print("PASS: Audio CNN forward pass successful.")