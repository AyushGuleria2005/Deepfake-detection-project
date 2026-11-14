import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class TemporalResNetLSTM(nn.Module):

    def __init__(
        self,
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
        resnet_checkpoint=None
    ):
        super().__init__()

        # -------------------------------------------------
        # ResNet18 spatial feature extractor
        # -------------------------------------------------

        self.resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        if resnet_checkpoint and torch.cuda.is_available():
            try:
                ckpt = torch.load(resnet_checkpoint, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                cleaned = {
                    k.replace("model.", ""): v
                    for k, v in state.items()
                    if not k.startswith("fc.") and not k.startswith("model.fc.")
                }
                self.resnet.load_state_dict(cleaned, strict=False)
                print(f"[TemporalResNetLSTM] Loaded fine-tuned visual backbone weights from {resnet_checkpoint}")
            except Exception as e:
                print(f"[TemporalResNetLSTM] Note: Could not load {resnet_checkpoint}: {e}")

        # Remove final classification layer.
        # ResNet18 output becomes 512-dimensional.
        self.resnet.fc = nn.Identity()

        # Freeze ResNet initially.
        for param in self.resnet.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # Temporal model
        # -------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout
        )

        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        # x shape:
        # [batch, sequence, channels, height, width]

        batch_size, sequence_length, channels, height, width = x.shape

        # Combine batch and time dimensions
        x = x.view(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        # Spatial features
        features = self.resnet(x)

        # Restore temporal dimension
        features = features.view(
            batch_size,
            sequence_length,
            512
        )

        # LSTM
        lstm_output, (hidden, cell) = self.lstm(
            features
        )

        # Last temporal state
        temporal_features = lstm_output[:, -1, :]

        # Binary classification logit
        output = self.classifier(
            temporal_features
        )

        return output


# ---------------------------------------------------------
# MODEL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TEMPORAL RESNET18 + LSTM TEST")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = TemporalResNetLSTM().to(device)

    # Dummy temporal batch
    test_input = torch.randn(
        2,
        5,
        3,
        224,
        224
    ).to(device)

    with torch.no_grad():

        output = model(
            test_input
        )

    print()
    print("Input shape:")
    print(test_input.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Expected:")
    print("Input  = [2, 5, 3, 224, 224]")
    print("Output = [2, 1]")

    print()
    print("Model parameters:")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total:",
        f"{total_params:,}"
    )

    print(
        "Trainable:",
        f"{trainable_params:,}"
    )

    print()
    print("=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)