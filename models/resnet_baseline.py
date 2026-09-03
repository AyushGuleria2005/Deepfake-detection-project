import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class ResNetBaseline(nn.Module):

    def __init__(self):
        super().__init__()

        # Load ImageNet-pretrained ResNet18
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Replace final classifier
        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            1
        )

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        """
        Extract the 512-dimensional feature vector
        immediately before the final classification layer.
        """

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        return x


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ResNetBaseline().to(device)

    test_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    output = model(
        test_input
    )

    features = model.extract_features(
        test_input
    )

    print("=" * 50)
    print("ResNet18 Baseline Test")
    print("=" * 50)

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Input shape:",
        test_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Feature shape:",
        features.shape
    )

    print(
        "Feature dimension:",
        features.shape[1]
    )

    print(
        "Model loaded successfully!"
    )