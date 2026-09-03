import torch
import torch.nn as nn


class FusionBaseline(nn.Module):
    """
    Simple multimodal fusion baseline.

    Inputs:
        visual_prob: [batch, 1]
        audio_prob:  [batch, 1]

    Output:
        final_logit: [batch, 1]
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, visual_prob, audio_prob):

        x = torch.cat(
            [visual_prob, audio_prob],
            dim=1
        )

        return self.network(x)


if __name__ == "__main__":

    model = FusionBaseline()

    visual = torch.rand(4, 1)
    audio = torch.rand(4, 1)

    output = model(
        visual,
        audio
    )

    print("Input visual:", visual.shape)
    print("Input audio:", audio.shape)
    print("Output:", output.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("PASS")