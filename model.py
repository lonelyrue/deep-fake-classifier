"""
Model definition: SE-ResNet18 for deepfake (real vs. StyleGAN) binary classification.

The final model is a ResNet-18 backbone augmented with Squeeze-and-Excitation (SE)
channel-attention blocks inserted into every residual block, plus a custom
regularised classification head. See README for the architecture rationale and
for the alternative (F3-Net, frequency-domain) approach explored during the project.
"""

import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Learns per-channel weights so the network can emphasise informative channels
    and suppress noisy ones — useful here because the dataset contains artificial
    artefacts, and GAN traces are not uniform across feature channels.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Squeeze: global average pooling -> a [B, C] descriptor per channel
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Excitation: bottleneck MLP producing a gate in [0, 1] per channel
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e.expand_as(x)  # rescale channels by learned weights


class SEBasicBlock(nn.Module):
    """Wraps a ResNet BasicBlock and applies SE attention to its output.

    The original ResNet block is left untouched; SE is applied after the
    standard residual forward pass.
    """

    def __init__(self, basic_block, channels):
        super().__init__()
        self.block = basic_block
        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.block(x)  # standard residual forward
        out = self.se(out)   # channel attention
        return out


class DeepfakeResNet18(nn.Module):
    """ResNet-18 + SE blocks + regularised classification head.

    Output: a single logit (use BCEWithLogitsLoss during training and a sigmoid
    at inference time).
    """

    def __init__(self, pretrained=True, dropout_rate=0.5):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)

        # Insert SE blocks into every residual block of all four stages.
        # Channel counts follow the standard ResNet-18 layout.
        se_channels = [64, 128, 256, 512]
        for layer_name, channels in zip(
            ["layer1", "layer2", "layer3", "layer4"], se_channels
        ):
            layer = getattr(self.resnet, layer_name)
            new_blocks = nn.Sequential(
                *[SEBasicBlock(block, channels) for block in layer]
            )
            setattr(self.resnet, layer_name, new_blocks)

        # Replace the final FC layer with a regularised head (dropout + MLP).
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    # Quick shape / parameter sanity check.
    model = DeepfakeResNet18(pretrained=False)
    dummy = torch.zeros(2, 3, 256, 256)
    out = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {out.shape}  (expected [2, 1])")
    print(f"Total parameters: {total:,}")
