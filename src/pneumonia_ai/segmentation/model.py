"""A compact configurable 2D U-Net for lung segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class DoubleConvolution(nn.Module):
    """Two convolution and ReLU operations used throughout the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the block."""

        return self.layers(inputs)


class UNet(nn.Module):
    """2D U-Net returning one raw logit channel per input pixel.

    ``base_channels=16`` and ``depth=3`` are deliberately small defaults for
    CPU development.  Skip features are resized before concatenation, so odd
    spatial dimensions remain valid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive integers.")
        if depth < 1:
            raise ValueError("depth must be at least one.")

        channel_widths = [base_channels * 2**level for level in range(depth)]
        self.encoder_blocks = nn.ModuleList()
        previous_channels = in_channels
        for channels in channel_widths:
            self.encoder_blocks.append(DoubleConvolution(previous_channels, channels))
            previous_channels = channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConvolution(previous_channels, previous_channels * 2)

        self.upconvolutions = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        decoder_channels = previous_channels * 2
        for skip_channels in reversed(channel_widths):
            self.upconvolutions.append(
                nn.ConvTranspose2d(decoder_channels, skip_channels, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConvolution(skip_channels * 2, skip_channels))
            decoder_channels = skip_channels
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return raw segmentation logits at exactly the input resolution."""

        if inputs.ndim != 4:
            raise ValueError("UNet inputs must have shape [B, C, H, W].")
        input_size = inputs.shape[-2:]
        skip_connections: list[Tensor] = []
        features = inputs
        for encoder in self.encoder_blocks:
            features = encoder(features)
            skip_connections.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)

        for upconvolution, decoder, skip in zip(
            self.upconvolutions, self.decoder_blocks, reversed(skip_connections)
        ):
            features = upconvolution(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = functional.interpolate(
                    features,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            features = decoder(torch.cat((skip, features), dim=1))
        logits = self.output(features)
        if logits.shape[-2:] != input_size:
            logits = functional.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
        return logits
