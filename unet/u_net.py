import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, in_channels, nb_labels=2):
        super().__init__()

        def DoubleConvRelu(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU()
            )

        # left part of the U-Net (downsampling)
        self.down1 = DoubleConvRelu(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConvRelu(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.latent = DoubleConvRelu(128, 256)

        # right part of the U-Net (upsampling)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_block1 = DoubleConvRelu(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block2 = DoubleConvRelu(128, 64)

        # final output layer labeling each pixel
        self.final = nn.Conv2d(64, nb_labels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        x = self.latent(self.pool2(d2))

        x = self.up2(x)
        x = self.up_block1(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.up_block2(torch.cat([x, d1], dim=1))

        return self.final(x)