import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# ✅ U-NET GENERATOR (Improved Autoencoder)
# =========================
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(512, 128)
        self.up1 = self.up_block(256, 64)

        # Final layer
        self.final = nn.Conv2d(128, 3, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # 128 → 64
        e2 = self.enc2(e1)  # 64 → 32
        e3 = self.enc3(e2)  # 32 → 16

        # Bottleneck
        b = self.bottleneck(e3)  # 16 → 8

        # Decoder
        d3 = self.up3(b)         # 8 → 16
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.up2(d3)        # 16 → 32
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.up1(d2)        # 32 → 64
        d1 = torch.cat([d1, e1], dim=1)

        # Final output
        out = self.final(d1)

        # ✅ IMPORTANT FIX: Resize output to match 128x128
        out = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)

        return torch.tanh(out)


# =========================
# ✅ DISCRIMINATOR
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)