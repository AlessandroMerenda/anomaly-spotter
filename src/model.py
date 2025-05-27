import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderUNetLite(nn.Module):
    """
    Autoencoder con skip connections (stile U-Net leggero), progettato per immagini 128x128 RGB.
    Input: Tensor (B, 3, 128, 128) normalizzato in [-1, 1]
    Output: Tensor ricostruito di stessa forma, con Tanh in uscita

    Note:
    - Le dimensioni devono essere multipli di 8 (es. 128, 256)
    - Le skip connections aiutano a preservare i dettagli locali
    """

    def __init__(self):
        super(AutoencoderUNetLite, self).__init__()

        # --- ENCODER ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 128 → 64

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 → 32

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)  # 32 → 16

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 16 → 32
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),  # concat con e3
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 32 → 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # concat con e2
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 64 → 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # concat con e1
            nn.ReLU(inplace=True)
        )

        # Output layer con attivazione Tanh per immagini normalizzate in [-1, 1]
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)       # (B, 64, 128, 128)
        p1 = self.pool1(e1)     # (B, 64, 64, 64)

        e2 = self.enc2(p1)      # (B, 128, 64, 64)
        p2 = self.pool2(e2)     # (B, 128, 32, 32)

        e3 = self.enc3(p2)      # (B, 256, 32, 32)
        p3 = self.pool3(e3)     # (B, 256, 16, 16)

        # --- Bottleneck ---
        b = self.bottleneck(p3)  # (B, 512, 16, 16)

        # --- Decoder + skip connections ---
        up3 = self.up3(b)                              # (B, 512, 32, 32)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))    # (B, 256, 32, 32)

        up2 = self.up2(d3)                             # (B, 256, 64, 64)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))    # (B, 128, 64, 64)

        up1 = self.up1(d2)                             # (B, 128, 128, 128)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))    # (B, 64, 128, 128)

        out = self.final(d1)                           # (B, 3, 128, 128)
        return out
