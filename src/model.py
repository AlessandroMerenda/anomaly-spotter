import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderUNetLite(nn.Module):
    """
    Autoencoder con skip connections (stile U-Net), progettato per immagini 128x128 RGB.
    Input: Tensor (B, 3, 128, 128) normalizzato in [-1, 1]
    Output: Tensor ricostruito di stessa forma, con Tanh in uscita

    Note:
    - Utilizza ConvTranspose2d per upsampling (migliore qualità)
    - Doppi layer Conv2d per migliore feature extraction
    - Skip connections per preservare i dettagli locali
    """

    def __init__(self):
        super(AutoencoderUNetLite, self).__init__()
        
        # --- ENCODER ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 → 64
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 → 32
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 → 16
        
        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # --- DECODER ---
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 16 → 32
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 = 256 + 256 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 32 → 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 64 → 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output finale
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # --- Encoder path ---
        enc1 = self.enc1(x)          # (B, 64, 128, 128)
        pool1 = self.pool1(enc1)     # (B, 64, 64, 64)
        
        enc2 = self.enc2(pool1)      # (B, 128, 64, 64)
        pool2 = self.pool2(enc2)     # (B, 128, 32, 32)
        
        enc3 = self.enc3(pool2)      # (B, 256, 32, 32)
        pool3 = self.pool3(enc3)     # (B, 256, 16, 16)
        
        # --- Bottleneck ---
        bottleneck = self.bottleneck(pool3)  # (B, 512, 16, 16)
        
        # --- Decoder path with skip connections ---
        up3 = self.upconv3(bottleneck)              # (B, 256, 32, 32)
        merge3 = torch.cat([up3, enc3], dim=1)      # (B, 512, 32, 32)
        dec3 = self.dec3(merge3)                    # (B, 256, 32, 32)
        
        up2 = self.upconv2(dec3)                    # (B, 128, 64, 64)
        merge2 = torch.cat([up2, enc2], dim=1)      # (B, 256, 64, 64)
        dec2 = self.dec2(merge2)                    # (B, 128, 64, 64)
        
        up1 = self.upconv1(dec2)                    # (B, 64, 128, 128)
        merge1 = torch.cat([up1, enc1], dim=1)      # (B, 128, 128, 128)
        dec1 = self.dec1(merge1)                    # (B, 64, 128, 128)
        
        # Output finale
        output = self.final(dec1)                   # (B, 3, 128, 128)
        return self.tanh(output)
