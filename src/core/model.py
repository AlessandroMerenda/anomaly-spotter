import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration constants
CONV_KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
UPSAMPLE_KERNEL = 2
UPSAMPLE_STRIDE = 2

class AutoencoderUNetLite(nn.Module):
    """
    Autoencoder con skip connections (stile U-Net), progettato per immagini 128x128 RGB.
    
    Input: Tensor (B, 3, 128, 128) normalizzato in [-1, 1] 
    Output: Tensor ricostruito (B, 3, 128, 128) in [-1, 1] con Tanh
    
    Architecture:
    - Encoder: 3 downsampling blocks (128→64→32→16)
    - Bottleneck: 512 channels at 16x16
    - Decoder: 3 upsampling blocks with skip connections (16→32→64→128)
    
    Note:
    - Utilizza ConvTranspose2d per upsampling (migliore qualità)
    - Doppi layer Conv2d per migliore feature extraction  
    - Skip connections per preservare i dettagli spaziali locali
    - Output range garantito [-1, 1] tramite Tanh
    """

    def __init__(self, input_channels: int = 3, output_channels: int = 3):
        """
        Initialize AutoencoderUNetLite.
        
        Args:
            input_channels: Number of input channels (default: 3 for RGB)
            output_channels: Number of output channels (default: 3 for RGB)
        """
        super(AutoencoderUNetLite, self).__init__()
        
        # Store configuration
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Validate input
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        
        # --- ENCODER ---
        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)  # 128 → 64
        
        self.encoder_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)  # 64 → 32
        
        self.encoder_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)  # 32 → 16
        
        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # --- DECODER ---
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=UPSAMPLE_KERNEL, stride=UPSAMPLE_STRIDE)  # 16 → 32
        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=CONV_KERNEL_SIZE, padding=1),  # 512 = 256 + 256 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=UPSAMPLE_KERNEL, stride=UPSAMPLE_STRIDE)  # 32 → 64
        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=CONV_KERNEL_SIZE, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=UPSAMPLE_KERNEL, stride=UPSAMPLE_STRIDE)  # 64 → 128
        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=CONV_KERNEL_SIZE, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output layer - maps to correct number of output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, padding=0)
        self.output_activation = nn.Tanh()  # Ensures output in [-1, 1] range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass attraverso U-Net autoencoder.
        
        Args:
            x: Input tensor [B, C, H, W], normalizzato in [-1, 1]
            
        Returns:
            Ricostruzione tensor [B, C, H, W] in [-1, 1]
            
        Raises:
            RuntimeError: Se le dimensioni input non sono supportate
            
        Note:
            - Skip connections preservano dettagli spaziali durante upsampling
            - Bottleneck a 512 canali cattura rappresentazioni ad alto livello
            - Output garantito in range [-1, 1] tramite Tanh activation
        """
        # Input validation
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D input tensor [B,C,H,W], got {x.dim()}D")
        
        if x.size(1) != self.input_channels:
            raise RuntimeError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        
        # --- ENCODER PATH ---
        # Block 1: 128x128 -> 64x64
        encoder_features_1 = self.encoder_block_1(x)        # (B, 64, 128, 128)
        pooled_1 = self.pool1(encoder_features_1)           # (B, 64, 64, 64)
        
        # Block 2: 64x64 -> 32x32  
        encoder_features_2 = self.encoder_block_2(pooled_1) # (B, 128, 64, 64)
        pooled_2 = self.pool2(encoder_features_2)           # (B, 128, 32, 32)
        
        # Block 3: 32x32 -> 16x16
        encoder_features_3 = self.encoder_block_3(pooled_2) # (B, 256, 32, 32)
        pooled_3 = self.pool3(encoder_features_3)           # (B, 256, 16, 16)
        
        # --- BOTTLENECK ---
        # Feature extraction at lowest spatial resolution
        bottleneck_features = self.bottleneck(pooled_3)     # (B, 512, 16, 16)
        
        # --- DECODER PATH WITH SKIP CONNECTIONS ---
        # Block 3: 16x16 -> 32x32
        upsampled_3 = self.upconv3(bottleneck_features)                    # (B, 256, 32, 32)
        skip_connection_3 = torch.cat([upsampled_3, encoder_features_3], dim=1)  # (B, 512, 32, 32)
        decoder_features_3 = self.decoder_block_3(skip_connection_3)       # (B, 256, 32, 32)
        
        # Block 2: 32x32 -> 64x64
        upsampled_2 = self.upconv2(decoder_features_3)                     # (B, 128, 64, 64)
        skip_connection_2 = torch.cat([upsampled_2, encoder_features_2], dim=1)  # (B, 256, 64, 64)
        decoder_features_2 = self.decoder_block_2(skip_connection_2)       # (B, 128, 64, 64)
        
        # Block 1: 64x64 -> 128x128
        upsampled_1 = self.upconv1(decoder_features_2)                     # (B, 64, 128, 128)
        skip_connection_1 = torch.cat([upsampled_1, encoder_features_1], dim=1)  # (B, 128, 128, 128)
        decoder_features_1 = self.decoder_block_1(skip_connection_1)       # (B, 64, 128, 128)
        
        # --- OUTPUT LAYER ---
        # Map to output channels and apply activation
        final_features = self.final_conv(decoder_features_1)  # (B, output_channels, 128, 128)
        reconstruction = self.output_activation(final_features)  # Range: [-1, 1]
        
        return reconstruction
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'model_name': 'AutoencoderUNetLite',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'input_size': '128x128',
            'output_range': '[-1, 1]',
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
