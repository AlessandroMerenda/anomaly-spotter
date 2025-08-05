"""
Advanced Loss Functions for Autoencoder Training.
Combines multiple loss types for better reconstruction and anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional
import logging

try:
    from pytorch_msssim import SSIM, MS_SSIM
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not available. SSIM loss will use custom implementation.")

from src.core.model_config import AutoencoderConfig
from src.utils.logging_utils import setup_logger


class SSIMCustom(nn.Module):
    """
    Custom SSIM implementation as fallback when pytorch_msssim is not available.
    """
    def __init__(self, window_size=11, channel=3):
        super(SSIMCustom, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, height, width) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class EdgePreservationLoss(nn.Module):
    """
    Edge preservation loss using Sobel operators.
    Useful for maintaining fine details in reconstructions.
    """
    def __init__(self):
        super(EdgePreservationLoss, self).__init__()
        
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Register as buffers (non-trainable parameters)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _get_edges(self, x):
        """Extract edges from image using Sobel operators."""
        # Convert to grayscale if needed
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Apply Sobel operators
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Combine edges
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        return edges
    
    def forward(self, pred, target):
        """Compute edge preservation loss."""
        pred_edges = self._get_edges(pred)
        target_edges = self._get_edges(target)
        
        return F.mse_loss(pred_edges, target_edges)


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    Useful when training on datasets with varying anomaly frequencies.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """Compute focal loss."""
        ce_loss = F.mse_loss(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Advanced combined loss for autoencoder training.
    
    Features:
    - Multiple reconstruction losses (L1, L2, Focal)
    - Structural similarity (SSIM/MS-SSIM)
    - Perceptual loss with VGG features
    - Edge preservation loss
    - Flexible weighting based on configuration
    - Automatic loss scaling and normalization
    """
    
    def __init__(self, config: AutoencoderConfig, logger: Optional[logging.Logger] = None):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.logger = logger or setup_logger("CombinedLoss")
        
        # Initialize reconstruction losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
        # Initialize SSIM loss
        self._init_ssim_loss()
        
        # Initialize perceptual loss
        self._init_perceptual_loss()
        
        # Initialize edge preservation loss
        if config.edge_weight > 0:
            self.edge_loss = EdgePreservationLoss()
        
        # Loss scaling factors (learned during training)
        self.register_buffer('loss_scales', torch.ones(5))  # For automatic loss balancing
        
        self.logger.info(f"CombinedLoss initialized:")
        self.logger.info(f"  Loss type: {config.loss_type}")
        self.logger.info(f"  Reconstruction weight: {config.reconstruction_weight}")
        self.logger.info(f"  SSIM weight: {config.ssim_weight}")
        self.logger.info(f"  Perceptual weight: {config.perceptual_weight}")
        self.logger.info(f"  Edge weight: {config.edge_weight}")
    
    def _init_ssim_loss(self):
        """Initialize SSIM loss function."""
        if SSIM_AVAILABLE:
            self.ssim = SSIM(
                data_range=1.0 if self.config.normalize_type == "tanh" else 2.0,
                size_average=True,
                channel=3,
                win_size=11
            )
            self.ms_ssim = MS_SSIM(
                data_range=1.0 if self.config.normalize_type == "tanh" else 2.0,
                size_average=True,
                channel=3,
                win_size=11
            )
            self.logger.info("Using pytorch_msssim for SSIM loss")
        else:
            self.ssim = SSIMCustom(window_size=11, channel=3)
            self.ms_ssim = None
            self.logger.info("Using custom SSIM implementation")
    
    def _init_perceptual_loss(self):
        """Initialize perceptual loss network."""
        if self.config.perceptual_weight > 0:
            # Use VGG16 features for perceptual loss
            vgg = models.vgg16(pretrained=True)
            self.perceptual_net = vgg.features[:16]  # Up to relu3_3
            self.perceptual_net.eval()
            
            # Freeze parameters
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            
            # Normalization for VGG (ImageNet stats)
            self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
            self.logger.info("Perceptual loss initialized with VGG16")
    
    def _normalize_for_vgg(self, x):
        """Normalize images for VGG network."""
        # Convert from [-1, 1] to [0, 1] if using tanh normalization
        if self.config.normalize_type == "tanh":
            x = (x + 1) / 2
        
        # Apply ImageNet normalization
        x = (x - self.vgg_mean) / self.vgg_std
        return x
    
    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various reconstruction losses."""
        losses = {}
        
        if self.config.loss_type in ["mse", "l2", "combined"]:
            losses['l2'] = self.l2_loss(pred, target)
        
        if self.config.loss_type in ["l1", "combined"]:
            losses['l1'] = self.l1_loss(pred, target)
        
        if self.config.loss_type == "focal":
            losses['focal'] = self.focal_loss(pred, target)
        
        # Combined MSE + L1 with specific weights
        if self.config.loss_type == "combined":
            losses['reconstruction'] = (
                self.config.mse_weight * losses['l2'] + 
                self.config.l1_weight * losses['l1']
            )
        elif self.config.loss_type == "mse":
            losses['reconstruction'] = losses['l2']
        elif self.config.loss_type == "l1":
            losses['reconstruction'] = losses['l1']
        elif self.config.loss_type == "focal":
            losses['reconstruction'] = losses['focal']
        else:
            # Default to MSE
            losses['reconstruction'] = self.l2_loss(pred, target)
        
        return losses
    
    def _compute_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss."""
        try:
            ssim_value = self.ssim(pred, target)
            # Convert similarity to loss (1 - SSIM)
            return 1 - ssim_value
        except Exception as e:
            self.logger.warning(f"SSIM computation failed: {e}. Using fallback.")
            # Fallback to simple pixel loss
            return self.l2_loss(pred, target)
    
    def _compute_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features."""
        if self.config.perceptual_weight == 0:
            return torch.tensor(0.0, device=pred.device)
        
        try:
            # Normalize inputs for VGG
            pred_norm = self._normalize_for_vgg(pred)
            target_norm = self._normalize_for_vgg(target)
            
            # Extract features
            pred_features = self.perceptual_net(pred_norm)
            target_features = self.perceptual_net(target_norm)
            
            # Compute feature loss
            return F.mse_loss(pred_features, target_features)
        
        except Exception as e:
            self.logger.warning(f"Perceptual loss computation failed: {e}")
            return torch.tensor(0.0, device=pred.device)
    
    def _compute_edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge preservation loss."""
        if self.config.edge_weight == 0:
            return torch.tensor(0.0, device=pred.device)
        
        try:
            return self.edge_loss(pred, target)
        except Exception as e:
            self.logger.warning(f"Edge loss computation failed: {e}")
            return torch.tensor(0.0, device=pred.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        
        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        losses = {}
        
        # Reconstruction losses
        recon_losses = self._compute_reconstruction_loss(pred, target)
        losses.update(recon_losses)
        
        # SSIM loss
        if self.config.ssim_weight > 0:
            losses['ssim'] = self._compute_ssim_loss(pred, target)
        else:
            losses['ssim'] = torch.tensor(0.0, device=pred.device)
        
        # Perceptual loss
        losses['perceptual'] = self._compute_perceptual_loss(pred, target)
        
        # Edge preservation loss
        losses['edge'] = self._compute_edge_loss(pred, target)
        
        # Combine all losses
        total_loss = (
            self.config.reconstruction_weight * losses['reconstruction'] +
            self.config.ssim_weight * losses['ssim'] +
            self.config.perceptual_weight * losses['perceptual'] +
            self.config.edge_weight * losses['edge']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights for logging."""
        return {
            'reconstruction': self.config.reconstruction_weight,
            'ssim': self.config.ssim_weight,
            'perceptual': self.config.perceptual_weight,
            'edge': self.config.edge_weight
        }


class AnomalyDetectionLoss(nn.Module):
    """
    Specialized loss for anomaly detection training.
    Focuses on reconstruction error as anomaly indicator.
    """
    
    def __init__(self, config: AutoencoderConfig):
        super(AnomalyDetectionLoss, self).__init__()
        self.config = config
        self.base_loss = CombinedLoss(config)
        
        # For anomaly-aware training
        self.register_buffer('normal_reconstruction_stats', torch.zeros(2))  # mean, std
        self.stats_initialized = False
    
    def update_normal_stats(self, reconstruction_errors: torch.Tensor):
        """Update statistics of normal reconstruction errors."""
        if reconstruction_errors.numel() > 0:
            mean_error = reconstruction_errors.mean()
            std_error = reconstruction_errors.std()
            
            if not self.stats_initialized:
                self.normal_reconstruction_stats[0] = mean_error
                self.normal_reconstruction_stats[1] = std_error
                self.stats_initialized = True
            else:
                # Exponential moving average
                alpha = 0.1
                self.normal_reconstruction_stats[0] = (
                    alpha * mean_error + (1 - alpha) * self.normal_reconstruction_stats[0]
                )
                self.normal_reconstruction_stats[1] = (
                    alpha * std_error + (1 - alpha) * self.normal_reconstruction_stats[1]
                )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute anomaly detection loss.
        
        Args:
            pred: Predicted images
            target: Target images  
            labels: Binary labels (0=normal, 1=anomaly) - optional
        
        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        # Compute base reconstruction loss
        total_loss, losses = self.base_loss(pred, target)
        
        # Update normal statistics if labels provided
        if labels is not None and self.training:
            normal_mask = (labels == 0)
            if normal_mask.any():
                normal_errors = F.mse_loss(pred[normal_mask], target[normal_mask], reduction='none')
                normal_errors = normal_errors.view(normal_errors.size(0), -1).mean(dim=1)
                self.update_normal_stats(normal_errors)
        
        return total_loss, losses
    
    def get_anomaly_scores(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error.
        
        Args:
            pred: Predicted images
            target: Target images
            
        Returns:
            Anomaly scores per sample
        """
        with torch.no_grad():
            # Compute pixel-wise reconstruction error
            pixel_errors = F.mse_loss(pred, target, reduction='none')
            
            # Average over spatial dimensions
            sample_errors = pixel_errors.view(pixel_errors.size(0), -1).mean(dim=1)
            
            # Normalize by normal statistics if available
            if self.stats_initialized:
                mean_normal = self.normal_reconstruction_stats[0]
                std_normal = self.normal_reconstruction_stats[1]
                normalized_scores = (sample_errors - mean_normal) / (std_normal + 1e-8)
                return normalized_scores
            else:
                return sample_errors


def create_loss_function(config: AutoencoderConfig) -> nn.Module:
    """
    Factory function to create appropriate loss function.
    
    Args:
        config: Model configuration
        
    Returns:
        Loss function instance
    """
    if hasattr(config, 'use_anomaly_loss') and config.use_anomaly_loss:
        return AnomalyDetectionLoss(config)
    else:
        return CombinedLoss(config)
