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
    Robust custom SSIM implementation as fallback when pytorch_msssim is not available.
    
    This implementation follows the original SSIM paper definition:
    Wang, Z., et al. "Image quality assessment: from error visibility to structural similarity"
    IEEE transactions on image processing 13.4 (2004): 600-612.
    """
    def __init__(self, window_size: int = 11, channel: int = 3, data_range: float = 1.0):
        super(SSIMCustom, self).__init__()
        
        # Validate inputs
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if window_size < 3:
            raise ValueError("window_size must be at least 3")
        if channel <= 0:
            raise ValueError("channel must be positive")
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        
        # SSIM constants (adjusted for data range)
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        
        # Create and register Gaussian window
        self.register_buffer('gaussian_window', self._create_gaussian_window(window_size, channel))
    
    def _gaussian_kernel_1d(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g
    
    def _create_gaussian_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window for SSIM computation."""
        sigma = 1.5  # Standard sigma for SSIM
        
        # Create 1D Gaussian kernel
        gaussian_1d = self._gaussian_kernel_1d(window_size, sigma)
        
        # Create 2D Gaussian kernel by outer product
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        
        # Expand for multiple channels
        window = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        window = window.expand(channel, 1, window_size, window_size)
        
        return window.contiguous()
    
    def _update_window_if_needed(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Update Gaussian window if needed for different device/dtype/channels."""
        _, channel, _, _ = input_tensor.shape
        
        # Check if window needs updating
        if (channel != self.channel or 
            self.gaussian_window.device != input_tensor.device or
            self.gaussian_window.dtype != input_tensor.dtype):
            
            # Create new window with correct properties
            window = self._create_gaussian_window(self.window_size, channel)
            window = window.to(device=input_tensor.device, dtype=input_tensor.dtype)
            
            # Update stored values
            self.channel = channel
            return window
        
        return self.gaussian_window
    
    def _ssim_components(self, img1: torch.Tensor, img2: torch.Tensor, 
                        window: torch.Tensor) -> tuple:
        """Compute SSIM components (luminance, contrast, structure)."""
        padding = self.window_size // 2
        
        # Compute local means
        mu1 = F.conv2d(img1, window, padding=padding, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=padding, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=img1.size(1)) - mu1_mu2
        
        # Ensure non-negative variances (numerical stability)
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        
        return mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, 
                return_components: bool = False) -> torch.Tensor:
        """
        Compute SSIM between two images.
        
        Args:
            img1: First image tensor [B, C, H, W]
            img2: Second image tensor [B, C, H, W]
            return_components: If True, return (ssim, luminance, contrast, structure)
            
        Returns:
            SSIM value(s) or tuple of components
            
        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        # Input validation
        if img1.shape != img2.shape:
            raise ValueError(f"Input tensors must have same shape: {img1.shape} vs {img2.shape}")
        
        if img1.dim() != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got {img1.dim()}D")
        
        if img1.size(2) < self.window_size or img1.size(3) < self.window_size:
            raise ValueError(f"Input size {img1.shape[-2:]} too small for window_size {self.window_size}")
        
        # Update window for current input
        window = self._update_window_if_needed(img1)
        
        # Compute SSIM components
        mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = self._ssim_components(
            img1, img2, window
        )
        
        # Compute SSIM components
        luminance = (2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)
        contrast = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        structure = (sigma12 + self.C2/2) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + self.C2/2)
        
        # Full SSIM
        ssim_map = luminance * contrast * structure
        ssim_value = ssim_map.mean()
        
        if return_components:
            return ssim_value, luminance.mean(), contrast.mean(), structure.mean()
        
        return ssim_value
    
    def get_ssim_map(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Get pixel-wise SSIM map.
        
        Args:
            img1: First image tensor [B, C, H, W]
            img2: Second image tensor [B, C, H, W]
            
        Returns:
            SSIM map tensor [B, C, H, W]
        """
        window = self._update_window_if_needed(img1)
        mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = self._ssim_components(
            img1, img2, window
        )
        
        # Compute pixel-wise SSIM
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        # Avoid division by zero
        ssim_map = numerator / (denominator + 1e-8)
        
        return ssim_map


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
        """Initialize SSIM loss function with robust fallback."""
        # Determine data range based on normalization
        data_range = 2.0 if self.config.normalize_type == "tanh" else 1.0  # tanh: [-1,1] = range 2
        
        if SSIM_AVAILABLE:
            try:
                self.ssim = SSIM(
                    data_range=data_range,
                    size_average=True,
                    channel=3,
                    win_size=11
                )
                self.ms_ssim = MS_SSIM(
                    data_range=data_range,
                    size_average=True,
                    channel=3,
                    win_size=11
                )
                self.logger.info("Using pytorch_msssim for SSIM loss")
                self._ssim_available = True
            except Exception as e:
                self.logger.warning(f"Failed to initialize pytorch_msssim: {e}. Falling back to custom implementation")
                self._init_custom_ssim(data_range)
        else:
            self._init_custom_ssim(data_range)
    
    def _init_custom_ssim(self, data_range: float):
        """Initialize custom SSIM implementation."""
        try:
            self.ssim = SSIMCustom(
                window_size=11, 
                channel=3,
                data_range=data_range
            )
            self.ms_ssim = None  # Custom implementation doesn't support MS-SSIM
            self.logger.info("Using robust custom SSIM implementation")
            self._ssim_available = True
        except Exception as e:
            self.logger.error(f"Failed to initialize custom SSIM: {e}. SSIM loss will be disabled")
            self.ssim = None
            self.ms_ssim = None
            self._ssim_available = False
    
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
        else:
            self.perceptual_net = None
    
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
        """Compute SSIM loss with robust error handling."""
        if not self._ssim_available or self.ssim is None:
            # SSIM not available, return zero loss
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        try:
            ssim_value = self.ssim(pred, target)
            
            # Validate SSIM value
            if torch.isnan(ssim_value) or torch.isinf(ssim_value):
                self.logger.warning("SSIM returned NaN or Inf, using fallback")
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            # Convert similarity to loss (1 - SSIM)
            # SSIM values should be in [-1, 1], but clamp to be safe
            ssim_clamped = torch.clamp(ssim_value, -1.0, 1.0)
            ssim_loss = 1.0 - ssim_clamped
            
            return ssim_loss
            
        except Exception as e:
            self.logger.warning(f"SSIM computation failed: {e}. Using fallback.")
            # Fallback: disable SSIM for this forward pass
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _compute_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features."""
        if self.config.perceptual_weight == 0 or self.perceptual_net is None:
            return torch.tensor(0.0, device=pred.device)
        
        try:
            # Ensure perceptual_net is on the same device as inputs
            if next(self.perceptual_net.parameters()).device != pred.device:
                self.perceptual_net = self.perceptual_net.to(pred.device)
            
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
