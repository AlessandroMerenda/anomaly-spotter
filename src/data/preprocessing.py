"""
Advanced preprocessing pipeline for MVTec AD dataset.
Combines robust error handling with state-of-the-art augmentation techniques.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging
from PIL import Image

# Safe import for albumentations (handle compatibility issues)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
    print("✅ Albumentations imported successfully")
except ImportError as e:
    print(f"⚠️  Albumentations not available: {e}")
    print("Using fallback torchvision transforms")
    ALBUMENTATIONS_AVAILABLE = False
    # Fallback imports
    from torchvision import transforms as T
    # Create dummy A namespace for fallback
    class A:
        pass

# Import utilities from existing codebase
try:
    # Try relative imports first
    from ..utils.logging_utils import setup_logger, DataError, handle_exception
except ImportError:
    # Fallback to absolute imports
    try:
        from utils.logging_utils import setup_logger, DataError, handle_exception
    except ImportError:
        # Final fallback - direct path imports
        from src.utils.logging_utils import setup_logger, DataError, handle_exception

class MVTecPreprocessor:
    """
    Advanced preprocessor for MVTec AD dataset.
    
    Features:
    - Robust error handling and logging
    - State-of-the-art augmentation with Albumentations
    - Separate train/test pipelines
    - Batch processing capabilities
    - Multiple image format support
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (128, 128),  # Match notebook default
                 normalize: bool = True,
                 normalization_type: str = "tanh",  # "imagenet" or "tanh"
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or setup_logger("MVTecPreprocessor")
        self.image_size = image_size
        self.normalize = normalize
        self.normalization_type = normalization_type
        
        # Supported image extensions
        self.SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # Initialize transforms
        self._setup_transforms()
        
        self.logger.info(f"MVTecPreprocessor initialized:")
        self.logger.info(f"  Image size: {image_size}")
        self.logger.info(f"  Normalization: {normalize} ({normalization_type})")
    
    def _setup_transforms(self):
        """Setup augmentation pipelines."""
        
        if ALBUMENTATIONS_AVAILABLE:
            # Base transforms
            base_transforms = [
                A.Resize(height=self.image_size[0], width=self.image_size[1])
            ]
            
            # Training augmentations
            train_augmentations = [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3),
                ], p=0.2),
            ]
            
            # Normalization
            if self.normalize:
                if self.normalization_type == "imagenet":
                    norm_transform = A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    )
                elif self.normalization_type == "tanh":
                    # Match notebook normalization [-1, 1]
                    norm_transform = A.Normalize(
                        mean=[0.5, 0.5, 0.5], 
                        std=[0.5, 0.5, 0.5]
                    )
                else:
                    raise ValueError(f"Unknown normalization type: {self.normalization_type}")
            else:
                norm_transform = A.NoOp()
            
            # Training pipeline
            self.train_transform = A.Compose(
                base_transforms + train_augmentations + [norm_transform, ToTensorV2()]
            )
            
            # Test pipeline (no augmentation)
            self.test_transform = A.Compose(
                base_transforms + [norm_transform, ToTensorV2()]
            )
            
        else:
            # Fallback to torchvision transforms
            if self.normalize:
                if self.normalization_type == "imagenet":
                    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                elif self.normalization_type == "tanh":
                    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                else:
                    raise ValueError(f"Unknown normalization type: {self.normalization_type}")
            else:
                normalize = T.Lambda(lambda x: x)  # No-op
            
            # Training pipeline with basic augmentations
            self.train_transform = T.Compose([
                T.Resize(self.image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.ToTensor(),
                normalize
            ])
            
            # Test pipeline (no augmentation)
            self.test_transform = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                normalize
            ])
        
        self.logger.info("Transform pipelines initialized successfully")
    
    def _load_image_safely(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Safely load an image with error handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of image or None if failed
        """
        try:
            image_path = Path(image_path)
            
            # Validate file exists and has valid extension
            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                return None
                
            if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                self.logger.warning(f"Unsupported file extension: {image_path}")
                return None
            
            # Try OpenCV first (faster for most cases)
            image = cv2.imread(str(image_path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            
            # Fallback to PIL
            with Image.open(image_path) as pil_image:
                image = np.array(pil_image.convert('RGB'))
                return image
                
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def preprocess_single(self, 
                         image_path: Union[str, Path], 
                         is_training: bool = True) -> Optional[torch.Tensor]:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
            is_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image tensor or None if failed
        """
        image = self._load_image_safely(image_path)
        if image is None:
            return None
        
        try:
            transform = self.train_transform if is_training else self.test_transform
            
            if ALBUMENTATIONS_AVAILABLE:
                # Albumentations format
                transformed = transform(image=image)
                return transformed['image']
            else:
                # Torchvision format - needs PIL Image
                from PIL import Image
                pil_image = Image.fromarray(image)
                transformed = transform(pil_image)
                return transformed
        
        except Exception as e:
            self.logger.error(f"Failed to transform image {image_path}: {e}")
            return None
    
    def preprocess_batch(self, 
                        image_paths: List[Union[str, Path]], 
                        is_training: bool = True,
                        skip_failed: bool = True) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of paths to image files
            is_training: Whether to apply training augmentations
            skip_failed: Whether to skip failed images or raise error
            
        Returns:
            Batch tensor of preprocessed images
        """
        transform = self.train_transform if is_training else self.test_transform
        
        successful_images = []
        failed_count = 0
        
        for path in image_paths:
            image = self._load_image_safely(path)
            if image is None:
                failed_count += 1
                if not skip_failed:
                    raise DataError(f"Failed to load image: {path}")
                continue
            
            try:
                if ALBUMENTATIONS_AVAILABLE:
                    # Albumentations format
                    transformed = transform(image=image)
                    successful_images.append(transformed['image'])
                else:
                    # Torchvision format - needs PIL Image
                    from PIL import Image
                    pil_image = Image.fromarray(image)
                    transformed = transform(pil_image)
                    successful_images.append(transformed)
            except Exception as e:
                failed_count += 1
                self.logger.error(f"Failed to transform {path}: {e}")
                if not skip_failed:
                    raise DataError(f"Failed to transform image: {path}")
        
        if failed_count > 0:
            self.logger.warning(f"Failed to process {failed_count}/{len(image_paths)} images")
        
        if len(successful_images) == 0:
            raise DataError("No images successfully processed")
        
        return torch.stack(successful_images)
    
    def get_transform(self, is_training: bool = True):
        """Get the transform pipeline for external use."""
        if ALBUMENTATIONS_AVAILABLE:
            return self.train_transform if is_training else self.test_transform
        else:
            # Return torchvision transform
            return self.train_transform if is_training else self.test_transform
