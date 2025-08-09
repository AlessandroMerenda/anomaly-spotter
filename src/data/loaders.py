"""
Advanced DataLoader for MVTec AD dataset.
Supports both single-category and multi-category training/testing with ground truth masks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
import cv2
import numpy as np
from PIL import Image
import logging

# Import preprocessing and utilities
try:
    # Try relative imports first
    from .preprocessing import MVTecPreprocessor
    from ..utils.logging_utils import setup_logger, DataError
except ImportError:
    # Fallback to absolute imports
    try:
        from data.preprocessing import MVTecPreprocessor
        from utils.logging_utils import setup_logger, DataError
    except ImportError:
        # Final fallback - direct path imports
        from src.data.preprocessing import MVTecPreprocessor
        from src.utils.logging_utils import setup_logger, DataError

class MVTecDataset(Dataset):
    """
    Advanced Dataset for MVTec AD with comprehensive features.
    
    Features:
    - Single or multi-category support
    - Automatic normal/anomaly labeling
    - Ground truth mask loading
    - Robust error handling
    - Flexible preprocessing pipeline
    """
    
    def __init__(self, 
                 root_dir: Union[str, Path],
                 categories: Union[str, List[str]] = None,
                 split: str = 'train',
                 preprocessor: Optional[MVTecPreprocessor] = None,
                 load_masks: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize MVTec dataset.
        
        Args:
            root_dir: Path to MVTec AD root directory
            categories: Single category string or list of categories. If None, auto-detect.
            split: 'train' or 'test'
            preprocessor: MVTecPreprocessor instance for image processing
            load_masks: Whether to load ground truth masks (test only)
            logger: Logger instance
        """
        self.logger = logger or setup_logger("MVTecDataset")
        self.root_dir = Path(root_dir)
        self.split = split
        self.preprocessor = preprocessor
        self.load_masks = load_masks and split == 'test'  # Masks only for test
        
        # Handle categories
        if categories is None:
            self.categories = self._auto_detect_categories()
        elif isinstance(categories, str):
            self.categories = [categories]
        else:
            self.categories = categories
            
        # Initialize data structures
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.mask_paths: List[Optional[Path]] = []
        self.categories_list: List[str] = []  # Track which category each sample belongs to
        
        # Load dataset
        self._load_dataset()
        
        self.logger.info(f"MVTecDataset initialized:")
        self.logger.info(f"  Categories: {self.categories}")
        self.logger.info(f"  Split: {split}")
        self.logger.info(f"  Total samples: {len(self.image_paths)}")
        self.logger.info(f"  Load masks: {self.load_masks}")
    
    def _auto_detect_categories(self) -> List[str]:
        """Auto-detect available categories in root directory."""
        categories = []
        for item in self.root_dir.iterdir():
            if item.is_dir() and (item / 'train').exists():
                categories.append(item.name)
        
        if not categories:
            raise DataError(f"No valid categories found in {self.root_dir}")
        
        self.logger.info(f"Auto-detected categories: {categories}")
        return categories
    
    def _load_dataset(self):
        """Load all image paths and labels based on split and categories."""
        
        for category in self.categories:
            category_path = self.root_dir / category
            
            if not category_path.exists():
                self.logger.warning(f"Category path not found: {category_path}")
                continue
            
            if self.split == 'train':
                self._load_train_data(category)
            else:
                self._load_test_data(category)
    
    def _load_train_data(self, category: str):
        """Load training data (only normal images)."""
        good_dir = self.root_dir / category / 'train' / 'good'
        
        if not good_dir.exists():
            self.logger.warning(f"Train good directory not found: {good_dir}")
            return
        
        # Load all normal images
        for img_path in good_dir.glob('*.png'):
            self.image_paths.append(img_path)
            self.labels.append(0)  # 0 = normal
            self.mask_paths.append(None)  # No masks for training
            self.categories_list.append(category)
        
        self.logger.debug(f"Loaded {len(list(good_dir.glob('*.png')))} training images for {category}")
    
    def _load_test_data(self, category: str):
        """Load test data (both normal and anomalous images)."""
        test_dir = self.root_dir / category / 'test'
        
        if not test_dir.exists():
            self.logger.warning(f"Test directory not found: {test_dir}")
            return
        
        # Load images from each defect type
        for defect_dir in test_dir.iterdir():
            if not defect_dir.is_dir():
                continue
                
            is_anomaly = defect_dir.name != 'good'
            
            for img_path in defect_dir.glob('*.png'):
                self.image_paths.append(img_path)
                self.labels.append(1 if is_anomaly else 0)
                self.categories_list.append(category)
                
                # Handle mask loading for anomalies
                if self.load_masks and is_anomaly:
                    mask_path = (self.root_dir / category / 'ground_truth' / 
                               defect_dir.name / f'{img_path.stem}_mask.png')
                    self.mask_paths.append(mask_path if mask_path.exists() else None)
                else:
                    self.mask_paths.append(None)
            
            self.logger.debug(f"Loaded {len(list(defect_dir.glob('*.png')))} {defect_dir.name} images for {category}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - image: Preprocessed image tensor [C, H, W]
            - label: 0 for normal, 1 for anomaly  
            - mask: Ground truth mask tensor (if available) or dummy mask
            - path: Original image path string
            - category: Category name string
            
        Raises:
            RuntimeError: If sample loading fails critically
            IndexError: If idx is out of bounds
        """
        # Validate index
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Sample index {idx} out of bounds [0, {len(self.image_paths)})")
            
        image_path = self.image_paths[idx]
        
        try:
            # Load image with detailed error context
            image = self._load_image(image_path)
            if image is None:
                raise DataError(f"Failed to load image: {image_path}")
            
            # Validate loaded image
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise DataError(f"Invalid image data loaded from: {image_path}")
                
            if len(image.shape) != 3 or image.shape[2] not in [1, 3, 4]:
                raise DataError(f"Invalid image shape {image.shape} from: {image_path}")
            
            # Load mask if available
            mask = None
            mask_path = self.mask_paths[idx] if idx < len(self.mask_paths) else None
            
            if mask_path is not None:
                mask = self._load_mask(mask_path)
                if mask is not None and (mask.size == 0 or len(mask.shape) not in [2, 3]):
                    self.logger.warning(f"Invalid mask shape {mask.shape} from: {mask_path}, ignoring")
                    mask = None
            
            # Apply preprocessing
            processed_image = self._apply_preprocessing(image, mask, image_path)
            if processed_image is None:
                raise DataError(f"Preprocessing failed for: {image_path}")
                
            # Handle mask processing
            processed_mask = self._process_mask(mask, processed_image.shape[-2:] if hasattr(processed_image, 'shape') else None)
            
            return {
                'image': processed_image,
                'label': self.labels[idx],
                'mask': processed_mask,
                'path': str(image_path),
                'category': self.categories_list[idx] if idx < len(self.categories_list) else 'unknown'
            }
            
        except (DataError, ValueError, RuntimeError) as e:
            # Expected errors that should fail fast
            self.logger.error(f"Data loading error for sample {idx} ({image_path}): {e}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}") from e
            
        except Exception as e:
            # Unexpected errors - log details and fail
            self.logger.error(f"Unexpected error loading sample {idx} ({image_path}): {type(e).__name__}: {e}")
            self.logger.error(f"Category: {self.categories_list[idx] if idx < len(self.categories_list) else 'unknown'}")
            self.logger.error(f"Split: {self.split}")
            raise RuntimeError(f"Unexpected error loading sample {idx}: {e}") from e
    
    def _apply_preprocessing(self, image: np.ndarray, mask: np.ndarray, image_path: Path) -> torch.Tensor:
        """Apply preprocessing to image and mask."""
        if self.preprocessor is not None:
            is_training = self.split == 'train'
            
            if mask is not None:
                # Use albumentations transforms that support masks
                try:
                    transform = self.preprocessor.get_transform(is_training)
                    transformed = transform(image=image, mask=mask)
                    return transformed['image']
                except Exception as e:
                    self.logger.warning(f"Mask preprocessing failed for {image_path}: {e}, trying image-only")
                    # Fall through to image-only preprocessing
            
            # Standard image preprocessing (image-only)
            processed = self.preprocessor.preprocess_single(image_path, is_training=is_training)
            if processed is None:
                raise DataError(f"Image preprocessing failed: {image_path}")
            return processed
        else:
            # Convert to tensor if no preprocessor
            if not isinstance(image, np.ndarray):
                raise DataError(f"Expected numpy array, got {type(image)}")
            
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                return image_tensor
            else:
                raise DataError(f"Unsupported image format: {image.shape}")
    
    def _process_mask(self, mask: Optional[np.ndarray], target_size: Optional[tuple] = None) -> torch.Tensor:
        """Process mask to tensor format."""
        if mask is None:
            # Return dummy mask - single pixel to minimize memory
            return torch.zeros(1, 1, 1, dtype=torch.float32)
        
        try:
            # Convert to tensor
            if isinstance(mask, np.ndarray):
                if len(mask.shape) == 2:
                    mask_tensor = torch.from_numpy(mask).float()
                elif len(mask.shape) == 3 and mask.shape[2] == 1:
                    mask_tensor = torch.from_numpy(mask.squeeze(-1)).float()
                else:
                    self.logger.warning(f"Unexpected mask shape: {mask.shape}, creating dummy mask")
                    return torch.zeros(1, 1, 1, dtype=torch.float32)
                
                # Normalize to [0, 1] if needed
                if mask_tensor.max() > 1.0:
                    mask_tensor = mask_tensor / 255.0
                
                return mask_tensor
            else:
                self.logger.warning(f"Unexpected mask type: {type(mask)}, creating dummy mask")
                return torch.zeros(1, 1, 1, dtype=torch.float32)
                
        except Exception as e:
            self.logger.warning(f"Mask processing failed: {e}, creating dummy mask")
            return torch.zeros(1, 1, 1, dtype=torch.float32)
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image with fallback methods."""
        try:
            # Try OpenCV first
            image = cv2.imread(str(image_path))
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Fallback to PIL
            with Image.open(image_path) as pil_image:
                return np.array(pil_image.convert('RGB'))
                
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """Load ground truth mask."""
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return (mask > 0).astype(np.float32)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load mask {mask_path}: {e}")
            return None
    
    def get_category_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics by category and split."""
        stats = {}
        for category in self.categories:
            category_indices = [i for i, cat in enumerate(self.categories_list) if cat == category]
            category_labels = [self.labels[i] for i in category_indices]
            
            stats[category] = {
                'total': len(category_indices),
                'normal': category_labels.count(0),
                'anomaly': category_labels.count(1)
            }
        
        return stats


def create_dataloaders(root_dir: Union[str, Path],
                      categories: Union[str, List[str]] = None,
                      preprocessor: Optional[MVTecPreprocessor] = None,
                      batch_size: int = 32,
                      num_workers: int = 2,
                      load_masks: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        root_dir: Path to MVTec AD root directory
        categories: Categories to include
        preprocessor: Preprocessing pipeline
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        load_masks: Whether to load ground truth masks for test set
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = MVTecDataset(
        root_dir=root_dir,
        categories=categories,
        split='train',
        preprocessor=preprocessor,
        load_masks=False  # No masks for training
    )
    
    test_dataset = MVTecDataset(
        root_dir=root_dir,
        categories=categories,
        split='test',
        preprocessor=preprocessor,
        load_masks=load_masks
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader
