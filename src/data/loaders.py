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
        
        Returns:
            Dictionary containing:
            - image: Preprocessed image tensor
            - label: 0 for normal, 1 for anomaly
            - mask: Ground truth mask (if available) or zeros
            - path: Original image path
            - category: Category name
        """
        try:
            # Load image
            image = self._load_image(self.image_paths[idx])
            if image is None:
                raise DataError(f"Failed to load image: {self.image_paths[idx]}")
            
            # Load mask if available
            mask = None
            if self.mask_paths[idx] is not None:
                mask = self._load_mask(self.mask_paths[idx])
            
            # Apply preprocessing
            if self.preprocessor is not None:
                is_training = self.split == 'train'
                
                if mask is not None:
                    # Use albumentations transforms that support masks
                    transform = self.preprocessor.get_transform(is_training)
                    transformed = transform(image=image, mask=mask)
                    image = transformed['image']
                    mask = transformed['mask']
                else:
                    # Standard image preprocessing
                    image = self.preprocessor.preprocess_single(
                        self.image_paths[idx], 
                        is_training=is_training
                    )
                    if image is None:
                        raise DataError(f"Preprocessing failed: {self.image_paths[idx]}")
            else:
                # Convert to tensor if no preprocessor
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                
                if mask is not None and isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float()
            
            # Ensure mask is tensor
            if mask is None:
                mask = torch.zeros(1, 1, 1)  # Dummy mask
            elif not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            
            return {
                'image': image,
                'label': self.labels[idx],
                'mask': mask,
                'path': str(self.image_paths[idx]),
                'category': self.categories_list[idx]
            }
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data to avoid training interruption
            dummy_size = 128 if self.preprocessor is None else self.preprocessor.image_size[0]
            return {
                'image': torch.zeros(3, dummy_size, dummy_size),
                'label': 0,
                'mask': torch.zeros(1, 1, 1),
                'path': str(self.image_paths[idx]),
                'category': self.categories_list[idx] if idx < len(self.categories_list) else 'unknown'
            }
    
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
