"""
Dataset classes for Ground Point Prediction.

Handles loading of images and annotations in YOLO-Pose format.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class GroundPointDataset(Dataset):
    """
    Dataset for Ground Point Prediction.
    
    Loads images with their bounding boxes and ground truth ground points.
    Annotations are expected in YOLO-Pose format:
        class_id x_center y_center width height keypoint_x keypoint_y [visibility]
    
    All coordinates are normalized (0-1).
    
    Attributes:
        samples: List of (image_path, annotations) tuples
        image_size: Target image size for resizing
        transform: Image transformation pipeline
    """
    
    def __init__(
        self, 
        data_dir: Path, 
        split: str, 
        image_size: int = 640,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset root (containing train/val/test folders)
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size
            transform: Optional transform pipeline (default: ImageNet normalization)
            augment: Whether to apply data augmentation (for training)
        """
        self.image_size = image_size
        self.augment = augment
        
        # Default transform: ImageNet normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load samples
        self.samples = self._load_samples(data_dir, split)
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {data_dir / split}")
    
    def _load_samples(
        self, 
        data_dir: Path, 
        split: str
    ) -> List[Tuple[Path, List[Tuple[Tuple, Tuple]]]]:
        """
        Load all samples from the dataset.
        
        Args:
            data_dir: Dataset root directory
            split: Dataset split
        
        Returns:
            List of (image_path, [(box_norm, ground_point_norm), ...]) tuples
        """
        samples = []
        
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        for label_file in sorted(labels_dir.glob("*.txt")):
            img_path = images_dir / f"{label_file.stem}.jpg"
            
            if not img_path.exists():
                continue
            
            # Parse annotations
            annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if len(parts) >= 7:
                        # YOLO-Pose format: class x_c y_c w h kp_x kp_y [vis]
                        box_norm = (
                            float(parts[1]),  # x_center
                            float(parts[2]),  # y_center
                            float(parts[3]),  # width
                            float(parts[4])   # height
                        )
                        ground_point = (
                            float(parts[5]),  # keypoint x
                            float(parts[6])   # keypoint y
                        )
                        
                        # Validate coordinates are in valid range
                        if (0 <= ground_point[0] <= 1 and 
                            0 <= ground_point[1] <= 1):
                            annotations.append((box_norm, ground_point))
            
            if annotations:
                samples.append((img_path, annotations))
        
        return samples
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.samples)
    
    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of:
                - image: Tensor (3, H, W)
                - boxes: Tensor (N, 4) in xyxy pixel format
                - ground_points: Tensor (N, 2) in normalized format
                - img_path: String path to original image
        """
        img_path, annotations = self.samples[idx]
        
        # Load and preprocess image
        image = cv2.imread(str(img_path))
        if image is None:
            # Handle corrupted images by returning next sample
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize to target size
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert annotations
        boxes_xyxy = []
        ground_points = []
        
        for box_norm, gp_norm in annotations:
            x_c, y_c, w, h = box_norm
            
            # Convert normalized xywh to xyxy in resized coordinates
            x1 = (x_c - w / 2) * self.image_size
            y1 = (y_c - h / 2) * self.image_size
            x2 = (x_c + w / 2) * self.image_size
            y2 = (y_c + h / 2) * self.image_size
            
            boxes_xyxy.append([x1, y1, x2, y2])
            ground_points.append([gp_norm[0], gp_norm[1]])  # Keep normalized
        
        # Convert to tensors
        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
        ground_points = torch.tensor(ground_points, dtype=torch.float32)
        
        # Apply image transform
        if self.transform:
            image = self.transform(image)
        
        return image, boxes_xyxy, ground_points, str(img_path)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Custom collate function for variable number of boxes per image.
    
    Args:
        batch: List of (image, boxes, ground_points, path) tuples
    
    Returns:
        Tuple of:
            - images: Stacked tensor (B, 3, H, W)
            - boxes: List of (N_i, 4) tensors
            - ground_points: List of (N_i, 2) tensors
            - paths: List of image paths
    """
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    ground_points = [item[2] for item in batch]
    paths = [item[3] for item in batch]
    
    return images, boxes, ground_points, paths


def create_dataloaders(
    data_dir: Path,
    image_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test splits.
    
    Args:
        data_dir: Path to dataset root
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = GroundPointDataset(
                data_dir=data_dir,
                split=split,
                image_size=image_size,
                augment=(split == 'train')
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            print(f"  {split.capitalize()}: {len(dataset)} images")
            
        except (FileNotFoundError, ValueError) as e:
            print(f"  {split.capitalize()}: Not found or empty ({e})")
            dataloaders[split] = None
    
    return dataloaders


