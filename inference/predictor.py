"""
Inference pipeline for Ground Point Prediction.

Provides easy-to-use interface for predicting ground points
from images and bounding boxes (e.g., from YOLO detector).
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import GroundPointPredictor


class GroundPointInference:
    """
    Inference pipeline for Ground Point Prediction.
    
    Provides a simple interface to predict ground points for objects
    detected by any YOLO model.
    
    Example:
        >>> predictor = GroundPointInference("weights/best_model.pt")
        >>> 
        >>> # With YOLO detections
        >>> boxes = yolo_model(image)  # Get bounding boxes
        >>> ground_points = predictor.predict(image, boxes)
        >>> 
        >>> # Or with image path
        >>> ground_points = predictor.predict_from_file("image.jpg", boxes)
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        backbone: str = "resnet34",
        device: str = "cuda",
        image_size: int = 640
    ):
        """
        Initialize inference pipeline.
        
        Args:
            weights_path: Path to trained model weights
            backbone: Backbone architecture (must match training)
            device: Device for inference ('cuda' or 'cpu')
            image_size: Image size for inference (must match training)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        
        # Load model
        self.model = GroundPointPredictor(
            backbone=backbone,
            pretrained=False
        ).to(self.device)
        
        # Load weights
        weights_path = Path(weights_path)
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        boxes: Union[np.ndarray, torch.Tensor, List],
        return_pixels: bool = True
    ) -> np.ndarray:
        """
        Predict ground points for given boxes.
        
        Args:
            image: Input image (BGR or RGB, HxWx3)
            boxes: Bounding boxes in xyxy format (N, 4)
                   Can be pixel coordinates or normalized (0-1)
            return_pixels: If True, return pixel coordinates,
                          otherwise return normalized (0-1)
        
        Returns:
            Ground points array (N, 2) - [x, y] for each box
        """
        # Validate input
        if image is None or image.size == 0:
            return np.array([])
        
        orig_h, orig_w = image.shape[:2]
        
        # Convert boxes to tensor
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes).float()
        
        if boxes.numel() == 0:
            return np.array([])
        
        # Check if boxes are normalized
        if boxes.max() <= 1.0:
            # Convert normalized to pixel coordinates
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
        
        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # Scale boxes to resized image
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        boxes_scaled = boxes.clone()
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y
        
        # Transform image
        image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        boxes_scaled = boxes_scaled.to(self.device)
        
        # Predict
        pred_points = self.model(image_tensor, [boxes_scaled])
        pred_points = pred_points.cpu().numpy()
        
        # Convert to original image coordinates
        if return_pixels:
            pred_points[:, 0] *= orig_w
            pred_points[:, 1] *= orig_h
        
        return pred_points
    
    def predict_from_file(
        self,
        image_path: Union[str, Path],
        boxes: Union[np.ndarray, torch.Tensor, List],
        return_pixels: bool = True
    ) -> np.ndarray:
        """
        Predict ground points from image file.
        
        Args:
            image_path: Path to image file
            boxes: Bounding boxes in xyxy format
            return_pixels: If True, return pixel coordinates
        
        Returns:
            Ground points array (N, 2)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        return self.predict(image, boxes, return_pixels)
    
    def predict_with_yolo(
        self,
        image: np.ndarray,
        yolo_results: "ultralytics.engine.results.Results"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict ground points from YOLO detection results.
        
        Args:
            image: Input image
            yolo_results: Results object from YOLO model
        
        Returns:
            Tuple of (boxes, ground_points) in pixel coordinates
        """
        # Extract boxes from YOLO results
        if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
        else:
            return np.array([]), np.array([])
        
        if len(boxes) == 0:
            return np.array([]), np.array([])
        
        ground_points = self.predict(image, boxes, return_pixels=True)
        
        return boxes, ground_points


