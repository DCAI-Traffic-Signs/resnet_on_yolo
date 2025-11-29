"""
Ground Point Predictor Model.

This module implements the GroundPointPredictor, a neural network that predicts
ground contact points for objects given their bounding boxes.

Architecture (based on YOLO-Pose):
    1. Backbone: Feature extraction from full image (ResNet/MobileNet)
    2. ROI-Align: Extract features for each bounding box
    3. Keypoint Head: Predict offset to ground point (like YOLO-Pose cv4)
    4. Decode: Convert offset to absolute coordinates (like kpts_decode)

The key insight from YOLO-Pose:
    - Process the FULL image through backbone (sees ground, horizon, context)
    - Use ROI-Align to get box-specific features
    - Predict keypoint as offset from box center, scaled by box size

References:
    - YOLO-Pose: ultralytics/nn/modules/head.py (Pose class, line 319-384)
    - Keypoint Loss: ultralytics/utils/loss.py (KeypointLoss, line 175-191)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision import models
from typing import List, Tuple, Optional, Union


class ConvBlock(nn.Module):
    """
    Standard Convolution Block (Conv + BatchNorm + Activation).
    
    This is the basic building block used in YOLO architectures.
    Uses SiLU activation (Swish) as in YOLOv8.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 1, 
        stride: int = 1, 
        padding: Optional[int] = None,
        groups: int = 1, 
        activation: bool = True
    ):
        """
        Initialize ConvBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding (auto-calculated if None)
            groups: Number of groups for grouped convolution
            activation: Whether to apply activation function
        """
        super().__init__()
        
        # Auto-calculate padding for 'same' output size
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv -> bn -> activation."""
        return self.act(self.bn(self.conv(x)))


class KeypointHead(nn.Module):
    """
    Keypoint Prediction Head.
    
    This is directly extracted from YOLO-Pose architecture.
    
    Original YOLO-Pose (ultralytics/nn/modules/head.py, line 353):
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) 
            for x in ch
        )
    
    We use a single scale version for 1 keypoint (ground point) with 2 dimensions (x, y).
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        """
        Initialize KeypointHead.
        
        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden layer channels
        """
        super().__init__()
        
        # Number of keypoint values: 1 keypoint × 2 dimensions (x, y)
        self.num_keypoint_values = 2
        
        # Architecture exactly as YOLO-Pose cv4
        self.head = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=3),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, self.num_keypoint_values, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through keypoint head.
        
        Args:
            x: ROI features with shape (N, C, H, W)
        
        Returns:
            Raw keypoint offsets with shape (N, 2)
        """
        # Apply convolution layers
        out = self.head(x)  # (N, 2, H, W)
        
        # Global Average Pooling to get single offset per box
        out = F.adaptive_avg_pool2d(out, 1)  # (N, 2, 1, 1)
        
        # Flatten to (N, 2)
        return out.view(out.size(0), -1)


class GroundPointPredictor(nn.Module):
    """
    Ground Point Predictor Model.
    
    Predicts ground contact points for objects given an image and bounding boxes.
    Can be used with any YOLO detector - just provide the detected boxes.
    
    Architecture:
        Image (B, 3, H, W) → Backbone → Features (B, C, H/32, W/32)
                                            ↓
        Boxes (N, 4) ────────────────→ ROI-Align → Box Features (N, C, 7, 7)
                                                        ↓
                                               Keypoint Head → Offsets (N, 2)
                                                        ↓
                                               Decode → Ground Points (N, 2)
    
    The decode step follows YOLO-Pose logic:
        ground_point = box_center + offset * 2.0 * box_size
    """
    
    # Supported backbone architectures
    SUPPORTED_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'mobilenet']
    
    def __init__(
        self, 
        backbone: str = 'resnet34', 
        pretrained: bool = True, 
        roi_size: int = 7,
        hidden_channels: int = 256
    ):
        """
        Initialize GroundPointPredictor.
        
        Args:
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50', 'mobilenet')
            pretrained: Whether to use ImageNet pretrained weights
            roi_size: Output size for ROI-Align
            hidden_channels: Hidden channels in keypoint head
        """
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {self.SUPPORTED_BACKBONES}")
        
        self.roi_size = roi_size
        self.backbone_name = backbone
        
        # Initialize backbone and get feature dimensions
        self.backbone, self.feature_dim, self.stride = self._build_backbone(
            backbone, pretrained
        )
        
        # Keypoint head (like YOLO-Pose cv4)
        self.keypoint_head = KeypointHead(self.feature_dim, hidden_channels)
    
    def _build_backbone(
        self, 
        backbone: str, 
        pretrained: bool
    ) -> Tuple[nn.Module, int, int]:
        """
        Build backbone network for feature extraction.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
        
        Returns:
            Tuple of (backbone_module, feature_dim, stride)
        """
        if backbone == 'resnet18':
            resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
            stride = 32
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
            stride = 32
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
            stride = 32
        elif backbone == 'mobilenet':
            mobilenet = models.mobilenet_v3_small(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            return mobilenet.features, 576, 16
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Build ResNet feature extractor (remove FC layers)
        backbone_module = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        return backbone_module, feature_dim, stride
    
    def forward(
        self, 
        images: torch.Tensor, 
        boxes: Union[List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass: predict ground points for given boxes.
        
        Args:
            images: Batch of images with shape (B, 3, H, W)
            boxes: Either:
                - List of (N_i, 4) tensors: boxes per image in xyxy format
                - Single (N, 5) tensor: [batch_idx, x1, y1, x2, y2]
        
        Returns:
            Ground points with shape (total_N, 2) in normalized coordinates (0-1)
        """
        batch_size = images.size(0)
        img_h, img_w = images.size(2), images.size(3)
        
        # Step 1: Extract features from full image
        features = self.backbone(images)  # (B, C, H/stride, W/stride)
        
        # Step 2: Prepare boxes for ROI-Align
        boxes_combined = self._prepare_boxes(boxes)
        
        if boxes_combined is None or boxes_combined.size(0) == 0:
            return torch.zeros(0, 2, device=images.device)
        
        # Step 3: ROI-Align to extract features for each box
        spatial_scale = 1.0 / self.stride
        roi_features = roi_align(
            features,
            boxes_combined,
            output_size=self.roi_size,
            spatial_scale=spatial_scale,
            aligned=True
        )  # (N, C, roi_size, roi_size)
        
        # Step 4: Predict offsets through keypoint head
        raw_offsets = self.keypoint_head(roi_features)  # (N, 2)
        
        # Step 5: Decode offsets to absolute coordinates
        boxes_xyxy = boxes_combined[:, 1:5]  # Remove batch index
        ground_points = self._decode_keypoints(raw_offsets, boxes_xyxy, img_h, img_w)
        
        return ground_points
    
    def _prepare_boxes(
        self, 
        boxes: Union[List[torch.Tensor], torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Prepare boxes for ROI-Align by adding batch indices.
        
        Args:
            boxes: Either list of tensors per image or single tensor with batch indices
        
        Returns:
            Tensor with shape (N, 5): [batch_idx, x1, y1, x2, y2]
        """
        if isinstance(boxes, list):
            # List of boxes per image -> combine with batch indices
            boxes_with_idx = []
            for batch_idx, box_tensor in enumerate(boxes):
                if box_tensor.numel() > 0:
                    idx_column = torch.full(
                        (box_tensor.size(0), 1), 
                        batch_idx,
                        dtype=box_tensor.dtype, 
                        device=box_tensor.device
                    )
                    boxes_with_idx.append(torch.cat([idx_column, box_tensor], dim=1))
            
            if boxes_with_idx:
                return torch.cat(boxes_with_idx, dim=0)
            return None
        else:
            # Already in correct format
            return boxes
    
    def _decode_keypoints(
        self, 
        offsets: torch.Tensor, 
        boxes_xyxy: torch.Tensor, 
        img_h: int, 
        img_w: int
    ) -> torch.Tensor:
        """
        Decode raw offsets to ground point coordinates.
        
        This follows YOLO-Pose kpts_decode logic (head.py, line 365-384):
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[0] - 0.5)) * strides
        
        Adapted for box-relative coordinates:
            - Box center serves as "anchor"
            - Box size serves as "stride"
        
        Args:
            offsets: Raw offsets from network (N, 2)
            boxes_xyxy: Boxes in pixel coordinates (N, 4)
            img_h, img_w: Image dimensions
        
        Returns:
            Ground points in normalized coordinates (N, 2)
        """
        # Extract box geometry
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
        
        box_cx = (x1 + x2) / 2  # Box center x
        box_cy = (y1 + y2) / 2  # Box center y
        box_w = x2 - x1         # Box width
        box_h = y2 - y1         # Box height
        
        # YOLO-Pose style decode:
        # Multiply by 2.0 to allow larger range (like YOLO-Pose)
        # Offset is relative to box size, centered at box center
        gp_x = box_cx + offsets[:, 0] * 2.0 * box_w
        gp_y = box_cy + offsets[:, 1] * 2.0 * box_h
        
        # Normalize to 0-1 range
        gp_x = gp_x / img_w
        gp_y = gp_y / img_h
        
        # Clamp to valid range
        gp_x = torch.clamp(gp_x, 0.0, 1.0)
        gp_y = torch.clamp(gp_y, 0.0, 1.0)
        
        return torch.stack([gp_x, gp_y], dim=1)
    
    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get number of model parameters.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


