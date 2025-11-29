"""
Loss functions for Ground Point Prediction.

Based on YOLO-Pose KeypointLoss implementation.

Reference: ultralytics/utils/loss.py (line 175-191)
"""

import torch
import torch.nn as nn
from typing import Optional


class KeypointLoss(nn.Module):
    """
    Keypoint Loss based on YOLO-Pose.
    
    Original implementation (ultralytics/utils/loss.py, line 183-191):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + 
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor * (1 - torch.exp(-e)) * kpt_mask).mean()
    
    This is an OKS-style loss that normalizes error by object area,
    so larger objects are allowed larger absolute errors.
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize KeypointLoss.
        
        Args:
            sigma: Keypoint sigma for OKS calculation (default 1.0 for single keypoint)
        """
        super().__init__()
        self.sigma = sigma
    
    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        img_h: int,
        img_w: int
    ) -> torch.Tensor:
        """
        Calculate keypoint loss.
        
        Args:
            pred_points: Predicted ground points (N, 2), normalized 0-1
            gt_points: Ground truth points (N, 2), normalized 0-1
            boxes_xyxy: Bounding boxes (N, 4) in pixel coordinates
            img_h: Image height
            img_w: Image width
        
        Returns:
            Scalar loss value
        """
        if pred_points.size(0) == 0:
            return torch.tensor(0.0, device=pred_points.device)
        
        # Convert to pixel coordinates for interpretable error
        pred_px = pred_points.clone()
        pred_px[:, 0] *= img_w
        pred_px[:, 1] *= img_h
        
        gt_px = gt_points.clone()
        gt_px[:, 0] *= img_w
        gt_px[:, 1] *= img_h
        
        # Squared Euclidean distance (as in YOLO-Pose)
        d = (pred_px[:, 0] - gt_px[:, 0]).pow(2) + \
            (pred_px[:, 1] - gt_px[:, 1]).pow(2)
        
        # Box area for normalization
        box_w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        box_h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        area = box_w * box_h
        
        # OKS-style normalization (as in YOLO-Pose)
        e = d / ((2 * self.sigma) ** 2 * (area + 1e-9) * 2)
        
        # OKS-style loss: 1 - exp(-e)
        loss = (1 - torch.exp(-e)).mean()
        
        return loss


class OKSLoss(nn.Module):
    """
    Object Keypoint Similarity (OKS) Loss.
    
    Alternative formulation that directly optimizes OKS metric.
    OKS = exp(-d^2 / (2 * s^2 * k^2))
    Loss = 1 - OKS
    """
    
    def __init__(self, sigma: float = 0.05):
        """
        Initialize OKS Loss.
        
        Args:
            sigma: Base sigma for OKS calculation
        """
        super().__init__()
        self.sigma = sigma
    
    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        img_h: int,
        img_w: int
    ) -> torch.Tensor:
        """
        Calculate OKS loss.
        
        Args:
            pred_points: Predicted ground points (N, 2), normalized 0-1
            gt_points: Ground truth points (N, 2), normalized 0-1
            boxes_xyxy: Bounding boxes (N, 4) in pixel coordinates
            img_h: Image height
            img_w: Image width
        
        Returns:
            Scalar loss value (1 - mean OKS)
        """
        if pred_points.size(0) == 0:
            return torch.tensor(0.0, device=pred_points.device)
        
        # Convert to pixel coordinates
        pred_px = pred_points.clone()
        pred_px[:, 0] *= img_w
        pred_px[:, 1] *= img_h
        
        gt_px = gt_points.clone()
        gt_px[:, 0] *= img_w
        gt_px[:, 1] *= img_h
        
        # Squared distance
        d_squared = (pred_px[:, 0] - gt_px[:, 0]).pow(2) + \
                    (pred_px[:, 1] - gt_px[:, 1]).pow(2)
        
        # Scale factor (box diagonal)
        box_w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        box_h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        scale_squared = box_w.pow(2) + box_h.pow(2)
        
        # OKS calculation
        oks = torch.exp(-d_squared / (2 * scale_squared * self.sigma ** 2 + 1e-9))
        
        # Loss is 1 - OKS (want to maximize OKS)
        loss = (1 - oks).mean()
        
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for keypoint regression.
    
    Simple alternative to OKS-style losses.
    Good for initial training, then switch to OKS.
    """
    
    def __init__(self, beta: float = 0.01):
        """
        Initialize Smooth L1 Loss.
        
        Args:
            beta: Threshold for switching between L1 and L2
        """
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)
    
    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        **kwargs  # Accept but ignore extra args
    ) -> torch.Tensor:
        """
        Calculate Smooth L1 loss on normalized coordinates.
        
        Args:
            pred_points: Predicted ground points (N, 2), normalized 0-1
            gt_points: Ground truth points (N, 2), normalized 0-1
        
        Returns:
            Scalar loss value
        """
        if pred_points.size(0) == 0:
            return torch.tensor(0.0, device=pred_points.device)
        
        return self.loss_fn(pred_points, gt_points)


