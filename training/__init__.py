"""
Training module for Ground Point Prediction.

Contains trainer class and loss functions.
"""

from .losses import KeypointLoss, OKSLoss
from .trainer import Trainer

__all__ = ["KeypointLoss", "OKSLoss", "Trainer"]


