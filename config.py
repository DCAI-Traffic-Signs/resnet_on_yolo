"""
Configuration for Ground Point Prediction Pipeline.

This module contains all configurable parameters for training and inference.
"""

from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/home/roman/Dokumente/DCAII/data/Mapillary/dataset")
WEIGHTS_DIR = BASE_DIR / "weights"

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelConfig:
    """Configuration for the GroundPointPredictor model."""
    
    # Backbone architecture: 'resnet18', 'resnet34', 'resnet50', 'mobilenet'
    BACKBONE: str = "resnet34"
    
    # Use pretrained ImageNet weights for backbone
    PRETRAINED: bool = True
    
    # ROI Align output size (features extracted per box)
    ROI_SIZE: int = 7
    
    # Hidden channels in keypoint head
    HIDDEN_CHANNELS: int = 256


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainingConfig:
    """Configuration for model training."""
    
    # Input image size (will be resized)
    IMAGE_SIZE: int = 640
    
    # Training hyperparameters
    EPOCHS: int = 50
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    
    # Learning rate scheduler
    SCHEDULER: str = "cosine"  # 'cosine', 'step', 'none'
    
    # Data loading
    NUM_WORKERS: int = 4
    
    # Checkpointing
    SAVE_EVERY: int = 10  # Save checkpoint every N epochs
    LOG_EVERY: int = 5    # Log metrics every N epochs
    
    # Device
    DEVICE: str = "cuda"  # 'cuda' or 'cpu'


# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

class InferenceConfig:
    """Configuration for model inference."""
    
    # Input image size (should match training)
    IMAGE_SIZE: int = 640
    
    # Confidence threshold for visualization
    VISUALIZE: bool = True
    
    # Device
    DEVICE: str = "cuda"


# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

class VisualizationConfig:
    """Configuration for result visualization."""
    
    # Colors (BGR for OpenCV, RGB for matplotlib)
    GT_COLOR = (0, 255, 0)       # Green for ground truth
    PRED_COLOR = (255, 0, 0)     # Red for prediction
    BOX_COLOR = (0, 255, 255)    # Cyan for bounding box
    
    # Marker sizes
    POINT_SIZE: int = 100
    LINE_WIDTH: int = 2
    
    # Output
    DPI: int = 120
    MAX_IMAGES: int = 20


