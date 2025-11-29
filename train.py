#!/usr/bin/env python3
"""
Training script for Ground Point Predictor.

Usage:
    python train.py                          # Train with default config
    python train.py --epochs 100             # Train for 100 epochs
    python train.py --backbone resnet50      # Use ResNet50 backbone
    python train.py --batch-size 16          # Use batch size 16

Example:
    python train.py --epochs 50 --backbone resnet34 --device cuda
"""

import argparse
from pathlib import Path
import torch

from config import ModelConfig, TrainingConfig, DATA_DIR, WEIGHTS_DIR
from models import GroundPointPredictor
from data import create_dataloaders
from training import Trainer
from utils import plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Ground Point Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--backbone", type=str, default=ModelConfig.BACKBONE,
        choices=["resnet18", "resnet34", "resnet50", "mobilenet"],
        help="Backbone architecture"
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=ModelConfig.PRETRAINED,
        help="Use ImageNet pretrained weights"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=TrainingConfig.EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=TrainingConfig.BATCH_SIZE,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=TrainingConfig.LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--image-size", type=int, default=TrainingConfig.IMAGE_SIZE,
        help="Input image size"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Path to dataset directory"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", type=Path, default=WEIGHTS_DIR,
        help="Directory for saving checkpoints"
    )
    
    # Device
    parser.add_argument(
        "--device", type=str, default=TrainingConfig.DEVICE,
        choices=["cuda", "cpu"],
        help="Device to train on"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("Ground Point Predictor - Training")
    print("=" * 60)
    
    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=TrainingConfig.NUM_WORKERS
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    if train_loader is None:
        raise RuntimeError("Training data not found!")
    
    # Create model
    print("\nCreating model...")
    model = GroundPointPredictor(
        backbone=args.backbone,
        pretrained=args.pretrained,
        roi_size=ModelConfig.ROI_SIZE,
        hidden_channels=ModelConfig.HIDDEN_CHANNELS
    )
    
    total_params, trainable_params = model.get_num_parameters()
    print(f"  Backbone: {args.backbone}")
    print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        device=device,
        save_dir=args.output_dir,
        log_every=TrainingConfig.LOG_EVERY,
        save_every=TrainingConfig.SAVE_EVERY
    )
    
    # Train
    history = trainer.train()
    
    # Plot training history
    plot_training_history(
        history,
        save_path=args.output_dir / "training_history.png",
        show=False
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model saved to: {args.output_dir / 'best_model.pt'}")
    print(f"Training history: {args.output_dir / 'training_history.png'}")


if __name__ == "__main__":
    main()


