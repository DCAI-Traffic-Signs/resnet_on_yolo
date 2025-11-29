"""
Trainer class for Ground Point Prediction.

Handles the training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .losses import KeypointLoss


class Trainer:
    """
    Trainer for GroundPointPredictor model.
    
    Handles:
        - Training loop with progress bar
        - Validation with metrics calculation
        - Learning rate scheduling
        - Checkpointing (best model, periodic saves)
        - Logging and history tracking
    
    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, config)
        >>> history = trainer.train()
        >>> trainer.save_checkpoint("final_model.pt")
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda",
        save_dir: Path = Path("weights"),
        log_every: int = 5,
        save_every: int = 10
    ):
        """
        Initialize Trainer.
        
        Args:
            model: GroundPointPredictor model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            device: Device to train on ('cuda' or 'cpu')
            save_dir: Directory for saving checkpoints
            log_every: Log metrics every N epochs
            save_every: Save checkpoint every N epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_every = log_every
        self.save_every = save_every
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (YOLO-Pose style)
        self.criterion = KeypointLoss(sigma=1.0)
        
        # Optimizer (AdamW as in modern vision models)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_error_mean': [],
            'val_error_median': [],
            'learning_rate': []
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Checkpoints: {self.save_dir}")
        print("-" * 60)
        
        for epoch in range(1, self.epochs + 1):
            # Training
            train_loss = self._train_epoch()
            
            # Validation
            if self.val_loader is not None:
                val_loss, val_error_mean, val_error_median = self._validate()
            else:
                val_loss, val_error_mean, val_error_median = 0, 0, 0
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_error_mean'].append(val_error_mean)
            self.history['val_error_median'].append(val_error_median)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.save_dir / "best_model.pt")
            
            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(self.save_dir / f"epoch_{epoch}.pt")
            
            # Logging
            if epoch % self.log_every == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train={train_loss:.5f}, "
                      f"Val={val_loss:.5f}, "
                      f"Error={val_error_mean:.1f}px "
                      f"(median: {val_error_median:.1f}px)")
        
        # Save final model
        self.save_checkpoint(self.save_dir / "final_model.pt")
        
        print("-" * 60)
        print(f"Training complete. Best val loss: {self.best_val_loss:.5f}")
        
        return self.history
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, boxes_list, gt_points_list, _ in pbar:
            # Move to device
            images = images.to(self.device)
            boxes_list = [b.to(self.device) for b in boxes_list]
            gt_points = torch.cat([gp.to(self.device) for gp in gt_points_list], dim=0)
            
            if gt_points.size(0) == 0:
                continue
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_points = self.model(images, boxes_list)
            
            # Compute loss
            boxes_flat = torch.cat(boxes_list, dim=0)
            loss = self.criterion(
                pred_points, gt_points, boxes_flat,
                images.size(2), images.size(3)
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track
            batch_samples = gt_points.size(0)
            total_loss += loss.item() * batch_samples
            num_samples += batch_samples
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_samples, 1)
    
    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        """
        Run validation.
        
        Returns:
            Tuple of (avg_loss, mean_error_px, median_error_px)
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        pixel_errors = []
        
        img_size = self.train_loader.dataset.image_size
        
        for images, boxes_list, gt_points_list, _ in self.val_loader:
            images = images.to(self.device)
            boxes_list = [b.to(self.device) for b in boxes_list]
            gt_points = torch.cat([gp.to(self.device) for gp in gt_points_list], dim=0)
            
            if gt_points.size(0) == 0:
                continue
            
            # Forward pass
            pred_points = self.model(images, boxes_list)
            
            # Compute loss
            boxes_flat = torch.cat(boxes_list, dim=0)
            loss = self.criterion(
                pred_points, gt_points, boxes_flat,
                images.size(2), images.size(3)
            )
            
            # Track loss
            batch_samples = gt_points.size(0)
            total_loss += loss.item() * batch_samples
            num_samples += batch_samples
            
            # Compute pixel errors
            pred_px = pred_points * img_size
            gt_px = gt_points * img_size
            errors = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=1))
            pixel_errors.extend(errors.cpu().numpy().tolist())
        
        avg_loss = total_loss / max(num_samples, 1)
        mean_error = np.mean(pixel_errors) if pixel_errors else 0
        median_error = np.median(pixel_errors) if pixel_errors else 0
        
        return avg_loss, mean_error, median_error
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)


