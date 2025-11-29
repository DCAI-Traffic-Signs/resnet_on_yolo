"""
Visualization utilities for Ground Point Prediction.

Provides functions to visualize predictions, training history, and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import torch


class Visualizer:
    """
    Visualization helper for ground point predictions.
    
    Provides methods to visualize:
        - Single image with boxes and ground points
        - Batch of predictions
        - Error distribution
    """
    
    # Default colors (RGB)
    COLORS = {
        'gt': (0, 255, 0),        # Green for ground truth
        'pred': (255, 0, 0),      # Red for prediction
        'box': (0, 255, 255),     # Cyan for bounding box
        'center': (255, 255, 0),  # Yellow for box center
        'error': (255, 165, 0)    # Orange for error line
    }
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        dpi: int = 120
    ):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.dpi = dpi
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_prediction(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        pred_points: np.ndarray,
        gt_points: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Visualize predictions on a single image.
        
        Args:
            image: Input image (RGB, HxWx3)
            boxes: Bounding boxes (N, 4) in xyxy pixel format
            pred_points: Predicted ground points (N, 2) in pixels
            gt_points: Ground truth points (N, 2) in pixels, optional
            title: Figure title
            save_path: Path to save figure
            show: Whether to display figure
        
        Returns:
            Annotated image as numpy array
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        
        for i in range(len(boxes)):
            box = boxes[i]
            pred = pred_points[i]
            
            # Draw bounding box
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor='cyan',
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Draw box center
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            ax.scatter([cx], [cy], c='cyan', s=50, marker='+', linewidths=2)
            
            # Draw prediction
            ax.scatter(
                [pred[0]], [pred[1]],
                c='red', s=150, marker='x',
                linewidths=3, zorder=5, label='Pred' if i == 0 else None
            )
            
            # Draw offset line (center to prediction)
            ax.plot(
                [cx, pred[0]], [cy, pred[1]],
                'r--', linewidth=1, alpha=0.7
            )
            
            # Draw ground truth if provided
            if gt_points is not None:
                gt = gt_points[i]
                ax.scatter(
                    [gt[0]], [gt[1]],
                    c='lime', s=150, marker='o',
                    edgecolors='black', linewidths=2,
                    zorder=5, label='GT' if i == 0 else None
                )
                
                # Error line
                error = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
                ax.plot(
                    [gt[0], pred[0]], [gt[1], pred[1]],
                    'yellow', linestyle=':', linewidth=2
                )
                
                # Error annotation
                ax.annotate(
                    f'{error:.1f}px',
                    xy=(pred[0], pred[1]),
                    xytext=(5, -5),
                    textcoords='offset points',
                    fontsize=8,
                    color='white',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7)
                )
        
        if title:
            ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        # Convert to numpy array
        fig.canvas.draw()
        result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return result
    
    def visualize_batch(
        self,
        images: List[np.ndarray],
        boxes_list: List[np.ndarray],
        pred_points_list: List[np.ndarray],
        gt_points_list: Optional[List[np.ndarray]] = None,
        save_dir: Optional[Path] = None,
        max_images: int = 20
    ) -> None:
        """
        Visualize a batch of predictions.
        
        Args:
            images: List of images
            boxes_list: List of box arrays
            pred_points_list: List of prediction arrays
            gt_points_list: List of ground truth arrays
            save_dir: Directory to save visualizations
            max_images: Maximum number of images to visualize
        """
        save_dir = Path(save_dir) if save_dir else self.output_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(min(len(images), max_images)):
            gt = gt_points_list[i] if gt_points_list else None
            
            save_path = save_dir / f"prediction_{i:04d}.jpg" if save_dir else None
            
            self.visualize_prediction(
                images[i],
                boxes_list[i],
                pred_points_list[i],
                gt,
                title=f"Image {i+1}",
                save_path=save_path
            )
        
        if save_dir:
            print(f"Saved {min(len(images), max_images)} visualizations to {save_dir}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Pixel error
    if 'val_error_mean' in history:
        axes[0, 1].plot(history['val_error_mean'], label='Mean', linewidth=2)
        if 'val_error_median' in history:
            axes[0, 1].plot(history['val_error_median'], label='Median', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Error (pixels)')
        axes[0, 1].set_title('Validation Pixel Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Final metrics summary
    axes[1, 1].axis('off')
    if history:
        summary_text = "Training Summary\n" + "-" * 30 + "\n"
        if 'train_loss' in history:
            summary_text += f"Final Train Loss: {history['train_loss'][-1]:.5f}\n"
        if 'val_loss' in history:
            summary_text += f"Final Val Loss: {history['val_loss'][-1]:.5f}\n"
        if 'val_error_mean' in history:
            summary_text += f"Final Mean Error: {history['val_error_mean'][-1]:.1f}px\n"
        if 'val_error_median' in history:
            summary_text += f"Final Median Error: {history['val_error_median'][-1]:.1f}px\n"
        
        axes[1, 1].text(
            0.5, 0.5, summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=14, fontfamily='monospace',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


