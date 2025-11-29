#!/usr/bin/env python3
"""
Inference script for Ground Point Predictor.

Usage:
    # Predict on single image with boxes
    python predict.py --image path/to/image.jpg --boxes "100,200,300,400;150,100,250,350"
    
    # Predict on test dataset and visualize
    python predict.py --test-data --visualize
    
    # Use with custom weights
    python predict.py --weights path/to/model.pt --image image.jpg

Example with YOLO (in Python):
    from predict import predict_with_yolo
    results = predict_with_yolo(yolo_model, image_path, gp_model_path)
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

from config import WEIGHTS_DIR, DATA_DIR, TrainingConfig, ModelConfig
from inference import GroundPointInference
from data import GroundPointDataset
from utils import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ground Point Prediction Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--boxes", type=str, default=None,
        help="Bounding boxes as 'x1,y1,x2,y2;x1,y1,x2,y2;...'"
    )
    parser.add_argument(
        "--test-data", action="store_true",
        help="Run on test dataset"
    )
    
    # Model options
    parser.add_argument(
        "--weights", type=Path, default=WEIGHTS_DIR / "best_model.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--backbone", type=str, default=ModelConfig.BACKBONE,
        help="Backbone architecture (must match training)"
    )
    
    # Output options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save visualization of predictions"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("predictions"),
        help="Directory for output visualizations"
    )
    
    # Device
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference"
    )
    
    return parser.parse_args()


def parse_boxes(boxes_str: str) -> np.ndarray:
    """Parse boxes from string format."""
    boxes = []
    for box_str in boxes_str.split(";"):
        coords = [float(x) for x in box_str.split(",")]
        if len(coords) == 4:
            boxes.append(coords)
    return np.array(boxes)


def predict_single_image(args, predictor: GroundPointInference):
    """Predict ground points for a single image."""
    print(f"\nImage: {args.image}")
    
    # Load image
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Parse boxes
    if args.boxes is None:
        raise ValueError("--boxes required when using --image")
    
    boxes = parse_boxes(args.boxes)
    print(f"Boxes: {len(boxes)}")
    
    # Predict
    ground_points = predictor.predict(image, boxes, return_pixels=True)
    
    # Print results
    print("\nResults:")
    print("-" * 40)
    for i, (box, gp) in enumerate(zip(boxes, ground_points)):
        print(f"  Box {i+1}: ({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")
        print(f"  Ground Point: ({gp[0]:.1f}, {gp[1]:.1f})")
        print()
    
    # Visualize if requested
    if args.visualize:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        vis = Visualizer(output_dir=args.output_dir)
        vis.visualize_prediction(
            image_rgb, boxes, ground_points,
            title=args.image.name,
            save_path=args.output_dir / f"{args.image.stem}_prediction.jpg"
        )
        print(f"Visualization saved to: {args.output_dir}")


def predict_test_dataset(args, predictor: GroundPointInference):
    """Predict on test dataset and evaluate."""
    print("\nLoading test dataset...")
    
    dataset = GroundPointDataset(
        data_dir=DATA_DIR,
        split="test",
        image_size=TrainingConfig.IMAGE_SIZE
    )
    
    print(f"Test samples: {len(dataset)}")
    
    errors = []
    visualizer = Visualizer(output_dir=args.output_dir / "visualizations")
    
    # Process samples
    max_vis = 20
    for i in range(len(dataset)):
        image_tensor, boxes, gt_points, img_path = dataset[i]
        
        # Load original image for visualization
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Convert boxes to original image coordinates
        scale = orig_w / TrainingConfig.IMAGE_SIZE
        boxes_orig = boxes.numpy() * scale
        
        # Predict
        pred_points = predictor.predict(image, boxes_orig, return_pixels=True)
        
        # Convert GT to pixels
        gt_points_px = gt_points.numpy()
        gt_points_px[:, 0] *= orig_w
        gt_points_px[:, 1] *= orig_h
        
        # Calculate errors
        for pred, gt in zip(pred_points, gt_points_px):
            error = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
            errors.append(error)
        
        # Visualize first N
        if args.visualize and i < max_vis:
            save_path = args.output_dir / "visualizations" / f"{Path(img_path).stem}.jpg"
            visualizer.visualize_prediction(
                image_rgb, boxes_orig, pred_points, gt_points_px,
                title=f"Sample {i+1}",
                save_path=save_path
            )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"  Samples: {len(errors)}")
    print(f"  Mean Error: {np.mean(errors):.1f} px")
    print(f"  Median Error: {np.median(errors):.1f} px")
    print(f"  90th Percentile: {np.percentile(errors, 90):.1f} px")
    print(f"  95th Percentile: {np.percentile(errors, 95):.1f} px")
    
    if args.visualize:
        print(f"\nVisualizations saved to: {args.output_dir / 'visualizations'}")


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 60)
    print("Ground Point Predictor - Inference")
    print("=" * 60)
    
    # Check weights exist
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    
    print(f"Weights: {args.weights}")
    print(f"Backbone: {args.backbone}")
    
    # Create predictor
    predictor = GroundPointInference(
        weights_path=args.weights,
        backbone=args.backbone,
        device=args.device,
        image_size=TrainingConfig.IMAGE_SIZE
    )
    
    # Run inference
    if args.image:
        predict_single_image(args, predictor)
    elif args.test_data:
        predict_test_dataset(args, predictor)
    else:
        print("\nNo input specified. Use --image or --test-data")
        print("Run with --help for usage information.")


if __name__ == "__main__":
    main()


