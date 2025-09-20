"""
Server-ready YOLOv8 pruning evaluation script.
Adapted for deployment with configurable paths and improved error handling.
"""

import sys
import os
from pathlib import Path

# Add the pruning directory to Python path
sys.path.append(str(Path(__file__).parent / 'pruning'))

import torch
import logging
import yaml
import cv2
import numpy as np
import glob
from typing import Dict, List, Any, Optional, Tuple
import json
import traceback
from datetime import datetime

from ultralytics import YOLO
from config import config
from pruning.pruning_yolo_v8 import (
    apply_activation_pruning_blocks_3_4, 
    apply_50_percent_gamma_pruning_blocks_3_4,
    prune_conv2d_in_block_with_activations,
    apply_pruning_v8,
    apply_gamma_pruning_iter, 
    apply_gamma_pruning_on_block_zeroed
)

logger = logging.getLogger("yolov8_server_pruning")

class YOLOv8PruningServer:
    """Server-ready YOLOv8 pruning service."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the pruning server with configuration."""
        self.config = config
        if config_file:
            # Reload config with custom file
            from config import PruningConfig
            self.config = PruningConfig(config_file)
        
        logger.info(f"Initialized YOLOv8 Pruning Server with config: {self.config.to_dict()}")
    
    def load_samples(self, image_dir: str, label_dir: str) -> List[Dict[str, Any]]:
        """
        Load image samples and their labels for training/validation.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing label files
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return samples
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        image_paths = sorted(image_paths)
        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                    
                h, w = img.shape[:2]
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, base + ".txt")
                
                labels = []
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            try:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    # YOLO format: class x_center y_center width height (normalized)
                                    labels.append({
                                        "class_id": class_id,
                                        "x_center": float(parts[1]),
                                        "y_center": float(parts[2]),
                                        "width": float(parts[3]),
                                        "height": float(parts[4])
                                    })
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Invalid label line in {label_path}: {line.strip()}")
                
                samples.append({
                    "image": img,
                    "label": labels,
                    "image_path": img_path,
                    "label_path": label_path
                })
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(samples)} samples")
        return samples
    
    def evaluate_model(self, model_path: str, data_yaml: str, 
                      imgsz: int = None, batch: int = None, 
                      device: str = None, conf: float = None, 
                      iou: float = None) -> Tuple[Dict[str, Any], Any]:
        """
        Evaluate a YOLOv8 model and return metrics.
        
        Args:
            model_path: Path to the model weights
            data_yaml: Path to dataset YAML configuration
            imgsz: Image size for evaluation
            batch: Batch size
            device: Device to use ('cpu' or 'cuda')
            conf: Confidence threshold
            iou: IoU threshold
            
        Returns:
            Tuple of (metrics dictionary, results object)
        """
        # Use config defaults if not specified
        imgsz = imgsz or self.config.img_size
        batch = batch or self.config.batch_size
        device = device or self.config.device
        conf = conf or self.config.conf_threshold
        iou = iou or self.config.iou_threshold
        
        try:
            logger.info(f"Evaluating model: {model_path}")
            model = YOLO(model_path)
            
            results = model.val(
                data=data_yaml,
                imgsz=imgsz,
                batch=batch,
                device=device,
                conf=conf,
                iou=iou,
                verbose=False
            )
            
            metrics = {
                "precision": results.results_dict.get("metrics/precision(B)", None),
                "recall": results.results_dict.get("metrics/recall(B)", None),
                "mAP_0.5": results.results_dict.get("metrics/mAP50(B)", None),
                "mAP_0.5:0.95": results.results_dict.get("metrics/mAP50-95(B)", None), 
                "per_class_mAP": results.maps.tolist() if hasattr(results, "maps") and results.maps is not None else None,
                "mean_mAP_0.5_0.95": float(results.maps.mean()) if hasattr(results, "maps") and results.maps is not None else None,
                "speed": results.speed if hasattr(results, "speed") else None,
                "inference_time": results.speed.get("inference", None) if hasattr(results, "speed") else None,
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Evaluation complete. mAP@0.5: {metrics.get('mAP_0.5', 'N/A')}")
            return metrics, results
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_path}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def prune_model(self, model_path: str, train_data: List[Dict], 
                   valid_data: List[Dict], classes: List[int],
                   last_layer_idx: int = None, save_path: str = None,
                   method: str = None) -> str:
        """
        Main entry point for pruning a YOLOv8 model.
        
        Args:
            model_path: Path to the original model
            train_data: Training dataset samples
            valid_data: Validation dataset samples
            classes: List of class indices
            last_layer_idx: Last layer index for pruning
            save_path: Path to save pruned model
            method: Pruning method to use
            
        Returns:
            Path to the saved pruned model
        """
        # Use config defaults if not specified
        last_layer_idx = last_layer_idx or self.config.last_layer_idx
        method = method or self.config.pruning_method
        save_path = save_path or self.config.get_model_save_path(f"_{method}")
        
        logger.info(f"Starting pruning pipeline with method: {method}")
        logger.info(f"Model: {model_path}, Save to: {save_path}")
        
        try:
            # Select pruning method
            if method == "activation_pruning_blocks_3_4":
                pruned_model = apply_activation_pruning_blocks_3_4(
                    model_path=model_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=classes,
                )
            elif method == "50_percent_gamma_pruning_blocks_3_4":
                pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
                    model_path=model_path,
                    layers_to_prune=last_layer_idx
                )
            elif method == "conv2d_with_activations":
                pruned_model = prune_conv2d_in_block_with_activations(
                    model_path=model_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=classes,
                )
            else:
                raise ValueError(f"Unknown pruning method: {method}")
            
            # Save the pruned model
            torch_model = pruned_model.model
            torch.save(torch_model.state_dict(), save_path)
            logger.info(f"Pruned model saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error during model pruning: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_pruning_pipeline(self, model_path: str = None, 
                           data_yaml: str = None,
                           evaluate_baseline: bool = True,
                           evaluate_pruned: bool = True) -> Dict[str, Any]:
        """
        Run the complete pruning pipeline.
        
        Args:
            model_path: Path to the model (uses config default if None)
            data_yaml: Path to dataset YAML (uses config default if None)
            evaluate_baseline: Whether to evaluate baseline model
            evaluate_pruned: Whether to evaluate pruned model
            
        Returns:
            Dictionary containing results
        """
        model_path = model_path or self.config.model_path
        data_yaml = data_yaml or self.config.data_yaml
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "model_path": model_path,
            "data_yaml": data_yaml
        }
        
        try:
            # Load dataset configuration
            with open(data_yaml, "r") as f:
                data_cfg = yaml.safe_load(f)
            
            classes_names = data_cfg["names"]
            classes = list(range(len(classes_names)))
            logger.info(f"Dataset classes: {classes_names}")
            
            # Get dataset paths
            train_img_dir = data_cfg["train"]
            val_img_dir = data_cfg["val"]
            train_label_dir = train_img_dir.replace("/images", "/labels")
            val_label_dir = val_img_dir.replace("/images", "/labels")
            
            # Load samples
            logger.info("Loading dataset samples...")
            train_data = self.load_samples(train_img_dir, train_label_dir)
            valid_data = self.load_samples(val_img_dir, val_label_dir)
            
            results["dataset_info"] = {
                "num_classes": len(classes),
                "class_names": classes_names,
                "train_samples": len(train_data),
                "val_samples": len(valid_data)
            }
            
            # Evaluate baseline model
            if evaluate_baseline:
                logger.info("Evaluating baseline model...")
                baseline_metrics, _ = self.evaluate_model(model_path, data_yaml)
                results["baseline_metrics"] = baseline_metrics
            
            # Prune model
            logger.info("Starting model pruning...")
            pruned_model_path = self.prune_model(
                model_path=model_path,
                train_data=train_data,
                valid_data=valid_data,
                classes=classes
            )
            results["pruned_model_path"] = pruned_model_path
            
            # Evaluate pruned model
            if evaluate_pruned:
                logger.info("Evaluating pruned model...")
                pruned_metrics, _ = self.evaluate_model(pruned_model_path, data_yaml)
                results["pruned_metrics"] = pruned_metrics
            
            results["status"] = "success"
            logger.info("Pruning pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in pruning pipeline: {e}")
            logger.error(traceback.format_exc())
            results["status"] = "error"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 Pruning Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Path to model weights")
    parser.add_argument("--data", type=str, help="Path to dataset YAML")
    parser.add_argument("--method", type=str, help="Pruning method", 
                       choices=["activation_pruning_blocks_3_4", "50_percent_gamma_pruning_blocks_3_4", "conv2d_with_activations"])
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--no-pruned", action="store_true", help="Skip pruned model evaluation")
    
    args = parser.parse_args()
    
    try:
        # Initialize server
        server = YOLOv8PruningServer(args.config)
        
        # Override config with command line arguments
        if args.model:
            server.config.model_path = args.model
        if args.data:
            server.config.data_yaml = args.data
        if args.method:
            server.config.pruning_method = args.method
        if args.output:
            server.config.output_dir = Path(args.output)
            server.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pruning pipeline
        results = server.run_pruning_pipeline(
            evaluate_baseline=not args.no_baseline,
            evaluate_pruned=not args.no_pruned
        )
        
        # Save results
        results_file = server.config.output_dir / f"pruning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        if results["status"] == "success":
            print("✅ Pruning pipeline completed successfully!")
            if "baseline_metrics" in results:
                print(f"📊 Baseline mAP@0.5: {results['baseline_metrics'].get('mAP_0.5', 'N/A'):.4f}")
            if "pruned_metrics" in results:
                print(f"📊 Pruned mAP@0.5: {results['pruned_metrics'].get('mAP_0.5', 'N/A'):.4f}")
        else:
            print("❌ Pruning pipeline failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 