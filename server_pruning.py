#!/usr/bin/env python3
"""
YOLOv8 Pruning Server Deployment Script
Optimized for GPU servers with simplified configuration and robust error handling.

Usage:
    python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt
    python server_pruning.py --method 50_percent_gamma_pruning_blocks_3_4 --model yolov8s.pt
    python server_pruning.py --help

Features:
- GPU detection and automatic device selection
- Multiple pruning methods available
- Comprehensive logging and progress tracking
- Automatic model saving and evaluation
- Server-optimized configuration
"""

import argparse
import sys
import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add pruning directory to path
pruning_dir = Path(__file__).parent / "pruning"
if pruning_dir.exists():
    sys.path.insert(0, str(pruning_dir))
    print(f"✅ Added pruning directory to path: {pruning_dir}")
else:
    print(f"❌ Pruning directory not found: {pruning_dir}")
    print("Please ensure the pruning/ directory exists with all required modules")
    sys.exit(1)

# Import required modules
try:
    import torch
    from ultralytics import YOLO
    import yaml
    import cv2
    import numpy as np
    import glob
    print("✅ Core dependencies imported successfully")
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install requirements: pip install -r server_requirements.txt")
    sys.exit(1)

# Import pruning functions with detailed error handling
try:
    # Test if the module can be imported
    import pruning_yolo_v8
    print(f"✅ Successfully imported pruning_yolo_v8 module")
    
    # Import specific functions
    from pruning_yolo_v8 import (
        apply_activation_pruning_blocks_3_4, 
        apply_50_percent_gamma_pruning_blocks_3_4,
        prune_conv2d_in_block_with_activations
    )
    print("✅ Successfully imported pruning functions")
    
    # Verify functions exist
    available_functions = [
        'apply_activation_pruning_blocks_3_4',
        'apply_50_percent_gamma_pruning_blocks_3_4', 
        'prune_conv2d_in_block_with_activations'
    ]
    
    for func_name in available_functions:
        if hasattr(pruning_yolo_v8, func_name):
            print(f"✅ Function available: {func_name}")
        else:
            print(f"❌ Function not found: {func_name}")
            
except ImportError as e:
    print(f"❌ Failed to import pruning functions: {e}")
    print("\nTroubleshooting steps:")
    print("1. Ensure pruning_yolo_v8.py exists in the pruning/ directory")
    print("2. Check that all dependencies are installed")
    print("3. Verify the pruning/ directory structure:")
    print("   pruning/")
    print("   ├── pruning_yolo_v8.py")
    print("   ├── yolov8_utils.py")
    print("   ├── clustering.py")
    print("   └── yolo_layer_pruner.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error importing pruning functions: {e}")
    sys.exit(1)

class ServerPruningConfig:
    """Server-optimized configuration for YOLOv8 pruning."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.setup_paths()
        self.setup_device()
        self.setup_logging()
        
    def setup_paths(self):
        """Setup file and directory paths."""
        # Model and data paths
        self.model_path = self.base_dir / "yolov8s.pt"
        self.data_yaml = self.base_dir / "pruning" / "data" / "VOC_adva.yaml"
        
        # Dataset paths
        self.train_images = self.base_dir / "pruning" / "dataset_voc" / "images" / "train"
        self.val_images = self.base_dir / "pruning" / "dataset_voc" / "images" / "val"
        self.train_labels = self.base_dir / "pruning" / "dataset_voc" / "labels" / "train"
        self.val_labels = self.base_dir / "pruning" / "dataset_voc" / "labels" / "val"
        
        # Output paths
        self.output_dir = self.base_dir / "server_output"
        self.logs_dir = self.base_dir / "server_logs"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_device(self):
        """Setup GPU/CPU device configuration."""
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU detected: {self.gpu_name} (Count: {self.gpu_count})")
        else:
            self.device = "cpu"
            self.gpu_count = 0
            self.gpu_name = "CPU"
            print("⚠️  No GPU detected, using CPU")
            
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / f"server_pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_config_dict(self):
        """Return configuration as dictionary."""
        return {
            "model_path": str(self.model_path),
            "data_yaml": str(self.data_yaml),
            "train_images": str(self.train_images),
            "val_images": str(self.val_images),
            "train_labels": str(self.train_labels),
            "val_labels": str(self.val_labels),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "device": self.device,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "batch_size": 16 if self.device == "cuda" else 8,
            "img_size": 640,
            "epochs": 20,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "last_layer_idx": 3
        }

class ServerPruningPipeline:
    """Main pruning pipeline for server deployment."""
    
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": config.get_config_dict(),
            "status": "started"
        }
        
    def check_requirements(self):
        """Check if all requirements are met."""
        self.logger.info("Checking requirements...")
        
        # Check model file
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
            
        # Check data config
        if not os.path.exists(self.config.data_yaml):
            raise FileNotFoundError(f"Dataset config not found: {self.config.data_yaml}")
            
        # Check dataset directories
        if not os.path.exists(self.config.train_images):
            raise FileNotFoundError(f"Training images not found: {self.config.train_images}")
            
        if not os.path.exists(self.config.val_images):
            raise FileNotFoundError(f"Validation images not found: {self.config.val_images}")
            
        self.logger.info("✅ All requirements met")
        
    def load_dataset_config(self):
        """Load dataset configuration."""
        with open(self.config.data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        
        self.classes_names = data_cfg["names"]
        self.classes = list(range(len(self.classes_names)))
        self.logger.info(f"Dataset classes: {self.classes_names}")
        
        return data_cfg
        
    def load_samples(self, image_dir, label_dir, max_samples=None):
        """Load image samples and their labels."""
        samples = []
        
        if not os.path.exists(image_dir):
            self.logger.error(f"Image directory not found: {image_dir}")
            return samples
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        image_paths = sorted(image_paths)
        
        # Limit samples for server efficiency
        if max_samples and len(image_paths) > max_samples:
            image_paths = image_paths[:max_samples]
            
        self.logger.info(f"Loading {len(image_paths)} images from {image_dir}")
        
        for i, img_path in enumerate(image_paths):
            if i % 100 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_paths)}")
                
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
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
                                    labels.append({
                                        "class_id": class_id,
                                        "x_center": float(parts[1]),
                                        "y_center": float(parts[2]),
                                        "width": float(parts[3]),
                                        "height": float(parts[4])
                                    })
                            except (ValueError, IndexError):
                                continue
                
                samples.append({
                    "image": img,
                    "label": labels,
                    "image_path": img_path,
                    "label_path": label_path
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing image {img_path}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(samples)} samples")
        return samples
        
    def evaluate_model(self, model_path):
        """Evaluate a YOLOv8 model and return metrics."""
        try:
            self.logger.info(f"Evaluating model: {model_path}")
            model = YOLO(model_path)
            
            results = model.val(
                data=str(self.config.data_yaml),
                imgsz=self.config.get_config_dict()["img_size"],
                batch=self.config.get_config_dict()["batch_size"],
                device=self.config.device,
                conf=self.config.get_config_dict()["conf_threshold"],
                iou=self.config.get_config_dict()["iou_threshold"],
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
            
            self.logger.info(f"Evaluation complete. mAP@0.5: {metrics.get('mAP_0.5', 'N/A')}")
            return metrics, results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_path}: {e}")
            raise
            
    def prune_model(self, method, model_path):
        """Prune the model using the specified method."""
        self.logger.info(f"Starting pruning with method: {method}")
        
        # Load training data (limit for server efficiency)
        train_data = self.load_samples(
            self.config.train_images, 
            self.config.train_labels, 
            max_samples=1000  # Limit for server efficiency
        )
        valid_data = self.load_samples(
            self.config.val_images, 
            self.config.val_labels, 
            max_samples=500   # Limit for server efficiency
        )
        
        if len(train_data) == 0 or len(valid_data) == 0:
            raise ValueError("No training or validation data found!")
        
        # Generate output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = self.config.output_dir / f"pruned_model_{method}_{timestamp}.pt"
        
        try:
            if method == "activation_pruning_blocks_3_4":
                pruned_model = apply_activation_pruning_blocks_3_4(
                    model_path=model_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=self.classes,
                )
            elif method == "50_percent_gamma_pruning_blocks_3_4":
                pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
                    model_path=model_path,
                    layers_to_prune=self.config.get_config_dict()["last_layer_idx"]
                )
            elif method == "conv2d_with_activations":
                pruned_model = prune_conv2d_in_block_with_activations(
                    model_path=model_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=self.classes,
                )
            else:
                raise ValueError(f"Unknown pruning method: {method}")
            
            # Save the pruned model
            pruned_model.save(str(save_path))
            self.logger.info(f"Pruned model saved to {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Error during model pruning: {e}")
            raise
            
    def run_pipeline(self, method, model_path):
        """Run the complete pruning pipeline."""
        start_time = time.time()
        
        try:
            # Check requirements
            self.check_requirements()
            
            # Load dataset config
            self.load_dataset_config()
            
            # Update results
            self.results.update({
                "method": method,
                "model_path": model_path,
                "dataset_info": {
                    "num_classes": len(self.classes),
                    "class_names": self.classes_names,
                    "train_samples": len(glob.glob(os.path.join(self.config.train_images, "*"))),
                    "val_samples": len(glob.glob(os.path.join(self.config.val_images, "*")))
                }
            })
            
            # Evaluate baseline model
            self.logger.info("📊 Evaluating baseline model...")
            baseline_metrics, _ = self.evaluate_model(model_path)
            self.results["baseline_metrics"] = baseline_metrics
            self.logger.info(f"✅ Baseline mAP@0.5: {baseline_metrics.get('mAP_0.5', 'N/A'):.4f}")
            
            # Prune model
            self.logger.info(f"✂️  Pruning model using method: {method}")
            pruned_model_path = self.prune_model(method, model_path)
            self.results["pruned_model_path"] = pruned_model_path
            
            # Evaluate pruned model
            self.logger.info("📊 Evaluating pruned model...")
            pruned_metrics, _ = self.evaluate_model(pruned_model_path)
            self.results["pruned_metrics"] = pruned_metrics
            self.logger.info(f"✅ Pruned mAP@0.5: {pruned_metrics.get('mAP_0.5', 'N/A'):.4f}")
            
            # Calculate performance metrics
            baseline_map = baseline_metrics.get('mAP_0.5', 0)
            pruned_map = pruned_metrics.get('mAP_0.5', 0)
            performance_retention = (pruned_map / baseline_map * 100) if baseline_map > 0 else 0
            
            self.results.update({
                "status": "completed",
                "performance_retention": performance_retention,
                "total_time_seconds": time.time() - start_time,
                "completion_timestamp": datetime.now().isoformat()
            })
            
            # Save results
            results_file = self.config.output_dir / f"pruning_results_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            self.logger.info(f"💾 Results saved to: {results_file}")
            
            # Print summary
            self.print_summary(baseline_metrics, pruned_metrics, pruned_model_path, performance_retention)
            
            return self.results
            
        except Exception as e:
            self.results.update({
                "status": "failed",
                "error": str(e),
                "total_time_seconds": time.time() - start_time,
                "completion_timestamp": datetime.now().isoformat()
            })
            self.logger.error(f"Pipeline failed: {e}")
            raise
            
    def print_summary(self, baseline_metrics, pruned_metrics, pruned_model_path, performance_retention):
        """Print final summary."""
        print("\n" + "="*60)
        print("🎉 PRUNING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Method: {self.results.get('method', 'N/A')}")
        print(f"Device: {self.config.device} ({self.config.gpu_name})")
        print(f"Total time: {self.results.get('total_time_seconds', 0):.1f} seconds")
        print("-"*60)
        print("PERFORMANCE METRICS:")
        print(f"Baseline mAP@0.5: {baseline_metrics.get('mAP_0.5', 'N/A'):.4f}")
        print(f"Pruned mAP@0.5:  {pruned_metrics.get('mAP_0.5', 'N/A'):.4f}")
        print(f"Performance retention: {performance_retention:.1f}%")
        print("-"*60)
        print("OUTPUTS:")
        print(f"Pruned model: {pruned_model_path}")
        print(f"Results log: {self.config.output_dir}")
        print("="*60)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Pruning Server Deployment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt
  python server_pruning.py --method 50_percent_gamma_pruning_blocks_3_4 --model yolov8n.pt
  python server_pruning.py --help

Available methods:
  - activation_pruning_blocks_3_4: Activation-based pruning for blocks 3-4
  - 50_percent_gamma_pruning_blocks_3_4: 50% gamma-based pruning for blocks 3-4
  - conv2d_with_activations: Conv2d pruning with activations
        """
    )
    
    parser.add_argument(
        "--method", 
        required=True,
        choices=["activation_pruning_blocks_3_4", "50_percent_gamma_pruning_blocks_3_4", "conv2d_with_activations"],
        help="Pruning method to use"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Path to the YOLOv8 model file (.pt)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: server_output/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ServerPruningConfig()
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(exist_ok=True)
    
    # Override model path if specified
    if args.model:
        config.model_path = Path(args.model)
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    print("🚀 YOLOv8 Server Pruning Pipeline")
    print("="*50)
    print(f"Method: {args.method}")
    print(f"Model: {config.model_path}")
    print(f"Device: {config.device} ({config.gpu_name})")
    print(f"Output: {config.output_dir}")
    print("="*50)
    
    # Run pipeline
    try:
        pipeline = ServerPruningPipeline(config)
        results = pipeline.run_pipeline(args.method, str(config.model_path))
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        config.logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()