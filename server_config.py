#!/usr/bin/env python3
"""
Server Configuration for YOLOv8 Pruning
Optimized for GPU server deployment with automatic path detection and validation.
"""

import os
import sys
from pathlib import Path
import torch
import yaml
from typing import Dict, Any, Optional

class ServerConfig:
    """Server-optimized configuration for YOLOv8 pruning."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.config_file = config_file
        self._config = {}
        self._load_config()
        self._setup_paths()
        self._setup_device()
        
    def _load_config(self):
        """Load configuration from file or use defaults."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self._config = yaml.safe_load(f)
                print(f"✅ Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"⚠️  Failed to load config file: {e}, using defaults")
                self._config = {}
        else:
            self._config = {}
            
    def _setup_paths(self):
        """Setup file and directory paths."""
        # Model paths - auto-detect if not specified
        self.model_path = self._get_path("model_path", self._find_model_file())
        self.data_yaml = self._get_path("data_yaml", self.base_dir / "pruning" / "data" / "VOC_adva.yaml")
        
        # Dataset paths
        self.train_images = self._get_path("train_images", self.base_dir / "pruning" / "dataset_voc" / "images" / "train")
        self.val_images = self._get_path("val_images", self.base_dir / "pruning" / "dataset_voc" / "images" / "val")
        self.train_labels = self._get_path("train_labels", self.base_dir / "pruning" / "dataset_voc" / "labels" / "train")
        self.val_labels = self._get_path("val_labels", self.base_dir / "pruning" / "dataset_voc" / "labels" / "val")
        
        # Output paths
        self.output_dir = self._get_path("output_dir", self.base_dir / "server_output")
        self.logs_dir = self._get_path("logs_dir", self.base_dir / "server_logs")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def _find_model_file(self) -> Path:
        """Auto-detect YOLOv8 model file."""
        model_files = list(self.base_dir.glob("*.pt"))
        if model_files:
            # Prefer yolov8s.pt, then yolov8n.pt, then any other .pt file
            for preferred in ["yolov8s.pt", "yolov8n.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]:
                for model_file in model_files:
                    if model_file.name == preferred:
                        return model_file
            return model_files[0]  # Return first found if no preferred model
        return self.base_dir / "yolov8s.pt"  # Default fallback
        
    def _get_path(self, key: str, default: Path) -> Path:
        """Get path from config or use default."""
        if key in self._config:
            return Path(self._config[key])
        return default
        
    def _setup_device(self):
        """Setup GPU/CPU device configuration."""
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        else:
            self.device = "cpu"
            self.gpu_count = 0
            self.gpu_name = "CPU"
            self.gpu_memory = 0
            
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration optimized for current hardware."""
        base_config = {
            "batch_size": 16 if self.device == "cuda" else 8,
            "img_size": 640,
            "epochs": 20,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "patience": 10,
            "save_period": 5,
            "workers": min(8, os.cpu_count() or 1),
            "device": self.device
        }
        
        # Optimize for GPU memory
        if self.device == "cuda":
            if self.gpu_memory < 8:  # Less than 8GB VRAM
                base_config["batch_size"] = 8
                base_config["workers"] = 4
            elif self.gpu_memory < 16:  # Less than 16GB VRAM
                base_config["batch_size"] = 12
                base_config["workers"] = 6
            else:  # 16GB+ VRAM
                base_config["batch_size"] = 16
                base_config["workers"] = 8
                
        return base_config
        
    def get_pruning_config(self) -> Dict[str, Any]:
        """Get pruning configuration."""
        return {
            "last_layer_idx": self._config.get("last_layer_idx", 3),
            "max_train_samples": self._config.get("max_train_samples", 1000),
            "max_val_samples": self._config.get("max_val_samples", 500),
            "pruning_methods": [
                "activation_pruning_blocks_3_4",
                "50_percent_gamma_pruning_blocks_3_4", 
                "conv2d_with_activations"
            ]
        }
        
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all required paths exist."""
        validation = {}
        
        # Check model file
        validation["model_exists"] = self.model_path.exists()
        if not validation["model_exists"]:
            print(f"❌ Model file not found: {self.model_path}")
            
        # Check data config
        validation["data_yaml_exists"] = self.data_yaml.exists()
        if not validation["data_yaml_exists"]:
            print(f"❌ Dataset config not found: {self.data_yaml}")
            
        # Check dataset directories
        validation["train_images_exist"] = self.train_images.exists()
        validation["val_images_exist"] = self.val_images.exists()
        validation["train_labels_exist"] = self.train_labels.exists()
        validation["val_labels_exist"] = self.val_labels.exists()
        
        if not validation["train_images_exist"]:
            print(f"❌ Training images not found: {self.train_images}")
        if not validation["val_images_exist"]:
            print(f"❌ Validation images not found: {self.val_images}")
        if not validation["train_labels_exist"]:
            print(f"❌ Training labels not found: {self.train_labels}")
        if not validation["val_labels_exist"]:
            print(f"❌ Validation labels not found: {self.val_labels}")
            
        return validation
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        info = {
            "data_yaml": str(self.data_yaml),
            "train_images": str(self.train_images),
            "val_images": str(self.val_images),
            "train_labels": str(self.train_labels),
            "val_labels": str(self.val_labels)
        }
        
        # Try to load dataset config
        try:
            if self.data_yaml.exists():
                with open(self.data_yaml, 'r') as f:
                    data_cfg = yaml.safe_load(f)
                info["classes"] = data_cfg.get("names", [])
                info["num_classes"] = len(info["classes"])
        except Exception as e:
            print(f"⚠️  Could not load dataset config: {e}")
            info["classes"] = []
            info["num_classes"] = 0
            
        return info
        
    def print_summary(self):
        """Print configuration summary."""
        print("🔧 SERVER CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Base directory: {self.base_dir}")
        print(f"Device: {self.device} ({self.gpu_name})")
        if self.device == "cuda":
            print(f"GPU memory: {self.gpu_memory:.1f} GB")
        print("-" * 50)
        print("PATHS:")
        print(f"Model: {self.model_path}")
        print(f"Data config: {self.data_yaml}")
        print(f"Train images: {self.train_images}")
        print(f"Val images: {self.val_images}")
        print(f"Output: {self.output_dir}")
        print(f"Logs: {self.logs_dir}")
        print("-" * 50)
        print("TRAINING CONFIG:")
        training_config = self.get_training_config()
        for key, value in training_config.items():
            print(f"{key}: {value}")
        print("-" * 50)
        print("PRUNING CONFIG:")
        pruning_config = self.get_pruning_config()
        for key, value in pruning_config.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
    def save_config(self, output_file: str = None):
        """Save current configuration to file."""
        if output_file is None:
            output_file = self.output_dir / "server_config.yaml"
            
        config_dict = {
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
            "gpu_memory": self.gpu_memory,
            "training": self.get_training_config(),
            "pruning": self.get_pruning_config()
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        print(f"✅ Configuration saved to {output_file}")
        
    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
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
            "gpu_memory": self.gpu_memory,
            "training": self.get_training_config(),
            "pruning": self.get_pruning_config(),
            "dataset": self.get_dataset_info()
        }

def main():
    """Test the configuration."""
    print("🔧 Testing Server Configuration")
    print("=" * 50)
    
    # Create config
    config = ServerConfig()
    
    # Print summary
    config.print_summary()
    
    # Validate paths
    print("\n🔍 Validating paths...")
    validation = config.validate_paths()
    
    all_valid = all(validation.values())
    if all_valid:
        print("✅ All paths are valid!")
    else:
        print("❌ Some paths are missing. Please check the configuration.")
        
    # Save config
    config.save_config()
    
    return config

if __name__ == "__main__":
    main()

