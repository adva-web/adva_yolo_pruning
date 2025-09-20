# YOLOv8 Pruning Server Deployment

A streamlined, GPU-optimized deployment solution for YOLOv8 model pruning on servers.

## 🚀 Quick Start

### 1. Setup
```bash
# Run the automated setup script
python server_setup.py

# Or quick setup (skip some checks)
python server_setup.py --quick

# Test imports specifically
python test_imports.py
```

### 2. Run Pruning
```bash
# Activation-based pruning (recommended)
python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt

# Gamma-based pruning
python server_pruning.py --method 50_percent_gamma_pruning_blocks_3_4 --model yolov8n.pt

# Conv2d with activations
python server_pruning.py --method conv2d_with_activations --model yolov8s.pt
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 10GB+ free space

### Required Files
- YOLOv8 model file (`.pt` format)
- Dataset in YOLO format
- Dataset configuration file (`VOC_adva.yaml`)

## 🛠️ Installation

### Option 1: Automated Setup (Recommended)
```bash
python server_setup.py
```

### Option 2: Manual Setup
```bash
# Install requirements
pip install -r server_requirements.txt

# Create directories
mkdir -p server_output server_logs
mkdir -p pruning/dataset_voc/{images,labels}/{train,val}

# Download a YOLOv8 model (if needed)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

## 📁 Directory Structure

```
your_project/
├── server_pruning.py          # Main pruning script
├── server_config.py           # Configuration management
├── server_setup.py            # Setup script
├── server_requirements.txt    # Server-optimized requirements
├── server_config.yaml         # Configuration file (auto-generated)
├── server_output/             # Output directory
├── server_logs/               # Log files
├── pruning/                   # Pruning modules
│   ├── pruning_yolo_v8.py
│   ├── yolov8_utils.py
│   └── ...
├── yolov8s.pt                 # YOLOv8 model file
└── pruning/dataset_voc/       # Dataset
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## ⚙️ Configuration

### Automatic Configuration
The system automatically detects:
- GPU availability and memory
- Model files
- Dataset structure
- Optimal batch sizes and parameters

### Manual Configuration
Edit `server_config.yaml`:
```yaml
# Model and data paths
model_path: "yolov8s.pt"
data_yaml: "pruning/data/VOC_adva.yaml"

# Dataset paths
train_images: "pruning/dataset_voc/images/train"
val_images: "pruning/dataset_voc/images/val"
train_labels: "pruning/dataset_voc/labels/train"
val_labels: "pruning/dataset_voc/labels/val"

# Output paths
output_dir: "server_output"
logs_dir: "server_logs"

# Pruning parameters
last_layer_idx: 3
max_train_samples: 1000
max_val_samples: 500
```

## 🎯 Available Pruning Methods

### 1. Activation-based Pruning (`activation_pruning_blocks_3_4`)
- **Description**: Prunes layers based on activation patterns
- **Best for**: General purpose pruning with good performance retention
- **Usage**: `--method activation_pruning_blocks_3_4`

### 2. Gamma-based Pruning (`50_percent_gamma_pruning_blocks_3_4`)
- **Description**: Prunes 50% of channels based on BatchNorm gamma values
- **Best for**: Aggressive pruning with significant size reduction
- **Usage**: `--method 50_percent_gamma_pruning_blocks_3_4`

### 3. Conv2d with Activations (`conv2d_with_activations`)
- **Description**: Prunes specific Conv2d layers using activation analysis
- **Best for**: Targeted pruning of specific layers
- **Usage**: `--method conv2d_with_activations`

## 📊 Usage Examples

### Basic Usage
```bash
# Test configuration
python server_config.py

# Run pruning with default settings
python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt
```

### Advanced Usage
```bash
# Custom output directory
python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt --output-dir /path/to/output

# Verbose logging
python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt --verbose

# Different model sizes
python server_pruning.py --method 50_percent_gamma_pruning_blocks_3_4 --model yolov8n.pt  # Nano
python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8m.pt        # Medium
python server_pruning.py --method conv2d_with_activations --model yolov8l.pt              # Large
```

## 📈 Output and Results

### Output Files
- **Pruned Model**: `server_output/pruned_model_{method}_{timestamp}.pt`
- **Results**: `server_output/pruning_results_{method}_{timestamp}.json`
- **Logs**: `server_logs/server_pruning_{timestamp}.log`

### Results JSON Structure
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "method": "activation_pruning_blocks_3_4",
  "config": {...},
  "baseline_metrics": {
    "mAP_0.5": 0.85,
    "mAP_0.5:0.95": 0.65,
    "precision": 0.82,
    "recall": 0.78
  },
  "pruned_metrics": {
    "mAP_0.5": 0.83,
    "mAP_0.5:0.95": 0.63,
    "precision": 0.80,
    "recall": 0.76
  },
  "performance_retention": 97.6,
  "total_time_seconds": 1800.5
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Out of Memory
- Reduce batch size in configuration
- Use smaller model (yolov8n.pt instead of yolov8s.pt)
- Reduce max_train_samples and max_val_samples

#### 3. Missing Dependencies
```bash
# Reinstall requirements
pip install -r server_requirements.txt

# Or install individually
pip install torch ultralytics opencv-python numpy PyYAML
```

#### 4. Dataset Issues
- Ensure images are in correct directories
- Check label file format (YOLO format)
- Verify VOC_adva.yaml configuration

#### 5. Import Issues
```bash
# Test imports specifically
python test_imports.py

# Check pruning directory structure
ls -la pruning/
# Should show:
# pruning_yolo_v8.py
# yolov8_utils.py
# clustering.py
# yolo_layer_pruner.py

# Manual import test
python -c "import sys; sys.path.insert(0, 'pruning'); import pruning_yolo_v8; print('Import successful')"
```

### Performance Optimization

#### GPU Optimization
- Use GPU with 8GB+ VRAM for best performance
- Enable mixed precision training (automatic)
- Use multiple GPUs if available

#### CPU Optimization
- Reduce batch size to 4-8
- Limit max_train_samples to 500
- Use fewer workers in configuration

## 📚 Advanced Usage

### Custom Configuration
```python
from server_config import ServerConfig

# Create custom config
config = ServerConfig("custom_config.yaml")
config.print_summary()

# Validate setup
validation = config.validate_paths()
```

### Programmatic Usage
```python
from server_pruning import ServerPruningPipeline, ServerPruningConfig

# Setup
config = ServerPruningConfig()
pipeline = ServerPruningPipeline(config)

# Run pruning
results = pipeline.run_pipeline("activation_pruning_blocks_3_4", "yolov8s.pt")
```

## 🤝 Support

### Getting Help
1. Check the logs in `server_logs/`
2. Run `python server_config.py` to validate setup
3. Test with `python server_setup.py --skip-test` to verify installation

### Performance Tips
- Use SSD storage for faster I/O
- Ensure sufficient RAM (16GB+ recommended)
- Monitor GPU memory usage during training
- Use appropriate batch sizes for your hardware

## 📄 License

This project uses the same license as the original YOLOv8 pruning implementation.

---

**Happy Pruning! 🚀**

