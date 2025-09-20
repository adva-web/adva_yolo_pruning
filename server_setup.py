#!/usr/bin/env python3
"""
Server Setup Script for YOLOv8 Pruning
Automated setup and validation for GPU server deployment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse

def print_header():
    """Print setup header."""
    print("🚀 YOLOv8 Pruning Server Setup")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 50)

def check_python_version():
    """Check Python version compatibility."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu_support():
    """Check GPU support."""
    print("\n🔍 Checking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Count: {gpu_count}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, will check after installation")
        return None

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    try:
        # Check if requirements file exists
        req_file = Path("server_requirements.txt")
        if not req_file.exists():
            req_file = Path("requirements.txt")
            
        if req_file.exists():
            print(f"Installing from {req_file}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(req_file)
            ])
        else:
            print("Installing basic requirements...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "ultralytics", "opencv-python", 
                "numpy", "PyYAML", "tqdm", "matplotlib", "scipy", "pandas"
            ])
        
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "server_output",
        "server_logs",
        "pruning/dataset_voc/images/train",
        "pruning/dataset_voc/images/val",
        "pruning/dataset_voc/labels/train",
        "pruning/dataset_voc/labels/val"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory: {dir_path}")
    
    return True

def check_model_files():
    """Check for YOLOv8 model files."""
    print("\n🔍 Checking for model files...")
    
    model_files = list(Path(".").glob("*.pt"))
    if model_files:
        print(f"✅ Found model files: {[f.name for f in model_files]}")
        return True
    else:
        print("⚠️  No .pt model files found")
        print("   You can download a model with:")
        print("   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt")
        return False

def check_dataset():
    """Check dataset availability."""
    print("\n🔍 Checking dataset...")
    
    dataset_paths = [
        "pruning/dataset_voc/images/train",
        "pruning/dataset_voc/images/val",
        "pruning/data/VOC_adva.yaml"
    ]
    
    has_data = True
    for path in dataset_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len(list(Path(path).glob("*")))
                print(f"✅ Dataset: {path} ({file_count} files)")
            else:
                print(f"✅ Config: {path}")
        else:
            print(f"❌ Missing: {path}")
            has_data = False
    
    if not has_data:
        print("\n⚠️  Dataset not complete. Please ensure:")
        print("   1. Images are in pruning/dataset_voc/images/train and val")
        print("   2. Labels are in pruning/dataset_voc/labels/train and val")
        print("   3. VOC_adva.yaml config exists in pruning/data/")
    
    return has_data

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    modules = [
        ("torch", "PyTorch"),
        ("ultralytics", "Ultralytics"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    all_imported = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_imported = False
    
    return all_imported

def test_pruning_imports():
    """Test pruning module imports."""
    print("\n🧪 Testing pruning modules...")
    
    # Add pruning directory to path
    pruning_dir = Path("pruning")
    if not pruning_dir.exists():
        print("❌ Pruning directory not found")
        return False
        
    sys.path.insert(0, str(pruning_dir))
    
    pruning_modules = [
        ("pruning_yolo_v8", "Main pruning functions"),
        ("yolov8_utils", "YOLOv8 utilities"),
        ("clustering", "Clustering algorithms"),
        ("yolo_layer_pruner", "Layer pruner")
    ]
    
    all_imported = True
    for module, name in pruning_modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_imported = False
    
    # Test specific function imports
    if all_imported:
        print("\n🔍 Testing specific function imports...")
        try:
            import pruning_yolo_v8
            required_functions = [
                'apply_activation_pruning_blocks_3_4',
                'apply_50_percent_gamma_pruning_blocks_3_4',
                'prune_conv2d_in_block_with_activations'
            ]
            
            for func_name in required_functions:
                if hasattr(pruning_yolo_v8, func_name):
                    print(f"✅ Function available: {func_name}")
                else:
                    print(f"❌ Function not found: {func_name}")
                    all_imported = False
        except Exception as e:
            print(f"❌ Error testing functions: {e}")
            all_imported = False
    
    return all_imported

def create_sample_config():
    """Create a sample configuration file."""
    print("\n📝 Creating sample configuration...")
    
    config_content = """# Server Configuration for YOLOv8 Pruning
# Modify these paths to match your setup

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
"""
    
    config_file = Path("server_config.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Sample config created: {config_file}")
    return True

def run_quick_test():
    """Run a quick test of the setup."""
    print("\n🧪 Running quick test...")
    
    try:
        # Test configuration
        from server_config import ServerConfig
        config = ServerConfig()
        print("✅ Configuration test passed")
        
        # Test basic YOLO import
        from ultralytics import YOLO
        print("✅ YOLO import test passed")
        
        # Test pruning imports
        sys.path.append(str(Path("pruning")))
        from pruning_yolo_v8 import apply_50_percent_gamma_pruning_blocks_3_4
        print("✅ Pruning import test passed")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 USAGE EXAMPLES")
    print("=" * 50)
    print("1. Test configuration:")
    print("   python server_config.py")
    print()
    print("2. Run pruning with different methods:")
    print("   python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt")
    print("   python server_pruning.py --method 50_percent_gamma_pruning_blocks_3_4 --model yolov8n.pt")
    print("   python server_pruning.py --method conv2d_with_activations --model yolov8s.pt")
    print()
    print("3. Run with custom output directory:")
    print("   python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt --output-dir /path/to/output")
    print()
    print("4. Run with verbose logging:")
    print("   python server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt --verbose")
    print("=" * 50)

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="YOLOv8 Pruning Server Setup")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test")
    parser.add_argument("--quick", action="store_true", help="Quick setup (skip some checks)")
    args = parser.parse_args()
    
    print_header()
    
    success = True
    
    # Run all checks
    success &= check_python_version()
    
    if not args.skip_install:
        success &= install_requirements()
    
    success &= setup_directories()
    success &= check_model_files()
    
    if not args.quick:
        success &= check_dataset()
    
    success &= test_imports()
    success &= test_pruning_imports()
    success &= create_sample_config()
    
    # Check GPU after installation
    if not args.skip_install:
        gpu_status = check_gpu_support()
        if gpu_status is True:
            print("🚀 GPU acceleration will be used")
        elif gpu_status is False:
            print("⚠️  CPU-only mode (slower but functional)")
    
    if not args.skip_test:
        success &= run_quick_test()
    
    if success:
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        show_usage_examples()
    else:
        print("\n❌ Setup failed. Please fix the issues above and run again.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r server_requirements.txt")
        print("2. Download a YOLOv8 model: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt")
        print("3. Ensure dataset is properly organized in pruning/dataset_voc/")
        sys.exit(1)

if __name__ == "__main__":
    main()

