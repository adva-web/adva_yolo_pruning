#!/bin/bash

# YOLOv8 Pruning Server Deployment Script
# Quick deployment for GPU servers

set -e  # Exit on any error

echo "🚀 YOLOv8 Pruning Server Deployment"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "server_pruning.py" ]; then
    echo "❌ server_pruning.py not found. Please run this script from the project root directory."
    exit 1
fi

# Make scripts executable
chmod +x server_setup.py
chmod +x server_pruning.py
chmod +x server_config.py

echo "✅ Made scripts executable"

# Run setup
echo "🔧 Running server setup..."
python3 server_setup.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "🎯 Ready to run pruning! Examples:"
    echo "  python3 server_pruning.py --method activation_pruning_blocks_3_4 --model yolov8s.pt"
    echo "  python3 server_pruning.py --method 50_percent_gamma_pruning_blocks_3_4 --model yolov8n.pt"
    echo ""
    echo "📚 For more information, see SERVER_README.md"
else
    echo "❌ Setup failed. Please check the errors above."
    exit 1
fi

