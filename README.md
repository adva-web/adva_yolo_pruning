# YOLOv8 Pruning Server

A server-ready implementation of YOLOv8 model pruning with configurable deployment options.

## 🚀 Quick Start

### 1. Deploy with Docker (Recommended)

```bash
# Clone and setup
git clone <your-repo>
cd adva_yolo_pruning

# Run deployment script
chmod +x deploy.sh
./deploy.sh

# Start the server
docker-compose up yolo-pruning
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python server_pruning.py --help
```

## 📁 Project Structure

```
adva_yolo_pruning/
├── server_pruning.py          # Main server application
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service orchestration
├── deploy.sh                  # Deployment automation
├── data/                      # Dataset directory
│   ├── dataset.yaml          # Dataset configuration
│   ├── images/               # Training/validation images
│   │   ├── train/
│   │   └── val/
│   └── labels/               # YOLO format labels
│       ├── train/
│       └── val/
├── models/                    # Model weights
│   └── *.pt                  # YOLOv8 model files
├── output/                    # Pruned models and results
├── logs/                      # Application logs
└── pruning/                   # Core pruning modules
    ├── pruning_yolo_v8.py    # Main pruning implementation
    ├── yolov8_utils.py       # Utilities
    └── ...                   # Other modules
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | Device for training/inference (`cpu` or `cuda`) |
| `BATCH_SIZE` | `16` | Batch size for training/evaluation |
| `IMG_SIZE` | `640` | Image size for processing |
| `EPOCHS` | `30` | Number of training epochs |
| `PRUNING_METHOD` | `activation_pruning_blocks_3_4` | Pruning algorithm |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `YOLO_MODEL_PATH` | `/app/models/yolov8s.pt` | Path to base model |
| `DATA_YAML_PATH` | `/app/data/dataset.yaml` | Dataset configuration |

### Configuration Methods

1. **Environment Variables**: Set in `.env` file or system environment
2. **Configuration File**: Pass YAML file with `--config` parameter
3. **Command Line**: Override with command-line arguments

## 🐳 Docker Deployment

### CPU Version

```bash
# Build and run
docker-compose up yolo-pruning

# Run specific job
docker-compose run --rm yolo-pruning python server_pruning.py \
  --model /app/models/yolov8s.pt \
  --data /app/data/dataset.yaml \
  --method activation_pruning_blocks_3_4
```

### GPU Version

```bash
# Requires nvidia-docker
docker-compose --profile gpu up yolo-pruning-gpu

# With custom configuration
docker-compose run --rm yolo-pruning-gpu python server_pruning.py \
  --model /app/models/your_model.pt \
  --data /app/data/your_dataset.yaml \
  --method 50_percent_gamma_pruning_blocks_3_4
```

## 🔧 Usage Examples

### Command Line Interface

```bash
# Basic usage
python server_pruning.py

# With custom parameters
python server_pruning.py \
  --model models/yolov8s.pt \
  --data data/dataset.yaml \
  --method activation_pruning_blocks_3_4 \
  --output results/

# Skip baseline evaluation
python server_pruning.py --no-baseline

# Skip pruned model evaluation
python server_pruning.py --no-pruned
```

### Python API

```python
from server_pruning import YOLOv8PruningServer

# Initialize server
server = YOLOv8PruningServer()

# Run complete pipeline
results = server.run_pruning_pipeline(
    model_path="models/yolov8s.pt",
    data_yaml="data/dataset.yaml",
    evaluate_baseline=True,
    evaluate_pruned=True
)

# Check results
if results["status"] == "success":
    print(f"Baseline mAP: {results['baseline_metrics']['mAP_0.5']}")
    print(f"Pruned mAP: {results['pruned_metrics']['mAP_0.5']}")
```

## 🧠 Pruning Methods

### Available Methods

1. **`activation_pruning_blocks_3_4`**: Activation-based pruning for blocks 3-4
2. **`50_percent_gamma_pruning_blocks_3_4`**: Gamma-based 50% pruning
3. **`conv2d_with_activations`**: Conv2D pruning with activation analysis

### Custom Pruning

To add custom pruning methods:

1. Implement in `pruning/pruning_yolo_v8.py`
2. Add to method selection in `server_pruning.py`
3. Update configuration options

## 📊 Output and Results

### Result Files

- **Pruned Models**: Saved in `output/` directory as `.pt` files
- **Metrics**: JSON files with detailed evaluation results
- **Logs**: Application logs in `logs/` directory

### Example Result Structure

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "status": "success",
  "config": {...},
  "dataset_info": {
    "num_classes": 20,
    "train_samples": 1000,
    "val_samples": 200
  },
  "baseline_metrics": {
    "mAP_0.5": 0.85,
    "mAP_0.5:0.95": 0.65,
    "precision": 0.82,
    "recall": 0.78
  },
  "pruned_metrics": {
    "mAP_0.5": 0.82,
    "mAP_0.5:0.95": 0.62,
    "precision": 0.80,
    "recall": 0.75
  },
  "pruned_model_path": "/app/output/pruned_model_activation_pruning_blocks_3_4.pt"
}
```

## 🔨 Development

### Local Development

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

### Adding Features

1. **New Pruning Methods**: Add to `pruning/pruning_yolo_v8.py`
2. **API Endpoints**: Extend `server_pruning.py` with Flask/FastAPI
3. **Monitoring**: Add metrics collection and monitoring

## 🐛 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Use CPU instead
   export DEVICE=cpu
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   
   # Monitor memory usage
   docker stats
   ```

3. **Dataset Loading Errors**
   - Ensure images and labels are properly formatted
   - Check paths in `dataset.yaml`
   - Verify file permissions

### Logs and Debugging

```bash
# View logs
docker-compose logs yolo-pruning

# Debug mode
export LOG_LEVEL=DEBUG

# Access container for debugging
docker-compose exec yolo-pruning bash
```

## 📈 Performance Optimization

### Server Optimization

1. **Resource Allocation**
   ```yaml
   # docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 16G
         cpus: '8'
   ```

2. **GPU Configuration**
   ```bash
   # Enable all GPUs
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

3. **Batch Processing**
   ```bash
   # Increase batch size for better GPU utilization
   export BATCH_SIZE=32
   ```

## 🚀 Production Deployment

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yolo-pruning-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: yolo-pruning
  template:
    metadata:
      labels:
        app: yolo-pruning
    spec:
      containers:
      - name: yolo-pruning
        image: yolo-pruning:latest
        env:
        - name: DEVICE
          value: "cuda"
        - name: BATCH_SIZE
          value: "32"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

### Monitoring

```bash
# Add monitoring with Prometheus/Grafana
# Health checks, metrics collection, alerting
```

## 📄 License

[Add your license information here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `logs/` directory
- Create an issue on GitHub
- Contact the development team 