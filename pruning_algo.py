import sys
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import torch.nn as nn

# Add YOLOv5 directory to path
yolov5_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'yolov5')
print("Yolov5 path:", yolov5_path)
sys.path.append(yolov5_path)
print("sys.path:", sys.path) 

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.metrics import box_iou
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import create_dataloader
from yolov5.val import run as validate_model

def main():
    # Configuration
    weights = '/Users/ahelman/adva_yolo_pruning/pruning/data/model/best.pt'  # model path
    data_yaml = '/Users/ahelman/adva_yolo_pruning/pruning/data/VOC_adva.yaml'  # dataset config
    device = select_device('cpu')  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    batch_size = 32
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45

    # 1. Load model
    print("Loading model...")
    model = attempt_load(weights, device=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    
    # 2. Evaluate baseline
    print("Evaluating baseline model...")
    baseline_results = evaluate_model(model, data_yaml, imgsz, batch_size, device)
    print(f"Baseline mAP@0.5: {baseline_results[0]:.4f}, mAP@0.5:0.95: {baseline_results[1]:.4f}")
    
    # # 3. Apply pruning
    # print("Applying pruning...")
    # pruned_model = prune_model(model, data_yaml, imgsz, batch_size, device)
    
    # # 4. Evaluate pruned model
    # print("Evaluating pruned model...")
    # pruned_results = evaluate_model(pruned_model, data_yaml, imgsz, batch_size, device)
    # print(f"Pruned mAP@0.5: {pruned_results[0]:.4f}, mAP@0.5:0.95: {pruned_results[1]:.4f}")
    
    # # 5. Save pruned model
    # torch.save(pruned_model.state_dict(), 'pruned_model.pt')
    # print("Pruned model saved to pruned_model.pt")

def evaluate_model(model, data_yaml, img_size, batch_size, device):
    """Evaluate model on validation set and return detailed metrics."""
    import yaml

    # Load YAML file as dictionary
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)

    results = validate_model(
        data=data_dict,
        weights=None,
        batch_size=batch_size,
        imgsz=img_size,
        conf_thres=0.001,
        iou_thres=0.65,
        device=device,
        model=model,
        dataloader=None,
        save_dir=Path(''),
        save_json=False,
        verbose=False
    )
    # results: (precision, recall, mAP_0.5, mAP_0.5:0.95, ...)

    metrics = {
        "precision": results[0],
        "recall": results[1],
        "mAP_0.5": results[2],
        "mAP_0.5:0.95": results[3],
        # Add more metrics if needed from results
    }
    return metrics

def prune_model(model, data_yaml, img_size, batch_size, device):
    """Apply pruning algorithm to model"""
    # TODO: Implement pruning based on pseudo code
    return model

if __name__ == "__main__":
    main()
