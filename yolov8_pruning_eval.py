import torch
import os
from ultralytics import YOLO
import logging
import yaml
import cv2
import numpy as np
import glob
import os

from pruning_yolo_v8 import apply_activation_pruning_blocks_3_4, apply_50_percent_gamma_pruning_blocks_3_4,prune_conv2d_in_block_with_activations,apply_pruning_v8,apply_gamma_pruning_iter, apply_gamma_pruning_on_block_zeroed

logger = logging.getLogger("yolov8_pruning")
logging.basicConfig(level=logging.INFO)

def train_model(path, data_yaml):
    """
    Train the YOLOv8 model on the provided training data.
    """
    model = YOLO(path)

    model.train(
    data=data_yaml,  # Path to your dataset YAML
    epochs=30,                 # Number of epochs
    imgsz=640,                 # Image size
    batch=16,                  # Batch size (adjust as needed)
    device='cpu'              # Use 'cpu' or 'cuda'
)
    return model


def load_samples(image_dir, label_dir):
    samples = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))  # or .png
    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    # YOLO format: class x_center y_center width height (normalized)
                    labels.append({
                        "class_id": class_id,
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4])
                    })
        samples.append({
            "image": img,
            "label": labels,
            "image_path": img_path,
            "label_path": label_path
        })
    return samples

def evaluate_model(model_path, data_yaml, imgsz=640, batch=32, device='cpu', conf=0.25, iou=0.45):
    """
    Evaluate a YOLOv8 model and return metrics.
    """
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
        "per_class_mAP": results.maps if hasattr(results, "maps") else None,
        "mean_mAP_0.5_0.95": float(results.maps.mean()) if hasattr(results, "maps") else None,
        "speed": results.speed if hasattr(results, "speed") else None,
        "inference_time": results.speed.get("inference", None) if hasattr(results, "speed") else None
    }
    return metrics, results

def prune_model(
    model_path,
    train_data,
    valid_data,
    classes,
    last_layer_idx=3,
    save_path="pruned_model.pt"
):
    """
    Main entry point for pruning a YOLOv8 model.
    Calls apply_pruning_v8 and saves the final pruned model.
    """
    logger.info("Starting full pruning pipeline for YOLOv8.")
    # pruned_model = prune_conv2d_in_block_with_activations(
    #     model_path=model_path,
    #     train_data=train_data,
    #     valid_data=valid_data,
    #     classes=classes,
    # )

    # pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
    #     model_path=model_path,
    #     layers_to_prune=3
    # )
    # Save the pruned model weights
    pruned_model = apply_activation_pruning_blocks_3_4(
        model_path=model_path,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
    )
    

    torch_model = pruned_model.model
    torch.save(torch_model.state_dict(), save_path)
    logger.info(f"Pruned model saved to {save_path}")
    print(f"DEBUG: Pruned model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    # # Paths
    # weights_yolo_n = "/Users/ahelman/adva_yolo_pruning/runs/detect/train/weights/best.pt"  # Path to YOLOv8 weights
    data_yaml = "/Users/ahelman/adva_yolo_pruning/pruning/data/VOC_adva.yaml" # Path to dataset YAML
    weights_yolo_s = "/Users/ahelman/adva_yolo_pruning/runs/detect/train128/weights/best.pt"
    # train_model(weights, data_yaml)

    # # 1. Evaluate baseline
    # print("Evaluating baseline model...")
    # baseline_metrics, results = evaluate_model(weights_yolo_s, data_yaml)
    # print("Baseline metrics:", baseline_metrics)
    # logger.info(f"Original metrics: {results.results_dict}")
    # print("DEBUG: Original evaluation complete.")


    # 2. Prune model   
    # Load class names
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    classes_names = data_cfg["names"]
    classes = list(range(len(classes_names)))
    print("Classes:", classes)
    # Load samples
    train_img_dir = data_cfg["train"]
    val_img_dir = data_cfg["val"]
    train_label_dir = train_img_dir.replace("/images", "/labels")
    val_label_dir = val_img_dir.replace("/images", "/labels")

    train_data = load_samples(train_img_dir, train_label_dir)
    valid_data = load_samples(val_img_dir, val_label_dir)
    print(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples.")

 
    print("Pruning model...")
    pruned_weights = prune_model(
        model_path=weights_yolo_s,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        last_layer_idx=3,
        save_path="pruned_v3_yolov8n.pt"
    )


    # # 3. Evaluate pruned model
    # # print("Evaluating pruned model...")
    # # pruned_metrics = evaluate_model("pruned_yolov8n.pt", data_yaml)
    # # print("Pruned metrics:", pruned_metrics)