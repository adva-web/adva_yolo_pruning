import torch
import torch.nn as nn
from collections import defaultdict
from tabulate import tabulate
from typing import Tuple, List
import torch.nn as nn
import logging
import numpy as np 
import math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Union



logger = logging.getLogger("yolov8_pruning")
logging.basicConfig(level=logging.INFO)

@dataclass
class DetectionMatch:
    image_path: str
    label_path: str
    object_index: int
    class_id: int
    gt_bbox: List[float]
    pred_bbox: List[float]
    patch_row: int
    patch_col: int
    iou: float
    patch_activation: List[float]


def extract_bn_gamma(bn: nn.BatchNorm2d) -> np.ndarray:
    """
    Extracts the gamma (weight) parameters from a BatchNorm2d layer.
    """
    return bn.weight.data.abs().cpu().numpy()

def get_conv_bn_pairs(model: nn.Module):
    """
    Extracts (Conv2d, BatchNorm2d) pairs from the model in order.
    Returns a list of (conv_layer, bn_layer) tuples.
    """
    pairs = []
    prev_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prev_conv = m
        elif isinstance(m, nn.BatchNorm2d) and prev_conv is not None:
            pairs.append((prev_conv, m))
            prev_conv = None
    return pairs
    
def prune_conv2d_layer_in_yolo(model: nn.Module,
                               conv_idx: int,
                               indices_to_keep: list):
    """
    Zero out all output channels except those in `indices_to_keep` for the Conv2d layer at conv_idx,
    doing the in-place ops inside torch.inference_mode() so that we can legally modify inference tensors.
    """
    all_conv_layers = get_all_conv2d_layers(model)
    if conv_idx < 0 or conv_idx >= len(all_conv_layers):
        logger.warning(f"conv_idx {conv_idx} is out of range for model with {len(all_conv_layers)} Conv2d layers.")
        return model

    target_conv = all_conv_layers[conv_idx]
    print(f"Pruning Conv2d layer at index {conv_idx}: id={id(target_conv)}, shape={tuple(target_conv.weight.shape)}")

    with torch.inference_mode():
        old_w = target_conv.weight.data.clone().detach()
        target_conv.weight.data.zero_()
        target_conv.weight.data[indices_to_keep, :, :, :] = old_w[indices_to_keep, :, :, :]

        if target_conv.bias is not None:
            old_b = target_conv.bias.data.clone().detach()
            target_conv.bias.data.zero_()
            target_conv.bias.data[indices_to_keep] = old_b[indices_to_keep]

        logger.info(
            f"Layer pruned. Kept {len(indices_to_keep)} / "
            f"{old_w.shape[0]} output channels."
        )

    return model

# def prune_conv2d_layer_in_yolo(model: nn.Module,
#                                target_conv: nn.Conv2d,
#                                indices_to_keep: list):
#     """
#     Zero out all output channels except those in `indices_to_keep`,
#     doing the in-place ops inside torch.inference_mode() so that we
#     can legally modify inference tensors.
#     """
#     found = False
#     print("Prune function model id:", id(model))

#     all_conv_layers = get_all_conv2d_layers(model)

#     print("=== Prune function : all_conv_layers ===")
#     for idx, layer in enumerate(all_conv_layers):
#         print(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

#     for m in model.modules():
#         print(f"Module: {type(m)}")
#         if isinstance(m, nn.Conv2d) :
#             print("m.weight.data_ptr():", m.weight.data_ptr())
#             print("target_conv.weight.data_ptr():", target_conv.weight.data_ptr())
#             print(f"id={id(m)}, shape={tuple(m.weight.shape)}")
#             print(f"id={id(target_conv)}, shape={tuple(target_conv.weight.shape)}")

#         if isinstance(m, nn.Conv2d) and \
#            m.weight.data_ptr() == target_conv.weight.data_ptr():
#             found = True
#             # Modify the existing weight & bias in-place,
#             # *inside* an inference_mode context:
#             with torch.inference_mode():
#                 # Clone old values (detached clone)
#                 old_w = m.weight.data.clone().detach()
#                 # Zero out everything
#                 m.weight.data.zero_()
#                 # Restore only the channels we want to keep
#                 m.weight.data[indices_to_keep, :, :, :] = old_w[indices_to_keep, :, :, :]

#                 if m.bias is not None:
#                     old_b = m.bias.data.clone().detach()
#                     m.bias.data.zero_()
#                     m.bias.data[indices_to_keep] = old_b[indices_to_keep]

#                 logger.info(
#                     f"Layer pruned. Kept {len(indices_to_keep)} / "
#                     f"{old_w.shape[0]} output channels."
#                 )
#             break

#     if not found:
#         logger.warning(
#             "Could not find the target_conv layer by data_ptr. "
#             "Make sure you're passing the exact same layer."
#         )

#     return model

def aggregate_activations_from_matches(matches: List[DetectionMatch], all_classes: List[int]) -> Dict[int, Dict[int, List[float]]]:
    if not matches:
        return {}
    num_channels = len(matches[0].patch_activation)
    aggregated = {ch: {cls: [] for cls in all_classes} for ch in range(num_channels)}
    for match in matches:
        for ch, act in enumerate(match.patch_activation):
            aggregated[ch][match.class_id].append(act)
    return aggregated
    

def get_all_conv2d_layers(model: nn.Module) -> List[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]

def get_partial_feature_map(x: torch.Tensor, mini_net: nn.Module) -> Tuple[torch.Tensor, float, float, int, int]:
    with torch.no_grad():
        feature_map = mini_net(x)
    _, _, fm_h, fm_w = feature_map.shape
    H, W = x.shape[2], x.shape[3]
    return feature_map, H / fm_h, W / fm_w, fm_h, fm_w


def build_mini_net(sliced_block: nn.Sequential, target_conv_layer: nn.Conv2d) -> nn.Sequential:
    layers = []
    for layer in sliced_block:
        layers.append(layer)
        # Check if this layer is the target; if so, stop here.
        if layer is target_conv_layer:
            break
    return nn.Sequential(*layers)

def extract_conv_weights_norm(conv: nn.Conv2d) -> np.ndarray:
    """
    Returns the L2 norm of each output channel's weights in a Conv2d layer.
    """
    weights = conv.weight  # shape: (out_channels, in_channels, kernel_H, kernel_W)
    norms = torch.norm(weights.view(weights.size(0), -1), p=2, dim=1)
    return norms.detach().cpu().numpy()

def convert_label_to_xyxy(label: Dict[str, Any], img_width: int, img_height: int) -> List[float]:
    xc = label['x_center'] * img_width
    yc = label['y_center'] * img_height
    w = label['width'] * img_width
    h = label['height'] * img_height
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

def compute_patch_indices(gt_bbox: List[float], stride_h: float, stride_w: float, fm_h: int, fm_w: int) -> Tuple[int, int]:
    x_center = (gt_bbox[0] + gt_bbox[2]) / 2.0
    y_center = (gt_bbox[1] + gt_bbox[3]) / 2.0
    patch_col = int(x_center // stride_w)
    patch_row = int(y_center // stride_h)
    return max(0, min(patch_row, fm_h - 1)), max(0, min(patch_col, fm_w - 1))

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def match_prediction_for_gt(gt_bbox: List[float], gt_class: int, pred_boxes: List[Dict[str, Any]], iou_threshold: float = 0.5) -> Tuple[Union[Dict[str, Any], None], float]:
    max_iou = 0.0
    best_pred = None
    for pred in pred_boxes:
        if pred['class'] == gt_class:
            iou = calculate_iou(gt_bbox, pred['bbox'])
            if iou > iou_threshold and iou > max_iou:
                max_iou = iou
                best_pred = pred
    return best_pred, max_iou


def process_sample_for_layer_idx_v8(sample: Dict[str, Any], model, mini_net: nn.Module) -> Tuple[List[DetectionMatch], List[DetectionMatch]]:
    """
    Process a single sample through the mini-net and YOLO model, returning matched
    and unmatched DetectionMatch lists.

    **Update**: we now resize/interpolate the input tensor so its spatial dimensions
    are multiples of the model stride, avoiding the "shape not divisible by stride" error.
    """
    matched_records = []
    unmatched_records = []
    image = sample['image']  # numpy HWC
    h, w = image.shape[:2]
    gt_labels = sample['label']
    image_path = sample['image_path']
    label_path = sample['label_path']

    # 1) Convert to tensor and normalize
    device = next(model.parameters()).device
    x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0  # 1×3×H×W

    # 2) Ensure H, W are divisible by model stride
    stride = int(model.model.stride.max())  # typically 32
    _, _, H, W = x.shape
    new_H = math.ceil(H / stride) * stride
    new_W = math.ceil(W / stride) * stride
    if new_H != H or new_W != W:
        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # 3) Get the feature map from the mini‐net (no grad)
    feature_map, stride_h, stride_w, fm_h, fm_w = get_partial_feature_map(x, mini_net)

    # 4) Run YOLO inference on the resized tensor
    predictions = model(x)[0]

    # 5) Extract bounding boxes and classes
    pred_boxes = [
        {'bbox': box.xyxy[0].tolist(), 'class': int(box.cls[0])}
        for box in predictions.boxes
    ]

    # 6) Match GT to predictions and collect activations
    for idx, gt in enumerate(gt_labels):
        gt_class = gt['class_id']
        gt_bbox = convert_label_to_xyxy(gt, w, h)  # original coords
        patch_row, patch_col = compute_patch_indices(gt_bbox, stride_h, stride_w, fm_h, fm_w)
        patch_activation = feature_map[0, :, patch_row, patch_col].tolist()
        best_pred, max_iou = match_prediction_for_gt(gt_bbox, gt_class, pred_boxes)

        record = DetectionMatch(
            image_path=image_path,
            label_path=label_path,
            object_index=idx,
            class_id=gt_class,
            gt_bbox=gt_bbox,
            pred_bbox=best_pred['bbox'] if best_pred else [],
            patch_row=patch_row,
            patch_col=patch_col,
            iou=max_iou,
            patch_activation=patch_activation
        )

        (matched_records if best_pred else unmatched_records).append(record)

    return matched_records, unmatched_records


def get_raw_objects_debug_v8(model, mini_net: nn.Sequential, dataset: list) -> Tuple[list, list]:
    """
    For YOLOv8: Processes each sample in the dataset, collects matched/unmatched detections,
    and prints per-class statistics for debugging.
    """
    layer_matched = []
    layer_unmatched = []

    stats = defaultdict(lambda: {
        "matched_count": 0,
        "unmatched_count": 0,
        "iou_sum_matched": 0.0,
        "iou_sum_all": 0.0,
        "total_objects": 0
    })

    for sample in dataset:
        # You must implement this for YOLOv8!
        matched, unmatched = process_sample_for_layer_idx_v8(sample, model, mini_net)
        layer_matched.extend(matched)
        layer_unmatched.extend(unmatched)

        for record in matched + unmatched:
            class_id = record.class_id
            stats[class_id]["total_objects"] += 1
            stats[class_id]["iou_sum_all"] += record.iou

            if record.iou > 0.5:
                stats[class_id]["matched_count"] += 1
                stats[class_id]["iou_sum_matched"] += record.iou
            else:
                stats[class_id]["unmatched_count"] += 1

    table = []
    headers = [
        "class_id",
        "matched_count",
        "unmatched_count",
        "total_objects",
        "avg_iou_all",
        "avg_iou_matched"
    ]

    class_ids = sorted(stats.keys())
    for class_id in class_ids:
        s = stats[class_id]
        avg_iou_all = s["iou_sum_all"] / s["total_objects"] if s["total_objects"] > 0 else 0.0
        avg_iou_matched = s["iou_sum_matched"] / s["matched_count"] if s["matched_count"] > 0 else 0.0
        table.append([
            class_id,
            s["matched_count"],
            s["unmatched_count"],
            s["total_objects"],
            round(avg_iou_all, 4),
            round(avg_iou_matched, 4)
        ])

    logger.info("\n" + tabulate(table, headers=headers, tablefmt="grid"))

    return layer_matched, layer_unmatched

