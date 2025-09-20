# from ultralytics import YOLO
# model = YOLO('/Users/ahelman/adva_yolo_pruning/pruned_v2_yolov8n.pt')  # or any YOLOv8 model
# # model.train(data="/Users/ahelman/adva_yolo_pruning/pruning/data/VOC_adva.yaml", epochs=3, imgsz=640, device='cpu')  # or 'cpu'
import torch

from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

original_path = '/Users/ahelman/adva_yolo_pruning/yolov8n.pt'
pruned_path = '/Users/ahelman/adva_yolo_pruning/pruned_v2_yolov8n.pt'

# Load state_dicts
orig_sd = torch.load(original_path, map_location='cpu', weights_only=True)
pruned_sd = torch.load(pruned_path, map_location='cpu', weights_only=True)

# Collect conv weights info
def get_conv_channels(sd):
    return {k: v.shape[0] for k, v in sd.items() if 'conv' in k and 'weight' in k}

orig_convs = get_conv_channels(orig_sd)
pruned_convs = get_conv_channels(pruned_sd)

print(f"{'Layer':40s} | {'Original':>8s} | {'Pruned':>8s}")
print('-'*60)
for k in orig_convs:
    orig_ch = orig_convs[k]
    pruned_ch = pruned_convs.get(k, 'N/A')
    print(f"{k:40s} | {str(orig_ch):>8s} | {str(pruned_ch):>8s}")