import numpy as np
from scipy.spatial.distance import jensenshannon
from itertools import combinations
import pandas as pd
import csv

import ast
from collections import defaultdict

def parse_activation_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    current_channel = None
    activations = defaultdict(lambda: defaultdict(list))  # channel → class → list of values

    for line in lines:
        line = line.strip()
        if line.startswith("Channel"):
            current_channel = int(line.split()[1].replace(":", ""))
        elif line.startswith("Class") and current_channel is not None:
            parts = line.split(":", 1)
            class_id = int(parts[0].split()[1])
            values = ast.literal_eval(parts[1].strip())
            activations[current_channel][class_id] = values

    return activations

def fix_distribution(vec, eps=1e-8):
    vec = np.clip(vec, 0, None)        # remove negative values
    total = np.sum(vec)
    if total == 0:
        return np.ones_like(vec) / len(vec)  # fallback to uniform
    return vec / total

def compute_mean_activations(activations):
    mean_dict = {}

    for ch in activations:
        mean_dict[ch] = {}
        for cls in activations[ch]:
            values = activations[ch][cls]
            # values = fix_distribution(values)
            if values:
                mean_val = float(np.mean(values))
            else:
                mean_val = 0.0  # or np.nan if you prefer
            mean_dict[ch][cls] = mean_val

    return mean_dict

def dense_matrix(mean_activations):
    """
    Create a dense matrix of activations.
    This function is a placeholder and does not perform any operations.
    """
    classes = sorted({cls for d in mean_activations.values() for cls in d.keys()})
    channels = sorted(mean_activations.keys())

    dist_matrix = np.zeros((len(channels), len(classes)))
    for i, ch in enumerate(channels):
        for j, cls in enumerate(classes):
            dist_matrix[i, j] = float(mean_activations[ch].get(cls, 0.0))  # fill missing classes with 0

    return channels,classes,dist_matrix
    


def compute_jensen_shannon(channels, dist_matrix):
    """
    Compute the Jensen-Shannon divergence between two distributions.
    This function is a placeholder and does not perform any operations.
    """
    # Distance matrix
    jsd_matrix = np.zeros((len(channels), len(channels)))

    for i in range(len(channels)):
        for j in range(len(channels)):
            if i == j:
                jsd_matrix[i, j] = 0.0
            else:
                jsd_matrix[i, j] = jensenshannon(dist_matrix[i], dist_matrix[j])  # returns sqrt(JS)

    # Optional: convert to DataFrame
    jsd_df = pd.DataFrame(jsd_matrix, index=channels, columns=channels)
    return jsd_df
 
if __name__ == "__main__":
    file_path = "/Users/ahelman/adva_yolo_pruning/channel_activation_distributions.txt"  # Replace with your file path
    activations = parse_activation_file(file_path)
    mean_activations = compute_mean_activations(activations)
    for channel, classes in mean_activations.items():
        print(f"Channel {channel}:")
        for class_id, mean_value in classes.items():
            print(f"  Class {class_id}: Mean Activation = {mean_value:.4f}")
    print("Mean activations computed successfully.")
    channels, classes, metrics = dense_matrix(mean_activations)

    jsd_matrix = compute_jensen_shannon(channels, metrics)
    print("Jensen-Shannon Divergence Matrix:")
    print(jsd_matrix)
    num_channels = jsd_matrix.shape[0]

    with open("jsd_matrix.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        header = [""] + [f"Channel {i+1}" for i in range(num_channels)]
        writer.writerow(header)
        # Write each row with row header
        for i in range(num_channels):
            row = [f"Channel {i+1}"] + list(jsd_matrix[i])
            writer.writerow(row)
    