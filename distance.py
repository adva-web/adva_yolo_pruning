import numpy as np
from typing import Union

from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calc_mss_value(space: np.ndarray, clustering: dict) -> Optional[float]:
    """
    Computes the Mean Simplified Silhouette (MSS) value to evaluate clustering quality.

    This function calculates the MSS value, which assesses clustering quality based on intra-cluster
    and nearest-cluster distances for each data point.

    Args:
        space (np.ndarray): A 2D array where each row represents a data point in the feature space.
        clustering (dict): Clustering information containing the following keys:
            - 'labels' (np.ndarray): Array of cluster labels for each data point in `space`.
            - 'medoids_loc' (np.ndarray): A 2D array where each row is the centroid of a cluster in the feature space.

    Returns:
        Optional[float]: The mean silhouette score, with higher scores indicating better clustering quality.
            Returns 1 if all intra-cluster distances are zero, indicating perfect clustering.
    """
    cluster_labels = clustering['labels']
    cluster_centers = clustering['medoids_loc']

    # Calculate intra-cluster distances for each data point
    intra_cluster_distances = euclidean_distances(
        space, cluster_centers[cluster_labels]
    ).diagonal()

    # Initialize nearest-cluster distances array
    nearest_cluster_distances = np.zeros_like(intra_cluster_distances)

    # Calculate the nearest-cluster distances
    for cluster_index, _ in enumerate(cluster_centers):
        cluster_members = space[cluster_labels == cluster_index]
        if cluster_members.size == 0:
            continue

        # Exclude the current cluster center
        non_current_centers = np.delete(cluster_centers, cluster_index, axis=0)
        distances_to_non_current_centers = euclidean_distances(
            cluster_members, non_current_centers
        )
        nearest_cluster_distances[cluster_labels == cluster_index] = (
            distances_to_non_current_centers.mean(axis=1)
        )

    # Check for non-zero intra-cluster distances
    valid_distances_mask = intra_cluster_distances != 0
    if not np.any(valid_distances_mask):
        return 1

    # Calculate silhouette scores for each valid point
    a = intra_cluster_distances[valid_distances_mask]
    b = nearest_cluster_distances[valid_distances_mask]
    silhouette_scores = (b - a) / np.maximum(a, b)

    # Return the mean silhouette score
    return np.mean(silhouette_scores)


def get_distance(metric: str, class1_data: np.ndarray, class2_data: np.ndarray) -> float:
    """
    Computes the distance between two classes based on the specified metric.

    This function serves as a dispatcher that selects the appropriate distance calculation
    method according to the specified metric.

    Supported metrics:
        - 'jm': Jeffries-Matusita (JM) distance
        - 'bhattacharyya': Bhattacharyya distance
        - 'wasserstein': Wasserstein distance

    Args:
        metric (str): The name of the metric to use for computing the distance.
                      Options are 'jm', 'bhattacharyya', and 'wasserstein'.
        class1_data (np.ndarray): Data points of the first class.
        class2_data (np.ndarray): Data points of the second class.

    Returns:
        float: The computed distance according to the selected metric.

    Raises:
        ValueError: If an unsupported metric name is provided.
    """
    if metric == 'jm':
        return jm_distance(class1_data, class2_data)
    elif metric == 'bhattacharyya':
        return bhattacharyya_distance(class1_data, class2_data)
    elif metric == 'wasserstein':
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(class1_data, class2_data)
    else:
        raise ValueError(f"Unsupported metric '{metric}' provided.")


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Hellinger distance between two probability distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The Hellinger distance between distributions `p` and `q`.
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2)


def jm_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Jeffries-Matusita (JM) distance between two probability distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The JM distance between distributions `p` and `q`.
    """
    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Bhattacharyya distance between two probability distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The Bhattacharyya distance between distributions `p` and `q`.
    """
    mean_p, mean_q = p.mean(), q.mean()
    std_p = p.std() if p.std() != 0 else 1e-10
    std_q = q.std() if q.std() != 0 else 1e-10

    var_p, var_q = std_p**2, std_q**2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + 0.5 * np.log(
        (var_p + var_q) / (2 * (std_p * std_q))
    )
    return b
