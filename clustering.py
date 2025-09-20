from typing import Dict, List

import numpy as np
from loguru import logger
from kmedoids import fasterpam
from kneed import KneeLocator
from sklearn.metrics.pairwise import pairwise_distances

from distance import calc_mss_value

def kmedoids_fasterpam(data: np.ndarray, k: int) -> dict:
    """
    Performs KMedoids clustering using the FasterPAM algorithm on the provided data.

    Args:
        data (np.ndarray): The dataset to cluster, as a NumPy array.
        k (int): The number of clusters to use for KMedoids clustering.

    Returns:
        dict: A dictionary containing:
            - 'labels' (np.ndarray): Cluster labels for each data point.
            - 'medoids' (np.ndarray): Indices of the medoids.
            - 'medoids_loc' (np.ndarray): Locations of the medoids in the data.
    """
    distances = pairwise_distances(data)
    k_medoids = fasterpam(distances, k)
    medoids_loc = data[k_medoids.medoids]

    return {
        'labels': k_medoids.labels,
        'medoids': k_medoids.medoids,
        'medoids_loc': medoids_loc,
    }


def get_knee(x: list, y: list, poly_deg: int = 2) -> int:
    """
    Identifies the 'knee' point in a curve using the KneeLocator.

    Args:
        x (list): The x-values of the curve.
        y (list): The y-values of the curve.
        poly_deg (int, optional): The degree of the polynomial used for interpolation. Defaults to 2.

    Returns:
        int: The x-value of the knee point if found; otherwise, None.
    """
    kn = KneeLocator(
        x,
        y,
        curve='concave',
        direction='increasing',
        interp_method='polynomial',
        polynomial_degree=poly_deg,
    )

    return kn.knee

def select_optimal_components(graph_space: Dict[str, np.ndarray], weights: np.ndarray, num_components: int, k_value: int, weight_form: bool = True) -> List[int]:
    """
    Selects optimal components based on the k-medoids clustering and MSS value.

    This function performs k-medoids clustering on the reduced matrix from the graph space
    for different values of k, calculates the MSS value for each clustering, and identifies
    the optimal number of components using the knee point of the MSS curve.

    Args:
        graph_space (Dict[str, np.ndarray]): Graph space containing the reduced matrix as a NumPy array.
        weights (np.ndarray): Weights of the components.
        num_components (int): The number of components to consider for selection.
        weight_form (bool, optional): Whether to consider weights in the clustering process. Defaults to True.

    Returns:
        np.ndarray: Indices of the optimal components as a NumPy array.
    """
    logger.info("Starting to find the knee point based on the MSS + Kneedle definition..")
    mss_values = []
    k_values = range(2, num_components)

    for i, k in enumerate(k_values):
        k_medoids = kmedoids_fasterpam(graph_space['reduced_matrix'], k)
        mss_value = calc_mss_value(clustering=k_medoids, space=graph_space['reduced_matrix'])
        mss_values.append(mss_value)

        if mss_value == 1.0 or mss_value == 1:
            extension_length = len(k_values) - i - 1
            mss_values.extend([1.0] * extension_length)
            break

    knee = get_knee(list(k_values), y=mss_values)
    optimal_kmedoids = kmedoids_fasterpam(graph_space['reduced_matrix'], int(knee))

    # weight_form = False
    if weight_form:
        return find_optimal_weighted_medoids(weights, optimal_kmedoids['labels'], optimal_kmedoids['medoids'])
    else:
        return optimal_kmedoids['medoids'].tolist()


def find_optimal_weighted_medoids(weights: np.ndarray, labels: np.ndarray, medoids: np.ndarray) -> List[int]:
    """
    Finds the optimal weighted medoids based on component weights within clusters.

    Args:
        weights (np.ndarray): Weights of all components.
        labels (np.ndarray): Cluster labels for each data point.
        medoids (np.ndarray): Indices of the initial medoids.

    Returns:
        list[int]: Indices of the data points with the highest weight in each cluster.
    """
    highest_weight_indices = np.empty_like(medoids)

    for i, medoid in enumerate(medoids):
        # Find all points assigned to the same cluster as this medoid
        cluster_indices = np.where(labels == labels[medoid])[0]

        # Extract the weights for these points
        cluster_weights = weights[cluster_indices]

        # Find the index of the maximum weight in this cluster
        max_weight_index = cluster_indices[np.argmax(cluster_weights)]

        # Store the index of the data point with the highest weight
        highest_weight_indices[i] = max_weight_index

    return highest_weight_indices.tolist()
