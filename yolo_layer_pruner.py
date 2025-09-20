import numpy as np
from loguru import logger
from itertools import combinations
from typing import Dict, Tuple, List
from distance import get_distance


class YoloLayerPruner:
    def __init__(self, activations):
        self.activations = activations

    def create_layer_space(self) -> Dict[str, np.ndarray]:
        """
        Creates the feature space (separability matrix) for the layer.
        """
        separation_matrix = self._calculate_separation_matrix()
        reduced_matrix = self._reduce_dimension(separation_matrix)

        return {
            'separation_matrix': separation_matrix,
            'reduced_matrix': reduced_matrix,
        }

    def _calculate_separation_matrix(self) -> np.ndarray:
        """
        Computes the separability matrix:
        Rows    = Channels
        Columns = Class pairs
        """
        class_labels = list(next(iter(self.activations.values())).keys())
        label_pairs = list(combinations(class_labels, 2))

        num_channels = len(self.activations)
        separation_matrix = np.zeros((num_channels, len(label_pairs)))

        for channel_idx in range(num_channels):
            separation_matrix[channel_idx] = self._calculate_channel_separation_scores(
                channel_idx, label_pairs
            )
            logger.debug(f"Separation scores computed for channel #{channel_idx}.")

        separation_matrix = self.replace_nan_with_zero(separation_matrix)
        return separation_matrix

    def _calculate_channel_separation_scores(
        self, channel_idx: int, label_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        For a given channel, compute JM scores for all class pairs.
        """
        separation_scores = np.zeros(len(label_pairs))

        for i, (label1, label2) in enumerate(label_pairs):
            class1_data = np.array(self.activations[channel_idx].get(label1, []))
            class2_data = np.array(self.activations[channel_idx].get(label2, []))

            score = get_distance('jm', class1_data, class2_data)
            separation_scores[i] = score

        return separation_scores

    @staticmethod
    def replace_nan_with_zero(matrix: np.ndarray) -> np.ndarray:
        return np.nan_to_num(matrix, nan=0.0)

    @staticmethod
    def _reduce_dimension(data: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensionality of the separability matrix using t-SNE.

        Args:
            data (np.ndarray): The separability matrix as a NumPy ndarray.

        Returns:
            np.ndarray: The dimensionally reduced form of the separability matrix as a NumPy ndarray.
        """
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, max_iter=1000, perplexity=10).fit_transform(data)
