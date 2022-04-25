import numpy as np
from numba import njit

from evolution.gene import Gene
from evolution.utils import get_neighbours

@njit
def get_centroids(phenotype: np.ndarray, image: np.ndarray, n_segments: int) -> np.ndarray:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]

    centroids = np.zeros((n_segments, image.shape[2]))
    counts = np.zeros(n_segments)
    for row in range(max_rows):
        for col in range(max_cols):
            segment = phenotype[row, col]
            node_rgb = image[row, col, :]
            centroids[segment, :] += node_rgb
            counts[segment] += 1
    print(centroids)
    print(counts)
    return centroids/counts.reshape((-1, 1))

@njit
def fitness(phenotype: np.ndarray, image: np.ndarray) -> tuple[float, float, float]:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]
    n_segments = np.max(phenotype) + 1

    segment_centroids = get_centroids(phenotype, image, n_segments)
    print(segment_centroids)

    edge_value = 0
    connectivity = 0
    deviation = 0
    for row in range(max_rows):
        for col in range(max_cols):
            segment = phenotype[row, col]
            node_rgb = image[row, col, :]
            neighbours = get_neighbours(row, col, max_rows, max_cols)
            for neighbour in neighbours:
                if phenotype[row, col] != phenotype[neighbour]:
                    neighbour_rgb = image[neighbour[0], neighbour[1], :]
                    edge_value -= np.linalg.norm(node_rgb - neighbour_rgb)
                    connectivity += (1/8)
            deviation += np.linalg.norm(node_rgb - segment_centroids[segment])

    return edge_value, connectivity, deviation
