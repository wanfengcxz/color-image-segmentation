from typing import Optional

import numba
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

from evolution.phenotype import to_phenotype
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
    return centroids/counts.reshape((-1, 1))


@njit
def phenotype_fitness(phenotype: np.ndarray, image: np.ndarray) -> np.ndarray:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]
    n_segments = np.max(phenotype) + 1
    print(n_segments)
    segment_centroids = get_centroids(phenotype, image, n_segments)

    edge_value = 0.0
    connectivity = 0.0
    deviation = 0.0
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

    return np.array([edge_value, connectivity, deviation])


@njit
def genotype_fitness(genotype: np.ndarray, image: np.ndarray) -> np.ndarray:
    return phenotype_fitness(to_phenotype(genotype, image.shape[0], image.shape[1]), image)


@njit
def population_fitness(population: np.ndarray, image: np.ndarray) -> np.ndarray:
    fitness = np.empty((population.shape[0], 3), dtype=numba.float64)
    for i, individual in enumerate(population):
        fitness[i] = genotype_fitness(individual, image)
    return fitness


def plot_normalized_fitness(population_fitness: np.ndarray, front_assignment: Optional[np.ndarray] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(population_fitness)
    xs = scaled[:, 0]
    ys = scaled[:, 1]
    zs = scaled[:, 2]

    Axes3D.scatter(xs=xs, ys=ys, zs=zs, c=front_assignment, ax=ax, zdir='z', s=20, depthshade=True)
    ax.set_xlabel('Edge Value')
    ax.set_ylabel('Connectivity')
    ax.set_zlabel('Deviation')
    plt.show()
