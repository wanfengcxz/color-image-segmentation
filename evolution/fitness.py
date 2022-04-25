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
def phenotype_fitness(phenotype: np.ndarray, image: np.ndarray) -> tuple[float, float, float]:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]
    n_segments = np.max(phenotype) + 1

    segment_centroids = get_centroids(phenotype, image, n_segments)

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


#@njit
def genotype_fitness(genotype: np.ndarray, image: np.ndarray) -> tuple[float, float, float]:
    return phenotype_fitness(to_phenotype(genotype, image.shape[0], image.shape[1]), image)


#@njit
def population_fitness(population: np.ndarray, image: np.ndarray) -> np.ndarray:
    return np.array([genotype_fitness(genotype, image) for genotype in population])


def plot_normalized_fitness(population_fitness: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scaler = MinMaxScaler(feature_range=(0, 1))
    xs = scaler.fit_transform(population_fitness[:, 0].reshape(-1, 1))
    ys = scaler.fit_transform(population_fitness[:, 1].reshape(-1, 1))
    zs = scaler.fit_transform(population_fitness[:, 2].reshape(-1, 1))

    Axes3D.scatter(xs=xs, ys=ys, zs=zs, ax=ax, zdir='z', s=20, c=None, depthshade=True)
    ax.set_xlabel('Edge Value')
    ax.set_ylabel('Connectivity')
    ax.set_zlabel('Deviation')
    plt.show()
