import numba
import numpy as np

from numba import njit

from evolution.individual.phenotype import to_phenotype
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
def phenotype_fitness(phenotype: np.ndarray,
                      image: np.ndarray) -> np.ndarray:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]
    n_segments = np.max(phenotype) + 1
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
    phenotype = to_phenotype(genotype, image.shape[0], image.shape[1])
    return phenotype_fitness(phenotype, image)


@njit
def population_fitness(population: np.ndarray, image: np.ndarray) -> np.ndarray:
    fitness = np.empty((population.shape[0], 3), dtype=numba.float64)
    for i, individual in enumerate(population):
        fitness[i] = genotype_fitness(individual, image)
    return fitness

@njit
def weighted_population_fitness(population: np.ndarray, image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    fitness = np.empty(population.shape[0], dtype=numba.float64)
    for i, individual in enumerate(population):
            fitness[i] = np.dot(genotype_fitness(individual, image), weights)
    return fitness

