import numba
import numpy as np
from numba import njit

from evolution.individual.gene import random_gene_value
from evolution.individual.genotype import initialize_genotype


@njit
def initialize_population(
    image: np.ndarray, population_size: int, n_segments: int = 24, moore: bool = True
) -> np.ndarray:
    population = np.zeros(
        (population_size, image.shape[0] * image.shape[1]), dtype=np.int16
    )
    for i in range(population_size):
        print(f"Initializing individual {i}")
        population[i] = initialize_genotype(image, n_segments=n_segments, moore=moore)
    return population


@njit
def uniform_crossover(population: np.ndarray, p_crossover: float = 0.7) -> np.ndarray:
    population = population.copy()
    for p1, p2 in zip(population[::2], population[1::2]):
        if np.random.random() < p_crossover:
            mask = np.random.randint(0, 2, size=p1.shape[0])
            temp1 = p1.copy()
            temp2 = p2.copy()
            temp1[np.where(mask == 1)] = p2[np.where(mask == 1)]
            temp2[np.where(mask == 1)] = p1[np.where(mask == 1)]
            p1[:], p2[:] = temp1, temp2
    return population


@njit
def mutate(population: np.ndarray, p_mutate: float = 0.1) -> np.ndarray:
    population = population.copy()
    for individual in population:
        if np.random.random() < p_mutate:
            idx = np.random.randint(individual.shape[0])
            individual[idx] = random_gene_value()
    return population


@njit
def new_population(
    population: np.ndarray,
    p_mutate: float = 0.1,
    p_crossover: float = 0.9,
    n_times: int = 1,
) -> np.ndarray:
    children = population.copy()
    np.random.shuffle(children)
    children = uniform_crossover(children, p_crossover)
    children = mutate(children, p_mutate)

    for n in range(n_times - 1):
        child = population.copy()
        np.random.shuffle(child)
        child = uniform_crossover(child, p_crossover)
        child = mutate(child, p_mutate)
        children = np.vstack((children, child))

    return children
