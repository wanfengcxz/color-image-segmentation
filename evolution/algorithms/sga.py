import pickle
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from prettytable import PrettyTable

from evolution.fitness import weighted_population_fitness
from evolution.population.population import initialize_population, new_population
from evolution.utils import numba_choice

@njit
def rank_selection(population: np.ndarray, fittest: np.ndarray):
    weights = np.array([1 / (i + 1) for i in range(fittest.shape[0])])
    parents = population[fittest]
    parents = numba_choice(parents, weights, k=population.shape[0])
    np.random.shuffle(parents)
    return parents

@njit
def sga(image: np.ndarray,
        weights: np.ndarray,
        n_segments: int = 24,
        population_size: int = 10,
        generations: int = 10,
        elite_frac: float = 0.1,
        p_mutate: float = 0.1,
        p_crossover: float = 0.9):

    n_elites = int(elite_frac * population_size)
    population = initialize_population(image, population_size, n_segments=n_segments)
    for g in range(generations):
        print(f'Generation {g}')
        fitness = weighted_population_fitness(population, image=image, weights=weights)
        fittest = np.argsort(fitness)
        elites = population[fittest[:n_elites]]
        population = rank_selection(population, fittest)
        population = new_population(population, p_mutate=p_mutate, p_crossover=p_crossover)
        np.random.shuffle(population)
        population[:n_elites] = elites

    return population

