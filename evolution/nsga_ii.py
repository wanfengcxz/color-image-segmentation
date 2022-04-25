import numpy as np

from evolution.fitness import population_fitness, plot_normalized_fitness
from evolution.genotype import initialize_genotype
from evolution.phenotype import to_phenotype


def initialize_population(image:np.ndarray, population_size: int, moore: bool = True) -> np.ndarray:
    population = np.zeros((population_size, image.shape[0] * image.shape[1]))
    for i in range(population_size):
        print(f'Initializing individual {i}')
        population[i] = initialize_genotype(image, n_segments=4, moore=moore)
    return population


def is_dominating(fitness1: np.ndarray, fitness2: np.ndarray) -> bool:
    for f1, f2 in zip(fitness1, fitness2):
        if f1 > f2:
            return False
    return True


def fast_non_dominated_sort(pop_fitness: np.ndarray) -> np.ndarray:
    print(pop_fitness)
    fronts = {1: []}
    dominated_by = np.zeros(pop_fitness.shape[0])
    dominates = {x: [] for x in range(pop_fitness.shape[0])}
    for p, fitness1 in enumerate(pop_fitness):
        for q, fitness2 in enumerate(pop_fitness):
            print(fitness1)
            print(fitness2)
            if p == q:
                continue
            if is_dominating(fitness1, fitness2):
                dominates[p].append(q)
            elif is_dominating(fitness2, fitness1):
                dominated_by[p] += 1

        if dominated_by[p] == 0:
            fronts[1].append(p)

    i = 1
    while len(fronts[i]) != 0:
        new_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    new_front.append(q)
        if len(new_front) == 0:
            break

        i += 1
        fronts[i] = new_front
    return fronts


def nsga_ii(image: np.ndarray, population_size: int):
    population = initialize_population(image, population_size)
    pop_fitness = population_fitness(population, image)
    print(fast_non_dominated_sort(pop_fitness))
    plot_normalized_fitness(pop_fitness)

