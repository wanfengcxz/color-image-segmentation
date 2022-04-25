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
    fronts = {1: []}
    front_assignment = np.ones(pop_fitness.shape[0], dtype=int)
    dominated_by = np.zeros(pop_fitness.shape[0])
    dominates = {x: [] for x in range(pop_fitness.shape[0])}
    for p, fitness1 in enumerate(pop_fitness):
        for q, fitness2 in enumerate(pop_fitness):
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
        front_assignment[new_front] = i
    print(fronts)
    return front_assignment


def crowding_distance(population_fitness: np.ndarray, front_assignment: np.ndarray) -> np.ndarray:
    distance_assignment = np.zeros(population_fitness.shape[0])
    i = 1
    while i in front_assignment:
        front = np.where(front_assignment == i)[0]
        front_fitness = population_fitness[front]
        for fitness in front_fitness.T:
            sorted_args = np.argsort(fitness)
            front = front[sorted_args]
            fitness = fitness[sorted_args]
            distance_assignment[front[0]] = np.inf
            distance_assignment[front[-1]] = np.inf

            for j in range(1, len(front)-1):
                distance_assignment[front[j]] += fitness[j+1] - fitness[j-1]

        i += 1

    return distance_assignment


def nsga_ii(image: np.ndarray, population_size: int):
    population = initialize_population(image, population_size)
    pop_fitness = population_fitness(population, image)
    front_assignment = fast_non_dominated_sort(pop_fitness)
    crowding_assignmnet = crowding_distance(pop_fitness, front_assignment)
    print(front_assignment)
    print(crowding_assignmnet)
    plot_normalized_fitness(pop_fitness)

