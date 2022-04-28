import numba
import numpy as np
from numba import njit
from numba.typed import List, Dict


from evolution.fitness import population_fitness, plot_normalized_fitness
from evolution.genotype import initialize_genotype
from evolution.phenotype import to_phenotype
from evolution.population import new_population
from evolution.utils import int_list_type


@njit
def initialize_population(image: np.ndarray, population_size: int, n_segments: int = 24, moore: bool = True) -> np.ndarray:
    population = np.zeros((population_size, image.shape[0] * image.shape[1]))
    for i in range(population_size):
        print(f'Initializing individual {i}')
        population[i] = initialize_genotype(image, n_segments=n_segments, moore=moore)
    return population


@njit
def is_dominating(fitness1: np.ndarray, fitness2: np.ndarray) -> bool:
    for f1, f2 in zip(fitness1, fitness2):
        if f1 > f2:
            return False
    return True

@njit
def fast_non_dominated_sort(pop_fitness: np.ndarray) -> np.ndarray:
    fronts = Dict.empty(
        key_type=numba.int8,
        value_type=int_list_type
    )
    fronts[1] = List.empty_list(numba.int8)

    dominates = Dict.empty(
        key_type=numba.int8,
        value_type=int_list_type
    )

    for x in range(pop_fitness.shape[0]):
        dominates[x] = List.empty_list(numba.int8)

    front_assignment = np.ones(pop_fitness.shape[0], dtype=numba.int8)
    dominated_by = np.zeros(pop_fitness.shape[0], dtype=numba.int8)

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
        new_front = List.empty_list(numba.int8)
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    new_front.append(q)

        if len(new_front) == 0:
            break

        i += 1
        fronts[i] = new_front
        for ind in new_front:
            front_assignment[ind] = i
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


def crowded_comparison(i: int, j: int, front_assignment: np.ndarray, crowding_assignment: np.ndarray) -> int:
    if front_assignment[i] < front_assignment[j]:
        return i
    elif front_assignment[j] < front_assignment[i]:
        return j
    else:
        if crowding_assignment[i] > crowding_assignment[j]:
            return i
        else:
            return j


def nsga_ii(image: np.ndarray, population_size: int, generations: int = 10, n_segments: int = 24) \
        -> tuple[np.ndarray, np.ndarray]:
    P = initialize_population(image, population_size, n_segments=n_segments)
    Q = new_population(P)
    front_assignment = np.ones(P.shape)
    for g in range(generations):
        print(f'Generation {g}')
        R = np.concatenate((P, Q), axis=0)
        fitness = population_fitness(R, image)
        front_assignment = fast_non_dominated_sort(fitness)
        crowding_assignment = crowding_distance(fitness, front_assignment)
        sorted_idx = np.lexsort((-crowding_assignment, front_assignment))
        P_next = R[sorted_idx][:population_size]
        Q_next = new_population(P_next)
        P = P_next
        Q = Q_next
        front_assignment = front_assignment[sorted_idx][:population_size]

    return P, front_assignment

