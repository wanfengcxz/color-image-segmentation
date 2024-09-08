from typing import Optional

import numba
import numpy as np
from numba import njit, jit
from numba.typed import List, Dict


from evolution.fitness import population_fitness
from evolution.population.population import (
    new_population,
    initialize_population,
    initialize_population_fcm,
)
from evolution.utils import int_list_type


@njit
def is_dominating(fitness1: np.ndarray, fitness2: np.ndarray) -> bool:
    if np.array_equal(fitness1, fitness2):
        return False
    for f1, f2 in zip(fitness1, fitness2):
        if f1 > f2:
            return False
    return True


@njit
def fast_non_dominated_sort(pop_fitness: np.ndarray) -> np.ndarray:
    fronts = Dict.empty(key_type=np.int16, value_type=int_list_type)
    fronts[1] = List.empty_list(np.int16)

    dominates = Dict.empty(key_type=np.int16, value_type=int_list_type)

    for x in range(pop_fitness.shape[0]):
        dominates[x] = List.empty_list(np.int16)

    front_assignment = np.ones(pop_fitness.shape[0], dtype=np.int16)
    dominated_by = np.zeros(pop_fitness.shape[0], dtype=np.int16)

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
        new_front = List.empty_list(np.int16)
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
    return front_assignment


@njit
def crowding_distance(
    population_fitness: np.ndarray, front_assignment: np.ndarray
) -> np.ndarray:
    distance_assignment = np.zeros(population_fitness.shape[0], dtype=np.float64)
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

            for j in range(1, len(front) - 1):
                distance_assignment[front[j]] += fitness[j + 1] - fitness[j - 1]

        i += 1

    return distance_assignment


@njit
def lexsort(
    front_assignment: np.ndarray, crowding_assignment: np.ndarray
) -> np.ndarray:
    crowding_assignment = -crowding_assignment
    lex_sorted = np.zeros(front_assignment.shape[0], dtype=np.int16)
    front_nr = 1
    max_front_nr = np.max(front_assignment)
    idxs_added = 0
    while front_nr <= max_front_nr:
        idxs = np.where(front_assignment == front_nr)[0]
        crowding_values = crowding_assignment[idxs]
        crowding_sorted_idxs = np.argsort(crowding_values)
        lex_sorted[idxs_added : idxs_added + idxs.shape[0]] = idxs[crowding_sorted_idxs]
        idxs_added += idxs.shape[0]
        front_nr += 1

    return lex_sorted


@njit
def crowded_comparison(
    i: int, j: int, front_assignment: np.ndarray, crowding_assignment: np.ndarray
) -> int:
    if front_assignment[i] < front_assignment[j]:
        return i
    elif front_assignment[j] < front_assignment[i]:
        return j
    else:
        if crowding_assignment[i] > crowding_assignment[j]:
            return i
        else:
            return j


@njit
def tournament_selection(
    population: np.ndarray,
    front_assignment: np.ndarray,
    crowding_assignment: np.ndarray,
):
    parents = np.empty(population.shape)
    for p in range(population.shape[0]):
        i = np.random.randint(population.shape[0])
        j = np.random.randint(population.shape[0])
        parents[p] = population[
            crowded_comparison(i, j, front_assignment, crowding_assignment)
        ]
    return parents


def nsga_ii(
    image: np.ndarray,
    n_segments: int = 24,
    population_size: int = 10,
    generations: int = 10,
    p_mutate: float = 0.1,
    p_crossover: float = 0.9,
    n_times: int = 1,
    fitness_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if fitness_path is not None:
        file = open(fitness_path, "w")
        file.write("generation,edge_value,connectivity,deviation,front\n")
        file.close()

    population_size = 10
    K = 20

    P = initialize_population_fcm(regions, population_size, K)
    Q = new_population(P, p_mutate=p_mutate, p_crossover=p_crossover, n_times=n_times)
    for g in range(generations):
        print(f"Generation {g}")
        R = np.vstack((P, Q))
        # 计算父群和子群的适应度# 计算父群和子群的适应度
        fitness = population_fitness(R, image)
        front_assignment = fast_non_dominated_sort(fitness)
        crowding_assignment = crowding_distance(fitness, front_assignment)
        sorted_idx = lexsort(front_assignment, crowding_assignment)
        P_next = R[sorted_idx][:population_size]
        P_front_assignment = front_assignment[sorted_idx][:population_size]
        P_crowding_assignment = crowding_assignment[sorted_idx][:population_size]
        parents = tournament_selection(
            P_next, P_front_assignment, P_crowding_assignment
        )
        Q_next = new_population(
            parents, p_mutate=p_mutate, p_crossover=p_crossover, n_times=n_times
        )
        P = P_next
        Q = Q_next

        if fitness_path is not None:
            generation_vector = np.expand_dims(
                np.ones(R.shape[0], dtype=int) * g, axis=1
            )
            front_vector = np.expand_dims(front_assignment, axis=1)
            save_array = np.hstack((generation_vector, fitness, front_vector))
            file = open(fitness_path, "a")
            np.savetxt(file, save_array, delimiter=",")
            file.close()

    return R, front_assignment
