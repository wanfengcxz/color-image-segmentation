from typing import Optional

import numba
import pandas as pd
import numpy as np
from numba import njit

from sklearn.cluster import MeanShift

int_list_type = numba.types.ListType(numba.int16)


def save_fitness(
    fitness: np.ndarray, path: str, front_assignment: Optional[np.ndarray] = None
) -> None:

    df = pd.DataFrame(fitness, columns=["Edge Value", "Connectivity", "Deviation"])
    if front_assignment is not None:
        df["Front"] = front_assignment

    df.to_csv(path, index=False)


def preprocess_cluster(image, cluster_algo="meanshift"):
    height, width, bands = image.shape
    meanshift = MeanShift()
    meanshift.fit(image.reshape(-1, bands))
    return meanshift.labels_, meanshift.cluster_centers_


@njit
def get_neighbours(
    row: int, col: int, max_rows: int, max_cols: int, moore: bool = True
) -> list:
    neighbours = []
    if col > 0:  # 1
        neighbours.append((row, col - 1))
    if col < (max_cols - 1):  # 2
        neighbours.append((row, col + 1))
    if row > 0:  # 3
        neighbours.append((row - 1, col))
    if row < (max_rows - 1):  # 4
        neighbours.append((row + 1, col))
    if moore:
        if row > 0 and col < (max_cols - 1):  # 5
            neighbours.append((row - 1, col + 1))
        if row < (max_rows - 1) and col < (max_cols - 1):  # 6
            neighbours.append((row + 1, col + 1))
        if row > 0 and col > 0:  # 7
            neighbours.append((row - 1, col - 1))
        if row < (max_rows - 1) and col > 0:  # 8
            neighbours.append((row + 1, col - 1))
    return neighbours


# https://stackoverflow.com/questions/64135020/speed-up-random-weighted-choice-without-replacement-in-python
@njit
def numba_choice(population, weights, k):
    # Get cumulative weights
    wc = np.cumsum(weights)
    # Total of weights
    m = wc[-1]
    # Arrays of sample and sampled indices
    sample = np.empty((k, population.shape[1]), population.dtype)
    # Sampling loop
    i = 0
    while i < k:
        # Pick random weight value
        r = m * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r, side="right")

        # Save sampled value and index
        sample[i] = population[idx]
        i += 1
    return sample
