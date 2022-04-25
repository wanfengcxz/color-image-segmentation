import numpy as np
import numba
from matplotlib import pyplot as plt
from numba import njit

from evolution.gene import Gene, to_diff


@njit
def to_phenotype(genotype: np.ndarray, rows: int, cols: int) -> np.ndarray:
    genotype_2d = np.reshape(genotype, (rows, cols))
    phenotype = -np.ones((rows, cols), dtype=numba.int8)
    segment_value = 0
    for row in range(rows):
        for col in range(cols):
            if phenotype[row, col] != -1:  # Node already added to phenotype
                continue

            node = (row, col)
            node_value = genotype_2d[node]
            path = [node]
            segment_merge_value = -1
            while node_value != Gene.none.value:
                row_diff, col_diff = to_diff(node_value)
                node = (node[0] + row_diff, node[1] + col_diff)

                node_value = genotype_2d[node]

                path.append(node)

                if phenotype[node] != -1:
                    segment_merge_value = phenotype[node]
                    break

            if segment_merge_value == -1:
                for node in path:
                    phenotype[node] = segment_value
                segment_value += 1
            else:
                for node in path:
                    phenotype[node] = segment_merge_value

    return phenotype


def visualize_phenotype(phenotype: np.ndarray) -> None:
    n_segments = np.max(phenotype) + 1
    colors = np.random.random((n_segments, 3))
    image = np.zeros((*phenotype.shape, 3))
    for row in range(phenotype.shape[0]):
        for col in range(phenotype.shape[1]):
            segment = phenotype[row, col]
            image[row, col, :] = colors[segment, :]

    plt.imshow(image)
    plt.show()



