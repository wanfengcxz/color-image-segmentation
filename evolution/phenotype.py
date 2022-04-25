import numpy as np
import numba
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



