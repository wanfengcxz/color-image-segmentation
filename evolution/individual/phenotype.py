import numpy as np
import numba
from numba import njit

from evolution.individual.gene import Gene, points_outwards, to_diff


@njit
def to_phenotype(genotype: np.ndarray, rows: int, cols: int) -> np.ndarray:
    genotype_2d = np.ascontiguousarray(genotype).reshape(rows, cols)
    phenotype = -np.ones((rows, cols), dtype=numba.int16)
    segment_value = 0
    for row in range(rows):
        for col in range(cols):
            if phenotype[row, col] != -1:  # Node already added to phenotype
                continue

            node = (row, col)
            node_value = genotype_2d[node]
            if points_outwards(node_value, row, col, rows, cols):
                node_value = Gene.none.value
            path = [node]
            segment_merge_value = -1
            while node_value != Gene.none.value:
                row_diff, col_diff = to_diff(node_value)
                node = (node[0] + row_diff, node[1] + col_diff)

                node_value = genotype_2d[node]
                if points_outwards(node_value, node[0], node[1], rows, cols):
                    node_value = Gene.none.value

                if phenotype[node] != -1 or node in path:
                    path.append(node)
                    segment_merge_value = phenotype[node]
                    break

                path.append(node)


            if segment_merge_value == -1:
                for node in path:
                    phenotype[node] = segment_value
                segment_value += 1
            else:
                for node in path:
                    phenotype[node] = segment_merge_value

    return phenotype


