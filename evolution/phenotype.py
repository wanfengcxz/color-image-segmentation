import numpy as np
import numba
from numba import njit

from evolution.gene import Gene, to_diff


def to_phenotype(genotype: np.ndarray, image_shape: tuple) -> np.ndarray:
    genotype = np.reshape(genotype, image_shape)
    phenotype = np.zeros(image_shape, dtype=int)

    segment_value = 1
    for row in range(image_shape[0]):
        for col in range(image_shape[1]):
            if phenotype[row, col] != 0:  # Node already added to phenotype
                continue

            node = (row, col)
            node_value = genotype[node]
            path = [node]
            segment_merge_value = 0
            while node_value != Gene.none.value:
                gene = Gene(genotype[node])
                row_diff, col_diff = to_diff(gene)
                node = (node[0] + row_diff, node[1] + col_diff)
                node_value = genotype[node]
                path.append(node)

                if phenotype[node] != 0:
                    segment_merge_value = phenotype[node]
                    break

            if segment_merge_value == 0:
                for node in path:
                    phenotype[node] = segment_value
                segment_value += 1
            else:
                for node in path:
                    phenotype[node] = segment_merge_value

    return phenotype



