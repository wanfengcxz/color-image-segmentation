from typing import Optional

import numpy as np
import numba
from matplotlib import pyplot as plt
from numba import njit

from evolution.gene import Gene, to_diff, points_outwards
from evolution.utils import get_neighbours


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


def to_color_segmentation(phenotype: np.ndarray) -> np.ndarray:
    n_segments = np.max(phenotype) + 1
    colors = np.random.random((n_segments, 3))
    image = np.zeros((*phenotype.shape, 3))
    for row in range(phenotype.shape[0]):
        for col in range(phenotype.shape[1]):
            segment = phenotype[row, col]
            image[row, col, :] = colors[segment, :]
    return image


def to_contour_segmentation(phenotype: np.ndarray) -> np.ndarray:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]
    image = np.ones(phenotype.shape) * 255
    for row in range(max_rows):
        for col in range(max_cols):
            if row == 0 or row == max_rows - 1 or col == 0 or col == max_cols - 1:
                image[row, col] = 0
                continue
            node = (row, col)
            neighbours = get_neighbours(row, col, max_rows, max_cols)
            for neighbour in neighbours:
                if phenotype[node] != phenotype[neighbour]:
                     image[row, col] = 0
    return image


def visualize_phenotype(phenotype: np.ndarray, ax: Optional[plt.Axes] = None) -> None:
    image = to_contour_segmentation(phenotype)
    if ax is None:
        plt.imshow(image)
        plt.show()
    else:
        ax.imshow(image)


def save_phenotype(phenotype: np.ndarray, path: str) -> None:
    image = to_contour_segmentation(phenotype)
    plt.imsave(path, image, cmap='gray')


