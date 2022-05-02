import time
from typing import Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from evolution.individual.gene import to_diff
from evolution.utils import get_neighbours


def visualize_genotype(genotype: np.ndarray, graph_shape: tuple[int, int], image: Optional[np.ndarray] = None) -> None:
    G = nx.DiGraph()
    genotype = np.reshape(genotype, graph_shape)
    max_rows = graph_shape[0]
    max_cols = graph_shape[1]
    for row in range(max_rows):
        for col in range(max_cols):
            row_diff, col_diff = to_diff(genotype[row, col])
            from_node = f'{row}-{col}'
            to_node = f'{row + row_diff}-{col + col_diff}'
            G.add_node(from_node, pos=(col, max_rows - row - 1))
            G.add_node(to_node, pos=(col + col_diff, max_rows - row - row_diff - 1))
            G.add_edge(from_node, to_node)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color='w', edgecolors='k', node_size=5, width=1, with_labels=False)
    if image is not None:
        plt.imshow(image)
    plt.show()


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
    image = np.ones(phenotype.shape, dtype=float)
    for row in range(max_rows):
        for col in range(max_cols):
            if row == 0 or row == max_rows - 1 or col == 0 or col == max_cols - 1:
                image[row, col] = 0.0
                continue
            node = (row, col)
            neighbours = get_neighbours(row, col, max_rows, max_cols)
            for neighbour in neighbours:
                if phenotype[node] != phenotype[neighbour]:
                     image[row, col] = 0.0
    return image

def to_contour_segmentation_v2(phenotype: np.ndarray) -> np.ndarray:
    max_rows = phenotype.shape[0]
    max_cols = phenotype.shape[1]
    image = np.ones(phenotype.shape, dtype=int)

    image[0, :] = 0
    image[max_rows - 1, :] = 0
    image[:, 0] = 0
    image[:, max_cols - 1] = 0
    for i in range(phenotype.max()):
        segmentation_nodes = zip(*np.where(phenotype == i))
        for node in segmentation_nodes:
            if node[0] == 0 or node[0] == max_rows - 1 or node[1] == 0 or node[1] == max_cols - 1:
                continue
            neighbours = get_neighbours(node[0], node[1], max_rows, max_cols)
            for neighbour in neighbours:
                if phenotype[node] != phenotype[neighbour] and phenotype[neighbour] > i:
                    image[node[0], node[1]] = 0
                    break
    return image


def visualize_type1(phenotype: np.ndarray, image: np.ndarray, ax: Optional[plt.axes] = None) -> None:
    type1 = image.copy()
    segmentation = to_contour_segmentation_v2(phenotype)
    type1[np.where(segmentation == 0)] = [0.0, 1.0, 0.0]
    if ax is None:
        plt.imshow(type1)
        plt.show()
    else:
        ax.imshow(type1)


def visualize_type2(phenotype: np.ndarray, ax: Optional[plt.axes] = None) -> None:
    segmentation = to_contour_segmentation_v2(phenotype)
    if ax is None:
        plt.imshow(segmentation, cmap='gray')
        plt.show()
    else:
        ax.imshow(segmentation)


def save_type1(arr: np.ndarray, image: np.ndarray, path: str, convert: bool = True) -> None:
    type1 = image.copy()
    if convert:
        segmentation = to_contour_segmentation_v2(arr)
    else:
        segmentation = arr
    type1[np.where(segmentation == 0)] = [0.0, 1.0, 0.0]
    plt.imsave(path, type1)


def save_type2(arr: np.ndarray, path: str, convert: bool = True) -> None:
    if convert:
        segmentation = to_contour_segmentation_v2(arr)
    else:
        segmentation = arr

    plt.imsave(path, segmentation, cmap='gray')