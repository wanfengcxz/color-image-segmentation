from heapq import heappush, heappop

import numpy as np
import numba
import matplotlib.pyplot as plt
import networkx as nx

from numba import njit

from evolution.gene import Gene, to_diff, from_diff
from evolution.phenotype import to_phenotype
from evolution.utils import get_neighbours, read_image


@njit
def initialize_genotype(image: np.ndarray, n_segments: int = 1, moore: bool = True) -> np.ndarray:
    graph_shape = image.shape[:2]
    n_added = 0
    n_total = graph_shape[0] * graph_shape[1]

    added = np.zeros(graph_shape, numba.int8)

    genotype = np.ones(n_total, dtype=numba.int8) * Gene.none.value
    genotype_weights = np.zeros(n_total)

    edge_queue = [(0.0, ((0, 0), (0, 0))) for x in range(0)]

    node = np.random.randint(graph_shape[0]), np.random.randint(graph_shape[1])
    print(node)
    n_added += 1
    while n_added < n_total:
        if not added[node]:
            added[node] = 1
            for neighbour in get_neighbours(node[0], node[1], graph_shape[0], graph_shape[1], moore=moore):
                node_rgb = image[node[0], node[1], :]
                neighbour_rgb = image[neighbour[0], neighbour[1], :]
                heappush(edge_queue, (np.linalg.norm(node_rgb - neighbour_rgb), (node, neighbour)))

        edge = heappop(edge_queue)
        from_node, to_node = edge[1]
        if not added[to_node]:
            gene = from_diff(from_node[0] - to_node[0], from_node[1] - to_node[1])
            idx = to_node[0] * image.shape[1] + to_node[1]
            genotype[idx] = gene.value
            genotype_weights[idx] = edge[0]

            n_added += 1
        node = to_node

    if n_segments > 1:
        highest_weights = np.argsort(genotype_weights)[-n_segments:]
        genotype[highest_weights] = Gene.none.value

    return genotype


def visualize_genotype(genotype: np.ndarray, graph_shape: tuple[int, int]) -> None:
    G = nx.DiGraph()
    genotype = np.reshape(genotype, graph_shape)
    max_rows = graph_shape[0]
    max_cols = graph_shape[1]
    for row in range(max_rows):
        for col in range(max_cols):
            row_diff, col_diff = to_diff(genotype[row, col])
            to_node = f'{row}-{col}'
            from_node = f'{row + row_diff}-{col + col_diff}'
            G.add_node(from_node, pos=(col + col_diff, max_rows - (row + row_diff)))
            G.add_node(to_node, pos=(col, max_rows - row))
            G.add_edge(to_node, from_node)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color='w', edgecolors='k', node_size=5, width=1, with_labels=False)
    plt.show()

if __name__ == '__main__':
    image = np.random.random((10, 10 , 3)) * 255  # read_image('training_images/86016/Test image.jpg') 241 x 161 x 3
    print('Creating genotype...')
    genotype = initialize_genotype(image, n_segments=10)
    print('Converting to phenotype...')
    print(to_phenotype(genotype, image.shape[:2]))
    print('Visualizing...')
    visualize_genotype(genotype, image.shape[:2])
