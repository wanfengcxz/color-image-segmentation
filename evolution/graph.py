from heapq import heappush, heappop
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from numba import njit

from evolution.individual import GeneType
from evolution.utils import read_image


class Node:
    def __init__(self, row: int, col: int, **kwargs):
        self.row = row
        self.col = col
        self.neighbours = []

        self.__dict__.update(kwargs)

    def __lt__(self, other):
        return self.row + self.col < other.row + other.col

def initialize_graph(shape: tuple[int, int]) -> np.ndarray:
    graph = np.empty(shape, dtype=Node)

    # Initialize graph
    for row in range(graph.shape[0]):
        for col in range(graph.shape[1]):
            graph[row][col] = Node(row=row, col=col)

    return graph

def add_neighbours(graph: np.ndarray) -> None:
    """

    Neighbour numbering:
    7 3 5
    2 P 1
    8 4 6
    """

    for row in range(graph.shape[0]):
        for col in range(graph.shape[1]):
            node = graph[row][col]
            neighbours = []
            if col > 0:  # 1
                neighbours.append(graph[row][col - 1])
            if col < (graph.shape[1] - 1):  # 2
                neighbours.append(graph[row][col + 1])
            if row > 0:  # 3
                neighbours.append(graph[row - 1][col])
            if row < (graph.shape[0] - 1):  # 4
                neighbours.append(graph[row + 1][col])
            if row > 0 and col < (graph.shape[1] - 1):  # 5
                neighbours.append(graph[row - 1][col + 1])
            if row < (graph.shape[0] - 1) and col < (graph.shape[1] - 1):  # 6
                neighbours.append(graph[row + 1][col + 1])
            if row > 0 and col > 0:  # 7
                neighbours.append(graph[row - 1][col - 1])
            if row < (graph.shape[0] - 1) and col > 0:  # 8
                neighbours.append(graph[row + 1][col - 1])
            # neighbours = [{'node': neighbour, 'weight': np.linalg.norm(node.rgb - neighbour.rgb)}
            #               for neighbour in neighbours]
            graph[row][col].neighbours = neighbours

def add_rgb(graph: np.ndarray, image: np.ndarray) -> None:
    # Initialize graph
    for row in range(graph.shape[0]):
        for col in range(graph.shape[1]):
            graph[row][col].rgb = image[row, col, :]


def initialize_genotype(image_graph: np.ndarray, n_segments: int = 1) -> np.ndarray:
    n_added = 0.0
    n_total = image_graph.shape[0] * image_graph.shape[1]
    added = np.zeros(image_graph.shape, dtype=np.float64)

    row, col = np.random.randint(image_graph.shape[0]), np.random.randint(image_graph.shape[1])
    current_node = image_graph[row, col]
    n_added += 1

    edge_queue = [(0.0, 0, 0) for x in range(0)]
    genotype = np.ones(n_total, dtype=int) * GeneType.none.value
    genotype_weights = np.zeros(n_total, dtype=float)
    while n_added < n_total:
        if not added[current_node.row, current_node.col]:
            added[current_node.row, current_node.col] = 1
            for neighbour in current_node.neighbours:
                heappush(edge_queue, (np.linalg.norm(current_node.rgb - neighbour.rgb), (current_node, neighbour)))

        edge = heappop(edge_queue)
        from_node, to_node = edge[1]
        if not added[to_node.row, to_node.col]:
            gene_type = GeneType.from_diff(from_node.row - to_node.row, from_node.col - to_node.col)
            idx = np.ravel_multi_index((to_node.row, to_node.col), image_graph.shape)
            genotype[idx] = gene_type.value
            genotype_weights[idx] = edge[0]

            n_added += 1
        current_node = to_node

    if n_segments > 1:
        highest_weights = np.argpartition(genotype_weights, -(n_segments-1))[-(n_segments-1):]
        genotype[highest_weights] = GeneType.none.value

    return genotype

def visualize_genotype(genotype: np.ndarray, image_shape: tuple[int, int]) -> None:
    G = nx.DiGraph()
    genotype = np.reshape(genotype, image_shape)
    max_rows = genotype.shape[0]
    max_cols = genotype.shape[1]
    for row in range(max_rows):
        for col in range(max_cols):
            row_diff, col_diff = GeneType.to_diff(GeneType(genotype[row, col]))
            to_node = f'{row}-{col}'
            from_node = f'{row + row_diff}-{col + col_diff}'
            # if to_node == from_node:
            #     continue
            G.add_node(from_node, pos=(col + col_diff, max_rows - (row + row_diff)))
            G.add_node(to_node, pos=(col, max_rows - row))
            G.add_edge(to_node, from_node)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color='w', edgecolors='k', node_size=5, width=1, with_labels=False)
    plt.show()

image = read_image('training_images/86016/Test image.jpg')#np.random.random((10,10,3)) * 255 # 241 x 161
print('Initializing image graph...')
image_graph = initialize_graph(image.shape[:2])
print('Adding rgb...')
add_rgb(image_graph, image)
print('Adding neighbours...')
add_neighbours(image_graph)

print('Creating genotype...')
genotype = initialize_genotype(image_graph, n_segments=10)
print(genotype)
print('Visualizing...')
#visualize_genotype(genotype, image_graph.shape[:2])
