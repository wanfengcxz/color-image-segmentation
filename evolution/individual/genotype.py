from heapq import heappush, heappop

import numpy as np
import numba

from numba import njit

from evolution.individual.gene import Gene, from_diff
from evolution.utils import get_neighbours


@njit
def initialize_genotype(
    image: np.ndarray, n_segments: int = 1, moore: bool = True
) -> np.ndarray:
    graph_shape = image.shape[:2]
    n_added = 0
    n_total = graph_shape[0] * graph_shape[1]

    added = np.zeros(graph_shape, dtype=np.bool)

    genotype = np.ones(n_total, dtype=np.int16) * Gene.none.value
    genotype_weights = np.zeros(n_total)

    edge_queue = [(0.0, ((0, 0), (0, 0))) for x in range(0)]

    node = np.random.randint(graph_shape[0]), np.random.randint(graph_shape[1])

    n_added += 1
    while n_added < n_total:
        if not added[node]:
            added[node] = 1
            node_rgb = image[node[0], node[1], :]
            # 获取邻居 默认为上下左右 moore=True时，还有左上，右上，左下，右下
            for neighbour in get_neighbours(
                node[0], node[1], graph_shape[0], graph_shape[1], moore=moore
            ):
                neighbour_rgb = image[neighbour[0], neighbour[1], :]
                heappush(
                    edge_queue,
                    (
                        np.linalg.norm(node_rgb - neighbour_rgb),  # RGB色彩空间中的距离
                        (node, neighbour),
                    ),
                )

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
        highest_weights = np.argsort(genotype_weights)
        idxs = np.random.choice(
            highest_weights[-n_total // 2 :], np.random.randint(n_segments - 1)
        )
        genotype[idxs] = Gene.none.value

    return genotype
