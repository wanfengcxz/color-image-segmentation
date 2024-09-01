import numpy as np
import numba
from numba import njit

from evolution.individual.gene import Gene, points_outwards, to_diff


@njit
def to_phenotype(genotype: np.ndarray, rows: int, cols: int) -> np.ndarray:
    genotype_2d = np.ascontiguousarray(genotype).reshape(rows, cols)
    phenotype = -np.ones((rows, cols), dtype=np.int16)
    segment_value = 0
    for row in range(rows):
        for col in range(cols):
            if phenotype[row, col] != -1:  # Node already added to phenotype
                continue

            node = (row, col)
            node_value = genotype_2d[node]
            # 如果该点方向不合法 则设置为环(none) （交叉变异会使得值不合法）
            if points_outwards(node_value, row, col, rows, cols):
                node_value = Gene.none.value
            path = [node]
            segment_merge_value = -1
            # 顺着方向一步一步走，并把走过的点加入path中，直到遇到环
            while node_value != Gene.none.value:
                row_diff, col_diff = to_diff(node_value)
                node = (node[0] + row_diff, node[1] + col_diff)  # 得到下一个点的坐标

                node_value = genotype_2d[node]
                if points_outwards(node_value, node[0], node[1], rows, cols):
                    node_value = Gene.none.value

                # 下一个点已经被归类 or 下一个点还是当前点(环)
                # 下一个点已经被归类，说明当前路径是子区域
                if phenotype[node] != -1 or node in path:
                    path.append(node)
                    segment_merge_value = phenotype[node]
                    break

                path.append(node)

            if segment_merge_value == -1:
                # 当前扫描到的路径还没有划分区域，则使用递增的新的区域值
                for node in path:
                    phenotype[node] = segment_value
                segment_value += 1
            else:
                # 当前扫描到的路径是之前区域的一部分，则赋值时需要和之前的区域值一样
                for node in path:
                    phenotype[node] = segment_merge_value

    return phenotype
