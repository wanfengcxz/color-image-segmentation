from enum import Enum
import random
from typing import Union

import numpy as np
from numba import njit


class Gene(Enum):
    right = 1
    left = 2
    up = 3
    down = 4
    upright = 5
    downright = 6
    upleft = 7
    downleft = 8
    none = 9

@njit
def random_gene_value(moore=True) -> int:
    genes = [Gene.right.value, Gene.left.value, Gene.up.value, Gene.down.value, Gene.none.value]
    if moore:
        genes.extend([Gene.upright.value, Gene.upleft.value, Gene.downleft.value, Gene.downright.value])
    gene = np.random.choice(np.array(genes))
    return gene

@njit
def points_outwards(gene_value: int, row: int, col: int, max_rows: int, max_cols: int) -> bool:
    if row == 0 and gene_value in [Gene.up.value, Gene.upright.value, Gene.upleft.value]:
        print('up', row, gene_value)
        return True
    if row == max_rows - 1 and gene_value in [Gene.down.value, Gene.downright.value, Gene.downleft.value]:
        print('down', row, gene_value)
        return True
    if col == 0 and gene_value in [Gene.left.value, Gene.upleft.value, Gene.downleft.value]:
        print('left', row, gene_value)
        return True
    if col == max_cols - 1 and gene_value in [Gene.right.value, Gene.downright.value, Gene.upright.value]:
        print('right', row, gene_value)
        return True
    return False

@njit
def to_diff(gene_value: int) -> tuple[int, int]:
    if gene_value == 1:
        return 0, 1
    if gene_value == 2:
        return 0, -1
    if gene_value == 3:
        return -1, 0
    if gene_value == 4:
        return 1, 0
    if gene_value == 5:
        return -1, 1
    if gene_value == 6:
        return 1, 1
    if gene_value == 7:
        return -1, -1
    if gene_value == 8:
        return 1, -1
    if gene_value == 9:
        return 0, 0


@njit
def from_diff(row_diff: int, col_diff: int) -> 'Gene':
    if row_diff == 0 and col_diff > 0:
        return Gene.right
    if row_diff == 0 and col_diff < 0:
        return Gene.left
    if row_diff < 0 and col_diff == 0:
        return Gene.up
    if row_diff > 0 and col_diff == 0:
        return Gene.down
    if row_diff < 0 and col_diff > 0:
        return Gene.upright
    if row_diff > 0 and col_diff > 0:
        return Gene.downright
    if row_diff < 0 and col_diff < 0:
        return Gene.upleft
    if row_diff > 0 and col_diff < 0:
        return Gene.downleft