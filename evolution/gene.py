from enum import Enum
from typing import Union

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