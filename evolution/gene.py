from enum import Enum

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
def to_diff(gene: Gene) -> tuple[int, int]:
    if gene == Gene.right:
        return 0, 1
    if gene == Gene.left:
        return 0, -1
    if gene == Gene.up:
        return -1, 0
    if gene == Gene.down:
        return 1, 0
    if gene == Gene.upright:
        return -1, 1
    if gene == Gene.downright:
        return 1, 1
    if gene == Gene.upleft:
        return -1, -1
    if gene == Gene.downleft:
        return 1, -1
    if gene == Gene.none:
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