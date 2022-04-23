from enum import Enum

import numpy as np


class GeneType(Enum):
    right = 1
    left = 2
    up = 3
    down = 4
    upright = 5
    downright = 6
    upleft = 7
    downleft = 8
    none = 9

    @staticmethod
    def from_diff(row_diff: int, col_diff: int) -> 'GeneType':
        if row_diff == 0 and col_diff > 0:
            return GeneType.right
        if row_diff == 0 and col_diff < 0:
            return GeneType.left
        if row_diff < 0 and col_diff == 0:
            return GeneType.up
        if row_diff > 0 and col_diff == 0:
            return GeneType.down
        if row_diff < 0 and col_diff > 0:
            return GeneType.upright
        if row_diff > 0 and col_diff > 0:
            return GeneType.downright
        if row_diff < 0 and col_diff < 0:
            return GeneType.upleft
        if row_diff > 0 and col_diff < 0:
            return GeneType.downleft

    @staticmethod
    def to_diff(gene_type: 'GeneType') -> tuple[int, int]:
        if gene_type == GeneType.right:
            return 0, 1
        if gene_type == GeneType.left:
            return 0, -1
        if gene_type == GeneType.up:
            return -1, 0
        if gene_type == GeneType.down:
            return 1, 0
        if gene_type == GeneType.upright:
            return -1, 1
        if gene_type == GeneType.downright:
            return 1, 1
        if gene_type == GeneType.upleft:
            return -1, -1
        if gene_type == GeneType.downleft:
            return 1, -1
        if gene_type == GeneType.none:
            return 0, 0



def initialize_individual(image_shape: tuple[int, int]) -> np.ndarray:
    pass