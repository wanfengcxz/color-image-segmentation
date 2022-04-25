

from operator import ge
import random
from typing import List, Tuple

import numpy as np


class Individual:

    def __init__(self, img_dim: Tuple[int, int]) -> None:
        self.img_dim = img_dim
        # List of ints from 0 to 4, representing the direction to the connected image segment
        # 0 = None
        # 1 = Up
        # 2 = Right
        # 3 = Down
        # 4 = Left
        self.genotype: List[int] = []

    def initialize_random(self) -> None:
        for i in range(self.img_dim[0] * self.img_dim[1]):
            self.genotype.append(random.randint(0, 4))

    def get_value_left(self, index: int) -> int:
        # If index in first column
        if index % self.img_dim[1] == 0:
            return None
        return self.genotype[index - 1]

    def get_value_right(self, index: int) -> int:
        # If index in last column
        if index % self.img_dim[1] == self.img_dim[1] - 1:
            return None
        return self.genotype[index + 1]

    def get_value_up(self, index: int) -> int:
        # If index in first row
        if index < self.img_dim[1]:
            return None
        return self.genotype[index - self.img_dim[1]]

    def get_value_down(self, index: int) -> int:
        # If index in last row
        if index >= self.img_dim[0] * self.img_dim[1] - self.img_dim[1]:
            return None
        return self.genotype[index + self.img_dim[1]]

    def get_2d_array(self):
        """ Get genotype as 2D array """
        return np.array(self.genotype).reshape(
            self.img_dim[0], self.img_dim[1])
