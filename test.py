import numpy as np

from evolution.individual.genotype import initialize_genotype
from evolution.individual.phenotype import to_phenotype
from utils import read_image
from visualization.individual import visualize_type2

image = np.array([
    [[1,1,1], [2,1,2]],
    [[1,1,1], [2,3,2]],
])
print(set(tuple(v) for m2d in image for v in m2d ))