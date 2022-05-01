import numpy as np

from evolution.algorithms.sga import sga
from evolution.individual.phenotype import to_phenotype
from utils import read_image
from visualization.individual import save_phenotype

image = read_image('training_images/86016/Test image.jpg')

population = sga(image=image,
                 weights=np.array([1.0, 1.0, 1.0]),
                 n_segments=4,
                 population_size=10,
                 generations=50,
                 p_mutate=0.25,
                 p_crossover=0.7
                 )

save_phenotype(to_phenotype(population[0], image.shape[0], image.shape[1]), 'images/sga.png')