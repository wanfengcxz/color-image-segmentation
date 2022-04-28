import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from evolution.fitness import population_fitness, plot_normalized_fitness
from evolution.genotype import initialize_genotype, visualize_genotype
from evolution.nsga_ii import nsga_ii
from evolution.phenotype import to_phenotype, visualize_phenotype, save_phenotype
from evolution.utils import read_image

image = read_image('training_images/176035/Test image.jpg')

population, front_assignment = nsga_ii(image=image, population_size=20, generations=10)

for i, individual in enumerate(population[np.where(front_assignment == 1)]):
    save_phenotype(to_phenotype(individual, image.shape[0], image.shape[1]), f'images/seg_{i}.png')

fitness = population_fitness(population, image)
plot_normalized_fitness(fitness, front_assignment)

