import numpy as np
from matplotlib import pyplot as plt

from evolution.algorithms.nsga_ii import nsga_ii
from evolution.fitness import population_fitness
from evolution.individual.phenotype import to_phenotype
from utils import read_image
from visualization.fitness import plot_fitness
from visualization.individual import save_phenotype

image = read_image('training_images/86016/Test image.jpg')

population, front_assignment = nsga_ii(image=image, n_segments=1, population_size=50, generations=50, p_mutate=0.25, p_crossover=0.7, fitness_path='fitness.csv')

# visualize_type2(to_phenotype(population[0], image.shape[0], image.shape[1]))
for i, individual in enumerate(population[np.where(front_assignment == 1)]):
    save_phenotype(to_phenotype(individual, image.shape[0], image.shape[1]), f'images/seg_{i}.png')

fitness = population_fitness(population, image)
#save_fitness(fitness, 'fitness.csv', front_assignment=front_assignment)
plot_fitness(fitness, front_assignment=front_assignment)
