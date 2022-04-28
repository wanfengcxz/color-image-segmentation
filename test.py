import numpy as np
from matplotlib import pyplot as plt

from evolution.fitness import population_fitness
from evolution.genotype import visualize_genotype, initialize_genotype
from evolution.nsga_ii import initialize_population
from evolution.phenotype import visualize_phenotype, to_phenotype
from evolution.population import mutate, uniform_crossover, new_population
from evolution.utils import read_image

image = read_image('training_images/147091/Test image.jpg')
P = initialize_population(image, 25)
#P = initialize_genotype(image, n_segments=24, moore=True)
fig, ax = plt.subplots(5, 5, figsize=(16, 16))
for i in range(5):
    for j in range(5):
        visualize_phenotype(to_phenotype(P[i*5 + j], rows=image.shape[0], cols=image.shape[1]), ax=ax[i][j])
plt.show()
# Q = new_population(P)
# R = np.concatenate((P, Q), axis=0)
#print(P)
# print(Q[0])
# print(R[0])
#print(P.shape, Q.shape, R.shape)
#print(population_fitness(P, image))
#visualize_phenotype(to_phenotype(P, rows=image.shape[0], cols=image.shape[1]))
# mask = np.array([1, 0, 1])
# p1 = np.array([1, 2, 3])
# p2 = np.array([4, 5, 6])
# temp1 = p1.copy()
# temp2 = p2.copy()
#
# temp1[np.where(mask == 1)] = p2[np.where(mask == 1)]
# temp2[np.where(mask == 1)] = p1[np.where(mask == 1)]
#
#
# print(temp1)
# print(temp2)
# print(population)
#visualize_genotype(population[0], graph_shape=(3,3))
# population = uniform_crossover(population, p_crossover=1)
# print(population)
# visualize_genotype(population[1], graph_shape=(3,3))
#visualize_phenotype(to_phenotype(population[0], rows=image.shape[0], cols=image.shape[1]))
