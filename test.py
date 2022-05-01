import numpy as np
from matplotlib import pyplot as plt

from evolution.fitness import plot_fitness
from evolution.algorithms.nsga_ii import fast_non_dominated_sort

fitness = np.array([
    [-4, 2, 3],
    [-2, 3, 4],
    [-4, 5, 6],
    [-2, 4, 4],
    [-2, 4, 3],
])

front_assignment = fast_non_dominated_sort(fitness)
print(front_assignment)
for i in range(2, 4):
    for w in fitness[np.where(front_assignment == i)]:
        dominated = False
        for b in fitness[np.where(front_assignment == i-1)]:
            if b[0] <= w[0] and  b[0] <= w[0] and b[0] <= w[0]:
                print(f'{b} dominates {w}')
                dominated = True
        if not dominated:
            print('WRONG!')
            break
    print('RIGHT!')

plot_fitness(fitness, front_assignment=front_assignment)
plt.show()