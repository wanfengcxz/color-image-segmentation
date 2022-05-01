import argparse
import os
import datetime

import numpy as np
from matplotlib import pyplot as plt

from evolution.algorithms.nsga_ii import nsga_ii
from evolution.algorithms.sga import sga
from evolution.fitness import population_fitness
from evolution.individual.phenotype import to_phenotype
from utils import read_image
from visualization.fitness import plot_fitness
from visualization.individual import save_type2

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path', help='path/to/image', type=str, default='training_images/86016/Test image.jpg')
parser.add_argument('-a', '--algorithm', help='Type of algorithm. Can be {nsga, sga}', type=str, default='nsga')
parser.add_argument('-o', '--output_dir', help='path/to/output/directory', type=str, default='output')
parser.add_argument('-ns', '--n_segments', help='Number of segments to initialize population with', type=int, default=4)
parser.add_argument('-n', '--population_size', help='Size of population', type=int, default=50)
parser.add_argument('-g', '--generations', help='Number of generations to run for', type=int, default=50)
parser.add_argument('-pc', '--p_crossover', help='Crossover probability', type=float, default=0.7)
parser.add_argument('-pm', '--p_mutate', help='Mutation probability', type=float, default=0.2)
parser.add_argument('-w', '--weights', help='Weights used for sga', type=list, default=[1.0, 1.0, 1.0])

args = parser.parse_args()

timestamp = datetime.datetime.now()


output_dir = os.path.join(args.output_dir, args.algorithm, timestamp.strftime('%d-%H%M'))

type1_dir = os.path.join(output_dir, 'segmentations', 'type1')
type2_dir = os.path.join(output_dir, 'segmentations', 'type2')
os.makedirs(type1_dir, exist_ok=True)
os.makedirs(type2_dir, exist_ok=True)

image = read_image(args.image_path)
plt.imsave(os.path.join(output_dir, 'image.png'), image)

if args.algorithm == 'nsga':
    population, front_assignment = nsga_ii(image=image,
                                           n_segments=args.n_segments,
                                           population_size=args.population_size,
                                           generations=args.generations,
                                           p_mutate=args.p_mutate,
                                           p_crossover=args.p_crossover,
                                           fitness_path=os.path.join(output_dir, 'fitness.csv'))

    for i, individual in enumerate(population[np.where(front_assignment == 1)]):
        phenotype = to_phenotype(individual, image.shape[0], image.shape[1])
        save_type2(phenotype, os.path.join(type2_dir, f'pareto_{i}.png'))

elif args.algorithm == 'sga':
    population = sga(image=image,
                     weights=np.array(args.weights),
                     n_segments=args.n_segments,
                     population_size=args.population_size,
                     generations=args.generations,
                     p_mutate=args.p_mutate,
                     p_crossover=args.p_crossover,
                     )

    phenotype = to_phenotype(population[0], image.shape[0], image.shape[1])
    save_type2(phenotype, os.path.join(type2_dir, 'best.png'))

