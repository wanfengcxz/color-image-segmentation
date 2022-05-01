import random
from typing import Tuple
from custom_types import Genotype, ImageDimensions
from individual import Individual
from utilities import get_flattened_neighbours


# CROSSOVER FUNCTIONS

def reproduce(parent1: Individual, parent2: Individual) -> Tuple[Genotype, Genotype]:
    """ Reproduce two individuals """
    # Reproduction
    random_value = random.random()
    # 80% chance of Single Point crossover
    if random_value <= 0.80:
        child1_genotype, child2_genotype = sp_crossover(
            parent1, parent2)
    # 10% chance of Uniform crossover
    elif random_value <= 0.90:
        child1_genotype, child2_genotype = uniform_crossover(
            parent1, parent2)
    # 10% chance to copy parents genotype
    else:
        child1_genotype = parent1.genotype
        child2_genotype = parent2.genotype
    return child1_genotype, child2_genotype


def sp_crossover(parent1: Individual, parent2: Individual) -> Tuple[Genotype, Genotype]:
    """ Single point crossover. Returns genotype for 2 children """
    crossover_point = random.randint(0, len(parent1.genotype) - 1)
    child1_genotype = list(parent1.genotype[:crossover_point]) + \
        list(parent2.genotype[crossover_point:])
    child2_genotype = list(parent2.genotype[:crossover_point]) + \
        list(parent1.genotype[crossover_point:])
    return child1_genotype, child2_genotype


def uniform_crossover(parent1: Individual, parent2: Individual) -> Tuple[Genotype, Genotype]:
    """ Uniform crossover. Returns genotype for 2 children """

    child1_genotype = []
    child2_genotype = []

    for i in range(len(parent1.genotype)):
        if random.random() <= 0.5:
            child1_genotype.append(parent1.genotype[i])
            child2_genotype.append(parent2.genotype[i])
        else:
            child1_genotype.append(parent2.genotype[i])
            child2_genotype.append(parent1.genotype[i])

    return child1_genotype, child2_genotype


# MUTATION FUNCTIONS
def genewise_mutation(genotype: Genotype, mutation_rate: float, image_dim: ImageDimensions) -> Genotype:
    """ Bitwise mutation. Returns mutated genotype """
    for i in range(len(genotype)):
        if random.random() <= mutation_rate:
            neighbours = get_flattened_neighbours(i, image_dim)
            chosen_neighbour = random.choice(neighbours)
            genotype[i] = chosen_neighbour
    return genotype


def mutate(genotype: Genotype, image_dim: ImageDimensions) -> Genotype:
    """ Bitwise mutation. Returns mutated genotype """
    random_number = random.random()
    # 10% chance of mutation
    if random_number <= 0.1:
        new_genotype = genewise_mutation(
            genotype, 0.02, image_dim)
    else:
        new_genotype = genotype
    return new_genotype
