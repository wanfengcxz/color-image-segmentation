import random
from typing import Tuple
from custom_types import Genotype, ImageDimensions
from individual import Individual
from utilities import dissolve_segment, get_flattened_neighbours


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
    # elif random_value <= 0.90:
    #     child1_genotype, child2_genotype = uniform_crossover(
    #         parent1, parent2)
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


def mutate(img_data, genotype: Genotype, image_dim: ImageDimensions, segment_constraints) -> Genotype:
    """ Bitwise mutation. Returns mutated genotype """
    random_number = random.random()
    segments = Individual(img_data, genotype).get_segments()
    num_segments = max(segments)+1
    # 10% chance of mutation
    if random_number <= 0.1 and num_segments <= int(0.9*segment_constraints['max']):
        new_genotype = genewise_mutation(
            genotype, 0.000002, image_dim)
    elif random_number <= 1.0:
        if num_segments > segment_constraints["max"]:

            # segment_id = random.randint(0, num_segments-1)
            # Count how many of element in segments
            counts = [0] * num_segments
            for i in range(len(segments)):
                counts[segments[i]] += 1

            segment_id = counts.index(min(counts))
            new_genotype = dissolve_segment(
                img_data, genotype, segments, segment_id)
        else:
            new_genotype = genotype
    # elif random_number <= 1.0:
    #     segments = Individual(img_data, genotype).get_segments()
    #     num_segments = max(segments)+1
    #     if num_segments > segment_constraints["max"]:
    #         # Select two random segment ids
    #         segment_id1 = random.randint(0, num_segments-1)
    #         segment_id2 = random.randint(0, num_segments-1)

    #         # Get index of pixel in segment 1
    #         done1 = False
    #         done2 = False
    #         for i in range(len(genotype)):
    #             if done1 and done2:
    #                 break
    #             if segments[i] == segment_id1:
    #                 segment_index1 = i
    #                 done1 = True
    #             if segments[i] == segment_id2:
    #                 segment_index2 = i
    #                 done2 = True
    #         new_genotype = genotype
    #         new_genotype[segment_index1] = segment_index2
    #     else:
    #         new_genotype = genotype
    else:
        new_genotype = genotype
    return new_genotype
