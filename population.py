import random
from typing import List
from custom_types import ImageDimensions

from individual import Individual
from ea_functions import mutate, reproduce


class Population:
    individuals: List[Individual]

    def __init__(self, img_data, population_size: int = 100, segment_constraints=None, weights=None):
        self.individuals = []
        self.img_data = img_data
        self.population_size = population_size
        self.segment_constraints = segment_constraints
        self.weights = weights

    def initialize_population(self) -> None:
        """ Initialize the population with random individuals """
        self.individuals = [Individual(self.img_data, weights=self.weights)
                            for _ in range(self.population_size)]

    def get_avg_fitness(self) -> float:
        """ Get the average fitness of the population """
        return sum(ind.fitness for ind in self.individuals) / self.population_size

    def get_best_individual(self) -> Individual:
        """ Get the best individual in the population """
        return max(self.individuals, key=lambda ind: ind.fitness)

    def tournament_selection(self, tournament_size: int = 2) -> Individual:
        """ Select a random subset of individuals and return the best one """
        if tournament_size > self.population_size:
            raise ValueError(
                "Tournament size cannot be larger than population size")
        tournament = random.sample(self.individuals, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def generate_new_population(self):
        """ Generate new population through crossover and mutation """
        image_dimensions = ImageDimensions(
            self.img_data.shape[0], self.img_data.shape[1])
        offsprings = []
        for i in range(self.population_size//2):
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Reproduction
            child1_genotype, child2_genotype = reproduce(parent1, parent2)

            # Mutation
            child1_genotype = mutate(child1_genotype, image_dimensions)
            child2_genotype = mutate(child2_genotype, image_dimensions)

            child1 = Individual(self.img_data, child1_genotype,
                                weights=self.weights)
            child2 = Individual(self.img_data, child2_genotype,
                                weights=self.weights)

            offsprings.append(child1)
            offsprings.append(child2)

        num_elite = int(self.population_size * 0.2)

        # Sort new generation by fitness and remove the worst individuals
        offsprings.sort(key=lambda x: x.fitness, reverse=True)
        offsprings = offsprings[:self.population_size - num_elite]

        # Sort self.individuals by fitness and keep the best individuals
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(num_elite):
            offsprings.append(self.individuals[i])

        if len(offsprings) != self.population_size:
            raise Exception('Population size is not correct')
        self.individuals = offsprings
