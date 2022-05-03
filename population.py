import random
from typing import List

from matplotlib import pyplot as plt
from tqdm import tqdm
from custom_types import ImageDimensions

from individual import Individual
from ea_functions import mutate, reproduce
from utilities import calculate_connectivity, calculate_deviation, calculate_edge_value, dominates, get_top_ranks, sort_based_on_crowding_distance


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
        return sum(ind.fitness_function() for ind in self.individuals) / self.population_size

    def get_best_individual(self) -> Individual:
        """ Get the best individual in the population """
        return max(self.individuals, key=lambda ind: ind.fitness_function())

    def tournament_selection(self, tournament_size: int = 2) -> Individual:
        """ Select a random subset of individuals and return the best one """
        if tournament_size > self.population_size:
            raise ValueError(
                "Tournament size cannot be larger than population size")
        tournament = random.sample(self.individuals, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness_function())

    def tournament_selection_using_rank(self, tournament_size: int = 2) -> Individual:
        """ Select a random subset of individuals and return the best one """
        if tournament_size > self.population_size:
            raise ValueError(
                "Tournament size cannot be larger than population size")
        tournament = random.sample(self.individuals, tournament_size)
        return min(tournament, key=lambda ind: ind.rank)

    def generate_new_population(self, final=True):
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
            child1_genotype = mutate(
                self.img_data, child1_genotype, image_dimensions, self.segment_constraints)
            child2_genotype = mutate(
                self.img_data, child2_genotype, image_dimensions, self.segment_constraints)

            child1 = Individual(self.img_data, child1_genotype,
                                weights=self.weights)
            child2 = Individual(self.img_data, child2_genotype,
                                weights=self.weights)

            offsprings.append(child1)
            offsprings.append(child2)

        num_elite = int(self.population_size * 0.2)

        # Sort new generation by fitness and remove the worst individuals
        offsprings.sort(key=lambda x: x.fitness_function(), reverse=True)
        offsprings = offsprings[:self.population_size - num_elite]

        if final:
            offsprings.extend(self.individuals)
            for ind in offsprings:
                if not self.satisfies_constraints(ind):
                    offsprings.remove(ind)
            offsprings.sort(key=lambda x: x.fitness_function(), reverse=True)
            offsprings = offsprings[:self.population_size]
            self.offsprings = offsprings
            return
        else:
            # Sort self.individuals by fitness and keep the best individuals
            self.individuals.sort(
                key=lambda x: x.fitness_function(), reverse=True)
            for i in range(num_elite):
                offsprings.append(self.individuals[i])

            if len(offsprings) != self.population_size:
                raise Exception('Population size is not correct')
            self.individuals = offsprings

    def nsga2_evolve(self, final=True):
        """ Evolve using NSGA 2 algorithm """
        image_dimensions = ImageDimensions(
            self.img_data.shape[0], self.img_data.shape[1])
        offsprings = []
        for i in range(self.population_size//2):
            parent1 = self.tournament_selection_using_rank()
            parent2 = self.tournament_selection_using_rank()

            # Reproduction
            child1_genotype, child2_genotype = reproduce(parent1, parent2)

            # Mutation
            child1_genotype = mutate(
                self.img_data, child1_genotype, image_dimensions, self.segment_constraints)
            child2_genotype = mutate(
                self.img_data, child2_genotype, image_dimensions, self.segment_constraints)

            child1 = Individual(self.img_data, child1_genotype,
                                weights=self.weights)
            child2 = Individual(self.img_data, child2_genotype,
                                weights=self.weights)

            offsprings.append(child1)
            offsprings.append(child2)

        new_population = []

        current_rank = 1
        remaining = self.individuals + offsprings

        while len(new_population) < self.population_size:
            # Get the non-dominated individuals
            nondominated_inds_index, remaining_index = get_top_ranks(
                remaining, self.segment_constraints)

            nondominated_inds = []
            new_remaining = []

            for i in range(len(remaining)):
                if i in nondominated_inds_index:
                    if final and not self.satisfies_constraints(remaining[i]):
                        continue
                    nondominated_inds.append(remaining[i])
                else:
                    new_remaining.append(remaining[i])

            remaining = new_remaining

            for ind in nondominated_inds:
                ind.rank = current_rank
                if final:
                    if not self.satisfies_constraints(ind):
                        nondominated_inds.remove(ind)

            # Sort based on crowding distance if not all can fit
            if len(new_population) + len(nondominated_inds) > self.population_size:
                nondominated_inds = sort_based_on_crowding_distance(
                    nondominated_inds)

            new_population.extend(nondominated_inds)
            current_rank += 1

        new_population = new_population[:self.population_size]
        self.individuals = new_population

    def population_summary(self):
        for i in range(len(self.individuals)):
            ecd = self.individuals[i].get_ecd_values()
            segments = self.individuals[i].get_segments()
            print(
                f"Individual {i}: edge_value: {ecd[0]}, connectivity: {-1*ecd[1]}, deviation: {-1*ecd[2]}, num_segments: {max(segments)+1}, rank: {self.individuals[i].rank}")

    def satisfies_constraints(self, individual: Individual) -> bool:
        """ Check if the individual satisfies the constraints """
        segments = individual.get_segments()
        num_segments = max(segments) + 1
        return self.segment_constraints['min'] <= num_segments <= self.segment_constraints['max']
