import pickle
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from fitness.fitness_functions import RealValueFitnessFunction, FitnessFunction


class GeneticAlgorithm(ABC):
    """The base class used to implement different kinds of genetic algorithms (e.g. SGA and crowding)."""

    def __init__(self,
                 population_size: int = 32,
                 n_bits: int = 8,
                 fitness_function: FitnessFunction = None,
                 p_cross_over: float = 0.6,
                 p_mutation: float = 0.05,
                 offspring_multiplier: int = 1):
        """
        :param population_size: Number of individuals in population.
        :param n_bits: Number of bits used to represent each individual.
        :param fitness_function: Fitness function to use during evolution.
        :param p_cross_over: Probability of crossover of two parents.
        :param p_mutation: Probability of mutating offspring.
        :param offspring_multiplier: Decides how many offspring are created in each generation:
                                     population_size is selected from population_size * offspring_multiplier offsprings
        """

        self.population_size: int = population_size
        self.n_bits: int = n_bits

        if fitness_function is None:
            raise TypeError('fitness_function must be specified')

        self.fitness_function = fitness_function
        self.p_cross_over: float = p_cross_over
        self.p_mutation: float = p_mutation
        self.offspring_multiplier = offspring_multiplier

        self.population: np.ndarray = self.init_population(population_size, n_bits)

        # Used to store histories during fit
        self.population_history: list[np.ndarray] = []
        self.fitness_history: list[np.ndarray] = []
        self.entropy_history: list[float] = []

    @staticmethod
    def init_population(population_size: int, n_bits: int) -> np.ndarray:
        """Initializes population of size (n_individuals x n_bits).
        Assignment task a).
        :param population_size: Number of individuals in population.
        :param n_bits: Number of bits used to represent each individual.
        :return: Numpy array of the whole population. Shape: (population_size x n_bits).
        """

        return np.random.randint(0, 2, (population_size, n_bits))

    @staticmethod
    def calculate_entropy(population: np.ndarray, epsilon: float = 1e-18) -> float:
        """Calculates entropy of a population.
        :param population: A (Nxb) numpy array of a population.
        :param epsilon: Minimum probability to avoid taking log of 0.
        :return: Entropy of population
        """
        probabilities = population.mean(axis=0).clip(min=epsilon)  # Clip to avoid taking log of 0.
        return -np.dot(probabilities, np.log2(probabilities))

    def _get_fitness_stats(self) -> np.ndarray:
        """Calculates fitness of whole population and returns sum, max/min, and mean of these.
        :return: A (3x1) numpy array of the sum, max, and mean of the fitness of the population.
        """

        fitness: np.ndarray = self.fitness_function(population=self.population)
        if self.fitness_function.maximizing:
            return np.array([fitness.sum(), fitness.max(), fitness.mean()])
        else:
            return np.array([fitness.sum(), fitness.min(), fitness.mean()])

    def _get_selection_probabilities(self, fitness: np.ndarray) -> np.ndarray:
        """Calculates selection probabilities from a fitness vector.
        The probabilities are calculated using the roulette wheel method.
        :param fitness: A (Nx1) numpy array specifying the fitness of each individual in the population.
        :return: A (Nx1) numpy array of the probabilities that an individual will be chosen as a parent.
        """

        if not self.fitness_function.maximizing:
            fitness *= -1

        return np.exp(fitness) / np.exp(fitness).sum()

    def _parent_selection(self, population: np.ndarray) -> np.ndarray:
        """Selects parents for the next generation of the population.
        :return: A multiset chosen from the current population.
        """

        parent_population = population.copy()

        fitness = self.fitness_function(population=parent_population)
        probabilities = self._get_selection_probabilities(fitness)
        indeces = np.random.choice(len(fitness),
                                   size=self.population_size,
                                   replace=True,
                                   p=probabilities)

        parent_population = parent_population[indeces]
        np.random.shuffle(parent_population)  # Shuffles the mating pool
        return parent_population

    def _cross_over(self, popultation: np.ndarray) -> np.ndarray:
        """Performs cross over for a whole population.
        Two and two individuals are crossed. If the population contains an odd
        number of individuals, the last one is not crossed, and just passes through.
        This should be done after selection.
        Assignment task c)
        :param popultation: A (Nxb) numpy array of a population.
        :return: A (Nxb) numpy array of the crossed over population.
        """

        crossed_population = popultation.copy()
        for p1, p2 in zip(crossed_population[::2], crossed_population[1::2]):
            if np.random.random() < self.p_cross_over:
                c = np.random.randint(1, self.n_bits)  # Random cross over point
                temp1 = p1[c:].copy()
                temp2 = p2[c:].copy()
                p1[c:], p2[c:] = temp2, temp1
        return crossed_population

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Mutates a population by randomly flipping bits.
        This should be done after cross over.
        Assignment task c)
        :param population: A (Nxb) numpy array of a population.
        :return: A (Nxb) numpy array of the mutated population.
        """

        mutated_population = population.copy()
        mask = np.random.choice([0, 1], size=population.shape, p=[1-self.p_mutation, self.p_mutation])
        idx = np.where(mask == 1)
        mutated_population[idx] = 1 - mutated_population[idx]
        return mutated_population

    @abstractmethod
    def _survivor_selection(self, parents: np.ndarray, offspring: np.ndarray) -> np.ndarray:
        """Selects and returns survivors for the next generation.
        :param parents: A numpy array of the parents of the current generation.
        :param offspring: A numpy array of the offsprings of the current generation.
        :return: A numpy array of the survivors for the next generation.
        """

        raise NotImplementedError('Subclass must implement _survivor_selection() method.')

    def fit(self,
            generations: int = 100,
            termination_fitness: Optional[float] = None,
            verbose: bool = False,
            visualize: bool = False,
            vis_sleep: float = 0.1,
            ) -> None:

        """Fits the population through a generational loop.
        For each generation the following is done:
        1. Selection
        2. Cross over
        3. Mutation
        4. Survivor selection
        :param generations: Number of generations the algorithm should run.
        :param termination_fitness: Fitting stops if termination_fitness has been reached.
                                    If None, all generations are performed.
        :param verbose: Whether or not additional data should be printed during fitting.
        :param visualize: Whether or not to visualize population during fitting.
        :param vis_sleep: Sleep timer between each generation. Controls speed of visualization.
        """

        # Only visualize if the fitness function is a real value fitness function.
        # Not visualizing for regression tasks.
        visualize = visualize and issubclass(type(self.fitness_function), RealValueFitnessFunction)

        self.population_history = []
        self.fitness_history = []
        self.entropy_history = []

        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 12))
            fig.suptitle(f'{self.__class__.__name__} on {self.fitness_function.__class__.__name__}')
            interval = self.fitness_function.interval
            x_func = np.linspace(interval[0], interval[1], 10*(interval[1] - interval[0]))
            y_func = self.fitness_function.fitness(x_func)
            ax.plot(x_func, y_func)
            ax.set_xlabel('Value')
            ax.set_ylabel('Fitness')
            points, = ax.plot(x_func, y_func, 'ro', label='Population')
            ax.legend()

        for g in range(generations):
            print(f'Generation {g} - {self.__class__.__name__}')
            fitness_stats = self._get_fitness_stats()
            self.fitness_history.append(fitness_stats)

            entropy = self.calculate_entropy(self.population)
            self.entropy_history.append(entropy)

            if termination_fitness is not None:
                if (self.fitness_function.maximizing and fitness_stats[-1] >= termination_fitness) or \
                   (not self.fitness_function.maximizing and fitness_stats[-1] <= termination_fitness):
                    break
            if verbose:
                print(f'Entropy: {round(entropy, 2)}')
                max_or_min = 'Max' if self.fitness_function.maximizing else 'Min'
                fitness_table = PrettyTable(['Sum', max_or_min, 'Mean'], title='Fitness')
                fitness_table.add_row([round(s, 4) for s in fitness_stats])
                print(fitness_table)
            if visualize:
                ax.set_title(f'Generation {g}')
                x = self.fitness_function.bits_to_scaled_nums(self.population)
                y = self.fitness_function.fitness(x)
                points.set_xdata(x)
                points.set_ydata(y)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(vis_sleep)

            self.population_history.append(self.population.copy())
            parents = self._parent_selection(self.population)
            crossed = self._cross_over(parents)
            mutated = self._mutate(crossed)
            self.population = self._survivor_selection(parents, mutated)

        self.population_history = np.asarray(self.population_history)
        self.fitness_history = np.asarray(self.fitness_history)
        self.entropy_history = np.asarray(self.entropy_history)

    def fittest_individual(self) -> np.ndarray:
        """Returns the currently fittest individual
        :return: A (1xb) numpy array of the fittest individual.
        """

        fitness = self.fitness_function(population=self.population)
        return self.population[np.argmax(fitness)]

    def save(self, file_name: str) -> None:
        """Saves the GeneticAlgorithm object to a file.
        Useful to save population and histories after fitting.
        :param file_name: File name of where to save. Expects that folder where file should be created exists.
        """

        with open(file_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str) -> 'Network':
        """Loads a GeneticAlgorithm object from a file.
        Useful to load population and histories from previously run fit.
        :param file_name: File name of saved object.
        :return: A GeneticAlgorithm object as specified by the file.
        """

        with open(file_name, 'rb') as file:
            network = pickle.load(file)
        return network