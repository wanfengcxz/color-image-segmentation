from typing import Tuple
from custom_types import Genotype, ImageDimensions
from utilities import calculate_connectivity, calculate_deviation, calculate_edge_value, prims_algorithm, separate_segments


class Individual:
    genotype: Genotype
    fitness: float

    def __init__(self, img_data, genotype=None, weights=None):

        self.img_data = img_data
        self.img_dim = ImageDimensions(img_data.shape[0], img_data.shape[1])
        self.weights = weights
        # Generate initial genotype
        if genotype is None:
            self.genotype = prims_algorithm(img_data)
        else:
            self.genotype = genotype
        # Segments dont need to be known initially
        self.segments = None
        # ECD - Tuple containing edge_value, connectivity_value, deviation_value
        self.ecd = None

        self.fitness = None
        self.crowding_distance = None
        self.rank = 100

    def fitness_function(self):
        """ Calculate fitness of individual """
        # TODO: Only calculate fitness if needed?
        if self.fitness is not None:
            return self.fitness
        if self.segments is None:
            self.segments = separate_segments(self.genotype)
        if self.ecd is None:
            self.get_ecd_values()

        # Weights
        edge_weight = self.weights[0]
        connectivity_weight = self.weights[1]
        deviation_weight = self.weights[2]

        self.fitness = edge_weight * \
            self.ecd[0] + connectivity_weight * \
            self.ecd[1] + deviation_weight*self.ecd[2]
        return self.fitness

    def get_segments(self):
        """ Get segments """
        if self.segments is None:
            self.segments = separate_segments(self.genotype)
        return self.segments

    def get_ecd_values(self) -> Tuple[float, float, float]:
        """ Get edge, connectivity and deviation values """
        if self.ecd is None:
            edge_value = calculate_edge_value(
                self.img_data, self.get_segments())
            connectivity_value = calculate_connectivity(
                self.segments, self.img_dim)
            deviation = calculate_deviation(self.img_data, self.segments)
            self.ecd = (edge_value, -connectivity_value, -deviation)
        return self.ecd
