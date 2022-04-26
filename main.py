import numpy as np
from population import Population
from individual import Individual
from tqdm.auto import trange, tqdm
from PIL import Image
from utilities import save_segmented_img


def genetic_algorithm(img_data, segment_constraints, population_size: int, num_generations: int = 40, debug=False) -> Individual:

    # Initialize population
    pop = Population(img_data, population_size, segment_constraints)
    pop.initialize_population()

    # Run evolution for num_generations
    for _ in trange(num_generations):
        if debug:
            tqdm.write(str(pop.get_avg_fitness()))
        pop.generate_new_population()

    # Get best individual solution
    return pop.get_best_individual()


if __name__ == "__main__":
    # Variables
    segment_constraints = {
        "min": 3,
        "max": 40
    }
    image_no = "86016"

    # Load img data
    img_data = np.asarray(Image.open(
        "training_images/"+image_no+"/Test image.jpg"), dtype='int64')

    # Run GA
    solution = genetic_algorithm(
        img_data, segment_constraints=segment_constraints, population_size=80, num_generations=10
    )

    save_segmented_img("training_images/"+image_no+"/solutions/",
                       img_data, solution.segments, 1)
    save_segmented_img("training_images/"+image_no+"/solutions/",
                       img_data, solution.segments, 2)
