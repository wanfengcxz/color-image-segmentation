import numpy as np
from population import Population
from individual import Individual
from tqdm.auto import trange, tqdm
from PIL import Image
from utilities import save_segmented_img
# from evaluator.run import main
import optuna


def genetic_algorithm(img_data, segment_constraints, population_size: int, num_generations: int = 40, weights=(0.33, 0.33, 0.33), debug=False, save: bool = False) -> Individual:

    # Initialize population
    pop = Population(img_data, population_size,
                     segment_constraints, weights=weights)
    pop.initialize_population()

    if save:
        b = pop.get_best_individual()
        save_segmented_img("training_images/"+image_no+"/solutions/gif/"+str(0),
                           img_data, b.get_segments(), 2)

    # Run evolution for num_generations
    for i in trange(num_generations, leave=True):
        if debug:
            tqdm.write(str(pop.get_avg_fitness()))
        pop.generate_new_population()

        if save:
            b = pop.get_best_individual()
            save_segmented_img("training_images/"+image_no+"/solutions/gif/"+str(i+1),
                               img_data, b.get_segments(), 2)

    # Get best individual solution
    return pop.get_best_individual()


def check_PRI_value(img_data, segment_ids):
    save_segmented_img("evaluator/evaluator/test_student/check_pri",
                       img_data, segment_ids, 2)
    PRI = main()
    print(f"PRI value: {PRI}")
    return PRI


def objective(trial):
    edge_weight = trial.suggest_float('edge_weight', 1e-8, 1, log=True)
    connectivity_weight = trial.suggest_float(
        'connectivity_weight', 1e-8, 1, log=True)
    deviation_weight = trial.suggest_float(
        'deviation_weight', 1e-8, 1, log=True)
    weights = (edge_weight, connectivity_weight, deviation_weight)

    img_data = np.asarray(Image.open(
        "training_images/"+image_no+"/Test image.jpg"), dtype='int64')

    s = genetic_algorithm(img_data, segment_constraints=segment_constraints,
                          population_size=40, num_generations=15, weights=weights)

    result = check_PRI_value(img_data, s.get_segments())

    return result


if __name__ == "__main__":
    # Variables
    segment_constraints = {
        "min": 3,
        "max": 40
    }
    # image_no = "86016"
    image_no = "118035"
    # image_no = "147091"
    # image_no = "176035"

    # Load img data
    img_data = np.asarray(Image.open(
        "training_images/"+image_no+"/Test image.jpg"), dtype='int64')

    weights = {'edge_weight': 0.0011748120300752,
               'connectivity_weight': 0.9985902255639098, 'deviation_weight': 2.349624060150376e-4}
    # weights = {'edge_weight': 0.001,
    #            'connectivity_weight': 0.998, 'deviation_weight': 0.001}
    # weights = {'edge_weight': 8.471792933811565e-06,
    #            'connectivity_weight': 0.2084033056268151, 'deviation_weight': 0.00048472170184982}
    weights_tuple = (
        weights['edge_weight'], weights['connectivity_weight'], weights['deviation_weight'])

    # Run GA
    solution = genetic_algorithm(
        img_data, segment_constraints=segment_constraints, population_size=100, num_generations=20, weights=weights_tuple, save=True)

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100)

    # study.best_params  # E.g. {'x': 2.002108042}

    save_segmented_img("training_images/"+image_no+"/solutions/",
                       img_data, solution.get_segments(), 1)
    save_segmented_img("training_images/"+image_no+"/solutions/",
                       img_data, solution.get_segments(), 2)
