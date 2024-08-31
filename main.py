import argparse
import os
import datetime
import shutil
from scipy.io import loadmat
import cv2

import numpy as np
from matplotlib import pyplot as plt

from evaluator.run import eval_files
from evolution.algorithms.nsga_ii import nsga_ii
from evolution.algorithms.sga import sga
from evolution.individual.phenotype import to_phenotype
from utils import read_image
from visualization.individual import (
    save_type2,
    save_type1,
    to_contour_segmentation_v2,
    to_color_segmentation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_path",
        help="path/to/image",
        type=str,
        default="training_images/353013/Test image.jpg",
    )
    parser.add_argument(
        "-gt", "--ground_truth", help="path/to/ground/truth/dir", type=str, default=None
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Type of algorithm. Can be {nsga, sga}",
        type=str,
        default="nsga",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="path/to/output/directory",
        type=str,
        default="output",
    )
    parser.add_argument(
        "-ns",
        "--n_segments",
        help="Number of segments to initialize population with",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n", "--population_size", help="Size of population", type=int, default=10
    )
    parser.add_argument(
        "-g",
        "--generations",
        help="Number of generations to run for",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-pc", "--p_crossover", help="Crossover probability", type=float, default=0.8
    )
    parser.add_argument(
        "-pm", "--p_mutate", help="Mutation probability", type=float, default=0.2
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Weights used for sga",
        type=list,
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--mode", help="pls choose from [hsi, rgb]", type=str, default="hsi"
    )
    parser.add_argument(
        "--r",
        "--reduce_algo",
        help="dimension reduce, support [pca]",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now()

    output_dir = os.path.join(
        args.output_dir, args.algorithm, timestamp.strftime("%d-%H%M%S")
    )

    type1_dir = os.path.join(output_dir, "segmentations", "type1")
    type2_dir = os.path.join(output_dir, "segmentations", "type2")
    color_dir = os.path.join(output_dir, "segmentations", "color")
    os.makedirs(type1_dir, exist_ok=True)
    os.makedirs(type2_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    # image = read_image(args.image_path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    image = loadmat("training_images/pavia/PaviaU.mat")["paviaU"]  # (610, 340, 103)
    image = image / image.max()
    # cv2.imshow("image", image[:, :, 50:53])
    # cv2.waitKey(0)

    # plt.imsave(os.path.join(output_dir, "image.png"), image)
    plt.imsave(os.path.join(output_dir, "image.png"), image[:, :, 50:53])
    # image = image[:, :, 50:53]

    if args.ground_truth is not None:
        gt_dir = os.path.join(output_dir, "gts")
        os.makedirs(gt_dir, exist_ok=True)
        for gt in os.listdir(args.ground_truth):
            shutil.copy(os.path.join(args.ground_truth, gt), gt_dir)

    if args.algorithm == "nsga":
        population, front_assignment = nsga_ii(
            image=image,
            n_segments=args.n_segments,
            population_size=args.population_size,
            generations=args.generations,
            p_mutate=args.p_mutate,
            p_crossover=args.p_crossover,
            fitness_path=os.path.join(output_dir, "fitness.csv"),
        )

        print("Saving images...")
        if image.shape[-1] != 3:
            image = image[:, :, 50:53]
        for i, individual in enumerate(population[np.where(front_assignment == 1)]):
            phenotype = to_phenotype(individual, image.shape[0], image.shape[1])
            segmentation = to_contour_segmentation_v2(phenotype)
            save_type1(
                segmentation,
                image,
                os.path.join(type1_dir, f"pareto_{i}.png"),
                convert=False,
            )
            save_type2(
                segmentation, os.path.join(type2_dir, f"pareto_{i}.png"), convert=False
            )
            plt.imsave(
                os.path.join(color_dir, f"pareto_{i}.png"),
                to_color_segmentation(phenotype),
            )

    elif args.algorithm == "sga":
        population = sga(
            image=image,
            weights=np.array(args.weights),
            n_segments=args.n_segments,
            population_size=args.population_size,
            generations=args.generations,
            p_mutate=args.p_mutate,
            p_crossover=args.p_crossover,
            fitness_path=os.path.join(output_dir, "fitness.csv"),
        )

        print("Saving images...")
        phenotype = to_phenotype(population[0], image.shape[0], image.shape[1])
        segmentation = to_contour_segmentation_v2(phenotype)
        save_type1(
            segmentation, image, os.path.join(type1_dir, f"best.png"), convert=False
        )
        save_type2(segmentation, os.path.join(type2_dir, "best.png"), convert=False)
        plt.imsave(
            os.path.join(color_dir, f"best.png"), to_color_segmentation(phenotype)
        )

    if args.ground_truth is not None:
        results = eval_files(gt_dir, type2_dir)
        with open(os.path.join(output_dir, "PRI.csv"), "w") as file:
            np.savetxt(file, results)

    print(f"Output saved to {output_dir}")


def display_Pavia():
    gt = loadmat("training_images/hsi/PaviaU_gt.mat")["paviaU_gt"].astype(np.float32)
    print(gt.shape)
    gt = gt / gt.max()
    cv2.imshow("gt", gt)
    cv2.waitKey()


if __name__ == "__main__":
    main()
    # display_Pavia()
