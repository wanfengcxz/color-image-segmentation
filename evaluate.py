import argparse

from evaluator.run import eval_files

if __name__ == "__main__":
    output_dir = "31-203457"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--optimal",
        help="path/to/optimal/dir",
        type=str,
        default="training_images\\pavia\\gt",
    )
    parser.add_argument(
        "-s",
        "--student",
        help="path/to/student/student",
        type=str,
        default=f"output\\nsga\\{output_dir}\\segmentations\\type2",
    )
    args = parser.parse_args()

    eval_files(args.optimal, args.student)
