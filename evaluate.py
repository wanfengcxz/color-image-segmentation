import argparse

from evaluator.run import eval_files

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optimal', help='path/to/optimal/dir', type=str)
parser.add_argument('-s', '--student', help='path/to/student/student', type=str)
args = parser.parse_args()

eval(args.optimal, args.student)

