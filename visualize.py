import argparse
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import read_image
from visualization.fitness import visualize_fitness_history

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', help='path/to/output/dir', type=str, default='output/nsga/01-2136')
parser.add_argument('-n', '--n_display', help='Number of images to display', type=int, default=5)
parser.add_argument('-f', '--display_fitness', action='store_true', help='Whether to visualize pareto front')
args = parser.parse_args()

fitness_path = os.path.join(args.output_dir, 'fitness.csv')
df = pd.read_csv(fitness_path, dtype={'generation': int, 'front': int})
df = df[(df['generation'] == df['generation'].max()) & (df['front'] == 1)]

type1_dir = os.path.join(args.output_dir, 'segmentations/type1')
type1_images = sorted(os.listdir(type1_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
n_type1 = len(type1_images)

type2_dir = os.path.join(args.output_dir, 'segmentations/type2')
type2_images = sorted(os.listdir(type2_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
n_type2 = len(type2_images)

color_dir = os.path.join(args.output_dir, 'segmentations/color')
color_images = sorted(os.listdir(color_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
n_color = len(color_images)

df['type1'] = np.array(type1_images)
df['type2'] = np.array(type2_images)
df['color'] = np.array(color_images)

n_display = min(args.n_display, df.shape[0])

fig, axes = plt.subplots(4, n_display, squeeze=False, figsize=(14, 7))

axes[0][0].set_ylabel('Type 1')
axes[1][0].set_ylabel('Type 2')
axes[2][0].set_ylabel('Color')
axes[3][0].set_ylabel('Fitness')

for n, idx in enumerate(np.random.choice(df.shape[0], n_display)):
    axes[0][n].imshow(read_image(os.path.join(type1_dir, df.iloc[idx]['type1'])))
    axes[1][n].imshow(read_image(os.path.join(type2_dir, df.iloc[idx]['type2'])))
    axes[2][n].imshow(read_image(os.path.join(color_dir, df.iloc[idx]['color'])))
    text = f'Edge Value: {df.iloc[idx]["edge value"]:.2f}\n' \
           f'Connectivitiy: {df.iloc[idx]["connectivity"]:.2f}\n' \
           f'Deviation: {df.iloc[idx]["deviation"]:.2f}\n'
    axes[3][n].axis('off')
    axes[3][n].text(0.2, 0.5, text)

    axes[0][n].set_xticks([])
    axes[0][n].set_yticks([])
    axes[1][n].set_xticks([])
    axes[1][n].set_yticks([])
    axes[2][n].set_xticks([])
    axes[2][n].set_yticks([])
    axes[3][n].set_xticks([])
    axes[3][n].set_yticks([])

plt.tight_layout()

if args.display_fitness:
    visualize_fitness_history(fitness_path)

plt.show()


