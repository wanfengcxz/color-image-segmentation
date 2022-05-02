import argparse
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import read_image, unique_pixels
from visualization.fitness import visualize_fitness_history

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', help='path/to/output/dir', type=str, default='output/nsga/02-195653')
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

evaluated = 'PRI.csv' in os.listdir(args.output_dir)

if evaluated:
    pri = pd.read_csv(os.path.join(args.output_dir, 'PRI.csv'), header=None).to_numpy()
    df['PRI'] = pri

n_display = min(args.n_display, df.shape[0])

fig, axes = plt.subplots(4, n_display, squeeze=False, figsize=(14, 7))

axes[0][0].set_ylabel('Type 1')
axes[1][0].set_ylabel('Type 2')
axes[2][0].set_ylabel('Color')
axes[3][0].set_ylabel('Fitness')

df = df.sample(frac=1).reset_index(drop=True)

if evaluated:
    best_idx = df['PRI'].idxmax()
    temp = df.iloc[0].copy()
    df.iloc[0] = df.iloc[best_idx].copy()
    df.iloc[best_idx] = temp

for n in range(n_display):
    color_image = read_image(os.path.join(color_dir, df.iloc[n]['color']))
    axes[0][n].imshow(read_image(os.path.join(type1_dir, df.iloc[n]['type1'])))
    axes[1][n].imshow(read_image(os.path.join(type2_dir, df.iloc[n]['type2'])))
    axes[2][n].imshow(color_image)
    text = f'Edge Value: {df.iloc[n]["edge_value"]:.2f}\n' \
           f'Connectivitiy: {df.iloc[n]["connectivity"]:.2f}\n' \
           f'Deviation: {df.iloc[n]["deviation"]:.2f}\n\n' \
           f'Segments: {unique_pixels(color_image)}\n'

    if evaluated:
        text += f'PRI: {df.iloc[n]["PRI"]:.2f}%'

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


