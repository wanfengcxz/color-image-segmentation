import argparse
import os

from matplotlib import pyplot as plt

from utils import read_image

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', help='path/to/output/dir', type=str, default='output/nsga/01-2136')
parser.add_argument('-n', '--n_display', help='Number of images to display', type=int, default=5)
args = parser.parse_args()

type1_dir = os.path.join(args.output_dir, 'segmentations/type1')
type1_images = os.listdir(type1_dir)
n_type1 = len(type1_images)

type2_dir = os.path.join(args.output_dir, 'segmentations/type2')
type2_images = os.listdir(type2_dir)
n_type2 = len(type2_images)

if len(type1_images) < args.n_display:
    print(f'Only {n_type1} type1 images exist. Showing {n_type1} images.')

if len(type2_images) < args.n_display:
    print(f'Only {n_type2} type2 images exist. Showing {n_type2} images.')

if len(type1_images) != len(type2_images):
    print('The number of type1 images and type2 images does not match. Images might not match.')

n_type1 = min(n_type1, args.n_display)
n_type2 = min(n_type2, args.n_display)
n_display = max(n_type1, n_type2)
fig, axes = plt.subplots(3, n_display, figsize=(16,16))

for n in range(n_display):
    axes[0][n].imshow(read_image(os.path.join(args.output_dir, 'image.png')))
    axes[0][n].set_xticks([])
    axes[0][n].set_yticks([])

for n in range(n_type1):
    axes[1][n].imshow(read_image(os.path.join(type1_dir, type1_images[n])))
    axes[1][n].set_xticks([])
    axes[1][n].set_yticks([])

for n in range(n_type2):
    axes[2][n].imshow(read_image(os.path.join(type2_dir, type2_images[n])))
    axes[2][n].set_xticks([])
    axes[2][n].set_yticks([])

plt.tight_layout()
plt.show()


