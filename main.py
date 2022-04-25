from PIL import Image
from numpy import asarray

from individual import Individual
# load the image
image = Image.open('training_images/train.png')
# convert image to numpy array
data = asarray(image)
print(data[3][3])


ind = Individual((4, 4))
ind.initialize_random()
print(ind.genotype)
print(ind.get_2d_array())
print(ind.get_value_down(10))


segmentation = [[0, 1, 2, 4, 5], [3, 6, 7, 10], [8, 9, 11, 12, 13, 14, 15]]


def get_deviation(segment):
    pass
