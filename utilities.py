from typing import Tuple
from PIL import Image
import numpy as np
from numba import jit

from custom_types import ImageDimensions


@jit(nopython=True)
def flattened_index(pixel_position: Tuple[int, int], img_dim: ImageDimensions) -> int:
    """ Get the index of a pixel in the flattened array """
    if pixel_position[0] < 0 or pixel_position[0] >= img_dim.rows:
        raise IndexError("Row index out of bounds")
    if pixel_position[1] < 0 or pixel_position[1] >= img_dim.cols:
        raise IndexError("Column index out of bounds")
    return pixel_position[0] * img_dim.cols + pixel_position[1]


def is_border_pixel(pixel_position: Tuple[int, int], img_dim: ImageDimensions) -> bool:
    """ Check if a pixel is on the border of the image """
    if pixel_position[0] == 0 or pixel_position[1] == 0:
        return True
    if pixel_position[0] == img_dim.rows-1 or pixel_position[1] == img_dim.cols-1:
        return True
    else:
        return False


@jit(nopython=True)
def is_segment_border_pixel(pixel_position: Tuple[int, int], img_dim, segments) -> bool:
    """ Check if a pixel is on the border of a segment """
    pixel_index = flattened_index(pixel_position, img_dim)
    for y in range(-1, 2):
        for x in range(-1, 2):
            pos = (pixel_position[0]+y, pixel_position[1]+x)
            try:
                neighbour_index = flattened_index(pos, img_dim)
                if segments[neighbour_index] != segments[pixel_index]:
                    return True
            except Exception:
                continue
    return False


def save_segmented_img(filepath: str, img_data, segments, type: int = 2):
    if type not in [1, 2]:
        raise ValueError("Type must be 1 or 2")

    new_img_data = img_data.copy()
    img_dimensions = ImageDimensions(img_data.shape[0], img_data.shape[1])

    # Check every pixel if it is a border pixel
    for y in range(img_dimensions.rows):
        for x in range(img_dimensions.cols):
            pixel_pos = (y, x)

            if type == 1:  # Type 1 segmentation
                if (is_border_pixel(pixel_pos, img_dimensions) or is_segment_border_pixel(pixel_pos, img_dimensions, segments)):
                    # Make border pixels green
                    new_img_data[y, x, 0] = 0
                    new_img_data[y, x, 1] = 255
                    new_img_data[y, x, 2] = 0

            elif type == 2:  # Type 2 segmentation
                if (is_border_pixel(pixel_pos, img_dimensions) or is_segment_border_pixel(pixel_pos, img_dimensions, segments)):
                    # Make border pixels black
                    new_img_data[y, x, 0] = 0
                    new_img_data[y, x, 1] = 0
                    new_img_data[y, x, 2] = 0
                else:
                    # And everything else white
                    new_img_data[y, x, 0] = 255
                    new_img_data[y, x, 1] = 255
                    new_img_data[y, x, 2] = 255

    new_img_data = new_img_data.astype(np.int8)
    Image.fromarray(new_img_data, "RGB").save(
        filepath+"type_"+str(type)+".jpg")
