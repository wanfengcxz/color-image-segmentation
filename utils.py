from PIL import Image
import numpy as np


def read_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return np.array(image, dtype=float) / 255


def unique_pixels(image: np.ndarray) -> int:
    return len(set(tuple(v) for m2d in image for v in m2d))