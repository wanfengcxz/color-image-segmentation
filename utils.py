from PIL import Image
import numpy as np
from scipy.io import loadmat


def read_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return np.array(image, dtype=float) / 255


def read_hsi_image(image_path: str, key: str) -> np.ndarray:
    image = loadmat(image_path)[key]  # (610, 340, 103)
    image = image / image.max()
    return image


def unique_pixels(image: np.ndarray) -> int:
    return len(set(tuple(v) for m2d in image for v in m2d))
