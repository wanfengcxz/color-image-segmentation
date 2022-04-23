from PIL import Image
import numpy as np


def read_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return np.asarray(image)