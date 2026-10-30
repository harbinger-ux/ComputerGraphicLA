"""Write a tiny checker JPEG/PNG so texture demos run without a photo."""
import os

import numpy as np
from PIL import Image


def write_checker(path, cells=8, px=256, c0=(32, 32, 38), c1=(200, 55, 55)):
    """Create ``path`` if missing. Returns ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.isfile(path):
        return path
    cell = max(px // cells, 1)
    img = np.zeros((px, px, 3), dtype=np.uint8)
    for y in range(px):
        for x in range(px):
            img[y, x] = c1 if ((x // cell) + (y // cell)) % 2 else c0
    Image.fromarray(img, mode="RGB").save(path)
    return path
