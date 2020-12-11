"""Read image files."""
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def read_rgb_mask(img_path: Union[str, Path]) -> np.ndarray:
    """Read the RGB mask and make 0-2 ints.

    Args:
        img_path (Pathlike): path to the mask file

    Returns:
        np.ndarray: the numpy array containing target values
    """
    mask_img = Image.open(img_path)
    mask_arr = np.array(mask_img.copy())

    new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)

    # Use RGB dictionary in 'RGBtoTarget.txt' to convert RGB to target
    new_mask_arr[np.where(np.all(mask_arr == [216, 124, 18], axis=-1))] = 0
    new_mask_arr[np.where(np.all(mask_arr == [255, 255, 255], axis=-1))] = 1
    new_mask_arr[np.where(np.all(mask_arr == [216, 67, 82], axis=-1))] = 2

    return new_mask_arr
