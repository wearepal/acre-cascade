"""Submission generation functions."""

from dataclasses import dataclass
from typing import List

import numpy as np

from src import Crop, Team


@dataclass
class Segmentation:
    """Segmentations form part of a Submission."""

    crop: str
    weed: str


@dataclass
class Submission:
    """Submission definition."""

    filename: str
    shape: List[int]
    team: Team
    crop: Crop
    segmentation: Segmentation


def rle_encode(img: np.ndarray) -> str:
    """Encode a numpy array to a string.

    Args:
        img (np.ndarray): 1 - foreground, 0 - background
    Returns:
        str: run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def sample_to_submission(
    filename: str, team_name: Team, crop_name: Crop, mask: np.ndarray
) -> Submission:
    """For a sample, convert to json ready to be zipped and uploaded.

    Args:
        filename (str): the filename of the image.
        team_name (str): Name of data collection team.
        crop_name (str): Name of crop.
        mask (np.ndarray): Mask to be submitted.

    Returns:
        (str, Submission): Tuple containing filename used to index and the submission object.
        May have to call dataclasses.asdict() on Submission.
    """
    segmentation_obj = Segmentation(crop=rle_encode(mask == 1), weed=rle_encode(mask == 2))
    return Submission(
        filename=filename,
        shape=list(mask.shape),
        team=team_name,
        crop=crop_name,
        segmentation=segmentation_obj,
    )
