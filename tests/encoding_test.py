"""Test the encoding step."""

from pathlib import Path

import numpy as np

from src.read_files import read_rgb_mask


def test_encoding():
    """Test the encoding."""
    base_path = Path(__file__).parent / "data"
    test_img_path = base_path / "rgb_mask_example.png"
    mask = read_rgb_mask(test_img_path)
    assert mask is not None

    given_mask = np.load(base_path / "arr_mask_example.npy")

    assert given_mask is not None
    np.testing.assert_array_equal(mask, given_mask)
