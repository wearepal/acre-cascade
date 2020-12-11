"""Test the encoding step."""

from pathlib import Path

from PIL import Image
import numpy as np
import torch

from src.data import IndexEncodeMask
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


def test_to_mask_tens():
    """Test tensor loading is same as provided function."""
    base_path = Path(__file__).parent / "data"
    test_img_path = base_path / "rgb_mask_example.png"
    mask_img = Image.open(test_img_path)
    tensor_mask = IndexEncodeMask()(mask_img)
    np_tens_mask = torch.as_tensor(read_rgb_mask(test_img_path)).t()
    torch.testing.assert_allclose(tensor_mask, np_tens_mask)
