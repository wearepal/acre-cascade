"""Test the submission."""
from dataclasses import asdict
import json
from pathlib import Path

from torch import from_numpy

from src.read_files import read_rgb_mask
from src.submission_generation import sample_to_submission


def test_mask_to_json():
    """Test that the mask can be wrapped up as specified."""
    base_path = Path(__file__).parent / "data"
    test_img_path = base_path / "rgb_mask_example.png"
    mask = read_rgb_mask(test_img_path)

    tensor_mask = from_numpy(mask)

    batch = ("arr_mask_example", "Bipbip", "Haricot", tensor_mask)
    submission_dict = sample_to_submission(*batch)

    assert submission_dict.filename == "arr_mask_example"
    assert submission_dict.shape == [1536, 2048]
    assert submission_dict.team == "Bipbip"
    assert submission_dict.crop == "Haricot"
    assert submission_dict.segmentation.crop is not None
    assert submission_dict.segmentation.weed is not None

    submission_dict = asdict(submission_dict)
    submission_dict = {submission_dict.pop("filename"): submission_dict}

    sample = base_path / "sample_submission.json"
    with open(f"{sample}") as json_file:
        sample_json = json.load(json_file)
    assert sample_json == submission_dict


def test_batch():
    """Test the batch-wise to submission."""
    base_path = Path(__file__).parent / "data"
    test_img_path = base_path / "rgb_mask_example.png"
    mask = read_rgb_mask(test_img_path)

    tensor_mask = from_numpy(mask)
    batch = [
        ("alpha.jpg", "Pead", "Haricot", tensor_mask),
        ("bravo.jpg", "Bipbip", "Haricot", tensor_mask),
        ("charlie.jpg", "Roseau", "Mais", tensor_mask),
    ]

    submission = {
        _sub.filename: asdict(_sub)
        for _sub in [sample_to_submission(a, b, c, d) for a, b, c, d in batch]
    }

    for _file, _team, _crop, _ in batch:
        assert _file in submission
        assert submission[_file]["shape"] == [1536, 2048]
        assert submission[_file]["team"] == _team
        assert submission[_file]["crop"] == _crop
        assert submission[_file]["segmentation"]["crop"] is not None
        assert submission[_file]["segmentation"]["weed"] is not None
