"""Script containing data-loading functionality."""

from abc import abstractmethod
from collections import defaultdict, namedtuple
import os
from pathlib import Path
import shutil
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from urllib.request import urlopen

from PIL import Image
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import torch
from torch.tensor import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, random_split
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from tqdm import tqdm
from typing_extensions import Literal, Protocol
from typing_inspect import get_args

from src.utils import implements

__all__ = ["AcreCascadeDataset", "AcreCascadeDataModule", "TrainBatch", "TestBatch", "Team", "Crop"]


Team = Literal["Bipbip", "Pead", "Roseau", "Weedelec"]
Crop = Literal["Haricot", "Mais"]
TrainBatch = namedtuple("TrainBatch", ["image", "mask", "team", "crop"])
TestBatch = namedtuple("TestBatch", ["image", "team", "crop", "filename"])
InputShape = namedtuple("InputShape", ["c", "h", "w"])
ImageSize = namedtuple("ImageSize", ["width", "height"])
Transform = Callable[[Union[Image.Image, Tensor]], Tensor]
Stage = Literal["fit", "test"]


def _download_from_url(url: str, dst: str) -> int:
    """Download from a url."""
    file_size = int(urlopen(url).info().get("Content-Length", -1))
    first_byte = os.path.getsize(dst) if os.path.exists(dst) else 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size,
        initial=first_byte,
        unit="B",
        unit_scale=True,
        desc=url.split("/")[-1],
    )
    req = requests.get(url, headers=header, stream=True)
    with (open(dst, "ab")) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


class _SizedDatasetProt(Protocol):
    """Typing Protocol for a SizedDataset (a Dataset defining a __len__ method)."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        ...


class _SizedDataset(Dataset):
    @abstractmethod
    def __len__(self) -> int:
        ...


class _DataTransformer(_SizedDataset):
    def __init__(self, base_dataset: _SizedDatasetProt, train: bool, transforms: Transform):
        self.base_dataset = base_dataset
        self.train = train
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Union[TrainBatch, TestBatch]:
        data = self.base_dataset[index]
        if self.transforms is not None:
            data = (self.transforms(data[0]),) + data[1:]
        if self.train:
            return TrainBatch(*data)
        return TestBatch(*data)


def _patches_from_image_mask_pair(
    image: Image, mask: Image, kernel_size: int, stride: int
) -> Iterator[Tuple[Image.Image, Image.Image]]:
    """Generates corresponding patches from an image-mask pair."""
    image_t = TF.to_tensor(image)
    mask_t = TF.to_tensor(mask)
    combined = torch.cat([image_t, mask_t], dim=0)  # pylint: disable=no-member

    patches = (
        combined.unfold(dimension=1, size=kernel_size, step=stride)
        .unfold(dimension=2, size=kernel_size, step=stride)
        .reshape(6, -1, kernel_size, kernel_size)
    )
    image_patches, mask_patches = patches.chunk(2, dim=0)
    for image_patch, mask_patch in zip(image_patches.unbind(dim=1), mask_patches.unbind(dim=1)):
        # Check that that the mask contains more than one class
        if torch.var(mask_patch) > 0:  # pylint: disable=no-member
            yield (TF.to_pil_image(image_patch), TF.to_pil_image(mask_patch))


class IndexEncodeMask:
    """Encode an RGB mask into its corresponding segmentation classes."""

    # We define the keys as the sum odf the RGB channels for computational efficiency.
    # This is possible because the sums are unique.
    mapping: ClassVar[Dict[int, int]] = {
        358: 0,  # Target 0 (background)
        396: 0,  # Other vegetation (which should be treated as Target 0)
        765: 1,  # Target 1 (crop)
        365: 2,  # Target 2 (weed)
    }

    def __call__(self, mask_img: Image.Image) -> Tensor:
        # sum over the RGB channels and convert from WH to HW format
        mask = torch.as_tensor(np.array(mask_img), dtype=torch.int64).sum(-1).t()  # type: ignore
        for sum_rgb, class_ in self.mapping.items():
            mask[mask == sum_rgb] = class_
        return mask


class AcreCascadeDataset(_SizedDataset):
    """PyTorch Dataset for the ACRE Cascade dataset."""

    url: ClassVar[
        str
    ] = "https://competitions.codalab.org/my/datasets/download/29a85805-2d8d-4701-a9ab-295180c89eb3"
    zipfile_name: ClassVar[str] = "images.zip"
    base_folder_name: ClassVar[str] = "crops"
    dataset_folder_name: ClassVar[str] = "Development_Dataset"
    train_folder_name: ClassVar[str] = "Training"
    test_folder_name: ClassVar[str] = "Test_Dev"

    def __init__(
        self,
        data_dir: Union[str, Path],
        download: bool = True,
        train: bool = True,
        teams: Optional[Union[Team, List[Team]]] = None,
        crop: Optional[Crop] = None,
        patch_size: int = 512,
        patch_stride: int = 256,
    ) -> None:
        super().__init__()

        self.root = Path(data_dir)
        self._base_folder = self.root / self.base_folder_name
        self._dataset_folder = self._base_folder / self.dataset_folder_name
        self._patch_dir = self._dataset_folder / self.train_folder_name / "Patches"
        self.download = download

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        if self.download:
            self._download_data()
        elif not self._check_downloaded():
            raise RuntimeError(
                f"Images don't exist at location {self._base_folder}. Have you downloaded them?"
            )

        self.train = train
        dtypes = {"images": "string", "team": "category", "crop": "category"}
        if self.train:
            split_folder = self._dataset_folder / self.train_folder_name
            dtypes["mask"] = "string"
        else:
            split_folder = self._dataset_folder / self.test_folder_name
        data = cast(
            pd.DataFrame,
            pd.read_csv(split_folder / "data.csv", dtype=dtypes, index_col=0),
        )
        # Filter the data by team(s)
        if teams is not None:
            if isinstance(teams, str):
                teams = [teams]
            data.query(expr=f"team == {teams}", inplace=True)  # type: ignore
        # Filter the data by crop, if a particular crop is specified
        if crop is not None:
            data.query(expr=f"crop == '{crop}'", inplace=True)  # type: ignore
        # Process the categorical values
        cat_cols = ["team", "crop"]
        # Construct  dictionaries to map back from categorical values to index values
        self.team_decoder = dict(enumerate(data["team"].cat.categories))  # type: ignore
        self.crop_decoder = dict(enumerate(data["crop"].cat.categories))  # type: ignore
        # Index-encode the categorical variables (team/crop)
        # dtype needs to be int64 for the labels to be compatible with CrossEntropyLoss
        data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes.astype("int64"))  # type: ignore

        # Divide up the dataframe into it's constituent arrays because indexing with pandas is
        # many times slower than indexing with numpy/torch
        self.image_fps = data["image"].values
        self.mask_fps = data["mask"].values if self.train else None
        self.team_data = torch.as_tensor(data["team"].values)  # pylint: disable=no-member
        self.crop_data = torch.as_tensor(data["crop"].values)  # pylint: disable=no-member

        # THe transformation applied to the mask
        self._target_transform = IndexEncodeMask()

    def _check_downloaded(self) -> bool:
        return self._dataset_folder.is_dir()

    def _generate_patches(self, image_fp: Path, team: str, crop: str) -> Dict[str, List[str]]:
        """Generate image patches for a single training image and it's associated mask."""
        mask_fp = (image_fp.parent.parent / "Masks" / image_fp.stem).with_suffix(".png")
        image_patch_dir = self._patch_dir / "Images" / team / crop / image_fp.stem
        image_patch_dir.mkdir(parents=True, exist_ok=True)
        mask_patch_dir = self._patch_dir / "Masks" / team / crop / image_fp.stem
        mask_patch_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_fp)
        mask = Image.open(mask_fp)
        filepaths: Dict[str, List[str]] = defaultdict(list)
        # Divide the images into patches and save them
        for patch_num, (image_patch, mask_patch) in enumerate(
            _patches_from_image_mask_pair(
                image=image,
                mask=mask,
                kernel_size=self.patch_size,
                stride=self.patch_stride,
            )
        ):
            image_patch_path = image_patch_dir / f"{image_fp.stem}_{patch_num}.png"
            with image_patch as file:
                file.save(image_patch_path)
            filepaths["image"].append(str(image_patch_path.relative_to(self._dataset_folder)))

            mask_patch_path = mask_patch_dir / f"{image_fp.stem}_{patch_num}.png"
            with mask_patch as file:
                file.save(mask_patch_path)
            filepaths["mask"].append(str(mask_patch_path.relative_to(self._dataset_folder)))

        return filepaths

    def _download_data(self) -> None:
        """Attempt to download data if files cannot be found in the base folder."""
        import zipfile

        # Check whether the data has already been downloaded - if it has and the integrity
        # of the files can be confirmed, then we are done
        if self._check_downloaded():
            print("Files already downloaded and verified")
            return
        # Create the directory and any required ancestors if not already existent
        if not self._base_folder.exists():
            self._base_folder.mkdir(parents=True)
        # Download the data from codalab
        _download_from_url(url=self.url, dst=str(self._base_folder / self.zipfile_name))
        # The downloaded data is in the form of a zipfile - extract it into its component directories
        with zipfile.ZipFile(self._base_folder / self.zipfile_name, "r") as fhandle:
            fhandle.extractall(str(self._base_folder))

    def process_files(self) -> None:
        """Store the filepaths of the images in a csv file, along with their mask, and labels."""
        # Compile the filepaths of the images, their associated massk and team/crop-type into a
        # .csv file which can be accessed by the dataset.
        extensions = ("*.png", "*.jpg", "*.jpeg")
        if self._patch_dir.exists():
            # Ask for confirmation before procedding to delete the existing image-patches
            while True:
                answer = input(
                    f"Image patches already exist at location '{self._patch_dir.resolve()}'. "
                    "Are you sure you want to overrite them (y/n)?"
                )
                if answer.lower().startswith("y"):
                    shutil.rmtree(self._patch_dir)
                    break
                elif answer.lower().startswith("n"):
                    exit()
        # The test data only needs to be processed once since we don't generate patches for it
        folders_to_check = [self.train_folder_name]
        if not (Path(self.test_folder_name) / "data.csv").exists():
            folders_to_check.append(self.test_folder_name)
        for split_folder_name in folders_to_check:
            data_dict: Dict[str, List[str]] = defaultdict(list)
            split_folder = self._dataset_folder / split_folder_name
            for team in get_args(Team):
                team = cast(str, team)
                for crop in get_args(Crop):
                    crop = cast(str, crop)
                    image_folder = split_folder / team / crop / "Images"
                    image_fps: List[Path] = []
                    # Images are not in a consistent format so multiple extensions need to be checked
                    for extension in extensions:
                        image_fps.extend(image_folder.glob(f"**/{extension}"))
                    pbar = tqdm(total=len(image_fps), desc=f"{split_folder_name}/{team}/{crop}")
                    for image_fp in tqdm(image_fps):
                        # Only the training data has masks available (these are our targets)
                        if split_folder_name == self.train_folder_name:
                            # We need to crop the training data for two reasons:
                            # 1) To enable batching of data from different teams, captured at
                            # different resolutions
                            # 2) To make training with batches more computationally tractable
                            patch_fps = self._generate_patches(
                                image_fp=image_fp, team=team, crop=crop
                            )
                            data_dict["image"].extend(patch_fps["image"])
                            data_dict["mask"].extend(patch_fps["mask"])
                            # Repeat the team/crop labels for each patch
                            num_patches = len(patch_fps["image"])
                            data_dict["team"].extend([team] * num_patches)
                            data_dict["crop"].extend([crop] * num_patches)
                        else:
                            data_dict["image"].append(
                                str(image_fp.relative_to(self._dataset_folder))
                            )
                            data_dict["team"].append(team)
                            data_dict["crop"].append(crop)
                        pbar.update()
                    pbar.close()
            data_df = pd.DataFrame(data_dict)  # type: ignore
            # Save the dataframe to the split-specific folder
            data_df.to_csv(split_folder / "data.csv")

    @implements(_SizedDataset)
    def __len__(self) -> int:
        return len(self.image_fps)

    @implements(_SizedDataset)
    def __getitem__(self, index: int) -> Union[TrainBatch, TestBatch]:
        image_fp = self.image_fps[index]
        image = Image.open(self._dataset_folder / image_fp)
        team = self.team_data[index]
        crop = self.crop_data[index]
        if self.mask_fps is not None:
            mask_t = Image.open(self._dataset_folder / self.mask_fps[index])
            mask = self._target_transform(mask_t)
            return TrainBatch(image=image, mask=mask, team=team, crop=crop)
        filename = Path(image_fp).stem
        return TestBatch(
            image=image,
            team=self.team_decoder[int(team)],
            crop=self.crop_decoder[int(crop)],
            filename=filename,
        )


def _prop_random_split(dataset: _SizedDataset, props: Sequence[float]) -> List[Subset]:
    """Splits a dataset based on proportions rather than on absolute sizes."""
    len_ = len(dataset)
    sum_ = np.sum(props)  # type: ignore
    if (sum_ > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if sum_ < 1:
        section_sizes.append(len_ - sum(section_sizes))
    return random_split(dataset, section_sizes)


class AcreCascadeDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module for the Acre Cascade dataset."""

    train_data: _SizedDataset
    val_data: _SizedDataset
    dims: InputShape
    num_classes: ClassVar[int] = 3  # background, crops, weeds

    def __init__(
        self,
        data_dir: Union[str, Path],
        train_batch_size: int,
        teams: Optional[Union[Team, List[Team]]] = None,
        test_teams: Optional[Union[Team, List[Team]]] = None,
        crop: Optional[Crop] = None,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        train_transforms: Transform = ToTensor(),
        test_transforms: Transform = ToTensor(),
        val_pcnt: float = 0.2,
        download: bool = True,
    ):
        super().__init__(
            train_transforms=train_transforms,
            test_transforms=test_transforms,
        )
        self.data_dir = data_dir
        self.download = download
        self.teams = teams
        self.test_teams = test_teams
        self.crop = crop

        if train_batch_size < 1:
            raise ValueError("train_batch_size must be a postivie integer.")
        self.train_batch_size = train_batch_size

        if val_batch_size is None:
            self.val_batch_size = train_batch_size
        else:
            if val_batch_size < 1:
                raise ValueError("val_batch_size must be a postivie integer.")
            self.val_batch_size = val_batch_size

        # num_workers == 0 means data-loading is done in the main process
        if num_workers < 0:
            raise ValueError("num_workers must be a non-negative number.")
        self.num_workers = num_workers

        if not (0.0 <= val_pcnt < 1.0):
            raise ValueError("val_pcnt must in the range [0, 1).")
        self.val_pcnt = val_pcnt

    @implements(pl.LightningDataModule)
    def prepare_data(self) -> None:
        """Download the ACRE Cascade Dataset if not already present in the root directory."""
        AcreCascadeDataset(data_dir=self.data_dir, download=self.download)

    @implements(pl.LightningDataModule)
    def setup(self, stage: Optional[Stage] = None) -> None:
        """Set up the data-module by instantiating the splits relevant to the given stage."""
        # Assign Train/Val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:  # fitting entails bothing training and validation
            labeled_data = AcreCascadeDataset(
                self.data_dir, train=True, download=False, teams=self.teams, crop=self.crop
            )
            val_data, train_data = _prop_random_split(labeled_data, props=(self.val_pcnt,))
            # Wrap the datasets in the DataTransformer class to allow for separate transformations
            # to be applied to the training and validation sets (this would not be possible if the
            # the transformations were a property of the dataset itself as random_split just creates
            # an index mask).
            self.train_data = _DataTransformer(
                train_data, train=True, transforms=self.train_transforms
            )
            self.val_data = _DataTransformer(val_data, train=True, transforms=self.test_transforms)
            self.dims = InputShape(*self.train_data[0].image.shape)

        # # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            test_data = AcreCascadeDataset(
                self.data_dir, train=False, download=False, teams=self.test_teams, crop=self.crop
            )
            self.test_data = _DataTransformer(
                test_data, train=False, transforms=self.test_transforms
            )
            self.dims = getattr(self, "dims", self.test_data[0].image.shape)

    @implements(pl.LightningDataModule)
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=1,  # The batch size needs to be 1 as the images are not consistent in size
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )
