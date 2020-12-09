"""Script containing data-loading functionality."""

from abc import abstractmethod
from collections import defaultdict
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    cast,
)
from urllib.request import urlopen

from PIL import Image
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
from torch.tensor import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, random_split
from torchvision.transforms import ToTensor
from tqdm import tqdm

from utils import implements


__all__ = ["AcreCascadeDataset", "AcreCascadeDataModule"]


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


Transform = Callable[[Union[Image.Image, Tensor]], Tensor]


class _SizedDatasetProt(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        ...


class _SizedDataset(Dataset):
    @abstractmethod
    def __len__(self) -> int:
        ...


class _DataTransformer(_SizedDataset):
    def __init__(self, base_dataset: _SizedDatasetProt, transforms: Transform):
        self.base_dataset = base_dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        data = self.base_dataset[index]
        if self.transforms is not None:
            data = (self.transforms(data[0]),) + data[1:]
        return data


class AcreCascadeDataset(_SizedDataset):
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
        download=True,
        train=True,
    ) -> None:
        super().__init__()

        self.root = Path(data_dir)
        self.download = download
        self._base_folder = self.root / self.base_folder_name
        self._dataset_folder = self._base_folder / self.dataset_folder_name

        if download:
            self._download()
        elif not self._check_downloaded():
            raise RuntimeError(
                f"Images don't exist at location {self._base_folder}. Have you downloaded them?"
            )

        self.train = train
        dtypes = {"images": "string", "group": "category", "crop": "category"}
        if self.train:
            split_folder = self._dataset_folder / self.train_folder_name
            dtypes["mask"] = "string"
        else:
            split_folder = self._dataset_folder / self.test_folder_name
        self.data = cast(
            pd.DataFrame, pd.read_csv(split_folder / "data.csv", dtype=dtypes, index_col=0)
        )
        # Index-encode the categorical variables (group/crop)
        cat_cols = self.data.select_dtypes(["category"]).columns
        # dtype needs to be int64 for the labels to be compatible with CrossEntropyLoss
        self.data[cat_cols] = self.data[cat_cols].apply(lambda x: x.cat.codes.astype("int64"))
        self._target_transform = ToTensor()

    def _check_downloaded(self) -> bool:
        return self._dataset_folder.is_dir()

    def _download(self) -> None:
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

        # Compile the filepaths of the images, their associated massk and group/crop-type into a .csv file which can be accessed by the dataset.
        for split_folder_name in [self.train_folder_name, self.test_folder_name]:
            data_dict: Dict[str, List[str]] = defaultdict(list)
            split_folder = self._dataset_folder / split_folder_name
            for group in split_folder.iterdir():
                for crop in group.iterdir():
                    for img in crop.glob("**/*.png"):
                        data_dict["image"].append(str(img.relative_to(self._dataset_folder)))
                        # ONly the training data has masks available (these are our targets)
                        if split_folder_name == self.train_folder_name:
                            data_dict["mask"].append(
                                str((crop / "Masks" / img.name).relative_to(self._dataset_folder))
                            )
                        data_dict["group"].append(group.stem)
                        data_dict["crop"].append(crop.stem)
            data_df = pd.DataFrame(data_dict)
            # Save the dataframe to the split-specific folder
            data_df.to_csv(split_folder / "data.csv")

    @implements(_SizedDataset)
    def __len__(self):
        return len(self.data)

    @implements(_SizedDataset)
    def __getitem__(self, index: int) -> Tuple[Image.Image, Optional[Tensor], int, int]:
        entry = cast(pd.DataFrame, self.data.iloc[index])
        img = Image.open(self._dataset_folder / entry["image"])  # type: ignore
        if self.train:
            mask_t = Image.open(self._dataset_folder / entry["mask"])  # type: ignore
            mask = self._target_transform(mask_t)
        else:
            mask = None
        return img, mask, entry["group"], entry["crop"]  # type: ignore


def _prop_random_split(dataset: _SizedDataset, props: Sequence[float]) -> List[Subset]:
    """Splits a dataset based on a proportions rather than on absolute sizes."""
    len_ = len(dataset)
    if (sum_ := (np.sum(props)) > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if sum_ < 1:
        section_sizes.append(len_ - sum(section_sizes))
    return random_split(dataset, section_sizes)


STAGE = Literal["fit", "test"]


class AcreCascadeDataModule(pl.LightningDataModule):
    train_data: _SizedDataset
    val_data: _SizedDataset
    dims: Tuple[int, int, int]

    def __init__(
        self,
        data_dir: Union[str, Path],
        train_batch_size: int = 1,
        test_batch_size: Optional[int] = None,
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

        if train_batch_size < 1:
            raise ValueError("train_batch_size must be a postivie integer.")
        self.train_batch_size = train_batch_size

        # Set the test batch-size to be the same as the train batch-size if unspecified
        if test_batch_size is None:
            self.test_batch_size = train_batch_size
        else:
            if test_batch_size < 1:
                raise ValueError("test_batch_size must be a postivie integer.")
            self.test_batch_size = test_batch_size

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
        AcreCascadeDataset(data_dir=self.data_dir, download=True)

    @implements(pl.LightningDataModule)
    def setup(self, stage: Optional[STAGE] = None) -> None:
        """Set up the data-module by instantiating the splits relevant to the given stage."""
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:  # fitting entails bothing training and validation
            labeled_data = AcreCascadeDataset(self.data_dir, train=True, download=False)
            val_data, train_data = _prop_random_split(labeled_data, props=(self.val_pcnt,))
            # Wrap the datasets in the DataTransformer class to allow for separate transformations
            # to be applied to the training and validation sets (this would not be possible if the
            # the transformations were a property of the dataset itself as random_split just creates
            # an index mask).
            self.train_data = _DataTransformer(train_data, transforms=self.train_transforms)
            self.val_data = _DataTransformer(val_data, transforms=self.test_transforms)
            self.dims = self.train_data[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            test_data = AcreCascadeDataset(self.data_dir, train=False, download=False)
            self.test_data = _DataTransformer(test_data, transforms=self.test_transforms)
            self.dims = getattr(self, "dims", self.test_data[0][0].shape)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )
