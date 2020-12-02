import importlib
import os
from pathlib import Path
from typing import Tuple, Optional, Final
from urllib.request import urlopen

from tqdm import tqdm

_BASE_FOLDER: Final = "crops"


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None

_FILE_LIST: Final = [
    (
        "1zmsC4yvw-e089uHXj5EdP0BSZ0AlDQRR",  # File ID
        "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
        "img_align_celeba.zip",  # Filename
    ),
    (
        "1gxmFoeEPgF9sT65Wpo85AnHl3zsQ4NvS",
        "75e246fa4810816ffd6ee81facbd244c",
        "list_attr_celeba.txt",
    ),
    (
        "1ih_VMokoI774ErNWrb26lDeWlanUBpnX",
        "d32c9cbf5e040fd4025c592c306e6668",
        "list_eval_partition.txt",
    ),
]

def crops(
    download_dir: str,
    download: bool = False,
    check_integrity: bool = True,
) -> Path:
    """Get Crops dataset."""
    root = Path(download_dir)

    base = root / _BASE_FOLDER
    img_dir = base / "Development_Dataset"
    if download:
        _download(base)
    elif check_integrity and not _check_downloaded(base):
            raise RuntimeError("Images don't exist at location. Have you downloaded them?")
    return img_dir


def _check_downloaded(base: Path) -> bool:
    print("checking integrity")
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError("Need torchvision to download data.")

    return (base / "Development_Dataset").is_dir()


import requests

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urlopen(url).info().get('Content-Length', -1))
    first_byte = os.path.getsize(dst) if os.path.exists(dst) else 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

def _download(base: Path) -> None:
    """Attempt to download data if files cannot be found in the base folder."""
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError("Need torchvision to download data.")
    import zipfile

    from torchvision.datasets.utils import download_file_from_google_drive

    if _check_downloaded(base):
        print("Files already downloaded and verified")
        return

    zipfile_name = "images.zip"

    download_from_url("https://competitions.codalab.org/my/datasets/download/29a85805-2d8d-4701-a9ab-295180c89eb3", base / zipfile_name)

    with zipfile.ZipFile(base / zipfile_name, "r") as fhandle:
        fhandle.extractall(str(base))


if __name__ == "__main__":
    dirpath = crops(".", True, True)
    print(dirpath)