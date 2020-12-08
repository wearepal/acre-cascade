"""Welcome to ACRE Cascade starter."""

from setuptools import find_packages, setup

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = "0"
_MINOR_VERSION = "0"
_PATCH_VERSION = "1"

_VERSION_SUFFIX = "alpha"

# Example, '0.4.0-rc1'
version = ".".join([_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION])
if _VERSION_SUFFIX:
    version = f"{version}-{_VERSION_SUFFIX}"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ACRE Cascade Starter",
    version=version,
    author="Predictive Analytics Lab - University of Sussex",
    description="A starter kit for the Crop Segmentation Competition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://competitions.codalab.org/competitions/27176",
    packages=find_packages(exclude=["tests.*", "tests"]),
    package_data={},
    python_requires=">=3.7",
    install_requires=[
        "pillow",
        "pytorch-lightning",
        "scikit_learn",
        "seaborn",
        "tqdm",
        "requests",
        "typing-extensions",
    ],
)
