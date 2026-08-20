"""Headless voice dataset preparation and training orchestration."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("voice-dataset-pipeline")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
