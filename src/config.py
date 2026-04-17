"""Project-wide path configuration using pathlib."""

from pathlib import Path

# Root of the repository (one level above this file's parent directory)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
RAW_DIR: Path = ROOT_DIR / "data" / "raw"
PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"

# Image output directories
IMAGES_DIR: Path = ROOT_DIR / "images"
EDA_IMAGES_DIR: Path = IMAGES_DIR / "eda"
MODEL_IMAGES_DIR: Path = IMAGES_DIR / "model"
