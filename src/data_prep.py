"""Data preparation utilities."""

from pathlib import Path


def clean_all_data(raw_dir: Path, processed_dir: Path) -> None:
    """Load, clean, and persist all raw datasets.

    Reads every raw data file from *raw_dir*, applies cleaning steps
    (e.g. type casting, missing-value imputation, deduplication), and
    writes the resulting DataFrames to *processed_dir* in a format
    suitable for downstream modelling.

    Parameters
    ----------
    raw_dir:
        Directory that contains the original, unmodified data files.
    processed_dir:
        Directory where cleaned datasets will be saved.

    Returns
    -------
    None
    """
    pass
