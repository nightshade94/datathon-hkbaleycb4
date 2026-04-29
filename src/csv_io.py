"""CSV input/output helpers for project datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def list_csv_files(directory: Path) -> list[Path]:
    """Return all CSV files in *directory* sorted by filename.

    Parameters
    ----------
    directory:
        Folder containing CSV files.

    Returns
    -------
    list[Path]
        Sorted CSV file paths.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    return sorted(directory.glob("*.csv"))


def read_csv_file(file_path: Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load one CSV file into a DataFrame.

    Parameters
    ----------
    file_path:
        CSV file path.
    **read_kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {file_path.name}")

    if "low_memory" not in read_kwargs:
        read_kwargs["low_memory"] = False

    return pd.read_csv(file_path, **read_kwargs)


def read_csv_directory(directory: Path, **read_kwargs: Any) -> dict[str, pd.DataFrame]:
    """Load every CSV file in a directory.

    Parameters
    ----------
    directory:
        Folder containing CSV files.
    **read_kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping ``{file_stem: dataframe}``.
    """
    csv_files = list_csv_files(directory)
    return {csv_file.stem: read_csv_file(csv_file, **read_kwargs) for csv_file in csv_files}


def write_csv_file(
    dataframe: pd.DataFrame,
    file_path: Path,
    *,
    index: bool = False,
    **write_kwargs: Any,
) -> Path:
    """Write one DataFrame to CSV.

    Parameters
    ----------
    dataframe:
        DataFrame to persist.
    file_path:
        Destination CSV path.
    index:
        Whether to write DataFrame index.
    **write_kwargs:
        Extra keyword arguments forwarded to :meth:`pandas.DataFrame.to_csv`.

    Returns
    -------
    Path
        Destination file path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(file_path, index=index, **write_kwargs)
    return file_path


def write_csv_directory(
    dataframes: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    index: bool = False,
    **write_kwargs: Any,
) -> list[Path]:
    """Write multiple DataFrames to CSV files in one folder.

    Parameters
    ----------
    dataframes:
        Mapping ``{file_stem: dataframe}``.
    output_dir:
        Destination folder for generated CSV files.
    index:
        Whether to write DataFrame index.
    **write_kwargs:
        Extra keyword arguments forwarded to :meth:`pandas.DataFrame.to_csv`.

    Returns
    -------
    list[Path]
        Paths to written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for file_stem, dataframe in dataframes.items():
        destination = output_dir / f"{file_stem}.csv"
        write_csv_file(dataframe, destination, index=index, **write_kwargs)
        written_files.append(destination)

    return written_files
