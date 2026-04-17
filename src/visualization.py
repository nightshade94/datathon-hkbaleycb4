"""Visualization utilities for EDA and model diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style() -> None:
    """Configure global Matplotlib / Seaborn aesthetics.

    Call this function once at the top of any script or notebook before
    generating plots.  Sets a consistent theme, font sizes, and figure
    DPI for all subsequent visualizations.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def generate_eda_plots(processed_dir: Path, output_dir: Path) -> None:
    """Create and save exploratory data analysis charts.

    Reads the processed datasets from *processed_dir*, produces a
    standard suite of EDA plots (distributions, correlations, missing-
    value heatmaps, etc.), and writes each figure as a PNG file to
    *output_dir*.

    Parameters
    ----------
    processed_dir:
        Directory containing the cleaned datasets produced by
        :func:`src.data_prep.clean_all_data`.
    output_dir:
        Directory where generated plot images will be saved.

    Returns
    -------
    None
    """
    pass
