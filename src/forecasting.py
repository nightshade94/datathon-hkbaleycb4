"""Time-series forecasting utilities."""

from pathlib import Path


def train_time_series_model(processed_dir: Path) -> None:
    """Train a time-series forecasting model and persist artefacts.

    Loads the time-series data from *processed_dir*, engineers temporal
    features, fits one or more forecasting models (e.g. LightGBM with
    lag features, XGBoost), evaluates performance on a hold-out window,
    and saves model artefacts and evaluation metrics alongside the
    processed data.

    Parameters
    ----------
    processed_dir:
        Directory containing the cleaned datasets produced by
        :func:`src.data_prep.clean_all_data`.

    Returns
    -------
    None
    """
    pass
