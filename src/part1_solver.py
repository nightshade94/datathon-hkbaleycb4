"""Solver for Part 1 multiple-choice questions."""

from pathlib import Path


def solve_mcq(processed_dir: Path) -> dict:
    """Derive answers for the multiple-choice section of the competition.

    Loads the relevant processed datasets from *processed_dir*, performs
    the required analysis or lookups, and returns a mapping of question
    identifiers to selected answer choices.

    Parameters
    ----------
    processed_dir:
        Directory containing the cleaned datasets produced by
        :func:`src.data_prep.clean_all_data`.

    Returns
    -------
    dict
        A dictionary mapping question IDs (``str``) to chosen answers
        (``str`` or ``int``), e.g. ``{"Q1": "B", "Q2": 3}``.
    """
    return {}
