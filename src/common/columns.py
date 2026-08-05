"""Small helpers for resolving optional column lists."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def resolve_present_columns(
    df: pd.DataFrame,
    value_columns: Sequence[str] | None,
    default_columns: Sequence[str],
) -> list[str]:
    """Return the candidate columns that are present in ``df``.

    Args:
        df: Input dataframe.
        value_columns: Optional explicit column list supplied by the caller.
        default_columns: Fallback column list used when ``value_columns`` is
            omitted.

    Returns:
        The ordered subset of candidate columns that exists in ``df``.
    """
    candidates = tuple(value_columns) if value_columns is not None else tuple(default_columns)
    return [column for column in candidates if column in df.columns]
