"""Generic helpers for detector and simulation ratio inputs.

This module centralizes access to the canonical adjusted Sharia ratios already
computed by the panel pipeline. Downstream code should prefer these columns
over local recomputations to avoid silent definition drift.
"""
from __future__ import annotations

import pandas as pd

CANONICAL_RATIO_COLUMNS: tuple[str, ...] = (
    "ratio_income",
    "ratio_debt_adj",
    "ratio_cash_adj",
)


def get_canonical_sharia_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical adjusted Sharia ratios available in ``df``.

    Args:
        df: Input dataframe produced by the panel pipeline.

    Returns:
        A numeric dataframe restricted to canonical ratio columns present in
        ``df``. If no canonical ratio is available, an empty dataframe indexed
        like ``df`` is returned.
    """
    ratio_columns = [column for column in CANONICAL_RATIO_COLUMNS if column in df.columns]
    if not ratio_columns:
        return pd.DataFrame(index=df.index)
    return df[ratio_columns].apply(pd.to_numeric, errors="coerce").copy()
