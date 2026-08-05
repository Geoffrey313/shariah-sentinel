"""Seasonal review-date gap detector (z9).

Detects firms that systematically report lower Sharia ratios at
fiscal year-end (when the SAC reviews) than during interim quarters.

For each firm, the raw statistic is:

    T9 = (median(ratio_interim) - median(ratio_yearend)) / MAD(ratio_all)

A large positive T9 means the ratio is consistently lower at
year-end — consistent with review-date management.

The statistic is computed once per firm (not per firm-quarter)
and broadcast to all quarters of that firm. This is a firm-level
detector, analogous to z4 (proximity) which also produces one
score per firm.

Empirical PIT on C maps T9 to N(0,1).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.common.methodology import SAC_MY, screening_thresholds_for_panel

from src.engine.pit import pit_empirical

log = logging.getLogger(__name__)

MIN_YE_QUARTERS: int = 4
MIN_INTERIM_QUARTERS: int = 4
MAD_CONSISTENCY: float = 1.4826
MAD_FLOOR: float = 1e-10
MIN_REF_FIRMS: int = 30

# Legacy alias — resolved per panel at detection time (same as d4).
SAC_THRESHOLDS: dict[str, float] = SAC_MY.ratio_thresholds()


def _t9_one_ratio_vectorized(vals: pd.Series, fqtr: pd.Series, firm_id: pd.Series) -> pd.Series:
    """Row-broadcast T9 for one ratio column, every firm at once.

    ``.groupby(firm_id).transform(...)`` computes each firm's median/count
    once and broadcasts it back to every row of that firm — the vectorized
    equivalent of ``_seasonal_gap_per_firm``'s per-firm median/MAD, without a
    Python-level loop over firms.
    """
    ye_vals = vals.where(fqtr == 4)
    interim_vals = vals.where(fqtr != 4)
    grouped_ye = ye_vals.groupby(firm_id, sort=False)
    grouped_interim = interim_vals.groupby(firm_id, sort=False)
    grouped_all = vals.groupby(firm_id, sort=False)

    n_ye = grouped_ye.transform("count")
    n_interim = grouped_interim.transform("count")
    med_ye = grouped_ye.transform("median")
    med_interim = grouped_interim.transform("median")
    med_all = grouped_all.transform("median")
    mad_all = MAD_CONSISTENCY * (vals - med_all).abs().groupby(firm_id, sort=False).transform("median")

    t9 = (med_interim - med_ye) / mad_all
    valid = (n_ye >= MIN_YE_QUARTERS) & (n_interim >= MIN_INTERIM_QUARTERS) & (mad_all >= MAD_FLOOR)
    return t9.where(valid)


def _seasonal_gap_vectorized(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    """Vectorized equivalent of looping ``_seasonal_gap_per_firm`` per firm.

    Args:
        df: Panel containing ``gvkey``, ``fqtr``, and the canonical Sharia
            ratio columns (same input contract as ``detect_seasonal_gap``).
        thresholds: Per-country ratio-threshold map (from
            ``screening_thresholds_for_panel``); only its keys are used here, to
            select which ratio columns participate.

    Returns:
        T9 broadcast to every row of its firm, aligned to ``df.index`` --
        matches ``detect_seasonal_gap``'s ``firm_raw`` dict once mapped back
        via ``gvkey``. NaN where a firm has too little year-end/interim data.

    Notes:
        Same math as ``_seasonal_gap_per_firm``, computed for every firm at
        once via ``groupby(...).transform(...)`` instead of a Python loop
        (see profiling notes).
    """
    firm_id = df["gvkey"]
    available_ratios = [c for c in thresholds if c in df.columns]
    per_ratio = [_t9_one_ratio_vectorized(df[col], df["fqtr"], firm_id) for col in available_ratios]
    if not per_ratio:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(per_ratio, axis=1).max(axis=1, skipna=True)


def _seasonal_gap_per_firm(
    ratios_firm: pd.DataFrame,
    fqtr_firm: pd.Series,
    thresholds: dict[str, float],
) -> float:
    """Compute T9 for one firm.

    Selects the ratio closest to its authority threshold (same logic
    as z4), then compares year-end vs interim medians.

    Returns NaN if insufficient data.
    """
    # Find the ratio closest to its threshold (max proximity signal)
    best_t9 = np.nan
    for ratio_col, threshold in thresholds.items():
        if ratio_col not in ratios_firm.columns:
            continue

        vals = ratios_firm[ratio_col].values
        fqtr = fqtr_firm.values

        # Year-end = fqtr == 4, interim = fqtr != 4
        ye_mask = (fqtr == 4) & np.isfinite(vals)
        interim_mask = (fqtr != 4) & np.isfinite(vals)

        n_ye = ye_mask.sum()
        n_interim = interim_mask.sum()

        if n_ye < MIN_YE_QUARTERS or n_interim < MIN_INTERIM_QUARTERS:
            continue

        ye_vals = vals[ye_mask]
        interim_vals = vals[interim_mask]
        all_vals = vals[np.isfinite(vals)]

        med_ye = np.median(ye_vals)
        med_interim = np.median(interim_vals)
        mad_all = MAD_CONSISTENCY * np.median(np.abs(all_vals - np.median(all_vals)))

        if mad_all < MAD_FLOOR:
            continue

        # Positive T9 = ratio is lower at year-end (suspicious)
        t9 = (med_interim - med_ye) / mad_all

        # Take the max across ratios (worst-offending ratio)
        if not np.isfinite(best_t9) or t9 > best_t9:
            best_t9 = t9

    return float(best_t9)


def detect_seasonal_gap(df: pd.DataFrame) -> pd.Series:
    """Compute the z9 seasonal review-date gap detector.

    Args:
        df: Input dataframe containing ``gvkey``, ``fqtr``, and
            the canonical Sharia ratio columns. If ``_split`` is
            present, the empirical PIT reference is restricted to
            C-firms.

    Returns:
        A series aligned on ``df.index``. Values are NaN when the
        firm has insufficient year-end/interim data or when the C
        reference is too small.
    """
    for col in ["gvkey", "fqtr"]:
        if col not in df.columns:
            log.warning("detect_seasonal_gap: missing %s; returning NaN.", col)
            return pd.Series(np.nan, index=df.index)

    thresholds = screening_thresholds_for_panel(df)
    available_ratios = [c for c in thresholds if c in df.columns]
    if not available_ratios:
        log.warning("detect_seasonal_gap: no Sharia ratios available; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    t9_broadcast = _seasonal_gap_vectorized(df, thresholds)
    firm_raw: dict[str, float] = t9_broadcast.groupby(df["gvkey"].astype(str), sort=False).first().to_dict()

    finite_firms = {k: v for k, v in firm_raw.items() if np.isfinite(v)}
    if not finite_firms:
        log.warning("detect_seasonal_gap: no firm has enough data; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    # Empirical PIT on C
    if "_split" in df.columns:
        c_firms = set(df.loc[df["_split"] == "C", "gvkey"].astype(str).unique())
    else:
        c_firms = set(finite_firms.keys())

    ref_values = np.array(
        [v for k, v in finite_firms.items() if k in c_firms],
        dtype=float,
    )
    if ref_values.size < MIN_REF_FIRMS:
        log.warning(
            "detect_seasonal_gap: only %d C reference firms (< %d); returning NaN.",
            ref_values.size, MIN_REF_FIRMS,
        )
        return pd.Series(np.nan, index=df.index)

    firm_ids = list(finite_firms.keys())
    firm_raw_array = np.array([finite_firms[k] for k in firm_ids], dtype=float)
    firm_z = pit_empirical(firm_raw_array, ref_values)
    firm_z_series = pd.Series(firm_z, index=firm_ids)

    result = df["gvkey"].astype(str).map(firm_z_series)
    n_scored = result.notna().sum()
    log.info(
        "detect_seasonal_gap: computed z9, coverage=%d/%d (%.1f%%), "
        "ref_size=%d.",
        n_scored, len(df), n_scored / len(df) * 100,
        ref_values.size,
    )
    return result
