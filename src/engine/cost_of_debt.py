"""Cost-of-debt break detector (z8).

Detects sudden spikes in the implied cost of debt:

    CoD = xintq / (dlttq + dlcq)

A firm that reduces reported debt without proportionally reducing
interest expense will show an elevated CoD relative to its own
recent history. The raw statistic is the MAD-standardised break:

    T8 = (CoD_t - median(CoD_{t-Q..t-1})) / MAD(CoD_{t-Q..t-1})

mapped to N(0,1) via empirical PIT on the reference sample C.

This detector is motivated by the contamination study's M4
mechanism, which reduces dlttq by a fixed fraction. No existing
detector (z1--z7) captures the resulting debt/interest
inconsistency because:
- z4 measures threshold proximity, not cost of debt
- z57 uses peer-relative features that dilute this specific ratio
- z3 uses Beneish accruals, not implied interest rates

CoD is empirically uncorrelated (r=0.000) with all six active
detectors, adding a genuine new dimension.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.engine.pit import pit_empirical

log = logging.getLogger(__name__)

Q_HIST: int = 8
MIN_HIST: int = 4
MAD_CONSISTENCY: float = 1.4826
MAD_FLOOR: float = 1e-10
MIN_REF_FIRMS: int = 30


def _implied_cost_of_debt(df: pd.DataFrame) -> pd.Series:
    """Compute CoD = xintq / (dlttq + dlcq)."""
    dlttq = pd.to_numeric(df["dlttq"], errors="coerce")
    dlcq = pd.to_numeric(df.get("dlcq", pd.Series(np.nan, index=df.index)), errors="coerce")
    debt = dlttq + dlcq
    xintq = df["xintq"] if "xintq" in df.columns else pd.Series(np.nan, index=df.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        cod = np.where(
            (debt > 0) & xintq.notna(),
            xintq / debt,
            np.nan,
        )
    return pd.Series(cod, index=df.index, dtype=float)


def _t8_vectorized(cod_sorted: np.ndarray, firm_codes_sorted: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of looping ``_t8_per_firm`` per firm.

    Builds an ``(n, Q_HIST)`` sliding-window matrix of each row's up to
    ``Q_HIST`` preceding same-firm CoD values via index arithmetic (no
    Python loop over firms or over quarters), then takes the
    nanmedian/MAD across the window axis. Requires ``cod_sorted`` and
    ``firm_codes_sorted`` to already be ordered by firm then quarter
    (see profiling notes).

    Args:
        cod_sorted: Implied cost-of-debt values, sorted by
            ``(gvkey, datacqtr)``.
        firm_codes_sorted: Integer firm codes (e.g. from
            ``pd.factorize``), aligned with ``cod_sorted``.

    Returns:
        T8 aligned with ``cod_sorted``. NaN where history is
        insufficient or MAD is zero.
    """
    n = len(cod_sorted)
    idx = np.arange(n)
    offsets = np.arange(-Q_HIST, 0)
    window_idx = idx[:, None] + offsets[None, :]
    in_bounds = window_idx >= 0
    window_idx_clipped = np.clip(window_idx, 0, n - 1)

    window_vals = cod_sorted[window_idx_clipped]
    same_firm = firm_codes_sorted[window_idx_clipped] == firm_codes_sorted[idx][:, None]
    valid = in_bounds & same_firm & np.isfinite(window_vals)
    window_vals = np.where(valid, window_vals, np.nan)

    count_finite = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        med = np.nanmedian(window_vals, axis=1)
        mad = MAD_CONSISTENCY * np.nanmedian(np.abs(window_vals - med[:, None]), axis=1)
        t8 = (cod_sorted - med) / mad

    ok = (count_finite >= MIN_HIST) & (mad >= MAD_FLOOR) & np.isfinite(cod_sorted)
    return np.where(ok, t8, np.nan)


def _t8_per_firm(cod_firm: np.ndarray) -> np.ndarray:
    """Compute raw T8 for one firm's CoD time series.

    Returns an array aligned with the input. NaN where history is
    insufficient or MAD is zero.
    """
    n = len(cod_firm)
    t8 = np.full(n, np.nan)

    for t in range(1, n):
        hist_start = max(0, t - Q_HIST)
        history = cod_firm[hist_start:t]
        finite = history[np.isfinite(history)]
        if len(finite) < MIN_HIST:
            continue

        current = cod_firm[t]
        if not np.isfinite(current):
            continue

        med = np.median(finite)
        mad = MAD_CONSISTENCY * np.median(np.abs(finite - med))
        if mad < MAD_FLOOR:
            continue

        t8[t] = (current - med) / mad

    return t8


def detect_cost_of_debt(df: pd.DataFrame) -> pd.Series:
    """Compute the z8 cost-of-debt break detector.

    Args:
        df: Input dataframe containing ``gvkey``, ``datacqtr``,
            ``dlttq``, ``dlcq``, and ``xintq``. If ``_split`` is
            present, the empirical PIT reference is restricted to
            C-rows.

    Returns:
        A series aligned on ``df.index``. Values are NaN when the
        firm has insufficient CoD history or when the C reference
        is too small.
    """
    for col in ["gvkey", "datacqtr"]:
        if col not in df.columns:
            log.warning("detect_cost_of_debt: missing %s; returning NaN.", col)
            return pd.Series(np.nan, index=df.index)

    if "dlttq" not in df.columns or "xintq" not in df.columns:
        log.warning("detect_cost_of_debt: missing dlttq or xintq; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    cod = _implied_cost_of_debt(df)

    # Sort by firm and quarter, compute T8 per firm
    df_sorted = df[["gvkey", "datacqtr"]].copy()
    df_sorted["_order"] = range(len(df_sorted))
    df_sorted["_cod"] = cod.values
    df_sorted = df_sorted.sort_values(["gvkey", "datacqtr"])

    firm_codes, _ = pd.factorize(df_sorted["gvkey"], sort=False)
    t8_sorted = _t8_vectorized(df_sorted["_cod"].to_numpy(dtype=float), firm_codes)
    t8_all = np.full(len(df), np.nan)
    t8_all[df_sorted["_order"].to_numpy()] = t8_sorted

    # Empirical PIT on C
    if "_split" in df.columns:
        c_mask = df["_split"].values == "C"
    else:
        c_mask = np.ones(len(df), dtype=bool)

    ref_values = t8_all[c_mask & np.isfinite(t8_all)]
    if ref_values.size < MIN_REF_FIRMS:
        log.warning(
            "detect_cost_of_debt: only %d C reference values (< %d); returning NaN.",
            ref_values.size, MIN_REF_FIRMS,
        )
        return pd.Series(np.nan, index=df.index)

    z8_all = np.full(len(df), np.nan)
    finite_mask = np.isfinite(t8_all)
    if finite_mask.any():
        z8_all[finite_mask] = pit_empirical(t8_all[finite_mask], ref_values)

    log.info(
        "detect_cost_of_debt: computed z8, coverage=%d/%d (%.1f%%), "
        "ref_size=%d.",
        finite_mask.sum(), len(df),
        finite_mask.mean() * 100,
        ref_values.size,
    )
    return pd.Series(z8_all, index=df.index)
