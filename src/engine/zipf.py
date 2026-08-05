"""Zipf detector.

This module implements a cautious version of the Working Paper's ``j=2``
detector. The core idea is to fit a log-log rank-size relationship on the
positive monetary figures available for each firm and map the residual variance
to the common anomaly-oriented z-scale.

The detector is intentionally conditional: if the rolling firm-quarter window
has too few positive figures, or if the calibration sample is too small, it
returns ``NaN`` instead of inventing a score. This keeps the implementation
honest for quarterly panel data while leaving the door open for richer
transaction-level inputs later.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.engine.pit import pit_chi2
from src.common.columns import resolve_present_columns

log = logging.getLogger(__name__)

DEFAULT_ZIPF_VALUE_COLUMNS: tuple[str, ...] = (
    "revtq",
    "iditq",
    "nopiq",
    "xintq",
    "xintdy",
    "niq",
    "oibdpq",
    "atq",
    "dlttq",
    "dlcq",
    "ltq",
    "cheq",
    "chq",
    "chsq",
    "actq",
    "lctq",
    "rectq",
    "invtq",
    "rectrq",
    "dd1q",
    "dlsq",
    "notesq",
)
MIN_POINTS: int = 8
MIN_REFERENCE_FIRMS: int = 10
POOLING_WINDOW_QUARTERS: int = 4

_ZIPF_RANK_CACHE: dict[int, tuple[np.ndarray, float, float]] = {}


def _collect_row_sizes_matrix(
    df: pd.DataFrame,
    value_columns: Sequence[str] | None,
) -> np.ndarray:
    """Collect positive monetary magnitudes into a dense NumPy matrix."""
    available = resolve_present_columns(df, value_columns, DEFAULT_ZIPF_VALUE_COLUMNS)
    if not available:
        return np.empty((len(df), 0), dtype=float)
    matrix = np.column_stack(
        [pd.to_numeric(df[column], errors="coerce").abs().to_numpy(dtype=float) for column in available]
    )
    matrix[~np.isfinite(matrix) | (matrix <= 0)] = np.nan
    return matrix


def _rolling_zipf_fits_fast(
    df: pd.DataFrame,
    row_sizes_matrix: np.ndarray,
    min_points: int,
    pooling_window_quarters: int,
) -> tuple[pd.Series, pd.Series]:
    """Fit one Zipf residual variance per firm-quarter rolling window using dense matrices."""
    sigma2_values = np.full(len(df), np.nan, dtype=float)
    df_resid_values = np.full(len(df), np.nan, dtype=float)
    if "gvkey" not in df.columns:
        return (
            pd.Series(sigma2_values, index=df.index, dtype=float),
            pd.Series(df_resid_values, index=df.index, dtype=float),
        )

    if "datacqtr" not in df.columns:
        for _, idx in df.groupby("gvkey", sort=False).groups.items():
            pos = df.index.get_indexer(pd.Index(idx))
            fit = _fit_zipf_residual_variance_fast(row_sizes_matrix[pos].reshape(-1), min_points=min_points)
            if fit is None:
                continue
            sigma_sq, n_points = fit
            sigma2_values[pos] = sigma_sq
            df_resid_values[pos] = n_points - 2
        return (
            pd.Series(sigma2_values, index=df.index, dtype=float),
            pd.Series(df_resid_values, index=df.index, dtype=float),
        )

    ordered = df.sort_values(["gvkey", "datacqtr"], kind="stable")
    ordered_pos = df.index.get_indexer(ordered.index)
    if len(ordered_pos) == 0:
        return (
            pd.Series(sigma2_values, index=df.index, dtype=float),
            pd.Series(df_resid_values, index=df.index, dtype=float),
        )

    sigma2_sorted, df_resid_sorted = _rolling_zipf_fits_vectorized(
        row_sizes_matrix[ordered_pos],
        pd.factorize(ordered["gvkey"], sort=False)[0],
        min_points=min_points,
        pooling_window_quarters=pooling_window_quarters,
    )
    sigma2_values[ordered_pos] = sigma2_sorted
    df_resid_values[ordered_pos] = df_resid_sorted
    return (
        pd.Series(sigma2_values, index=df.index, dtype=float),
        pd.Series(df_resid_values, index=df.index, dtype=float),
    )


def _rolling_zipf_fits_vectorized(
    matrix_sorted: np.ndarray,
    firm_codes_sorted: np.ndarray,
    min_points: int,
    pooling_window_quarters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of looping ``_fit_zipf_residual_variance_fast`` per row.

    ``pooling_window_quarters`` is a fixed count of *time* periods, so the
    same sliding-window index-arithmetic trick used for z6/z8 applies here
    to gather each row's up-to-``pooling_window_quarters`` preceding
    same-firm value blocks (``(n_rows, pooling_window_quarters, n_cols)``,
    no Python loop). The number of *valid* (finite, positive) pooled figures
    still varies per row though, so the OLS log-rank/log-size fit itself is
    batched by grouping rows that share the same valid-point count ``n``
    (their log-rank design is identical, mirroring ``_ZIPF_RANK_CACHE``) --
    this replaces a Python loop over every row with a Python loop over the
    handful of distinct ``n`` values that occur in the panel.

    Args:
        matrix_sorted: ``(n_rows, n_cols)`` positive-magnitude matrix,
            already sorted by ``(gvkey, datacqtr)``.
        firm_codes_sorted: Integer firm codes (e.g. from
            ``pd.factorize``), aligned with ``matrix_sorted``.
        min_points: Minimum pooled positive magnitudes required to fit.
        pooling_window_quarters: Rolling window length in quarters
            (includes the current row).

    Returns:
        ``(sigma2, df_resid)`` aligned with ``matrix_sorted``.
    """
    n_rows, n_cols = matrix_sorted.shape
    sigma2_sorted = np.full(n_rows, np.nan, dtype=float)
    df_resid_sorted = np.full(n_rows, np.nan, dtype=float)
    if n_rows == 0 or n_cols == 0:
        return sigma2_sorted, df_resid_sorted

    idx = np.arange(n_rows)
    offsets = np.arange(-(pooling_window_quarters - 1), 1)
    window_idx = idx[:, None] + offsets[None, :]
    in_bounds = window_idx >= 0
    window_idx_clipped = np.clip(window_idx, 0, n_rows - 1)
    same_firm = firm_codes_sorted[window_idx_clipped] == firm_codes_sorted[idx][:, None]
    valid_time = in_bounds & same_firm

    gathered = matrix_sorted[window_idx_clipped]
    gathered = np.where(valid_time[:, :, None], gathered, np.nan)
    flat = gathered.reshape(n_rows, -1)

    valid_point = np.isfinite(flat) & (flat > 0)
    counts = valid_point.sum(axis=1)

    # Sort descending with invalid entries pushed to the tail: negate so an
    # ascending sort puts the largest original values first, masking
    # invalid entries to +inf beforehand so they land after negation.
    masked_for_sort = np.where(valid_point, -flat, np.inf)
    sorted_neg = np.sort(masked_for_sort, axis=1)
    sorted_desc = np.where(np.isfinite(sorted_neg), -sorted_neg, np.nan)

    max_n = int(counts.max()) if n_rows else 0
    if max_n == 0:
        return sigma2_sorted, df_resid_sorted
    with np.errstate(invalid="ignore"):
        log_sizes_full = np.log(sorted_desc[:, :max_n])

    for n in np.unique(counts):
        if n < min_points:
            continue
        group_mask = counts == n
        log_ranks = np.log(np.arange(1, n + 1, dtype=float))
        x_mean = log_ranks.mean()
        x_centered = log_ranks - x_mean
        denom = float(np.dot(x_centered, x_centered))
        if denom <= 0.0:
            continue
        df_resid = n - 2
        if df_resid <= 0:
            continue

        log_sizes_grp = log_sizes_full[group_mask, :n]
        y_mean = log_sizes_grp.mean(axis=1)
        centered = log_sizes_grp - y_mean[:, None]
        beta1 = (centered @ x_centered) / denom
        beta0 = y_mean - beta1 * x_mean
        resid = log_sizes_grp - (beta0[:, None] + beta1[:, None] * log_ranks[None, :])
        sigma2_grp = (resid * resid).sum(axis=1) / df_resid

        sigma2_sorted[group_mask] = sigma2_grp
        df_resid_sorted[group_mask] = df_resid

    return sigma2_sorted, df_resid_sorted


def _fit_zipf_residual_variance_fast(values: np.ndarray, min_points: int) -> tuple[float, int] | None:
    """Fast closed-form log-rank/log-size residual variance."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = len(arr)
    if n < min_points:
        return None

    sizes = np.sort(arr)[::-1]
    log_sizes = np.log(sizes)
    log_ranks, x_mean, denom = _ZIPF_RANK_CACHE.get(n, (np.empty(0, dtype=float), 0.0, 0.0))
    if log_ranks.size == 0:
        log_ranks = np.log(np.arange(1, n + 1, dtype=float))
        x_mean = float(log_ranks.mean())
        x_centered = log_ranks - x_mean
        denom = float(np.dot(x_centered, x_centered))
        _ZIPF_RANK_CACHE[n] = (log_ranks, x_mean, denom)
    y_mean = float(log_sizes.mean())
    if denom <= 0.0:
        return None
    x_centered = log_ranks - x_mean
    beta1 = float(np.dot(x_centered, log_sizes - y_mean) / denom)
    beta0 = y_mean - beta1 * x_mean
    resid = log_sizes - (beta0 + beta1 * log_ranks)
    df_resid = n - 2
    if df_resid <= 0:
        return None
    sigma2 = float(np.sum(resid ** 2) / df_resid)
    return sigma2, n


def _estimate_null_scale(
    sigma2: pd.Series,
    mask_calibration: pd.Series,
    min_reference_firms: int,
) -> float | None:
    """Estimate the null residual variance for the Zipf detector."""
    sigma2_values = sigma2.loc[mask_calibration]
    sigma2_values = sigma2_values[np.isfinite(sigma2_values) & (sigma2_values > 0)]
    if sigma2_values.empty:
        return None
    if len(sigma2_values) < min_reference_firms:
        log.warning(
            "detect_zipf: weak calibration reference (%d firm(s) < %d); using a pooled fallback scale.",
            len(sigma2_values),
            min_reference_firms,
        )
    return float(np.median(sigma2_values))


def detect_zipf(
    df: pd.DataFrame,
    value_columns: Sequence[str] | None = None,
    row_sizes_matrix: np.ndarray | None = None,
    min_points: int = MIN_POINTS,
    min_reference_firms: int = MIN_REFERENCE_FIRMS,
    pooling_window_quarters: int = POOLING_WINDOW_QUARTERS,
) -> pd.Series:
    """Compute the ``z2`` Zipf detector.

    Args:
        df: Input dataframe containing ``gvkey`` and monetary figure columns.
        value_columns: Optional explicit list of figure columns. When omitted, a
            conservative default list of monetary columns from the panel schema
            is used.
        min_points: Minimum number of positive magnitudes required to fit one
            firm-level Zipf regression.
        min_reference_firms: Minimum number of calibration rows required to
            estimate the null scale from split ``C``.
        pooling_window_quarters: Rolling window used when ``datacqtr`` is
            available. When quarter labels are absent, the detector falls back
            to one pooled firm-level fit.

    Returns:
        A series aligned on ``df.index``. Firms with too few positive magnitudes
        or an unstable calibration reference remain ``NaN``.
    """
    if "gvkey" not in df.columns:
        log.warning("detect_zipf: missing `gvkey`; returning NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    if row_sizes_matrix is None:
        row_sizes_matrix = _collect_row_sizes_matrix(df, value_columns)
    if row_sizes_matrix.size == 0 or not np.isfinite(row_sizes_matrix).any():
        log.warning("detect_zipf: no monetary figure columns available; returning NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    sigma2, df_resid = _rolling_zipf_fits_fast(
        df=df,
        row_sizes_matrix=row_sizes_matrix,
        min_points=min_points,
        pooling_window_quarters=pooling_window_quarters,
    )
    mask_cal = df["_split"] == "C" if "_split" in df.columns else pd.Series(True, index=df.index)
    sigma0_sq = _estimate_null_scale(
        sigma2=sigma2,
        mask_calibration=mask_cal,
        min_reference_firms=min_reference_firms,
    )
    if sigma0_sq is None or not np.isfinite(sigma0_sq) or sigma0_sq <= 0:
        log.warning("detect_zipf: calibration scale is not estimable; returning NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    valid = np.isfinite(sigma2) & np.isfinite(df_resid) & (df_resid > 0)
    if not valid.any():
        log.warning("detect_zipf: no firm reached the minimum point count (%d).", min_points)
        return pd.Series(np.nan, index=df.index, dtype=float)

    z = pd.Series(np.nan, index=df.index, dtype=float)
    for dof in sorted(df_resid.loc[valid].astype(int).unique()):
        dof_mask = (df_resid.astype("Int64") == dof).to_numpy(dtype=bool, na_value=False)
        mask = valid & dof_mask
        chi2_stat = (df_resid.loc[mask] * sigma2.loc[mask]) / sigma0_sq
        z.loc[mask] = pit_chi2(chi2_stat.to_numpy(dtype=float), degrees_of_freedom=int(dof))

    skipped = int((~valid).sum())
    if skipped:
        log.debug("detect_zipf: skipped %d row(s) with insufficient positive magnitudes.", skipped)
    return z
