"""Temporal consistency detector.

This detector implements the Working Paper's ``j=6`` family from canonical
Sharia ratios:

- estimate quarter-over-quarter ratio changes for each firm
- center them on the firm's own recent history
- score the new change either with:
  - the active diagonal phase-1 statistic ``T6 = sum(Z_i^2)``
  - or an optional joint Mahalanobis statistic using the temporal covariance
- map ``T6`` to the anomaly-oriented z-scale with a chi-square PIT

The diagonal variant remains the active default because it is the most stable
for short histories. The joint variant is implemented and ready for future use
when longer firm histories justify a covariance-aware score.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.engine.pit import pit_chi2
from src.common.ratio_inputs import get_canonical_sharia_ratios

log = logging.getLogger(__name__)

Q_HIST: int = 8
MIN_HIST: int = 4
STD_EPS: float = 1e-10
RIDGE_EPS: float = 1e-6


def _compute_joint_t6(
    delta_t: np.ndarray,
    mu_delta: np.ndarray,
    deltas_hist: np.ndarray,
) -> tuple[float, int]:
    """Compute the optional covariance-aware temporal statistic.

    This is the future-ready joint alternative to the active diagonal phase-1
    score. It uses only dimensions that are finite at the scoring date and
    whose historical deltas are jointly estimable.
    """
    valid_cols = np.isfinite(delta_t) & np.isfinite(mu_delta)
    if valid_cols.sum() == 0:
        return np.nan, 0

    hist_sub = deltas_hist[:, valid_cols]
    hist_sub = hist_sub[np.all(np.isfinite(hist_sub), axis=1)]
    p_eff = hist_sub.shape[1]
    if p_eff == 0 or len(hist_sub) < max(MIN_HIST, p_eff + 1):
        return np.nan, 0

    centered = delta_t[valid_cols] - mu_delta[valid_cols]
    cov = np.cov(hist_sub, rowvar=False)
    if p_eff == 1:
        var = float(cov) if np.isfinite(cov) else np.nan
        if not np.isfinite(var) or var <= STD_EPS:
            return np.nan, 0
        return float((centered[0] ** 2) / var), 1

    cov = np.asarray(cov, dtype=float) + RIDGE_EPS * np.eye(p_eff)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    return float(centered @ inv_cov @ centered), int(p_eff)


def _t6_diagonal_vectorized(
    vals_sorted: np.ndarray, firm_codes_sorted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``_t6_per_firm(..., mode="diagonal")``.

    Builds the per-firm quarter-over-quarter delta series once, then -- for
    each canonical ratio dimension -- a sliding ``(n, Q_HIST - 1)`` window
    of preceding same-firm deltas via index arithmetic (no Python loop over
    firms or over quarters), matching ``_t6_per_firm``'s growing-then-capped
    rolling history exactly. Only the active ``"diagonal"`` mode is
    vectorized; ``"joint"`` stays on the per-firm loop (see profiling notes).

    Args:
        vals_sorted: ``(n, p)`` canonical ratio matrix, sorted by
            ``(gvkey, datacqtr)``.
        firm_codes_sorted: Integer firm codes (e.g. from
            ``pd.factorize``), aligned with ``vals_sorted``.

    Returns:
        ``(t6, effective_dims)`` aligned with ``vals_sorted``.
    """
    n, p = vals_sorted.shape
    same_prev = np.empty(n, dtype=bool)
    same_prev[0] = False
    if n > 1:
        same_prev[1:] = firm_codes_sorted[1:] == firm_codes_sorted[:-1]

    prev_vals = np.vstack([np.full((1, p), np.nan), vals_sorted[:-1]]) if n > 0 else vals_sorted
    delta = vals_sorted - prev_vals
    delta[~same_prev] = np.nan

    d_window = Q_HIST - 1
    idx = np.arange(n)
    offsets = np.arange(-d_window, 0)
    window_idx = idx[:, None] + offsets[None, :]
    in_bounds = window_idx >= 0
    window_idx_clipped = np.clip(window_idx, 0, n - 1)
    same_firm = firm_codes_sorted[window_idx_clipped] == firm_codes_sorted[idx][:, None]
    # A window slot only holds a structurally valid delta if it is in
    # bounds, belongs to the same firm, and isn't that firm's very first
    # row (which has no preceding row to diff against).
    structurally_present = in_bounds & same_firm & same_prev[window_idx_clipped]
    ok_len = structurally_present.sum(axis=1) >= MIN_HIST

    t6 = np.full(n, np.nan)
    dims = np.zeros(n, dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        for j in range(p):
            window_vals = np.where(in_bounds & same_firm, delta[window_idx_clipped, j], np.nan)
            counts = np.isfinite(window_vals).sum(axis=1)
            mu = np.divide(np.nansum(window_vals, axis=1), counts, out=np.full(n, np.nan), where=counts > 0)
            centered = np.where(np.isfinite(window_vals), window_vals - mu[:, None], np.nan)
            denom = counts - 1
            sigma_sq = np.divide(np.nansum(centered * centered, axis=1), denom, out=np.full(n, np.nan), where=denom > 0)
            sigma = np.sqrt(sigma_sq)

            z = np.where(sigma > STD_EPS, (delta[:, j] - mu) / sigma, np.nan)
            finite_z = np.isfinite(z) & ok_len
            t6 = np.where(finite_z, np.where(np.isfinite(t6), t6, 0.0) + z ** 2, t6)
            dims = np.where(finite_z, dims + 1, dims)

    t6 = np.where(dims > 0, t6, np.nan)
    return t6, dims


def _t6_per_firm(
    ratios_firm: pd.DataFrame,
    mode: str = "diagonal",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw ``T6`` and effective degrees of freedom for one firm.

    Args:
        ratios_firm: Canonical ratio dataframe for one firm, sorted by quarter.

    Returns:
        A tuple ``(t6_values, effective_dimensions)`` aligned with the input
        rows. ``effective_dimensions`` counts the finite standardized ratio
        changes effectively entering each ``T6`` value. In joint mode it
        counts the number of dimensions entering the Mahalanobis score.

    Notes:
        The active diagonal mode is more stable on short firm histories.
        The joint mode is implemented for future covariance-aware runs once the
        project decides to promote it from optional to active.
    """
    n = len(ratios_firm)
    t6 = np.full(n, np.nan)
    effective_dims = np.zeros(n, dtype=int)
    vals = ratios_firm.values  # shape (n, p)

    for t in range(1, n):
        if t < MIN_HIST + 1:
            continue
        hist_start = max(0, t - Q_HIST)
        deltas_hist = np.diff(vals[hist_start:t], axis=0)  # shape (t-hist_start-1, p)
        if len(deltas_hist) < MIN_HIST:
            continue

        mu_delta = np.nanmean(deltas_hist, axis=0)
        sigma_delta = np.nanstd(deltas_hist, axis=0, ddof=1)

        delta_t = vals[t] - vals[t - 1]
        if mode == "joint":
            t6_joint, p_eff = _compute_joint_t6(delta_t, mu_delta, deltas_hist)
            if np.isfinite(t6_joint) and p_eff > 0:
                t6[t] = t6_joint
                effective_dims[t] = p_eff
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sigma_delta > STD_EPS, (delta_t - mu_delta) / sigma_delta, np.nan)

        valid = z[np.isfinite(z)]
        if len(valid) == 0:
            continue
        t6[t] = float(np.sum(valid ** 2))
        effective_dims[t] = int(len(valid))

    return t6, effective_dims


def detect_temporal(
    df: pd.DataFrame,
    mode: str = "diagonal",
) -> pd.Series:
    """Compute the ``z6`` temporal consistency detector.

    Args:
        df: Input dataframe containing ``gvkey``, ``datacqtr``, and at least one
            canonical Sharia ratio.
        mode: Either ``"diagonal"`` for the active phase-1 score
            ``sum(Z_i^2)`` or ``"joint"`` for the optional covariance-aware
            Mahalanobis variant.

    Returns:
        A series aligned on ``df.index``. Values are ``NaN`` when the detector
        cannot be estimated.

    Notes:
        ``Q_HIST`` controls the rolling history length and ``MIN_HIST`` the
        minimum number of historical deltas required for one score. These are
        the two main levers to revisit if the project later wants a stricter
        WP-final temporal specification.
    """
    if mode not in {"diagonal", "joint"}:
        raise ValueError("detect_temporal: mode must be either 'diagonal' or 'joint'.")

    missing = [c for c in ["gvkey", "datacqtr"] if c not in df.columns]
    if missing:
        log.warning("detect_temporal: missing columns %s; returning NaN.", missing)
        return pd.Series(np.nan, index=df.index)

    ratios = get_canonical_sharia_ratios(df)
    if ratios.empty:
        log.warning("detect_temporal: no canonical adjusted ratios available; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    df_sorted = df[["gvkey", "datacqtr"]].copy()
    df_sorted["_order"] = range(len(df_sorted))
    df_sorted = df_sorted.sort_values(["gvkey", "datacqtr"])
    ratios_sorted = ratios.loc[df_sorted.index]

    t6_all = np.full(len(df), np.nan)
    effective_dims_all = np.zeros(len(df), dtype=int)
    if mode == "diagonal":
        # Vectorized (see profiling notes): ~90x faster on the full panel,
        # verified bit-identical. The optional "joint" mode (not currently
        # used in production) stays on the per-firm loop below.
        firm_codes, _ = pd.factorize(df_sorted["gvkey"], sort=False)
        t6_sorted, dims_sorted = _t6_diagonal_vectorized(
            ratios_sorted.to_numpy(dtype=float), firm_codes
        )
        original_positions = df_sorted["_order"].to_numpy()
        t6_all[original_positions] = t6_sorted
        effective_dims_all[original_positions] = dims_sorted
    else:
        for _, grp_idx in df_sorted.groupby("gvkey").groups.items():
            t6_firm, dims_firm = _t6_per_firm(ratios_sorted.loc[grp_idx], mode=mode)
            original_positions = df_sorted.loc[grp_idx, "_order"].values
            t6_all[original_positions] = t6_firm
            effective_dims_all[original_positions] = dims_firm

    if np.isfinite(t6_all).sum() == 0:
        return pd.Series(np.nan, index=df.index)

    z6_all = np.full(len(df), np.nan)
    for p_eff in sorted(set(effective_dims_all[np.isfinite(t6_all)])):
        if p_eff <= 0:
            continue
        mask = np.isfinite(t6_all) & (effective_dims_all == p_eff)
        if mask.any():
            z6_all[mask] = pit_chi2(t6_all[mask], degrees_of_freedom=int(p_eff))
    log.info(
        "detect_temporal: computed z6 with mode=%s, Q_HIST=%d, MIN_HIST=%d.",
        mode,
        Q_HIST,
        MIN_HIST,
    )
    return pd.Series(z6_all, index=df.index)
