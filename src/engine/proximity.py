"""Threshold proximity detector.

This detector implements the Working Paper's ``j=4`` idea: firms that manage
their Sharia ratios to stay just below a threshold should exhibit abnormally
small normalized distances to that threshold.

Framework §9.2 D4 derives the raw statistic ``T = √(12Q)(1/2 − d̄)`` from the
null model ``r_t ∼ Uniform(0, τ)``. That null describes unmanaged firms, not
the SAC-compliant sample the project uses as reference ``C`` — compliant firms
deliberately keep buffer below ``τ``, so ``d̄ > 1/2`` systematically and the
raw statistic is structurally negative on ``C``.

To stay consistent with the other detectors that already self-calibrate on
``C`` (z₃ M-score), the implementation computes ``T`` parametrically and then
maps it to the common ``N(0, 1)`` scale via **empirical PIT on ``C``**. This
turns the "honest firm" baseline into mean ≈ 0, std ≈ 1 by construction,
while preserving the manipulation direction (``T`` still grows positively
when a firm clusters near ``τ``).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.common.methodology import SAC_MY, screening_thresholds_for_panel

from src.engine.pit import pit_empirical

log = logging.getLogger(__name__)

# Legacy aliases — the actual per-country thresholds are resolved at
# detection time from the panel's ``methodology_key`` column via
# ``screening_thresholds_for_panel`` (SAC fallback keeps MYS behaviour identical).
# Kept for external callers that imported these constants.
THRESHOLD_DEBT: float = SAC_MY.threshold_debt
THRESHOLD_CASH: float = SAC_MY.threshold_cash
THRESHOLD_INCOME: float = SAC_MY.threshold_income

SAC_THRESHOLDS: dict[str, float] = SAC_MY.ratio_thresholds()
MIN_QUARTERS: int = 4

# Minimum number of distinct-firm raw statistics required on ``C`` before the
# empirical PIT reference is considered usable. The framework §9.2 D3 floor
# of 30 is reused here — same logic (a stable empirical CDF needs enough
# independent samples); the detector falls back to NaN otherwise so Phase 4
# can detect the miss.
MIN_REF_FIRMS: int = 30


def _proximity_raw_statistic(ratios: np.ndarray, threshold: float) -> float:
    """Raw §9.2 D4 statistic ``T = √(12Q)(1/2 − d̄)`` for one ratio history.

    Compliant firms with ``r_t`` concentrated near 0 produce ``d̄ > 1/2`` and
    therefore negative ``T``; manipulators clustering near ``τ`` produce
    ``d̄ → 0`` and large positive ``T``. The subsequent empirical PIT on
    ``C`` re-centres the scale without flipping the sign.
    """
    r = ratios[np.isfinite(ratios) & (ratios >= 0)]
    q = len(r)
    if q < MIN_QUARTERS:
        return np.nan
    # Ratios at or above the cap have zero remaining buffer. Do not drop them:
    # they are exactly the cases a threshold-proximity detector must preserve.
    d = np.maximum((threshold - r) / threshold, 0.0)  # ∈ [0, 1]
    d_bar = d.mean()
    return float(np.sqrt(12 * q) * (0.5 - d_bar))


def _get_ratio_matrix(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Return the precomputed canonical Sharia ratios available in ``df``."""
    available = {name for name in thresholds if name in df.columns}
    if not available:
        return pd.DataFrame(index=df.index)
    return df[list(available)].copy()


def _proximity_raw_statistic_vectorized(
    vals: pd.Series, threshold: float, firm_id: pd.Series
) -> pd.Series:
    """Row-broadcast ``T`` for one ratio column, every firm at once.

    Vectorized equivalent of ``_proximity_raw_statistic``: this is a
    whole-firm-history aggregate (no rolling window), so
    ``groupby(firm_id).transform(...)`` computes each firm's ``q``/``d̄``
    once and broadcasts it back to every row of that firm.
    """
    mask = vals.notna() & (vals >= 0)
    d = ((threshold - vals) / threshold).clip(lower=0.0)
    d_masked = d.where(mask)
    grouped = d_masked.groupby(firm_id, sort=False)
    q = grouped.transform("count")
    d_bar = grouped.transform("mean")
    t = np.sqrt(12 * q) * (0.5 - d_bar)
    return t.where(q >= MIN_QUARTERS)


def _firm_raw_statistics(
    df: pd.DataFrame,
    ratios: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """One raw ``T`` per firm — max over available SAC ratios.

    Taking the max keeps the detector sensitive to the worst-offending ratio
    at the cost of a small positive-tail bias (``E[max_{3} N(0, 1)] ≈ 0.85``).
    The subsequent empirical PIT absorbs that bias into the reference CDF.

    Vectorized via ``groupby(...).transform(...)`` instead of a per-firm
    Python loop (see profiling notes); verified bit-identical on the full
    real panel.
    """
    firm_id = df["gvkey"]
    per_ratio = [
        _proximity_raw_statistic_vectorized(ratios[name], threshold, firm_id)
        for name, threshold in thresholds.items()
        if name in ratios.columns
    ]
    if not per_ratio:
        return {str(gid): np.nan for gid in df["gvkey"].unique()}
    t4_broadcast = pd.concat(per_ratio, axis=1).max(axis=1, skipna=True)
    return t4_broadcast.groupby(firm_id.astype(str), sort=False).first().to_dict()


def _reference_firms(df: pd.DataFrame) -> set[str]:
    """Set of firm ids that contribute at least one row to ``C``.

    When ``_split`` is absent (ad-hoc callers without Phase 0), every firm
    is treated as part of the reference — the detector still returns a score
    but its calibration reduces to "against itself".
    """
    if "_split" not in df.columns:
        return set(df["gvkey"].astype(str).unique())
    return set(df.loc[df["_split"] == "C", "gvkey"].astype(str).unique())


def detect_threshold_proximity(df: pd.DataFrame) -> pd.Series:
    """Compute the ``z4`` threshold proximity detector.

    Args:
        df: Input dataframe containing ``gvkey`` and the canonical Sharia
            ratio columns produced by the panel pipeline. If ``_split`` is
            present, the PIT reference is restricted to firms with at least
            one row labelled ``"C"`` so the ``N(0, 1)`` scale is calibrated
            on the honest compliant sample.

    Returns:
        A series aligned on ``df.index``. Values are ``NaN`` when no ratio
        reaches ``MIN_QUARTERS`` samples or when the ``C`` reference has
        fewer than ``MIN_REF_FIRMS`` distinct firms.
    """
    if "gvkey" not in df.columns:
        log.warning("detect_threshold_proximity: missing `gvkey`; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    thresholds = screening_thresholds_for_panel(df)
    ratios = _get_ratio_matrix(df, thresholds)
    if ratios.empty:
        log.warning(
            "detect_threshold_proximity: no precomputed canonical ratios available; returning NaN."
        )
        return pd.Series(np.nan, index=df.index)

    firm_raw = _firm_raw_statistics(df, ratios, thresholds)
    finite_firm_raw = {k: v for k, v in firm_raw.items() if np.isfinite(v)}
    if not finite_firm_raw:
        return pd.Series(np.nan, index=df.index)

    ref_firms = _reference_firms(df)
    reference_values = np.asarray(
        [v for k, v in finite_firm_raw.items() if k in ref_firms],
        dtype=float,
    )
    if reference_values.size < MIN_REF_FIRMS:
        log.warning(
            "detect_threshold_proximity: only %d reference firms in C (< %d); returning NaN.",
            int(reference_values.size),
            MIN_REF_FIRMS,
        )
        return pd.Series(np.nan, index=df.index)

    firm_ids = list(finite_firm_raw.keys())
    firm_raw_array = np.asarray([finite_firm_raw[k] for k in firm_ids], dtype=float)
    firm_z = pit_empirical(firm_raw_array, reference_values)
    firm_z_series = pd.Series(firm_z, index=firm_ids)
    return df["gvkey"].astype(str).map(firm_z_series)
