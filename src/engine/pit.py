"""Centralized PIT transformations for detector statistics.

Each helper maps a raw detector statistic ``T_j`` to the common anomaly-oriented
z-scale through ``z_j = Phi^{-1}(F_j(T_j; H0_hat))``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f, norm

CLIP_EPS: float = 1e-9


def pit_chi2(T: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Apply a chi-square PIT transformation.

    Args:
        T: Raw detector statistics.
        degrees_of_freedom: Degrees of freedom of the chi-square null.

    Returns:
        Detector z-scores on the common standard normal scale.
    """
    T_arr = np.asarray(T, dtype=float)
    p = np.where(np.isfinite(T_arr), chi2.cdf(T_arr, df=degrees_of_freedom), np.nan)
    return norm.ppf(np.clip(p, CLIP_EPS, 1.0 - CLIP_EPS))


def pit_f(T: np.ndarray, dfn: int, dfd: int) -> np.ndarray:
    """Apply an F-based PIT transformation.

    Args:
        T: Raw detector statistics already scaled for the F null.
        dfn: Numerator degrees of freedom.
        dfd: Denominator degrees of freedom.

    Returns:
        Detector z-scores on the common standard normal scale.
    """
    T_arr = np.asarray(T, dtype=float)
    p = np.where(np.isfinite(T_arr), f.cdf(T_arr, dfn=dfn, dfd=dfd), np.nan)
    return norm.ppf(np.clip(p, CLIP_EPS, 1.0 - CLIP_EPS))


def pit_empirical(T: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Apply an empirical PIT transformation from a calibration sample.

    Args:
        T: Raw detector statistics to transform.
        reference: Calibration statistics from the estimated null regime.

    Returns:
        Approximate detector z-scores on the standard normal scale.

    Notes:
        The empirical CDF ``F(t) = mean(reference <= t)`` is computed via a
        sort + ``searchsorted`` (O(n log m + m log m)) instead of comparing
        every ``t`` against the full reference array in a Python loop
        (O(n * m)) -- this is one of the hottest primitives in the scoring
        pipeline (every detector's second-stage calibration, plus the
        Family-3 counterfactual reference lookups, funnel through here).
        Verified bit-identical to the loop-based reference implementation,
        including ties and out-of-range values (see profiling notes).
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) < 2:
        return np.full(np.asarray(T).size, np.nan)
    T_arr = np.asarray(T, dtype=float)
    ref_sorted = np.sort(ref)
    finite = np.isfinite(T_arr)
    counts = np.searchsorted(ref_sorted, np.where(finite, T_arr, 0.0), side="right")
    p = np.where(finite, counts / len(ref), np.nan)
    return norm.ppf(np.clip(p, CLIP_EPS, 1.0 - CLIP_EPS))


def pit_already_normal(z_approx: np.ndarray) -> np.ndarray:
    """Pass through statistics that are already on the z-scale.

    Args:
        z_approx: Statistics already approximately distributed as ``N(0, 1)``.

    Returns:
        The same numeric array without further transformation.
    """
    return np.asarray(z_approx, dtype=float)
