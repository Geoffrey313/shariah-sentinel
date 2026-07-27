"""Shared mathematical helpers used across the project."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance


def first_digit(x: float | int | None) -> float:
    """Return the first significant digit of a numeric value.

    Args:
        x: Input numeric value.

    Returns:
        The first significant digit as a float, or ``np.nan`` when the input is
        zero, missing, or not finite.
    """
    if pd.isna(x):
        return np.nan
    x = abs(float(x))
    if not np.isfinite(x) or x == 0:
        return np.nan
    exponent = math.floor(math.log10(x))
    scaled = x / (10 ** exponent)
    return float(int(np.clip(scaled, 1.0, 9.999999999)))


def force_first_digit(x: float | int | None, target_digit: int) -> float | int | None:
    """Replace the first significant digit while keeping the scale unchanged."""
    if pd.isna(x) or x == 0:
        return x
    sign = 1 if x >= 0 else -1
    ax = abs(float(x))
    exponent = np.floor(np.log10(ax))
    mantissa = ax / (10 ** exponent)
    frac = mantissa - np.floor(mantissa)
    new_mantissa = target_digit + frac
    return sign * new_mantissa * (10 ** exponent)


def safe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-8) -> pd.Series:
    """Safely divide two numeric series.

    Args:
        num: Numerator series.
        den: Denominator series.
        eps: Minimum absolute denominator magnitude.

    Returns:
        A float series with ``NaN`` where the denominator is missing or too
        close to zero.
    """
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=num.index, dtype=float)
    mask = den.abs() > eps
    out.loc[mask] = num.loc[mask] / den.loc[mask]
    return out


def kl_divergence_kde(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    grid_size: int = 512,
    eps: float = 1e-10,
) -> float:
    """Estimate the 1D KL divergence ``D_KL(P_real || P_synth)`` via KDE."""
    x_real = np.asarray(x_real, dtype=float)
    x_synth = np.asarray(x_synth, dtype=float)
    x_real = x_real[np.isfinite(x_real)]
    x_synth = x_synth[np.isfinite(x_synth)]

    if len(x_real) < 2 or len(x_synth) < 2:
        return np.nan

    lo = min(x_real.min(), x_synth.min())
    hi = max(x_real.max(), x_synth.max())
    span = hi - lo

    if span <= 0:
        return 0.0

    lo -= 0.05 * span
    hi += 0.05 * span
    grid = np.linspace(lo, hi, grid_size)

    p = np.maximum(gaussian_kde(x_real)(grid), eps)
    q = np.maximum(gaussian_kde(x_synth)(grid), eps)

    dx = grid[1] - grid[0]
    p = p / (np.sum(p) * dx)
    q = q / (np.sum(q) * dx)

    return float(np.sum(p * np.log(p / q)) * dx)


def wasserstein_distance_1d(
    x_real: np.ndarray,
    x_synth: np.ndarray,
) -> float:
    """Compute the 1D Wasserstein-1 distance between two distributions."""
    x_real  = np.asarray(x_real, dtype=float)
    x_synth = np.asarray(x_synth, dtype=float)
    x_real  = x_real[np.isfinite(x_real)]
    x_synth = x_synth[np.isfinite(x_synth)]

    if len(x_real) < 2 or len(x_synth) < 2:
        return np.nan

    return float(wasserstein_distance(x_real, x_synth))


def corr_matrix_deviation(
    corr_real: np.ndarray,
    corr_synth: np.ndarray,
) -> dict[str, float]:
    """Measure the deviation between two correlation matrices."""
    D = corr_real - corr_synth
    baseline_frob = float(np.linalg.norm(corr_real, "fro"))
    n = corr_real.shape[0]

    # off-diagonal mask
    off_diag = ~np.eye(n, dtype=bool)

    return {
        "frobenius_norm":      float(np.linalg.norm(D, "fro")),
        "frobenius_norm_relative": float(np.linalg.norm(D, "fro") / baseline_frob) if baseline_frob > 0 else np.nan,
        "max_abs_diff":        float(np.max(np.abs(D))),
        "mean_abs_diff":       float(np.mean(np.abs(D))),
        "off_diag_frobenius":  float(np.linalg.norm(D[off_diag])),
    }
