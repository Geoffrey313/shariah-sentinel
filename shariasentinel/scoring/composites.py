"""Composite statistics from the seven (k ≤ 7) z-scores — framework §3–§6.

Five composites are computed side-by-side:

- ``Z+``        (§3.2)  ``Σ w_j max(z_j, 0)``                 — targeted, dilution-prone.
- ``Z+_renorm`` (§6.2)  ``(Σ_A w_j z_j) / (Σ_A w_j)``         — dilution-free intensity.
- ``B_A``       (§6.4)  ``|A| / k``                           — breadth.
- ``Z²_Mah``    (§4.1)  ``z' Σ̂⁻¹ z``                          — joint anomaly.
- ``T_IUT``     (§5.2)  ``min_j z_j``                         — unanimity test.

All functions accept a 2-D ``numpy`` z-matrix of shape ``(n, k)`` and
return a 1-D array of length ``n`` (scalar per row). The module stays free
of I/O — Phase 4 / Phase 4b wrap it with panel data and persistence.

NaN handling matches framework semantics:

- ``Z+``        — a silent detector contributes ``0``; a NaN detector is
  treated as silent.
- ``Z+_renorm`` — NaN detectors drop out of the active set ``A``; if
  ``|A| < min_active``, the row is NaN.
- ``B_A``       — counts active detectors out of ``k``; NaN → silent.
- ``Z²_Mah``    — requires a complete z-vector. NaN → NaN.
- ``T_IUT``     — ``min`` over finite entries only; NaN when fewer than
  ``min_active_for_iut`` detectors are finite.
"""
from __future__ import annotations

import numpy as np


def _as_2d_float(z: np.ndarray) -> np.ndarray:
    """Cast a z-matrix to 2-D float without copying when possible."""
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"z must be 2-D, got shape {arr.shape}.")
    return arr


def _validate_weights(weights: np.ndarray, k: int) -> np.ndarray:
    """Check ``w_j ≥ 0, Σ w_j = 1`` and dimension matches ``k``."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (k,):
        raise ValueError(f"weights shape {w.shape} ≠ ({k},).")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative (framework §3.2).")
    s = w.sum()
    if not np.isfinite(s) or abs(s - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1 (got {s}).")
    return w


def truncated_sum(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``Z+ = Σ w_j max(z_j, 0)`` (framework §3.2).

    Silent detectors (``z_j ≤ 0`` or NaN) contribute ``0``; a row where
    every detector is NaN returns NaN so downstream reports can distinguish
    "no data" from "all calm".
    """
    arr = _as_2d_float(z)
    k = arr.shape[1]
    w = _validate_weights(weights, k)
    positive = np.where(np.isfinite(arr), np.clip(arr, 0.0, None), 0.0)
    row_has_data = np.isfinite(arr).any(axis=1)
    out = positive @ w
    return np.where(row_has_data, out, np.nan)


def active_set_indicator(z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean mask ``(n, k)`` — ``True`` where ``z_j > threshold`` and finite."""
    arr = _as_2d_float(z)
    return np.where(np.isfinite(arr), arr > threshold, False)


def breadth(z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """``B_A = |A| / k`` (framework §6.4)."""
    active = active_set_indicator(z, threshold=threshold)
    k = active.shape[1]
    return active.sum(axis=1).astype(float) / float(k)


def renormalised_truncated_sum(
    z: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.0,
    min_active: int = 1,
) -> np.ndarray:
    """``Z+_renorm = Σ_A w_j z_j / Σ_A w_j`` (framework §6.2).

    Rows with fewer than ``min_active`` active detectors (``z_j > threshold``)
    return NaN — ``|A|`` is random, and the denominator is undefined at
    ``|A| = 0``.
    """
    arr = _as_2d_float(z)
    k = arr.shape[1]
    w = _validate_weights(weights, k)
    active = active_set_indicator(arr, threshold=threshold)
    n_active = active.sum(axis=1)
    numerator = np.where(active, np.where(np.isfinite(arr), arr, 0.0) * w, 0.0).sum(axis=1)
    denominator = np.where(active, w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / denominator
    out = np.where(n_active >= min_active, out, np.nan)
    return out


def mahalanobis_squared(z: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``Z²_Mah = z' Σ̂⁻¹ z`` (framework §4.1).

    Rows with any NaN entry are returned as NaN — the quadratic form is
    only defined on a complete z-vector. The quadratic form is evaluated
    via ``np.linalg.solve`` rather than an explicit inverse; singular
    covariance inputs degrade gracefully to ``NaN`` instead of crashing
    Phase 4.
    """
    arr = _as_2d_float(z)
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.shape != (arr.shape[1], arr.shape[1]):
        raise ValueError(
            f"Σ shape {sigma_arr.shape} ≠ ({arr.shape[1]}, {arr.shape[1]})."
        )
    complete = np.isfinite(arr).all(axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if complete.any():
        block = arr[complete]
        try:
            solved = np.linalg.solve(sigma_arr, block.T).T
        except np.linalg.LinAlgError:
            return out
        out[complete] = np.einsum("ij,ij->i", block, solved)
    return out


def iut_statistic(z: np.ndarray, min_active: int = 1) -> np.ndarray:
    """``T_IUT = min_j z_j`` (framework §5.2).

    Computed over finite entries only. When fewer than ``min_active``
    detectors produced a finite score the test is undefined → NaN. Under
    ``H_1^IUT`` every detector has positive mean; the test rejects when
    the minimum is large.
    """
    arr = _as_2d_float(z)
    finite = np.isfinite(arr)
    n_finite = finite.sum(axis=1)
    # Replace NaN with +inf so they do not win the min; fallback to NaN
    # if no finite entry.
    safe = np.where(finite, arr, np.inf)
    minimum = safe.min(axis=1)
    minimum = np.where(n_finite >= min_active, minimum, np.nan)
    # If every detector was NaN, min returns +inf → mask back to NaN.
    minimum = np.where(np.isfinite(minimum), minimum, np.nan)
    return minimum
