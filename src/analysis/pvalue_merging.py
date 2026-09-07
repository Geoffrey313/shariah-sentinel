"""Vovk--Wang (2020) p-value merging functions.

Combine ``K`` p-values that test one hypothesis into a single valid p-value under
**arbitrary, unknown dependence**, via the generalized (power) mean
``M_{r,K}(p) = ((1/K) sum_j p_j^r)^{1/r}`` scaled by a sharp correction factor
``a_{r,K}`` (Vovk & Wang, *Combining p-values via averaging*, Biometrika 107(4),
791--808, DOI 10.1093/biomet/asaa027). Also provides the order-statistic combiners
(Bonferroni, and the arbitrary-dependence Hommel and the positive-dependence Simes).

The special cases of the ``r``-mean family are

    r = +inf : maximum           (factor 1, precise)
    r =  1   : arithmetic mean   (factor 2; Rueschendorf/Meng -- not improvable)
    r =  0   : geometric mean    (factor e; Mattner)
    r = -1   : harmonic mean     (factor e*ln K conservative all-K bound, OR the
                                  sharp finite a_{-1,K} -- see harmonic_sharp_factor)
    r = -inf : minimum           (factor K = Bonferroni)

The harmonic member has two valid factors, both selectable explicitly:
``e*ln K`` is a conservative bound valid for all ``K``; the sharp finite factor
``a_{-1,K}`` (Gui, Jiang & Wang, Biometrika 2025, DOI 10.1093/biomet/asaf038,
attributed there to Vovk & Wang 2020) is smaller and ``K``-specific
(``a_{-1,8} ~= 4.2498`` vs ``e*ln 8 ~= 5.6525``).

Self-contained numerical library: imports only NumPy (and stdlib functools), touches
no data, no configuration and no pipeline. It is the shared foundation for the
within-firm aggregation and the p-value merging baselines.

Scope note (do not overclaim): nothing here improves on Vovk--Wang. Their
worst-case correction factors are sharp -- the arithmetic-mean factor 2 "cannot be
improved in general" (Rueschendorf/Meng). Combining under estimable dependence for
more power is a *different* problem, handled elsewhere (the empirical bootstrap),
not by this module.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "correction_factor",
    "harmonic_sharp_factor",
    "generalized_mean",
    "merge_generalized",
    "bonferroni",
    "arithmetic",
    "geometric",
    "harmonic",
    "harmonic_sharp",
    "maximum",
    "hommel",
    "simes",
    "merge",
    "ARBITRARY_DEPENDENCE_METHODS",
]

_E = float(np.e)


def _clean(p) -> np.ndarray:
    """Validate and return a 1-D float array of p-values in ``[0, 1]``."""
    arr = np.asarray(p, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("no p-values given")
    if np.any(~np.isfinite(arr)):
        raise ValueError("p-values must be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    return arr


def correction_factor(r: float, K: int) -> float:
    """``a_{r,K}`` making ``a_{r,K} * M_{r,K}`` a valid merging function under
    arbitrary dependence (Vovk & Wang, 2020).

    Reduces to 2 (arithmetic, ``r=1``), ``e`` (geometric, ``r=0``), ``e*ln K``
    (harmonic, ``r=-1`` -- the conservative all-K bound; see
    :func:`harmonic_sharp_factor` for the sharp finite factor), ``K`` (Bonferroni,
    ``r=-inf``) and 1 (maximum, ``r=+inf``).
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if r == np.inf:
        return 1.0
    if r == -np.inf:
        return float(K)
    if r == -1.0:
        return _E * np.log(K) if K > 1 else 1.0
    if r > -1.0:
        # covers the arithmetic (r=1 -> 2) and geometric (r=0 -> e) cases
        return _E if r == 0.0 else float((r + 1.0) ** (1.0 / r))
    # r in (-inf, -1): a_{r,K} = (r/(r+1)) * K^{1 + 1/r}
    return float((r / (r + 1.0)) * K ** (1.0 + 1.0 / r))


@lru_cache(maxsize=None)
def _harmonic_sharp_root(K: int) -> float:
    """Unique positive root ``y_K`` of ``y^2 = K*((y+1)*ln(y+1) - y)``.

    Returns ``nan`` when no positive root exists (small ``K``), so the caller can
    fall back to the conservative ``e*ln K`` bound. Solved by bisection (NumPy-only,
    no SciPy)."""
    if K < 3:
        return float("nan")

    def f(y: float) -> float:
        return y * y - K * ((y + 1.0) * np.log(y + 1.0) - y)

    lo, hi = 1e-9, 4.0 * K
    if f(lo) >= 0.0:  # no adverse dip near 0 -> no positive root
        return float("nan")
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=None)
def harmonic_sharp_factor(K: int) -> float:
    """Sharp finite Vovk--Wang harmonic correction factor ``a_{-1,K}`` under
    arbitrary dependence,

        ``a_{-1,K} = (y_K + K)^2 / (K * (y_K + 1))``,

    with ``y_K`` the positive root of ``y^2 = K*((y+1)*ln(y+1) - y)`` (Gui, Jiang &
    Wang, Biometrika 2025, DOI 10.1093/biomet/asaf038; attributed there to Vovk &
    Wang 2020). Smaller than the conservative ``e*ln K`` all-``K`` bound
    (``a_{-1,8} ~= 4.2498`` vs ``e*ln 8 ~= 5.6525``). Falls back to ``e*ln K`` when
    the root does not exist (``K < 3``)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return 1.0
    y = _harmonic_sharp_root(K)
    if not np.isfinite(y):
        return _E * np.log(K)
    a = (y + K) ** 2 / (K * (y + 1.0))
    # the sharp factor never exceeds the conservative all-K bound (numeric guard)
    return float(min(a, _E * np.log(K)))


def generalized_mean(p, r: float) -> float:
    """The power mean ``M_{r,K}(p) = ((1/K) sum p_j^r)^{1/r}`` with the standard
    min (``r=-inf``), max (``r=+inf``) and geometric (``r=0``) limits.

    A zero p-value drives every non-positive-order mean to 0 (so a single exact
    zero dominates), which is the intended behaviour.
    """
    arr = _clean(p)
    K = arr.size
    if r == -np.inf:
        return float(arr.min())
    if r == np.inf:
        return float(arr.max())
    if r == 0.0:
        if np.any(arr <= 0.0):
            return 0.0
        return float(np.exp(np.mean(np.log(arr))))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        powered = np.power(arr, r)
        m = float(np.mean(powered))
        if not np.isfinite(m):
            # r < 0 with a zero p-value: p^r -> +inf, mean -> +inf, M -> 0
            return 0.0
        return float(m ** (1.0 / r))


def merge_generalized(p, r: float) -> float:
    """Valid merged p-value ``min(1, a_{r,K} * M_{r,K}(p))`` under arbitrary
    dependence. For ``K = 1`` returns the single p-value unchanged.

    The harmonic member (``r = -1``) uses the conservative ``e*ln K`` factor; for
    the sharp finite factor call :func:`harmonic` with ``mode='sharp'``."""
    arr = _clean(p)
    K = arr.size
    if K == 1:
        return float(arr[0])
    return float(min(1.0, correction_factor(r, K) * generalized_mean(arr, r)))


def bonferroni(p) -> float:
    """``min(1, K * min p)`` -- the ``r = -inf`` member. Valid under any dependence."""
    return merge_generalized(p, -np.inf)


def arithmetic(p) -> float:
    """``min(1, 2 * mean p)`` -- the ``r = 1`` member. Valid under any dependence."""
    return merge_generalized(p, 1.0)


def geometric(p) -> float:
    """``min(1, e * geomean p)`` -- the ``r = 0`` member. Valid under any dependence."""
    return merge_generalized(p, 0.0)


def harmonic(p, mode: str = "conservative") -> float:
    """``min(1, a_{-1,K} * harmonic-mean p)`` -- the ``r = -1`` member, valid under
    any dependence.

    ``mode='conservative'`` (default, backward-compatible) uses the ``e*ln K``
    all-``K`` bound; ``mode='sharp'`` uses the sharp finite factor
    :func:`harmonic_sharp_factor`. The mode is always explicit -- there is no hidden
    default that silently changes which factor produced a number."""
    arr = _clean(p)
    K = arr.size
    if K == 1:
        return float(arr[0])
    if mode == "conservative":
        factor = _E * np.log(K)
    elif mode == "sharp":
        factor = harmonic_sharp_factor(K)
    else:
        raise ValueError(
            f"unknown harmonic mode {mode!r}; use 'conservative' or 'sharp'"
        )
    return float(min(1.0, factor * generalized_mean(arr, -1.0)))


def harmonic_sharp(p) -> float:
    """Harmonic merge with the **sharp finite** factor -- :func:`harmonic` with
    ``mode='sharp'``. Exposed as a distinct named combiner so it appears explicitly
    in the baseline tables and reporting (never as a silent variant of harmonic)."""
    return harmonic(p, mode="sharp")


def maximum(p) -> float:
    """``max p`` -- the ``r = +inf`` member (precise). Valid under any dependence."""
    return merge_generalized(p, np.inf)


def hommel(p) -> float:
    """Hommel (1983): ``(sum_{k=1}^{K} 1/k) * min_k (K/k) p_(k)``. Valid under
    **arbitrary** dependence (the arbitrary-dependence analogue of Simes)."""
    arr = np.sort(_clean(p))
    K = arr.size
    if K == 1:
        return float(arr[0])
    c = float(np.sum(1.0 / np.arange(1, K + 1)))
    ranked = (K / np.arange(1, K + 1)) * arr
    return float(min(1.0, c * float(ranked.min())))


def simes(p) -> float:
    """Simes (1986): ``min_k (K/k) p_(k)``. Valid under independence and PRDS
    positive dependence, **not** under arbitrary dependence -- use only with a
    stated dependence assumption; :func:`hommel` is the arbitrary-dependence analogue."""
    arr = np.sort(_clean(p))
    K = arr.size
    if K == 1:
        return float(arr[0])
    ranked = (K / np.arange(1, K + 1)) * arr
    return float(min(1.0, float(ranked.min())))


# Combiners valid under arbitrary dependence (Simes is deliberately excluded).
ARBITRARY_DEPENDENCE_METHODS: tuple[str, ...] = (
    "bonferroni", "arithmetic", "geometric", "harmonic", "harmonic_sharp",
    "maximum", "hommel",
)

_METHODS = {
    "bonferroni": bonferroni,
    "arithmetic": arithmetic,
    "geometric": geometric,
    "harmonic": harmonic,
    "harmonic_sharp": harmonic_sharp,
    "maximum": maximum,
    "hommel": hommel,
    "simes": simes,
}


def merge(p, method: str = "harmonic") -> float:
    """Dispatch to a named merging function. Default ``harmonic`` -- the
    dependence-robust choice used for within-firm aggregation, with the conservative
    ``e*ln K`` factor. Use ``method='harmonic_sharp'`` for the sharp finite factor.

    For an arbitrary generalized-mean order, call :func:`merge_generalized` with ``r``.
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_METHODS)} "
            f"or use merge_generalized(p, r)"
        ) from None
    return fn(p)
