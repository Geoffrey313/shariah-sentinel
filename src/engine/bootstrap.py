"""Joint bootstrap — framework §3.3, §5.3, §6.3.

One bootstrap produces the simultaneous null distributions of all five
composites (``Z+``, ``Z+_renorm``, ``B_A``, ``Z²_Mah``, ``T_IUT``). This
matters because ``B_A`` is a random denominator of ``Z+_renorm``; computing
the two separately would double the Monte-Carlo cost and desynchronise the
replicate indexing required for concordance diagnostics.

Two variants:

- **Parametric** — ``z^(b) ∼ N(0, Σ̂_LW)``. Assumes joint normality of
  ``z``; fast, requires only ``Σ̂``.
- **Non-parametric** — resample complete z-vectors from the reference
  sample ``C``. Makes no distributional assumption; recommended when
  Phase 1 flagged one or more detectors as non-normal (which is our case).

The bootstrap draws are generated with ``numpy``'s modern ``Generator``
API seeded from :attr:`BootstrapSettings.random_seed` so results are
reproducible across runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.engine.composites import (
    breadth,
    iut_statistic,
    mahalanobis_squared,
    renormalised_truncated_sum,
    truncated_sum,
)
from src.engine.detector_composite import orthogonal_softmax, softmax_active
from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapNullDistributions:
    """All five composite null distributions for one bootstrap variant.

    Each array has shape ``(B,)``. Stored sorted ascending so p-values
    reduce to ``np.searchsorted`` — ``O(log B)`` per row.
    """

    z_plus_sorted: np.ndarray
    z_plus_renorm_sorted: np.ndarray
    breadth_sorted: np.ndarray
    z_mahalanobis_sorted: np.ndarray
    t_iut_sorted: np.ndarray
    z_plus_softmax_sorted: np.ndarray
    z_plus_orth_sorted: np.ndarray
    n_replicates: int
    variant: str


def _simulate_parametric(
    sigma: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``B`` vectors ``z^(b) ∼ N(0, Σ̂)``.

    Returns an array of shape ``(B, k)``. Uses the multivariate normal
    sampler with ``method="cholesky"`` for determinism across numpy
    versions.
    """
    k = sigma.shape[0]
    mean = np.zeros(k, dtype=float)
    return rng.multivariate_normal(mean, sigma, size=n_replicates, method="cholesky")


def _simulate_non_parametric(
    reference_rows: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample ``B`` complete z-vectors from ``C``.

    Each bootstrap replicate picks one row of ``reference_rows`` with
    replacement — the whole vector is kept intact so the resampled joint
    distribution matches ``C`` by construction.
    """
    n_ref = reference_rows.shape[0]
    idx = rng.integers(low=0, high=n_ref, size=n_replicates)
    return reference_rows[idx]


def _composite_null_from_draws(
    draws: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
    variant: str,
) -> BootstrapNullDistributions:
    """Apply the five composite functions to a ``(B, k)`` draw matrix."""
    comp = settings.composites
    zp = truncated_sum(draws, weights)
    zpr = renormalised_truncated_sum(
        draws, weights,
        threshold=comp.active_set_threshold,
        min_active=comp.min_active_for_renorm,
    )
    b = breadth(draws, threshold=comp.active_set_threshold)
    zm = mahalanobis_squared(draws, sigma)
    t = iut_statistic(draws, min_active=comp.min_active_for_iut)
    finite_mask = np.isfinite(draws)
    z_fill = np.where(finite_mask, draws, 0.0)
    z_softmax = softmax_active(z_fill, finite_mask, weights, gamma=comp.softmax_gamma)
    z_orth = orthogonal_softmax(z_fill, finite_mask, weights, gamma=comp.softmax_gamma, Sigma=sigma)

    return BootstrapNullDistributions(
        z_plus_sorted=np.sort(zp[np.isfinite(zp)]),
        z_plus_renorm_sorted=np.sort(zpr[np.isfinite(zpr)]),
        breadth_sorted=np.sort(b[np.isfinite(b)]),
        z_mahalanobis_sorted=np.sort(zm[np.isfinite(zm)]),
        t_iut_sorted=np.sort(t[np.isfinite(t)]),
        z_plus_softmax_sorted=np.sort(z_softmax[np.isfinite(z_softmax)]),
        z_plus_orth_sorted=np.sort(z_orth[np.isfinite(z_orth)]),
        n_replicates=int(draws.shape[0]),
        variant=variant,
    )


def run_parametric_bootstrap(
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> BootstrapNullDistributions:
    """Generate the parametric joint null distribution."""
    rng = np.random.default_rng(settings.bootstrap.random_seed)
    draws = _simulate_parametric(
        sigma=sigma,
        n_replicates=settings.bootstrap.n_replicates,
        rng=rng,
    )
    return _composite_null_from_draws(
        draws=draws,
        sigma=sigma,
        weights=weights,
        settings=settings,
        variant="parametric",
    )


def run_non_parametric_bootstrap(
    reference_rows: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> BootstrapNullDistributions:
    """Generate the non-parametric joint null distribution.

    Args:
        reference_rows: Complete z-vectors from ``C``, shape ``(n_ref, k)``.
        sigma: ``Σ̂`` used for ``Z²_Mah``. Pass the same matrix used by the
            parametric variant so the two are comparable.
        weights: Detector weights for ``Z+`` / ``Z+_renorm``.
        settings: Configuration.
    """
    rng = np.random.default_rng(settings.bootstrap.random_seed)
    draws = _simulate_non_parametric(
        reference_rows=reference_rows,
        n_replicates=settings.bootstrap.n_replicates,
        rng=rng,
    )
    return _composite_null_from_draws(
        draws=draws,
        sigma=sigma,
        weights=weights,
        settings=settings,
        variant="non_parametric",
    )


def upper_tail_pvalue(
    observed: np.ndarray,
    null_sorted: np.ndarray,
    n_replicates: int,
) -> np.ndarray:
    """Framework §3.3 p-value ``(1 + Σ 1{sim ≥ obs}) / (B + 1)``.

    Uses ``np.searchsorted`` on the sorted null for ``O(n log B)`` speed
    rather than ``O(n · B)``. NaN observed → NaN p-value.
    """
    obs = np.asarray(observed, dtype=float)
    if null_sorted.size == 0:
        return np.full(obs.shape, np.nan, dtype=float)
    # Number of null ≥ obs  ≡  total - number of null < obs  ≡  size -
    # searchsorted(null_sorted, obs, side="left").
    idx = np.searchsorted(null_sorted, obs, side="left")
    count_ge = null_sorted.size - idx
    p = (1.0 + count_ge) / (n_replicates + 1.0)
    return np.where(np.isfinite(obs), p, np.nan)


def lower_tail_pvalue(
    observed: np.ndarray,
    null_sorted: np.ndarray,
    n_replicates: int,
) -> np.ndarray:
    """Lower-tail variant — used for ``T_IUT`` where the rejection region
    is ``T > c_α`` under the §5.2 convention but the *null* distribution
    is still one-sided. Kept symmetric for completeness.
    """
    # For the IUT we still want P(T_null ≥ T_obs); provided separately so
    # the caller's intent stays explicit.
    return upper_tail_pvalue(observed, null_sorted, n_replicates)
