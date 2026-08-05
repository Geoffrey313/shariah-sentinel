"""Phase 2 — covariance estimation and dependence structure.

Framework §4.2 designates Ledoit-Wolf shrinkage as the production ``Σ̂``
consumed by:

- the parametric joint bootstrap of Phase 4 (``z^(b) ∼ N(0, Σ̂_LW)``),
- the Mahalanobis composite ``Z²_Mah = z' Σ̂_LW⁻¹ z`` (§4.1),
- the IUT p-value under dependence (§5.3).

This module estimates ``Σ̂`` three ways so Phase 4 can sanity-check
sensitivity, and exposes dependence diagnostics that detect pathological
cases early (quasi-singular covariance, detectors dominated by sector/year
confounds).

Deliverables (under ``outputs/scores/<country>/phase2_dependence/``):

- ``dependence.json`` — shrinkage intensity ``λ``, ``K_eff``, Pearson/Spearman
  matrices, PCA explained-variance and singular-flag.
- ``cov_ledoit_wolf.parquet`` — production ``Σ̂_LW``.
- ``cov_mcd.parquet`` — robust MCD covariance (sensitivity check).
- ``pearson.parquet`` / ``spearman.parquet`` — raw correlation matrices.
- ``partial_correlations.parquet`` — partial correlations net of configured
  control columns (framework §4.2).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA

from src.engine.zscores import load_zscores
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DependenceOutcome:
    """Return value of :func:`run_phase2`."""

    summary: dict
    ledoit_wolf: pd.DataFrame
    mcd: pd.DataFrame | None
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    partial: pd.DataFrame
    paths: dict[str, Path]


def _complete_rows(
    zscores_c: pd.DataFrame,
    detectors: tuple[str, ...],
) -> pd.DataFrame:
    """Rows of ``C`` where every detector produced a finite score.

    Σ̂ estimation requires complete observations — partial rows would
    collapse the effective sample size asymmetrically across entries of the
    matrix.
    """
    block = zscores_c.loc[:, list(detectors)]
    mask = np.isfinite(block).all(axis=1)
    return block.loc[mask]


def _wrap_matrix(mat: np.ndarray, detectors: tuple[str, ...]) -> pd.DataFrame:
    """Wrap a raw numpy matrix with detector labels on both axes."""
    return pd.DataFrame(mat, index=list(detectors), columns=list(detectors))


def _pearson_spearman(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson + Spearman correlation matrices on complete rows."""
    return df.corr(method="pearson"), df.corr(method="spearman")


def _ledoit_wolf(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Shrinkage-regularised covariance [Ledoit & Wolf 2004]."""
    estimator = LedoitWolf().fit(df.to_numpy(dtype=float))
    cov = _wrap_matrix(estimator.covariance_, tuple(df.columns))
    return cov, float(estimator.shrinkage_)


def _mcd(df: pd.DataFrame, support_fraction: float | None) -> pd.DataFrame | None:
    """Robust covariance via Minimum Covariance Determinant.

    Returns ``None`` if sklearn rejects the configuration (typical for
    tiny samples — MCD needs ``n > p + 1``).
    """
    try:
        estimator = MinCovDet(support_fraction=support_fraction).fit(
            df.to_numpy(dtype=float)
        )
    except ValueError as err:
        log.warning("phase2: MCD unavailable (%s); skipping robust Σ̂.", err)
        return None
    return _wrap_matrix(estimator.covariance_, tuple(df.columns))


def _pca_diagnostics(df: pd.DataFrame, variance_cutoff: float) -> dict:
    """PCA explained-variance summary + effective dimensionality.

    ``K_eff`` is the smallest number of principal components whose
    cumulative explained-variance share exceeds ``variance_cutoff``. It is
    the working definition of "how many independent directions does the
    detector battery actually produce" (framework §4.2).
    """
    pca = PCA().fit(df.to_numpy(dtype=float))
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(explained)
    k_eff = int(np.searchsorted(cumulative, variance_cutoff) + 1)
    k_eff = min(k_eff, len(explained))
    return {
        "explained_variance_ratio": [float(v) for v in explained],
        "cumulative_variance": [float(v) for v in cumulative],
        "k_eff": k_eff,
        "variance_cutoff": variance_cutoff,
    }


def _partial_correlation_matrix(
    zscores: pd.DataFrame,
    controls: pd.DataFrame,
    detectors: tuple[str, ...],
) -> pd.DataFrame:
    """Partial Pearson correlations among detectors, net of controls.

    For each detector pair we regress both sides on the control matrix
    (OLS with intercept) and correlate the residuals. Non-finite rows are
    dropped globally so the control design is identical across pairs.
    """
    data = pd.concat([zscores.loc[:, list(detectors)], controls], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return pd.DataFrame(index=list(detectors), columns=list(detectors), dtype=float)

    z_block = data.loc[:, list(detectors)].to_numpy(dtype=float)
    x = data.drop(columns=list(detectors))
    # One-hot encode non-numeric controls (sector) and keep numeric ones
    # (year) as-is. Add an intercept column.
    x_design = pd.get_dummies(x, drop_first=True).astype(float)
    x_design.insert(0, "__intercept__", 1.0)
    x_np = x_design.to_numpy(dtype=float)

    # Residuals per detector: r_j = z_j - X β_j with β_j from lstsq.
    beta, *_ = np.linalg.lstsq(x_np, z_block, rcond=None)
    residuals = z_block - x_np @ beta
    residual_df = pd.DataFrame(residuals, columns=list(detectors))
    return residual_df.corr(method="pearson")


def run_phase2(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    zscores: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> DependenceOutcome:
    """Estimate ``Σ̂`` and dependence diagnostics on reference sample ``C``.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0`.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        zscores: Optional pre-computed z-score frame. When ``None`` the
            cache written by :func:`compute_zscores` is loaded.
        write_outputs: If ``True``, persist matrices + JSON summary under
            ``settings.output_layout.phase2_dir()``.

    Returns:
        A :class:`DependenceOutcome` wrapping the production Σ̂_LW, the
        MCD robust Σ̂, correlation matrices, partial correlations, JSON
        summary and resolved paths.

    Raises:
        KeyError: If the panel lacks ``_split``.
        ValueError: If ``C`` has fewer complete rows than
            :attr:`DependenceSettings.min_observations`.
    """
    settings = settings or AnalysisSettings()
    if "_split" not in panel.columns:
        raise KeyError("phase2: panel lacks `_split` — run Phase 0 first.")

    zscores = zscores if zscores is not None else load_zscores(settings)
    if len(zscores) != len(panel):
        raise ValueError(
            f"phase2: zscores length {len(zscores)} != panel length "
            f"{len(panel)}; re-run `compute_zscores`."
        )
    zscores = zscores.copy()
    zscores.index = panel.index

    configured_detectors = tuple(settings.zscores.active_detectors())
    mask_c = panel["_split"] == SPLIT_LABEL_INCLUDED
    zscores_c = zscores.loc[mask_c]

    # A detector with zero finite values on ``C`` cannot participate in Σ̂
    # estimation (it would force "complete rows" to zero). Drop it with a
    # warning — Phase 4 can still report which detectors were excluded.
    dropped_detectors: list[str] = []
    detectors: tuple[str, ...] = tuple(
        d for d in configured_detectors
        if np.isfinite(zscores_c[d].to_numpy(dtype=float)).any()
    )
    for d in configured_detectors:
        if d not in detectors:
            dropped_detectors.append(d)
            log.warning(
                "phase2: detector %s has zero finite scores on C — excluded from Σ̂.",
                d,
            )
    if not detectors:
        raise ValueError(
            "phase2: no detector produced finite scores on C; Σ̂ not estimable."
        )

    complete = _complete_rows(zscores_c, detectors)
    if len(complete) < settings.dependence.min_observations:
        raise ValueError(
            f"phase2: only {len(complete)} complete z-vectors in C "
            f"(< {settings.dependence.min_observations}); Σ̂ not estimable."
        )

    pearson, spearman = _pearson_spearman(complete)
    lw_cov, lw_shrinkage = _ledoit_wolf(complete)
    mcd_cov = _mcd(complete, settings.dependence.mcd_support_fraction)
    pca_info = _pca_diagnostics(complete, settings.dependence.pca_variance_cutoff)

    control_cols = [
        c for c in settings.dependence.partial_corr_controls if c in panel.columns
    ]
    missing_controls = [
        c for c in settings.dependence.partial_corr_controls if c not in panel.columns
    ]
    controls = panel.loc[mask_c, control_cols] if control_cols else pd.DataFrame(index=zscores_c.index)
    partial = _partial_correlation_matrix(zscores_c, controls, detectors)

    k_eff = pca_info["k_eff"]
    k_eff_warning = k_eff < settings.dependence.k_eff_warn_below

    summary: dict = {
        "c_rows": int(mask_c.sum()),
        "complete_rows": int(len(complete)),
        "ledoit_wolf_shrinkage": lw_shrinkage,
        "shrinkage_warning": lw_shrinkage < settings.dependence.shrinkage_block_if_lt,
        "pca": pca_info,
        "k_eff_warning": bool(k_eff_warning),
        "partial_correlation_controls_used": control_cols,
        "partial_correlation_controls_missing": missing_controls,
        "detectors": list(detectors),
        "dropped_detectors_all_nan": dropped_detectors,
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase2_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["json"] = out_dir / "dependence.json"
        paths["ledoit_wolf"] = out_dir / "cov_ledoit_wolf.parquet"
        paths["pearson"] = out_dir / "pearson.parquet"
        paths["spearman"] = out_dir / "spearman.parquet"
        paths["partial"] = out_dir / "partial_correlations.parquet"
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        lw_cov.to_parquet(paths["ledoit_wolf"])
        pearson.to_parquet(paths["pearson"])
        spearman.to_parquet(paths["spearman"])
        partial.to_parquet(paths["partial"])
        if mcd_cov is not None:
            paths["mcd"] = out_dir / "cov_mcd.parquet"
            mcd_cov.to_parquet(paths["mcd"])
        log.info("phase2: wrote %d deliverables to %s", len(paths), out_dir)

    return DependenceOutcome(
        summary=summary,
        ledoit_wolf=lw_cov,
        mcd=mcd_cov,
        pearson=pearson,
        spearman=spearman,
        partial=partial,
        paths=paths,
    )
