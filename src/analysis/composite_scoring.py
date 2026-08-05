"""Phase 4 — composites on full panel + joint bootstrap p-values.

Ingests the z-score cache (from ``compute_zscores``) and Σ̂_LW (from Phase 2),
then:

1. Picks the active detector set — those that produced any finite score on
   ``C``. Detectors dropped by Phase 2 for being all-NaN (e.g. z1 under the
   current Benford floor) are excluded here too.
2. Builds the weight vector according to
   :attr:`CompositeSettings.weighting_mode`.
3. Computes the base per-row composites on the full panel, plus the
   ``softmax`` and ``orthogonal`` score extensions used in reporting.
4. Runs one or two joint bootstraps (parametric and/or non-parametric)
   producing the simultaneous null distributions.
5. Converts observed composites to bootstrap p-values via searchsorted.
6. Writes ``composites_panel.parquet`` (per-row composites + p-values),
   ``null_distributions.parquet`` (quantile summaries of the null), and
   a JSON report of the Phase.

Concordance of top-N rankings across composites is emitted as a small
auxiliary CSV.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.bootstrap import (
    BootstrapNullDistributions,
    run_non_parametric_bootstrap,
    run_parametric_bootstrap,
    upper_tail_pvalue,
)
from src.engine.composites import (
    active_set_indicator,
    breadth,
    iut_statistic,
    mahalanobis_squared,
    renormalised_truncated_sum,
    truncated_sum,
)
from src.engine.detector_composite import orthogonal_softmax, softmax_active
from src.engine.zscores import load_zscores
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)


# Panel column names — exposed so downstream phases filter without guessing.
COL_Z_PLUS: str = "z_plus"
COL_Z_PLUS_RENORM: str = "z_plus_renorm"
COL_BREADTH: str = "breadth"
COL_Z_MAHALANOBIS: str = "z_mahalanobis_sq"
COL_T_IUT: str = "t_iut"
COL_Z_PLUS_SOFTMAX: str = "z_plus_softmax"
COL_Z_PLUS_ORTH: str = "z_plus_orth"

COL_P_Z_PLUS: str = "p_z_plus"
COL_P_Z_PLUS_RENORM: str = "p_z_plus_renorm"
COL_P_BREADTH: str = "p_breadth"
COL_P_Z_MAHALANOBIS: str = "p_z_mahalanobis_sq"
COL_P_T_IUT: str = "p_t_iut"
COL_P_Z_PLUS_SOFTMAX: str = "p_z_plus_softmax"
COL_P_Z_PLUS_ORTH: str = "p_z_plus_orth"


@dataclass(frozen=True)
class Phase4Outcome:
    """Return value of :func:`run_phase4`."""

    composites: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Detector & weight handling
# ─────────────────────────────────────────────────────────────────────────────
def _active_detectors(
    zscores: pd.DataFrame,
    panel: pd.DataFrame,
    configured: tuple[str, ...],
) -> tuple[tuple[str, ...], list[str]]:
    """Pick detectors with at least one finite score on ``C``.

    Returns ``(active, dropped)``. ``active`` preserves the configured
    order so every downstream matrix uses the same column layout.
    """
    mask_c = panel["_split"] == SPLIT_LABEL_INCLUDED
    active: list[str] = []
    dropped: list[str] = []
    for name in configured:
        if name not in zscores.columns:
            dropped.append(name)
            continue
        has_any = np.isfinite(zscores.loc[mask_c, name].to_numpy(dtype=float)).any()
        (active if has_any else dropped).append(name)
    return tuple(active), dropped


def _build_weights(
    active: tuple[str, ...],
    settings: AnalysisSettings,
) -> np.ndarray:
    """Return a ``(k,)`` weight vector for the active detectors."""
    k = len(active)
    mode = settings.composites.weighting_mode
    if mode == "uniform":
        if k == 0:
            return np.zeros(0, dtype=float)
        return np.full(k, 1.0 / k, dtype=float)
    raise ValueError(f"unknown weighting_mode {mode!r} — extend _build_weights.")


def _load_sigma(path: Path, active: tuple[str, ...]) -> np.ndarray:
    """Load Σ̂_LW from Phase 2 parquet and align to active detectors."""
    cov = pd.read_parquet(path)
    missing = [c for c in active if c not in cov.columns]
    if missing:
        raise KeyError(
            f"phase4: active detector(s) {missing} missing from Σ̂_LW — "
            f"re-run Phase 2 after detector-set changes."
        )
    return cov.loc[list(active), list(active)].to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Composite columns on the panel
# ─────────────────────────────────────────────────────────────────────────────
def _composite_columns(
    z_matrix: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> dict[str, np.ndarray]:
    """Compute the base composites plus reporting extensions for the full panel."""
    comp = settings.composites
    finite_mask = np.isfinite(z_matrix)
    z_fill = np.where(finite_mask, z_matrix, 0.0)
    return {
        COL_Z_PLUS: truncated_sum(z_matrix, weights),
        COL_Z_PLUS_RENORM: renormalised_truncated_sum(
            z_matrix, weights,
            threshold=comp.active_set_threshold,
            min_active=comp.min_active_for_renorm,
        ),
        COL_BREADTH: breadth(z_matrix, threshold=comp.active_set_threshold),
        COL_Z_MAHALANOBIS: mahalanobis_squared(z_matrix, sigma),
        COL_T_IUT: iut_statistic(z_matrix, min_active=comp.min_active_for_iut),
        COL_Z_PLUS_SOFTMAX: softmax_active(
            z_fill,
            finite_mask,
            weights,
            gamma=comp.softmax_gamma,
        ),
        COL_Z_PLUS_ORTH: orthogonal_softmax(
            z_fill,
            finite_mask,
            weights,
            gamma=comp.softmax_gamma,
            Sigma=sigma,
        ),
    }


def _attach_pvalues(
    comp: dict[str, np.ndarray],
    null: BootstrapNullDistributions,
) -> dict[str, np.ndarray]:
    """Build a mapping ``p_<composite>`` → array of p-values."""
    B = null.n_replicates
    return {
        COL_P_Z_PLUS: upper_tail_pvalue(comp[COL_Z_PLUS], null.z_plus_sorted, B),
        COL_P_Z_PLUS_RENORM: upper_tail_pvalue(comp[COL_Z_PLUS_RENORM], null.z_plus_renorm_sorted, B),
        COL_P_BREADTH: upper_tail_pvalue(comp[COL_BREADTH], null.breadth_sorted, B),
        COL_P_Z_MAHALANOBIS: upper_tail_pvalue(comp[COL_Z_MAHALANOBIS], null.z_mahalanobis_sorted, B),
        COL_P_T_IUT: upper_tail_pvalue(comp[COL_T_IUT], null.t_iut_sorted, B),
        COL_P_Z_PLUS_SOFTMAX: upper_tail_pvalue(comp[COL_Z_PLUS_SOFTMAX], null.z_plus_softmax_sorted, B),
        COL_P_Z_PLUS_ORTH: upper_tail_pvalue(comp[COL_Z_PLUS_ORTH], null.z_plus_orth_sorted, B),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _null_distribution_quantiles(
    null: BootstrapNullDistributions,
    probs: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.999),
) -> pd.DataFrame:
    """Summary quantiles per composite null — for downstream plotting."""
    arrays = {
        COL_Z_PLUS: null.z_plus_sorted,
        COL_Z_PLUS_RENORM: null.z_plus_renorm_sorted,
        COL_BREADTH: null.breadth_sorted,
        COL_Z_MAHALANOBIS: null.z_mahalanobis_sorted,
        COL_T_IUT: null.t_iut_sorted,
        COL_Z_PLUS_SOFTMAX: null.z_plus_softmax_sorted,
        COL_Z_PLUS_ORTH: null.z_plus_orth_sorted,
    }
    rows = []
    for name, arr in arrays.items():
        if arr.size == 0:
            continue
        q = np.quantile(arr, probs)
        row = {"composite": name, "variant": null.variant, "n_finite": int(arr.size)}
        row.update({f"q{int(p * 1000)}": float(qv) for p, qv in zip(probs, q)})
        rows.append(row)
    return pd.DataFrame(rows)


def _top_n_concordance(
    composites: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Jaccard similarity of each composite's top-N row set.

    Top-N is taken on the *observed composite value* (not p-value), which
    gives an ordering that ``NaN`` rows do not contaminate. Lower = more
    disagreement between composites.
    """
    composite_cols = [
        COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_BREADTH, COL_Z_MAHALANOBIS, COL_T_IUT,
        COL_Z_PLUS_SOFTMAX, COL_Z_PLUS_ORTH,
    ]
    sets = {
        c: set(composites[c].nlargest(top_n).index)
        for c in composite_cols
        if c in composites.columns
    }
    rows = []
    for a in sets:
        for b in sets:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            rows.append(
                {"a": a, "b": b, "jaccard": (inter / union) if union else 0.0}
            )
    return pd.DataFrame(rows)


def _pvalue_calibration_on_c(
    composites: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict:
    """KS-style uniformity check of ``C``-row p-values.

    Under ``H_0`` the bootstrap p-values on ``C`` should be roughly
    uniform. A large deviation means the bootstrap null does not match
    the empirical null — useful as a red flag.
    """
    from scipy.stats import kstest

    mask_c = panel["_split"] == SPLIT_LABEL_INCLUDED
    out: dict[str, dict] = {}
    for col in [COL_P_Z_PLUS, COL_P_Z_PLUS_RENORM, COL_P_BREADTH,
                COL_P_Z_MAHALANOBIS, COL_P_T_IUT,
                COL_P_Z_PLUS_SOFTMAX, COL_P_Z_PLUS_ORTH]:
        vals = composites.loc[mask_c, col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            out[col] = {"n": 0, "ks_stat": None, "ks_pvalue": None}
            continue
        ks = kstest(vals, "uniform")
        out[col] = {
            "n": int(vals.size),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_phase4(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    zscores: pd.DataFrame | None = None,
    sigma_override: pd.DataFrame | np.ndarray | None = None,
    null_override: BootstrapNullDistributions | None = None,
    write_outputs: bool = True,
) -> Phase4Outcome:
    """Compute composites + bootstrap p-values for the full panel.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0`.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        zscores: Optional pre-computed z-scores. When ``None`` the Phase 2
            cache is loaded from disk.
        sigma_override: Optional in-memory Σ̂_LW aligned either as a labelled
            DataFrame or raw ndarray. When provided, Phase 4 does not reload
            the covariance from the Phase 2 output path.
        write_outputs: If ``True``, persist per-row composites, null-
            distribution quantiles, concordance table and JSON summary
            under ``settings.output_layout.phase4_dir()``.

    Returns:
        A :class:`Phase4Outcome` with the composites frame, summary and
        paths.

    Raises:
        KeyError: If ``_split`` is missing or Σ̂_LW is missing required
            detectors.
        ValueError: If no detector produced finite scores on ``C``.
    """
    settings = settings or AnalysisSettings()
    if "_split" not in panel.columns:
        raise KeyError("phase4: panel lacks `_split` — run Phase 0 first.")

    zscores = zscores if zscores is not None else load_zscores(settings)
    if len(zscores) != len(panel):
        raise ValueError(
            f"phase4: zscores length {len(zscores)} != panel length "
            f"{len(panel)}; rebuild the zscores cache."
        )
    zscores = zscores.copy()
    zscores.index = panel.index

    configured = tuple(settings.zscores.active_detectors())
    active, dropped = _active_detectors(zscores, panel, configured)
    if not active:
        raise ValueError("phase4: no detector has finite scores on C.")
    if dropped:
        log.warning("phase4: dropped detectors with no C coverage: %s", dropped)

    weights = _build_weights(active, settings)
    if sigma_override is None:
        sigma_path = settings.output_layout.phase2_dir() / "cov_ledoit_wolf.parquet"
        sigma = _load_sigma(sigma_path, active)
    elif isinstance(sigma_override, pd.DataFrame):
        missing = [c for c in active if c not in sigma_override.columns or c not in sigma_override.index]
        if missing:
            raise KeyError(
                f"phase4: active detector(s) {missing} missing from in-memory Σ̂_LW."
            )
        sigma = sigma_override.loc[list(active), list(active)].to_numpy(dtype=float)
    else:
        sigma = np.asarray(sigma_override, dtype=float)
        if sigma.shape != (len(active), len(active)):
            raise ValueError(
                f"phase4: sigma_override has shape {sigma.shape}, expected {(len(active), len(active))}."
            )

    z_matrix = zscores.loc[:, list(active)].to_numpy(dtype=float)
    composite_cols = _composite_columns(z_matrix, sigma, weights, settings)

    # Build reference for non-parametric bootstrap (complete rows on C).
    mask_c = panel["_split"] == SPLIT_LABEL_INCLUDED
    z_c = z_matrix[mask_c.to_numpy()]
    complete_c = z_c[np.isfinite(z_c).all(axis=1)]

    if null_override is not None:
        # Fast path: reuse a pre-computed null — skip bootstrap entirely.
        primary_null = null_override
        primary_variant = "override"
        nulls: dict[str, BootstrapNullDistributions] = {}
    else:
        nulls = {}
        if settings.bootstrap.parametric:
            log.info("phase4: running parametric bootstrap (B=%d)", settings.bootstrap.n_replicates)
            nulls["parametric"] = run_parametric_bootstrap(
                sigma=sigma, weights=weights, settings=settings,
            )
        if settings.bootstrap.non_parametric:
            if complete_c.shape[0] < settings.dependence.min_observations:
                log.warning(
                    "phase4: non-parametric bootstrap skipped — only %d complete rows on C (< %d).",
                    int(complete_c.shape[0]), settings.dependence.min_observations,
                )
            else:
                log.info(
                    "phase4: running non-parametric bootstrap on %d C rows (B=%d)",
                    int(complete_c.shape[0]), settings.bootstrap.n_replicates,
                )
                nulls["non_parametric"] = run_non_parametric_bootstrap(
                    reference_rows=complete_c,
                    sigma=sigma, weights=weights, settings=settings,
                )

        if not nulls:
            raise ValueError("phase4: no bootstrap variant enabled / feasible.")

        primary_variant = settings.bootstrap.primary_mode
        if primary_variant not in nulls:
            fallback = next(iter(nulls))
            log.warning(
                "phase4: primary bootstrap variant %r unavailable; using %r.",
                primary_variant, fallback,
            )
            primary_variant = fallback
        primary_null = nulls[primary_variant]

    pvalue_cols = _attach_pvalues(composite_cols, primary_null)

    schema = settings.panel_schema
    composites = pd.DataFrame(
        {
            schema.firm_id: panel[schema.firm_id].values,
            schema.quarter: panel[schema.quarter].values,
        },
        index=panel.index,
    )
    for name, arr in composite_cols.items():
        composites[name] = arr
    for name, arr in pvalue_cols.items():
        composites[name] = arr

    # Diagnostics — skipped on the fast path (null_override provided).
    if nulls:
        null_quantiles = pd.concat(
            [_null_distribution_quantiles(null) for null in nulls.values()],
            ignore_index=True,
        )
        concordance = _top_n_concordance(composites, top_n=100)
        calibration = _pvalue_calibration_on_c(composites, panel)
    else:
        null_quantiles = pd.DataFrame()
        concordance = pd.DataFrame()
        calibration = {}

    summary: dict = {
        "active_detectors": list(active),
        "dropped_detectors": dropped,
        "weights": {d: float(w) for d, w in zip(active, weights)},
        "weighting_mode": settings.composites.weighting_mode,
        "bootstrap": {
            "primary_mode": primary_variant,
            "n_replicates": settings.bootstrap.n_replicates,
            "seed": settings.bootstrap.random_seed,
            "variants_run": list(nulls.keys()),
            "n_complete_c_rows": int(complete_c.shape[0]),
        },
        "pvalue_calibration_on_c": calibration,
        "rows": {
            "total": int(len(panel)),
            "finite_per_composite": {
                c: int(np.isfinite(composites[c]).sum())
                for c in (COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_BREADTH,
                          COL_Z_MAHALANOBIS, COL_T_IUT,
                          COL_Z_PLUS_SOFTMAX, COL_Z_PLUS_ORTH)
            },
        },
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase4_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["composites"] = out_dir / "composites_panel.parquet"
        paths["null_quantiles"] = out_dir / "null_distribution_quantiles.csv"
        paths["concordance"] = out_dir / "top100_concordance.csv"
        paths["json"] = out_dir / "phase4_composites.json"
        composites.to_parquet(paths["composites"], index=False)
        null_quantiles.to_csv(paths["null_quantiles"], index=False)
        concordance.to_csv(paths["concordance"], index=False)
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("phase4: wrote %d deliverables to %s", len(paths), out_dir)

    return Phase4Outcome(composites=composites, summary=summary, paths=paths)
