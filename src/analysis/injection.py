"""Phase 5 — injection-based power analysis (main research contribution).

Validates the framework's detection claim:

    "theoretical bounds on detection sensitivity as a function of
     manipulation magnitude δ and sample size n" — paper abstract.

First-ship approach — **z-score-level injection**. For each
(archetype, δ) cell, Phase 5:

1. Samples ``N_inject`` honest rows from ``C``.
2. Shifts the z-scores of the archetype's ``affected_detectors`` by ``+δ``
   on those rows (z-score units; ``δ = 1`` ≡ a one-σ anomaly on the
   affected dimensions).
3. Recomputes the five composites on the perturbed vectors.
4. Looks up bootstrap p-values in the **baseline** null distributions
   (representing ``H_0``) and measures the detection rate — the fraction
   of injected rows with ``p < α``.

This validates framework §5.5 directly: with the ``systematic``
archetype (uniform shift on all detectors), the empirical ``T_IUT``
detection rate should match the theoretical curve
``π(δ) = Φ(δ − c_α)^k``. Raw-data-level injection (perturb Compustat →
rebuild panel → rerun detectors) is a follow-up extension.

Deliverables (under ``outputs/scores/<country>/phase5_injection/``):

- ``power_curves.csv``       — detection rate per (archetype, δ,
  composite, α).
- ``detector_specificity.csv`` — fraction of rows where each detector
  alone flags at ``α = primary_alpha_for_mde``.
- ``minimum_detectable_effect.csv`` — smallest δ such that detection
  rate ≥ ``power_target`` at the primary α, per (archetype, composite).
- ``theoretical_empirical_comparison.csv`` — framework §5.5 benchmark
  vs empirical ``T_IUT`` detection rate on the ``systematic`` archetype.
- ``phase5_injection.json``  — summary + timings.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.engine.bootstrap import (
    BootstrapNullDistributions,
    run_non_parametric_bootstrap,
    run_parametric_bootstrap,
    upper_tail_pvalue,
)
from src.engine.composites import (
    breadth,
    iut_statistic,
    mahalanobis_squared,
    renormalised_truncated_sum,
    truncated_sum,
)
from src.engine.detector_composite import orthogonal_softmax, softmax_active
from src.engine.zscores import load_zscores
from src.common.config import AnalysisSettings, ArchetypeSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED
from src.analysis.composite_scoring import (
    COL_BREADTH, COL_T_IUT, COL_Z_MAHALANOBIS, COL_Z_PLUS, COL_Z_PLUS_ORTH,
    COL_Z_PLUS_RENORM, COL_Z_PLUS_SOFTMAX,
    _build_weights,
    _load_sigma,
)

log = logging.getLogger(__name__)


COMPOSITE_COLUMNS: tuple[str, ...] = (
    COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_BREADTH, COL_Z_MAHALANOBIS, COL_T_IUT,
    COL_Z_PLUS_SOFTMAX, COL_Z_PLUS_ORTH,
)


@dataclass(frozen=True)
class Phase5Outcome:
    """Return value of :func:`run_phase5`."""

    power_curves: pd.DataFrame
    detector_specificity: pd.DataFrame
    mde: pd.DataFrame
    theoretical_vs_empirical: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Injection
# ─────────────────────────────────────────────────────────────────────────────
def _inject_zscores(
    base_rows: np.ndarray,
    affected_columns: list[int],
    delta: float,
) -> np.ndarray:
    """Return a copy of ``base_rows`` with ``+δ`` added on ``affected_columns``.

    NaN entries stay NaN — a detector that could not score the honest row
    cannot score the manipulated one either. Using a pure copy keeps the
    injection deterministic and side-effect-free, which matters when the
    caller sweeps thousands of (archetype, δ, replicate) cells.
    """
    injected = base_rows.copy()
    if affected_columns:
        injected[:, affected_columns] = injected[:, affected_columns] + delta
    return injected


def _merge_rewrite_map(settings: AnalysisSettings) -> dict[str, str]:
    """Build a ``raw_input → merge_output`` map for archetype rewriting.

    Archetypes declare affected detectors by their raw names (``z5``, ``z7``).
    When merges are active, those names no longer exist in the post-merge
    battery — they must be rewritten to the merge output (``z57``) so the
    injection still hits the correct column.
    """
    mapping: dict[str, str] = {}
    for rule in settings.zscores.merge_rules:
        if not rule.drop_inputs:
            continue
        for raw in rule.inputs:
            mapping[raw] = rule.output_name
    return mapping


def _resolve_affected_indices(
    archetype: ArchetypeSettings,
    active: tuple[str, ...],
    only_active: bool,
    merge_map: dict[str, str] | None = None,
) -> list[int]:
    """Map detector names in an archetype to column indices in ``active``.

    ``merge_map`` rewrites raw detector names to their merge output, so a
    legacy archetype referencing e.g. ``z7`` still injects into ``z57`` when
    the z5/z7 merge is active. Duplicate indices (multiple raw inputs
    mapping to the same output) are de-duplicated so a single column is not
    perturbed twice on the same row.
    """
    merge_map = merge_map or {}
    seen: set[int] = set()
    indices: list[int] = []
    for raw in archetype.affected_detectors:
        name = merge_map.get(raw, raw)
        if name not in active:
            if only_active:
                continue
            raise KeyError(
                f"archetype {archetype.name!r} targets unknown detector "
                f"{name!r} (active={active})."
            )
        col = active.index(name)
        if col in seen:
            continue
        seen.add(col)
        indices.append(col)
    return indices


def _composites_matrix(
    z_rows: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> dict[str, np.ndarray]:
    """Compute all composites on a raw z-matrix (no dataframe wrap)."""
    comp = settings.composites
    finite_mask = np.isfinite(z_rows)
    z_fill = np.where(finite_mask, z_rows, 0.0)
    return {
        COL_Z_PLUS: truncated_sum(z_rows, weights),
        COL_Z_PLUS_RENORM: renormalised_truncated_sum(
            z_rows, weights,
            threshold=comp.active_set_threshold,
            min_active=comp.min_active_for_renorm,
        ),
        COL_BREADTH: breadth(z_rows, threshold=comp.active_set_threshold),
        COL_Z_MAHALANOBIS: mahalanobis_squared(z_rows, sigma),
        COL_T_IUT: iut_statistic(z_rows, min_active=comp.min_active_for_iut),
        COL_Z_PLUS_SOFTMAX: softmax_active(z_fill, finite_mask, weights, gamma=comp.softmax_gamma),
        COL_Z_PLUS_ORTH: orthogonal_softmax(z_fill, finite_mask, weights, gamma=comp.softmax_gamma, Sigma=sigma),
    }


def _pvalues_for_composites(
    composites: dict[str, np.ndarray],
    null: BootstrapNullDistributions,
) -> dict[str, np.ndarray]:
    """Bootstrap p-values for each composite, using ``null`` as reference."""
    B = null.n_replicates
    return {
        COL_Z_PLUS: upper_tail_pvalue(composites[COL_Z_PLUS], null.z_plus_sorted, B),
        COL_Z_PLUS_RENORM: upper_tail_pvalue(composites[COL_Z_PLUS_RENORM], null.z_plus_renorm_sorted, B),
        COL_BREADTH: upper_tail_pvalue(composites[COL_BREADTH], null.breadth_sorted, B),
        COL_Z_MAHALANOBIS: upper_tail_pvalue(composites[COL_Z_MAHALANOBIS], null.z_mahalanobis_sorted, B),
        COL_T_IUT: upper_tail_pvalue(composites[COL_T_IUT], null.t_iut_sorted, B),
        COL_Z_PLUS_SOFTMAX: upper_tail_pvalue(composites[COL_Z_PLUS_SOFTMAX], null.z_plus_softmax_sorted, B),
        COL_Z_PLUS_ORTH: upper_tail_pvalue(composites[COL_Z_PLUS_ORTH], null.z_plus_orth_sorted, B),
    }


def _per_detector_pvalues(z_rows: np.ndarray) -> np.ndarray:
    """Univariate p-values ``P(Z ≥ z)`` per detector — Φ upper-tail.

    Used only for the ``detector_specificity`` report; the composite
    p-values drive the paper-level power claim.
    """
    p = 1.0 - norm.cdf(z_rows)
    return p


def _theoretical_iut_power(
    deltas: tuple[float, ...],
    alpha: float,
    k: int,
) -> np.ndarray:
    """Framework §5.5 power formula ``π(δ) = Φ(δ − c_α)^k``.

    ``c_α`` is the one-sided critical value from the null CDF
    ``P(T_IUT > c_α) = α`` under joint independence with ``z_j ∼ N(0, 1)``;
    ``P_{H_0}(T_IUT ≤ t) = 1 − (1 − Φ(t))^k`` gives ``c_α = Φ^{-1}(1 − α^{1/k})``.
    """
    c_alpha = norm.ppf(1.0 - alpha ** (1.0 / k))
    return np.array([norm.cdf(delta - c_alpha) ** k for delta in deltas])


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def _active_detectors_for_phase(
    zscores: pd.DataFrame,
    mask_c: np.ndarray,
    configured: tuple[str, ...],
) -> tuple[tuple[str, ...], list[str]]:
    active, dropped = [], []
    for name in configured:
        if name not in zscores.columns:
            dropped.append(name)
            continue
        vals = zscores[name].to_numpy(dtype=float)
        (active if np.isfinite(vals[mask_c]).any() else dropped).append(name)
    return tuple(active), dropped


def _rebuild_null(
    complete_c: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> BootstrapNullDistributions:
    """Rebuild the primary bootstrap null with the Phase 4 settings.

    Phase 5 re-computes the null rather than reading Phase 4's cached
    arrays (which are only saved as quantile summaries). The bootstrap
    is deterministic for a fixed seed so this is stable across runs.
    """
    mode = settings.bootstrap.primary_mode
    if mode == "non_parametric":
        if complete_c.shape[0] < settings.dependence.min_observations:
            raise ValueError(
                f"phase5: not enough complete C rows "
                f"({complete_c.shape[0]}) for non-parametric bootstrap."
            )
        return run_non_parametric_bootstrap(
            reference_rows=complete_c,
            sigma=sigma,
            weights=weights,
            settings=settings,
        )
    if mode == "parametric":
        return run_parametric_bootstrap(
            sigma=sigma, weights=weights, settings=settings,
        )
    raise ValueError(f"phase5: unknown primary_mode {mode!r}.")


def _sample_base_rows(
    complete_c: np.ndarray,
    n_inject: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``n_inject`` rows (with replacement) from the complete-on-C pool.

    Sampling with replacement lets ``n_inject`` exceed the pool size; at
    the current panel it's 500 vs ~5,000 complete rows, so with- or
    without-replacement would be near-identical. With-replacement stays
    robust to smaller reference samples on other country panels.
    """
    idx = rng.integers(low=0, high=complete_c.shape[0], size=n_inject)
    return complete_c[idx]


def _power_row(
    archetype: str,
    delta: float,
    composite: str,
    alpha: float,
    pvals: np.ndarray,
) -> dict:
    """One row of the power-curves long-form table."""
    finite = np.isfinite(pvals)
    n = int(finite.sum())
    detection = float((pvals[finite] < alpha).mean()) if n else float("nan")
    return {
        "archetype": archetype,
        "delta": delta,
        "composite": composite,
        "alpha": alpha,
        "n_finite": n,
        "detection_rate": detection,
    }


def _specificity_row(
    archetype: str,
    delta: float,
    detector: str,
    alpha: float,
    pvals: np.ndarray,
) -> dict:
    finite = np.isfinite(pvals)
    n = int(finite.sum())
    return {
        "archetype": archetype,
        "delta": delta,
        "detector": detector,
        "alpha": alpha,
        "n_finite": n,
        "flag_rate": float((pvals[finite] < alpha).mean()) if n else float("nan"),
    }


def _mde_from_power_curve(
    power_curves: pd.DataFrame,
    alpha: float,
    target_power: float,
) -> pd.DataFrame:
    """Smallest δ with ``detection_rate ≥ target_power`` per (archetype, composite)."""
    rows: list[dict] = []
    sub = power_curves[power_curves["alpha"] == alpha]
    for (arch, comp), grp in sub.groupby(["archetype", "composite"]):
        grp_sorted = grp.sort_values("delta")
        ok = grp_sorted[grp_sorted["detection_rate"] >= target_power]
        mde = float(ok["delta"].iloc[0]) if not ok.empty else float("nan")
        rows.append(
            {
                "archetype": arch,
                "composite": comp,
                "alpha": alpha,
                "power_target": target_power,
                "mde_delta": mde,
            }
        )
    return pd.DataFrame(rows)


def _theoretical_table(
    empirical_power: pd.DataFrame,
    settings: AnalysisSettings,
    k: int,
) -> pd.DataFrame:
    """Empirical vs framework §5.5 power for the ``systematic`` archetype.

    The framework derives ``π(δ) = Φ(δ − c_α)^k`` under independent
    ``N(0, 1)`` detectors. Any gap is attributable to the observed
    detector correlations (z5–z7 in particular, see Sprint 2 findings).
    """
    sub = empirical_power[
        (empirical_power["archetype"] == "systematic")
        & (empirical_power["composite"] == COL_T_IUT)
    ]
    rows: list[dict] = []
    for alpha in settings.injection.alpha_grid:
        alpha_sub = sub[sub["alpha"] == alpha].sort_values("delta")
        if alpha_sub.empty:
            continue
        deltas = tuple(alpha_sub["delta"].to_list())
        empirical = alpha_sub["detection_rate"].to_numpy(dtype=float)
        theoretical = _theoretical_iut_power(deltas=deltas, alpha=alpha, k=k)
        for d, e, t in zip(deltas, empirical, theoretical):
            rows.append(
                {
                    "alpha": alpha,
                    "delta": d,
                    "empirical_t_iut_power": float(e),
                    "theoretical_t_iut_power": float(t),
                    "gap": float(e - t),
                }
            )
    return pd.DataFrame(rows)


def run_phase5(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    zscores: pd.DataFrame | None = None,
    sigma_override: pd.DataFrame | np.ndarray | None = None,
    write_outputs: bool = True,
) -> Phase5Outcome:
    """Run the full injection sweep and emit power diagnostics.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0`.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        zscores: Optional pre-computed z-score frame.
        sigma_override: Optional in-memory covariance matrix aligned either
            as a labelled DataFrame or raw ndarray. When provided, Phase 5
            does not reload the Ledoit-Wolf covariance from disk.
        write_outputs: If ``True``, persist deliverables under
            ``settings.output_layout.phase5_dir()``.

    Returns:
        A :class:`Phase5Outcome` bundling power curves, detector
        specificity, MDE table, theoretical-vs-empirical comparison, a
        JSON summary and resolved paths.
    """
    settings = settings or AnalysisSettings()
    if "_split" not in panel.columns:
        raise KeyError("phase5: panel lacks `_split` — run Phase 0 first.")

    zscores = zscores if zscores is not None else load_zscores(settings)
    if len(zscores) != len(panel):
        raise ValueError(
            f"phase5: zscores length {len(zscores)} != panel length {len(panel)}."
        )
    zscores = zscores.copy()
    zscores.index = panel.index

    configured = tuple(settings.zscores.active_detectors())
    mask_c = (panel["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()
    active, dropped = _active_detectors_for_phase(zscores, mask_c, configured)
    if not active:
        raise ValueError("phase5: no detector has finite scores on C.")
    if dropped:
        log.warning("phase5: dropped detectors without C coverage: %s", dropped)

    weights = _build_weights(active, settings)
    if sigma_override is None:
        sigma = _load_sigma(
            settings.output_layout.phase2_dir() / "cov_ledoit_wolf.parquet",
            active,
        )
    elif isinstance(sigma_override, pd.DataFrame):
        missing = [c for c in active if c not in sigma_override.columns or c not in sigma_override.index]
        if missing:
            raise KeyError(
                f"phase5: active detector(s) {missing} missing from in-memory covariance."
            )
        sigma = sigma_override.loc[list(active), list(active)].to_numpy(dtype=float)
    else:
        sigma = np.asarray(sigma_override, dtype=float)
        if sigma.shape != (len(active), len(active)):
            raise ValueError(
                f"phase5: sigma_override has shape {sigma.shape}, expected {(len(active), len(active))}."
            )

    z_matrix = zscores.loc[:, list(active)].to_numpy(dtype=float)
    complete_c = z_matrix[mask_c]
    complete_c = complete_c[np.isfinite(complete_c).all(axis=1)]

    null = _rebuild_null(
        complete_c=complete_c, sigma=sigma, weights=weights, settings=settings,
    )

    rng = np.random.default_rng(settings.injection.random_seed)
    power_rows: list[dict] = []
    specificity_rows: list[dict] = []
    inj = settings.injection

    log.info(
        "phase5: %d archetypes × %d δ values × n_inject=%d, primary α=%s, k=%d",
        len(inj.archetypes), len(inj.delta_grid), inj.n_inject,
        inj.alpha_grid, len(active),
    )

    merge_map = _merge_rewrite_map(settings)
    n_total = len(inj.archetypes) * len(inj.delta_grid)
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_total, desc="phase5: injection", unit="cell")
    except ImportError:
        pbar = None

    for arch in inj.archetypes:
        affected = _resolve_affected_indices(
            arch, active, inj.only_active_detectors, merge_map=merge_map,
        )
        if not affected:
            log.warning(
                "phase5: archetype %r has no affected detectors in the active set; skipping.",
                arch.name,
            )
            if pbar:
                pbar.update(len(inj.delta_grid))
            continue
        for delta in inj.delta_grid:
            if pbar:
                pbar.set_postfix(arch=arch.name, delta=delta)
                pbar.update(1)
            base_rows = _sample_base_rows(complete_c, inj.n_inject, rng)
            injected = _inject_zscores(base_rows, affected, delta)
            composites = _composites_matrix(injected, sigma, weights, settings)
            pvals = _pvalues_for_composites(composites, null)

            for composite, alpha in ((c, a) for c in COMPOSITE_COLUMNS for a in inj.alpha_grid):
                power_rows.append(
                    _power_row(arch.name, delta, composite, alpha, pvals[composite])
                )

            per_det_p = _per_detector_pvalues(injected)
            for j, name in enumerate(active):
                for alpha in inj.alpha_grid:
                    specificity_rows.append(
                        _specificity_row(arch.name, delta, name, alpha, per_det_p[:, j])
                    )

    power_curves = pd.DataFrame(power_rows)
    specificity = pd.DataFrame(specificity_rows)
    mde = _mde_from_power_curve(
        power_curves=power_curves,
        alpha=inj.primary_alpha_for_mde,
        target_power=inj.power_target,
    )
    theoretical = _theoretical_table(
        empirical_power=power_curves, settings=settings, k=len(active),
    )

    summary: dict = {
        "active_detectors": list(active),
        "n_inject": inj.n_inject,
        "delta_grid": list(inj.delta_grid),
        "alpha_grid": list(inj.alpha_grid),
        "primary_alpha_for_mde": inj.primary_alpha_for_mde,
        "power_target": inj.power_target,
        "archetypes": [a.name for a in inj.archetypes],
        "bootstrap": {
            "primary_mode": settings.bootstrap.primary_mode,
            "n_replicates": settings.bootstrap.n_replicates,
        },
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase5_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["power_curves"] = out_dir / "power_curves.csv"
        paths["specificity"] = out_dir / "detector_specificity.csv"
        paths["mde"] = out_dir / "minimum_detectable_effect.csv"
        paths["theoretical"] = out_dir / "theoretical_empirical_comparison.csv"
        paths["json"] = out_dir / "phase5_injection.json"
        power_curves.to_csv(paths["power_curves"], index=False)
        specificity.to_csv(paths["specificity"], index=False)
        mde.to_csv(paths["mde"], index=False)
        theoretical.to_csv(paths["theoretical"], index=False)
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("phase5: wrote %d deliverables to %s", len(paths), out_dir)

    return Phase5Outcome(
        power_curves=power_curves,
        detector_specificity=specificity,
        mde=mde,
        theoretical_vs_empirical=theoretical,
        summary=summary,
        paths=paths,
    )
