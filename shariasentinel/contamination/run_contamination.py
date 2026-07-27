"""Contamination study orchestrator.

For each (mechanism, contamination_rate) cell:
1. Generate H₀ from C (bootstrap).
2. Apply the mechanism to produce H₁ contaminated rows.
3. Run the full detector pipeline on the mixed benchmark.
4. Compute composites and p-values.
5. Evaluate AUC, precision, recall against ground-truth labels.

Usage::

    python -m server.scoring.contamination.run_contamination --country mys

Outputs go to ``outputs/scores/<country>/contamination/``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from shariasentinel.scoring.config import AnalysisSettings
from shariasentinel.scoring.compute_zscores import _run_detector, _DETECTOR_DISPATCH
from shariasentinel.scoring.composites import (
    truncated_sum, renormalised_truncated_sum,
    breadth, mahalanobis_squared, iut_statistic,
)
from shariasentinel.scoring.phase4_composites import _build_weights, _load_sigma
from shariasentinel.scoring.bootstrap import run_non_parametric_bootstrap, upper_tail_pvalue
from shariasentinel.scoring.sector_filter import apply_sector_filter, exclusion_suffix

from .config import ContaminationSettings
from .evaluate import compute_auc, evaluate_cell
from .h0_generator import generate_h0
from .mechanisms import MECHANISM_DISPATCH

log = logging.getLogger(__name__)


def _compute_zscores_on_benchmark(
    benchmark: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Run all detectors on the benchmark dataset."""
    active = list(settings.zscores.include_detectors)
    result = pd.DataFrame(index=benchmark.index)
    for name in active:
        log.debug("contamination: running detector %s", name)
        series = _run_detector(name, benchmark, settings)
        result[name] = series.reindex(benchmark.index).astype(float).values

    # Apply merges
    for rule in settings.zscores.merge_rules:
        inputs_present = [c for c in rule.inputs if c in result.columns]
        if len(inputs_present) < 2:
            continue
        mat = result[inputs_present].to_numpy(dtype=float)
        if rule.method == "max":
            merged = np.nanmax(mat, axis=1)
        else:
            merged = np.nanmean(mat, axis=1)
        all_nan = np.isnan(mat).all(axis=1)
        merged = np.where(all_nan, np.nan, merged)
        result[rule.output_name] = merged
        if rule.drop_inputs:
            result = result.drop(columns=[c for c in rule.inputs if c in result.columns])

    return result


def _compute_composites_on_benchmark(
    zscores: pd.DataFrame,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Compute all five composites on the benchmark z-scores."""
    active = tuple(settings.zscores.active_detectors())
    active_present = [d for d in active if d in zscores.columns]
    z_matrix = zscores[active_present].to_numpy(dtype=float)

    comp = settings.composites
    composites = pd.DataFrame(index=zscores.index)
    composites["z_plus"] = truncated_sum(z_matrix, weights)
    composites["z_plus_renorm"] = renormalised_truncated_sum(
        z_matrix, weights,
        threshold=comp.active_set_threshold,
        min_active=comp.min_active_for_renorm,
    )
    composites["breadth"] = breadth(z_matrix, threshold=comp.active_set_threshold)
    composites["z_mahalanobis_sq"] = mahalanobis_squared(z_matrix, sigma)
    composites["t_iut"] = iut_statistic(z_matrix, min_active=comp.min_active_for_iut)
    return composites


def run_contamination_study(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    contam_settings: ContaminationSettings | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Run the full contamination study.

    Args:
        panel: Panel with ``_split`` column (output of Phase 0).
        settings: Scoring pipeline settings.
        contam_settings: Contamination-specific settings.
        write_outputs: Write results to disk.

    Returns:
        Long-form DataFrame with one row per (mechanism, ρ, composite)
        cell, containing AUC and precision/recall at each α.
    """
    settings = settings or AnalysisSettings()
    contam_settings = contam_settings or ContaminationSettings()

    # Apply sector exclusion if configured
    panel = apply_sector_filter(panel, settings)

    # Load Σ̂ and weights from Phase 2
    active = tuple(settings.zscores.active_detectors())
    sigma = _load_sigma(
        settings.output_layout.phase2_dir() / "cov_ledoit_wolf.parquet",
        active,
    )
    weights = _build_weights(active, settings)

    # Build bootstrap null from C (for p-value computation)
    mask_c = (panel["_split"] == "C").to_numpy()
    zscores_cache = pd.read_parquet(
        settings.output_layout.zscores_dir() / settings.zscores.filename
    )
    zscores_cache.index = panel.index
    z_c = zscores_cache.loc[mask_c, list(active)].to_numpy(dtype=float)
    complete_c = z_c[np.isfinite(z_c).all(axis=1)]

    null = run_non_parametric_bootstrap(complete_c, sigma, weights, settings)
    null_distributions = {
        "z_plus": null.z_plus_sorted,
        "z_plus_renorm": null.z_plus_renorm_sorted,
        "z_mahalanobis_sq": null.z_mahalanobis_sorted,
        "t_iut": null.t_iut_sorted,
    }

    # Mechanism dispatch
    mechanisms = {}
    for mech_cfg in [contam_settings.m1, contam_settings.m2,
                     contam_settings.m3, contam_settings.m4,
                     contam_settings.m5]:
        if mech_cfg.enabled:
            mechanisms[mech_cfg.name] = MECHANISM_DISPATCH[mech_cfg.name]

    rng = np.random.default_rng(contam_settings.random_seed)
    all_results: list[dict] = []

    n_total = len(mechanisms) * len(contam_settings.contamination_rates)
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_total, desc="contamination", unit="cell")
    except ImportError:
        pbar = None

    for mech_name, mech_fn in mechanisms.items():
        for rho in contam_settings.contamination_rates:
            if pbar:
                pbar.set_postfix(mechanism=mech_name, rho=f"{rho:.0%}")
                pbar.update(1)
            log.info("contamination: %s, ρ=%.2f", mech_name, rho)

            # 1. Generate H₀
            h0 = generate_h0(panel, contam_settings)

            # 2. Apply mechanism
            benchmark = mech_fn(h0, rho, contam_settings, rng)
            labels = benchmark["y"].to_numpy(dtype=int)

            # 3. Run detectors
            zscores = _compute_zscores_on_benchmark(benchmark, settings)

            # 4. Compute composites
            composites = _compute_composites_on_benchmark(
                zscores, sigma, weights, settings,
            )

            # 5. Evaluate
            cell_results = evaluate_cell(
                composites, labels, null_distributions,
                contam_settings.alpha_grid,
            )
            # Per-detector z-score shift (for figure 3)
            active_det = [d for d in settings.zscores.active_detectors()
                          if d in zscores.columns]
            for det in active_det:
                z_vals = zscores[det].to_numpy(dtype=float)
                mean_contam = np.nanmean(z_vals[labels == 1])
                mean_clean = np.nanmean(z_vals[labels == 0])
                cell_results[f"dz_{det}"] = float(mean_contam - mean_clean)

            cell_results["mechanism"] = mech_name
            cell_results["rho"] = rho
            cell_results["n_total"] = len(benchmark)
            cell_results["n_contaminated"] = int(labels.sum())
            all_results.append(cell_results)

            log.info(
                "  AUC: z_mah=%.3f, t_iut=%.3f, z_plus=%.3f",
                cell_results.get("auc_z_mahalanobis_sq", float("nan")),
                cell_results.get("auc_t_iut", float("nan")),
                cell_results.get("auc_z_plus", float("nan")),
            )

    results_df = pd.DataFrame(all_results)

    if write_outputs:
        out_dir = (
            settings.output_layout.root
            / Path(str(settings.output_layout.scores_relative).format(
                country=settings.output_layout.country_code_lower
            ))
            / "contamination"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_dir / "auc_by_mechanism.csv", index=False)

        # AUC summary table (mechanism × ρ) for the headline composite
        auc_pivot = results_df.pivot_table(
            index="mechanism", columns="rho",
            values="auc_z_mahalanobis_sq",
        )
        auc_pivot.to_csv(out_dir / "auc_summary.csv")

        # Detector shifts at ρ=5% for the shift figure
        dz_cols = [c for c in results_df.columns if c.startswith("dz_")]
        if dz_cols:
            rho_target = 0.05
            closest_rho = min(contam_settings.contamination_rates,
                              key=lambda r: abs(r - rho_target))
            shift_df = results_df[results_df["rho"] == closest_rho][
                ["mechanism"] + dz_cols
            ].reset_index(drop=True)
            shift_df.to_csv(out_dir / "detector_shift.csv", index=False)

        summary = {
            "mechanisms": list(mechanisms.keys()),
            "contamination_rates": list(contam_settings.contamination_rates),
            "h0_sample_size": contam_settings.h0_sample_size,
            "n_cells": len(all_results),
        }
        (out_dir / "contamination_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
        log.info("contamination: wrote results to %s", out_dir)

    return results_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the contamination benchmark study."
    )
    parser.add_argument(
        "--country",
        default=AnalysisSettings().output_layout.country_code_lower,
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Sector names to exclude. Example: --exclude 'Financial Services'",
    )
    parser.add_argument(
        "--strict-compliance",
        action="store_true",
        default=False,
        help="Use the strict-compliance reference sample (ratio violations excluded from C).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    defaults = AnalysisSettings()

    # Route outputs to the correct directory (with or without exclusion)
    country = args.country
    ref_update = {}
    if args.strict_compliance:
        ref_update["require_ratio_compliance"] = True

    settings = defaults.model_copy(
        update={
            "exclude_sectors": tuple(args.exclude),
            "output_layout": defaults.output_layout.model_copy(
                update={"country_code_lower": country}
            ),
            **({"reference_sample": defaults.reference_sample.model_copy(
                update=ref_update
            )} if ref_update else {}),
        }
    )

    # If sectors are excluded, use the excluded-sector scoring outputs
    suffix = exclusion_suffix(settings)
    if suffix:
        settings = settings.model_copy(
            update={
                "output_layout": settings.output_layout.model_copy(
                    update={"country_code_lower": country + suffix}
                )
            }
        )

    # Load panel from base country (panel itself is never sector-filtered on disk)
    panel_path = defaults.output_layout.root / Path(
        str(defaults.output_layout.panel_relative).format(country=country)
    )
    if not panel_path.exists():
        # Try the phase0 panel_with_split
        panel_path = settings.output_layout.phase0_dir() / "panel_with_split.parquet"
    log.info("Loading panel from %s", panel_path)
    panel = pd.read_parquet(panel_path)

    # Add _split if missing (needed for H0 generation)
    if "_split" not in panel.columns:
        base_split_path = (
            defaults.output_layout.root
            / Path(str(defaults.output_layout.scores_relative).format(country=country))
            / "phase0_reference_sample" / "panel_with_split.parquet"
        )
        if base_split_path.exists():
            panel = pd.read_parquet(base_split_path)

    run_contamination_study(panel=panel, settings=settings)


if __name__ == "__main__":
    main()
