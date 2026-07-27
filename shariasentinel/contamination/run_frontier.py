"""Option B — Detection frontier.

Sweep over mechanism magnitudes to find the minimum perturbation
that achieves a target AUC.  Produces a CSV with one row per
(mechanism, magnitude, ρ) cell.

Usage::

    python -m server.contamination.run_frontier --country mys

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
from shariasentinel.scoring.phase4_composites import _build_weights, _load_sigma
from shariasentinel.scoring.bootstrap import run_non_parametric_bootstrap
from shariasentinel.scoring.sector_filter import apply_sector_filter, exclusion_suffix

from .config import ContaminationSettings, M1Settings, M3Settings, M4Settings
from .evaluate import compute_auc
from .h0_generator import generate_h0
from .mechanisms import MECHANISM_DISPATCH
from .run_contamination import (
    _compute_zscores_on_benchmark,
    _compute_composites_on_benchmark,
)

log = logging.getLogger(__name__)

# ── Magnitude grids per mechanism ──────────────────────────────────────
# Each key maps to a list of (param_name, value) dicts that will override
# the default ContaminationSettings.
_FRONTIER_GRID: dict[str, list[dict]] = {
    "M1_threshold": [
        {"band_fraction": v}
        for v in [0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
    ],
    "M2_smoothing": [
        {"window": w}
        for w in [2, 4, 6, 8, 12]
    ],
    "M3_benford": [
        {"rounding_magnitude": m}
        for m in [1, 2, 3, 4, 5]
    ],
    "M4_inconsistency": [
        {"debt_reduction": d}
        for d in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
    ],
}

# Human-readable labels for the magnitude axis
_MAGNITUDE_LABELS: dict[str, str] = {
    "M1_threshold": "band_fraction",
    "M2_smoothing": "window",
    "M3_benford": "rounding_magnitude",
    "M4_inconsistency": "debt_reduction",
}


def _make_contam_settings(
    base: ContaminationSettings,
    mechanism: str,
    overrides: dict,
) -> ContaminationSettings:
    """Create a ContaminationSettings with one mechanism's params overridden."""
    field_map = {
        "M1_threshold": "m1",
        "M2_smoothing": "m2",
        "M3_benford": "m3",
        "M4_inconsistency": "m4",
    }
    field = field_map[mechanism]
    current = getattr(base, field)
    updated_mech = current.model_copy(update=overrides)
    return base.model_copy(update={field: updated_mech})


def run_detection_frontier(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    contam_settings: ContaminationSettings | None = None,
    rho_values: tuple[float, ...] = (0.05, 0.10),
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Sweep mechanism magnitudes and measure AUC at each point.

    Returns:
        DataFrame with columns: mechanism, magnitude_param, magnitude_value,
        rho, auc_z_mahalanobis_sq, auc_t_iut, auc_z_plus, auc_z_plus_renorm.
    """
    settings = settings or AnalysisSettings()
    contam_settings = contam_settings or ContaminationSettings()

    panel = apply_sector_filter(panel, settings)

    # Load Sigma and weights
    active = tuple(settings.zscores.active_detectors())
    sigma = _load_sigma(
        settings.output_layout.phase2_dir() / "cov_ledoit_wolf.parquet",
        active,
    )
    weights = _build_weights(active, settings)

    # Bootstrap null from C
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

    rng = np.random.default_rng(contam_settings.random_seed)
    all_results: list[dict] = []

    for mech_name, grid in _FRONTIER_GRID.items():
        mech_fn = MECHANISM_DISPATCH[mech_name]
        mag_label = _MAGNITUDE_LABELS[mech_name]

        for overrides in grid:
            mag_value = list(overrides.values())[0]
            cs = _make_contam_settings(contam_settings, mech_name, overrides)

            for rho in rho_values:
                log.info(
                    "frontier: %s, %s=%s, rho=%.2f",
                    mech_name, mag_label, mag_value, rho,
                )

                h0 = generate_h0(panel, cs)
                benchmark = mech_fn(h0, rho, cs, rng)
                labels = benchmark["y"].to_numpy(dtype=int)

                zscores = _compute_zscores_on_benchmark(benchmark, settings)
                composites = _compute_composites_on_benchmark(
                    zscores, sigma, weights, settings,
                )

                row = {
                    "mechanism": mech_name,
                    "magnitude_param": mag_label,
                    "magnitude_value": float(mag_value),
                    "rho": rho,
                    "n_total": len(benchmark),
                    "n_contaminated": int(labels.sum()),
                }

                for comp in ["z_plus", "z_plus_renorm",
                             "z_mahalanobis_sq", "t_iut"]:
                    if comp in composites.columns:
                        scores = composites[comp].to_numpy(dtype=float)
                        row[f"auc_{comp}"] = compute_auc(scores, labels)

                # Per-detector shifts
                active_det = [d for d in settings.zscores.active_detectors()
                              if d in zscores.columns]
                for det in active_det:
                    z_vals = zscores[det].to_numpy(dtype=float)
                    mean_c = np.nanmean(z_vals[labels == 1])
                    mean_cl = np.nanmean(z_vals[labels == 0])
                    row[f"dz_{det}"] = float(mean_c - mean_cl)

                all_results.append(row)

                log.info(
                    "  AUC: z_mah=%.3f, t_iut=%.3f",
                    row.get("auc_z_mahalanobis_sq", float("nan")),
                    row.get("auc_t_iut", float("nan")),
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
        results_df.to_csv(out_dir / "detection_frontier.csv", index=False)

        summary = {
            "study": "detection_frontier",
            "mechanisms": list(_FRONTIER_GRID.keys()),
            "rho_values": list(rho_values),
            "n_cells": len(all_results),
        }
        (out_dir / "frontier_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
        log.info("frontier: wrote %d cells to %s", len(all_results), out_dir)

    return results_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the detection frontier study (Option B)."
    )
    parser.add_argument(
        "--country",
        default=AnalysisSettings().output_layout.country_code_lower,
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Sector names to exclude.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    defaults = AnalysisSettings()

    country = args.country
    settings = defaults.model_copy(
        update={
            "exclude_sectors": tuple(args.exclude),
            "output_layout": defaults.output_layout.model_copy(
                update={"country_code_lower": country}
            ),
        }
    )

    suffix = exclusion_suffix(settings)
    if suffix:
        settings = settings.model_copy(
            update={
                "output_layout": settings.output_layout.model_copy(
                    update={"country_code_lower": country + suffix}
                )
            }
        )

    panel_path = defaults.output_layout.root / Path(
        str(defaults.output_layout.panel_relative).format(country=country)
    )
    if not panel_path.exists():
        panel_path = settings.output_layout.phase0_dir() / "panel_with_split.parquet"
    log.info("Loading panel from %s", panel_path)
    panel = pd.read_parquet(panel_path)

    if "_split" not in panel.columns:
        base_split_path = (
            defaults.output_layout.root
            / Path(str(defaults.output_layout.scores_relative).format(country=country))
            / "phase0_reference_sample" / "panel_with_split.parquet"
        )
        if base_split_path.exists():
            panel = pd.read_parquet(base_split_path)

    run_detection_frontier(panel=panel, settings=settings)


if __name__ == "__main__":
    main()
