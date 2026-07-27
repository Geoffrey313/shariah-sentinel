"""Option C — Firm-level contamination evaluation.

Instead of measuring row-level AUC, this study:
1. Selects specific firms from C.
2. Contaminates ALL their quarters with a given mechanism.
3. Re-runs the full pipeline.
4. Applies firm-level BH and checks whether contaminated firms are flagged.

This tests the operationally relevant detection path: weak row-level
signals accumulating over many quarters into firm-level discoveries.

Usage::

    python -m server.contamination.run_firm_level --country mys

Outputs go to ``outputs/scores/<country>/contamination/``.
"""
from __future__ import annotations

import argparse
import json
import logging
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from shariasentinel.scoring.config import AnalysisSettings
from shariasentinel.scoring.phase4_composites import _build_weights, _load_sigma
from shariasentinel.scoring.bootstrap import run_non_parametric_bootstrap, upper_tail_pvalue
from shariasentinel.scoring.sector_filter import apply_sector_filter, exclusion_suffix

from .config import ContaminationSettings
from .mechanisms import MECHANISM_DISPATCH
from .run_contamination import (
    _compute_zscores_on_benchmark,
    _compute_composites_on_benchmark,
)

log = logging.getLogger(__name__)


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (same as phase7_fdr but self-contained)."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    vals = p[finite]
    m = vals.size
    order = np.argsort(vals, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)
    raw_q = vals[order] * m / ranks
    adjusted_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    unsort = np.empty(m, dtype=int)
    unsort[order] = np.arange(m, dtype=int)
    finite_q = np.clip(adjusted_sorted[unsort], 0.0, 1.0)
    q[finite] = finite_q
    return q


def run_firm_level_study(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    contam_settings: ContaminationSettings | None = None,
    firm_fractions: tuple[float, ...] = (0.02, 0.05, 0.10),
    q_levels: tuple[float, ...] = (0.01, 0.05, 0.10),
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Run firm-level contamination evaluation.

    Args:
        panel: Panel with ``_split`` and ``gvkey`` columns.
        settings: Scoring pipeline settings.
        contam_settings: Contamination settings.
        firm_fractions: Fraction of firms to contaminate.
        q_levels: BH q thresholds for flagging.
        write_outputs: Write results to disk.

    Returns:
        DataFrame with one row per (mechanism, firm_fraction, q_level).
    """
    settings = settings or AnalysisSettings()
    contam_settings = contam_settings or ContaminationSettings()

    panel = apply_sector_filter(panel, settings)

    # Load Sigma, weights, bootstrap null
    active = tuple(settings.zscores.active_detectors())
    sigma = _load_sigma(
        settings.output_layout.phase2_dir() / "cov_ledoit_wolf.parquet",
        active,
    )
    weights = _build_weights(active, settings)

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

    schema = settings.panel_schema
    firm_col = schema.firm_id

    # Work on C rows only — build a firm-aware benchmark
    c_panel = panel[mask_c].copy()
    if firm_col not in c_panel.columns:
        raise KeyError(f"firm_level: panel lacks {firm_col!r}")

    firms_in_c = c_panel[firm_col].unique()
    rng = np.random.default_rng(contam_settings.random_seed)

    # Mechanism dispatch — use all enabled
    mechanisms = {}
    for mech_cfg in [contam_settings.m1, contam_settings.m2,
                     contam_settings.m3, contam_settings.m4]:
        if mech_cfg.enabled:
            mechanisms[mech_cfg.name] = MECHANISM_DISPATCH[mech_cfg.name]

    all_results: list[dict] = []

    for mech_name, mech_fn in mechanisms.items():
        for frac in firm_fractions:
            n_contam_firms = max(1, floor(len(firms_in_c) * frac))
            contam_firms = set(
                rng.choice(firms_in_c, size=n_contam_firms, replace=False)
            )

            log.info(
                "firm_level: %s, fraction=%.2f (%d/%d firms)",
                mech_name, frac, n_contam_firms, len(firms_in_c),
            )

            # Build benchmark: all C rows, mark contaminated firms
            benchmark = c_panel.copy()
            benchmark["y"] = 0
            benchmark["_firm_contam"] = benchmark[firm_col].isin(contam_firms).astype(int)

            # Apply mechanism to contaminated firms' rows
            # We set rho=1.0 but only on the subset of contaminated firms
            contam_mask = benchmark["_firm_contam"] == 1
            n_contam_rows = contam_mask.sum()

            if n_contam_rows == 0:
                log.warning("firm_level: no contaminated rows; skipping.")
                continue

            # Apply mechanism: extract contaminated rows, modify, put back
            clean_part = benchmark[~contam_mask].copy()
            dirty_part = benchmark[contam_mask].copy()
            dirty_part["y"] = 0  # Will be set to 1 by mechanism

            # Apply mechanism with rho=1.0 (contaminate all selected rows)
            dirty_modified = mech_fn(dirty_part, 1.0, contam_settings, rng)

            benchmark = pd.concat(
                [clean_part, dirty_modified], ignore_index=True,
            )
            # Ground truth: firm-level label
            firm_labels = benchmark.groupby(firm_col, observed=True).agg(
                y_firm=("y", "max"),
                _firm_contam=("_firm_contam", "max"),
            )

            # Run detectors
            zscores = _compute_zscores_on_benchmark(benchmark, settings)

            # Compute composites
            composites = _compute_composites_on_benchmark(
                zscores, sigma, weights, settings,
            )
            composites[firm_col] = benchmark[firm_col].values

            # Compute row-level p-values
            composite_cols = ["z_plus", "z_plus_renorm",
                              "z_mahalanobis_sq", "t_iut"]
            for comp in composite_cols:
                if comp not in composites.columns or comp not in null_distributions:
                    continue
                scores = composites[comp].to_numpy(dtype=float)
                pvals = upper_tail_pvalue(
                    scores, null_distributions[comp], null.n_replicates,
                )
                composites[f"p_{comp}"] = pvals

            # Firm-level aggregation: min p-value per firm
            p_cols = [f"p_{c}" for c in composite_cols if f"p_{c}" in composites.columns]
            firm_pvals = composites.groupby(firm_col, observed=True)[p_cols].min()

            # BH correction at firm level
            for p_col in p_cols:
                q_col = f"q_{p_col}"
                firm_pvals[q_col] = _benjamini_hochberg(
                    firm_pvals[p_col].to_numpy(dtype=float)
                )

            # Join with ground truth
            firm_eval = firm_pvals.join(firm_labels, how="left")
            n_firms_total = len(firm_eval)
            n_firms_contam = int(firm_eval["_firm_contam"].sum())
            n_firms_clean = n_firms_total - n_firms_contam

            # Evaluate at each q-level
            for q_thresh in q_levels:
                row = {
                    "mechanism": mech_name,
                    "firm_fraction": frac,
                    "n_firms_total": n_firms_total,
                    "n_firms_contaminated": n_firms_contam,
                    "n_firms_clean": n_firms_clean,
                    "q_threshold": q_thresh,
                }

                for comp in composite_cols:
                    q_col = f"q_p_{comp}"
                    if q_col not in firm_eval.columns:
                        continue

                    flagged = firm_eval[q_col] <= q_thresh
                    is_contam = firm_eval["_firm_contam"] == 1

                    tp = int((flagged & is_contam).sum())
                    fp = int((flagged & ~is_contam).sum())
                    fn = int((~flagged & is_contam).sum())
                    tn = int((~flagged & ~is_contam).sum())

                    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

                    row[f"tp_{comp}"] = tp
                    row[f"fp_{comp}"] = fp
                    row[f"fn_{comp}"] = fn
                    row[f"precision_{comp}"] = precision
                    row[f"recall_{comp}"] = recall
                    row[f"fpr_{comp}"] = fpr

                all_results.append(row)

            log.info(
                "  firms: %d contam / %d total, best recall(z_mah,q=0.05)=%.3f",
                n_firms_contam, n_firms_total,
                next(
                    (r["recall_z_mahalanobis_sq"]
                     for r in all_results[-len(q_levels):]
                     if r.get("q_threshold") == 0.05),
                    float("nan"),
                ),
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
        results_df.to_csv(out_dir / "firm_level_evaluation.csv", index=False)

        summary = {
            "study": "firm_level_contamination",
            "mechanisms": list(mechanisms.keys()),
            "firm_fractions": list(firm_fractions),
            "q_levels": list(q_levels),
            "n_cells": len(all_results),
        }
        (out_dir / "firm_level_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
        log.info("firm_level: wrote %d cells to %s", len(all_results), out_dir)

    return results_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the firm-level contamination study (Option C)."
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

    run_firm_level_study(panel=panel, settings=settings)


if __name__ == "__main__":
    main()
