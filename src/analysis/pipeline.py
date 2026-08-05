"""Unified scoring orchestrator — Sprint 1 through Sprint 5.

Runs the full Phase 0–7 pipeline with optional sector exclusion.
When ``exclude_sectors`` is set, outputs go to a separate directory
(e.g. ``outputs/scores/mys_ex_financial_services/``) so full-panel
and excluded-sector results coexist for comparison.

Usage::

    # Full panel (default)
    python -m server.scoring.run_analysis

    # Exclude Financial Services
    python -m server.scoring.run_analysis --exclude "Financial Services"

    # Exclude multiple sectors
    python -m server.scoring.run_analysis --exclude "Financial Services" "SPAC"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.engine.zscores import compute_zscores
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import resolve_label_column, run_phase0
from src.analysis.calibration import run_phase1
from src.analysis.dependence import run_phase2
from src.analysis.detector_qc import run_phase3
from src.analysis.composite_scoring import run_phase4
from src.analysis.confidence import run_phase4b
from src.analysis.injection import run_phase5
from src.analysis.robustness_phases import run_phase6
from src.analysis.fdr import run_phase7
from src.analysis.qualitative import run_qualitative
from src.analysis.sector_filter import apply_sector_filter, compliant_suffix, exclusion_suffix

log = logging.getLogger(__name__)


def run_full_analysis(
    settings: AnalysisSettings | None = None,
) -> dict:
    """Run Phase 0 through Phase 7 end-to-end.

    Args:
        settings: Configuration. When ``exclude_sectors`` is non-empty,
            the panel is filtered before Phase 0 and outputs are routed
            to a distinct directory.

    Returns:
        Dictionary of per-phase summary dicts for programmatic access.
    """
    settings = settings or AnalysisSettings()

    # Route outputs to a separate directory if sectors are excluded
    # or compliant-only is active.
    suffix = exclusion_suffix(settings) + compliant_suffix(settings)
    if suffix:
        current_country = settings.output_layout.country_code_lower
        settings = settings.model_copy(
            update={
                "output_layout": settings.output_layout.model_copy(
                    update={"country_code_lower": current_country + suffix}
                )
            }
        )
        log.info("run_analysis: outputs routed to %s/", current_country + suffix)

    # Load panel
    panel_path = settings.output_layout.panel_path()
    if not panel_path.exists():
        # Fall back to the base country path (without suffix) for the
        # source panel — the panel itself is never modified by sector
        # exclusion.
        base_settings = settings.model_copy(
            update={
                "output_layout": settings.output_layout.model_copy(
                    update={
                        "country_code_lower": settings.output_layout.country_code_lower.split("_ex_")[0].split("_compliant_only")[0]
                    }
                )
            }
        )
        panel_path = base_settings.output_layout.panel_path()

    log.info("run_analysis: reading panel %s", panel_path)
    panel = pd.read_parquet(panel_path)

    # Apply sector filter
    panel = apply_sector_filter(panel, settings)
    log.info(
        "run_analysis: panel after filter = %d rows, %d firms, %d sectors",
        len(panel),
        panel[settings.panel_schema.firm_id].nunique(),
        panel[settings.panel_schema.sector].nunique()
        if settings.panel_schema.sector in panel.columns else 0,
    )

    summaries = {}

    # Sprint 1 — Phase 0 + Phase 3
    log.info("run_analysis: Phase 0 — reference sample")
    p0 = run_phase0(panel=panel, settings=settings, write_outputs=True)
    summaries["phase0"] = p0.summary
    log.info("run_analysis: |C| = %d", p0.summary["global"]["size"])

    log.info("run_analysis: Phase 3 — detector validity masks")
    p3 = run_phase3(panel=p0.panel, settings=settings, write_outputs=True)
    summaries["phase3"] = p3.summary

    # Sprint 2 — z-scores + Phase 1 + Phase 2
    log.info("run_analysis: computing z-scores")
    zc = compute_zscores(panel=p0.panel, settings=settings, write_outputs=True)

    log.info("run_analysis: Phase 1 — null calibration")
    p1 = run_phase1(
        panel=p0.panel, settings=settings,
        zscores=zc.zscores, write_outputs=True,
    )
    summaries["phase1"] = p1.summary
    log.info("run_analysis: forced bootstrap = %s", p1.summary["forced_bootstrap"])

    log.info("run_analysis: Phase 2 — dependence")
    p2 = run_phase2(
        panel=p0.panel, settings=settings,
        zscores=zc.zscores, write_outputs=True,
    )
    summaries["phase2"] = p2.summary
    log.info("run_analysis: K_eff = %d", p2.summary["pca"]["k_eff"])

    # Sprint 3 — Phase 4 + Phase 4b
    log.info("run_analysis: Phase 4 — composites + bootstrap")
    p4 = run_phase4(
        panel=p0.panel, settings=settings,
        zscores=zc.zscores, write_outputs=True,
    )
    summaries["phase4"] = p4.summary

    log.info("run_analysis: Phase 4b — confidence qualifier")
    p4b = run_phase4b(
        panel=p0.panel, settings=settings,
        zscores=zc.zscores, write_outputs=True,
    )
    summaries["phase4b"] = p4b.summary

    # Sprint 4 — Phase 5
    log.info("run_analysis: Phase 5 — injection power study")
    p5 = run_phase5(
        panel=p0.panel, settings=settings,
        zscores=zc.zscores, write_outputs=True,
    )
    summaries["phase5"] = p5.summary

    # Sprint 5 — Phase 6 + Phase 7
    log.info("run_analysis: Phase 6 — robustness")
    p6 = run_phase6(
        panel=p0.panel, settings=settings,
        composites=p4.composites, write_outputs=True,
    )
    summaries["phase6"] = p6.summary

    # Optional: restrict verdict reporting to ratio-compliant rows
    report_panel = p0.panel
    report_composites = p4.composites
    if settings.compliant_only:
        # Thresholds follow the panel's methodology (SAC 33/33/5,
        # DFM 30/30/10) via the central registry — never hardcoded.
        from src.common.methodology import thresholds_for_panel

        ratio_ok = pd.Series(True, index=report_panel.index, dtype=bool)
        for col, cap in thresholds_for_panel(report_panel).items():
            if col in report_panel.columns:
                exceeds = report_panel[col] > cap
                ratio_ok &= ~exceeds.fillna(False)
        report_panel = report_panel[ratio_ok].copy()
        report_composites = report_composites[ratio_ok].copy()
        label_col = resolve_label_column(report_panel, settings)
        n_sac = (report_panel[label_col].fillna(0).astype(int) == 1).sum()
        n_nonsac = len(report_panel) - n_sac
        log.info(
            "run_analysis: --compliant-only restricted reporting to "
            "%d ratio-compliant rows (%d firms): %d SAC + %d non-SAC.",
            len(report_panel),
            report_panel[settings.panel_schema.firm_id].nunique(),
            n_sac, n_nonsac,
        )

    log.info("run_analysis: Phase 7 — FDR")
    p7 = run_phase7(
        panel=report_panel, settings=settings,
        composites=report_composites, write_outputs=True,
    )
    summaries["phase7"] = p7.summary

    # Qualitative cross-sectional analysis
    log.info("run_analysis: qualitative analysis")
    qual = run_qualitative(
        panel=report_panel, composites=report_composites,
        settings=settings, write_outputs=True,
    )
    summaries["qualitative"] = {k: len(v) for k, v in qual.items()}

    # Optional: robustness benchmark (Families 1-4)
    if settings.run_robustness:
        log.info("run_analysis: robustness benchmark")
        from src.analysis import run_robustness_benchmark
        rb = run_robustness_benchmark(
            panel=p0.panel, settings=settings, write_outputs=True,
        )
        summaries["robustness_benchmark"] = {
            "family1_rows": len(rb.family1),
            "family2_rows": len(rb.family2),
            "family3_rows": len(rb.family3),
            "family4_rows": len(rb.family4),
        }

    log.info("run_analysis: all phases complete.")
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full scoring analysis (Phase 0–7) with optional sector exclusion."
    )
    parser.add_argument(
        "--country",
        default=AnalysisSettings().output_layout.country_code_lower,
        help="Lower-case country code subfolder.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Sector names to exclude. Example: --exclude 'Financial Services' 'SPAC'",
    )
    parser.add_argument(
        "--strict-compliance",
        action="store_true",
        default=False,
        help="Exclude from reference sample C any SAC-labeled row whose "
        "recomputed ratios exceed the SAC thresholds (debt > 33%%, "
        "cash > 33%%, income > 5%%). Tightens the null distribution.",
    )
    parser.add_argument(
        "--compliant-only",
        action="store_true",
        default=False,
        help="Restrict verdict reporting (FDR, qualitative, headline "
        "numbers) to ratio-compliant rows only. Detectors and composites "
        "are still computed on the full panel.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        default=False,
        help="Run the cross-family robustness benchmark (adversarial "
        "evasion, AnoShift, realistic manipulations) after Phase 7.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool size for scoring contamination cells (Families 1 "
        "& 2). 1 = serial (default). Cap BLAS threads per worker "
        "(OMP_NUM_THREADS) and bound total CPU with a cgroup so "
        "workers × threads cannot oversubscribe the host.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    defaults = AnalysisSettings()
    ref_update = {}
    if args.strict_compliance:
        ref_update["require_ratio_compliance"] = True

    top_update = {
        "exclude_sectors": tuple(args.exclude),
        "output_layout": defaults.output_layout.model_copy(
            update={"country_code_lower": args.country}
        ),
    }
    if ref_update:
        top_update["reference_sample"] = defaults.reference_sample.model_copy(
            update=ref_update
        )
    if args.compliant_only:
        top_update["compliant_only"] = True
    if args.robustness:
        top_update["run_robustness"] = True
    if args.workers and args.workers > 1:
        top_update["robustness_benchmark"] = defaults.robustness_benchmark.model_copy(
            update={"workers": args.workers}
        )

    settings = defaults.model_copy(update=top_update)
    run_full_analysis(settings=settings)


if __name__ == "__main__":
    main()
