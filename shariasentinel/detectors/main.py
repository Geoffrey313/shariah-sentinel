"""
detectors/main.py
─────────────────
CLI entry point for running anomaly detectors on the SAC-anchored panel.

Usage
-----
    # Run all detectors + composite
    python -m detectors.main --detectors all --composite

    # Run a single detector
    python -m detectors.main --detectors d3

    # Run a subset
    python -m detectors.main --detectors d1,d3,d5

Outputs (under `data_clean/`):
    scores_<country>.parquet   One row per panel row, columns z1..z7 for the
                               requested detectors.
    scores_<country>.csv       Same content, CSV.
    composites_<country>.csv   (if --composite) Z_plus, Z_plus_renorm, B_A, T_IUT.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .d1_benford import detect_benford
from .d2_zipf import detect_zipf
from .d3_mscore import detect_mscore
from .d4_proximity import detect_threshold_proximity
from .d5_coherence import detect_cross_statement_coherence
from .d6_temporal import detect_temporal
from .d7_peer import detect_peer
from .composite import compute_composite

log = logging.getLogger(__name__)

# Canonical mapping: short name → (z-column, detector function)
DETECTOR_REGISTRY: dict[str, tuple[str, callable]] = {
    "d1": ("z1", detect_benford),
    "d2": ("z2", detect_zipf),
    "d3": ("z3", detect_mscore),
    "d4": ("z4", detect_threshold_proximity),
    "d5": ("z5", detect_cross_statement_coherence),
    "d6": ("z6", detect_temporal),
    "d7": ("z7", detect_peer),
}


def parse_detector_spec(spec: str) -> list[str]:
    """Parse `--detectors` argument: 'all' or comma-separated list like 'd1,d3,d5'."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(DETECTOR_REGISTRY.keys())
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in DETECTOR_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown detector(s): {unknown}. "
            f"Valid options: {sorted(DETECTOR_REGISTRY.keys())} or 'all'."
        )
    return names


def run_detectors(panel: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Run the requested detectors and return a DataFrame indexed like `panel`
    with one column per detector (z1..z7) plus `gvkey`/`datacqtr` for joining.

    Detectors that fail are logged and their column filled with NaN.
    """
    scores = pd.DataFrame(index=panel.index)

    for name in names:
        z_col, fn = DETECTOR_REGISTRY[name]
        log.info("Running %s → %s …", name, z_col)
        try:
            scores[z_col] = fn(panel)
            n_valid = int(scores[z_col].notna().sum())
            log.info("  %s: %d / %d valid scores (%.1f%%)",
                     z_col, n_valid, len(scores), 100 * n_valid / max(len(scores), 1))
        except Exception as e:
            log.error("  %s FAILED: %s", z_col, e)
            scores[z_col] = pd.NA

    # Prepend identity columns for easy joining
    id_cols = [c for c in ("gvkey", "datacqtr", "datadate", "conm") if c in panel.columns]
    return pd.concat([panel[id_cols], scores], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run anomaly detectors on the SAC-anchored panel.",
    )
    parser.add_argument(
        "--country", default="mys",
        help="Country tag used to build default panel and output paths (default: mys).",
    )
    parser.add_argument(
        "--panel", default=None,
        help="Path to the panel parquet. Default: outputs/panel/<country>/compustat_quarterly.parquet.",
    )
    parser.add_argument(
        "--detectors", default="all",
        help="Comma-separated detectors (e.g. 'd1,d3,d5') or 'all'. Default: all.",
    )
    parser.add_argument(
        "--composite", action="store_true",
        help="Compute composite scores (Z_plus, Z_plus_renorm, B_A, T_IUT) from z-scores.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory. Default: outputs/scores/<country>.",
    )
    args = parser.parse_args()

    c = args.country.lower()
    if args.panel is None:
        args.panel = f"outputs/panel/{c}/compustat_quarterly.parquet"
    if args.out_dir is None:
        args.out_dir = f"outputs/scores/{c}"

    names = parse_detector_spec(args.detectors)
    log.info("Selected detectors: %s", names)

    panel_path = Path(args.panel)
    log.info("Loading panel from %s", panel_path)
    panel = pd.read_parquet(panel_path)
    log.info("Panel shape: %d rows × %d columns", panel.shape[0], panel.shape[1])

    scores = run_detectors(panel, names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "z_scores.parquet"
    csv_path     = out_dir / "z_scores.csv"
    scores.to_parquet(parquet_path, index=False)
    scores.to_csv(csv_path, index=False)
    log.info("Wrote scores → %s", parquet_path)

    if args.composite:
        log.info("Computing composite scores …")
        z_cols = [c for c in ("z1", "z2", "z3", "z4", "z5", "z6", "z7") if c in scores.columns]
        if not z_cols:
            log.error("No z-score columns found — cannot compute composite.")
            return
        composites = compute_composite(scores[z_cols])
        id_cols = [c for c in ("gvkey", "datacqtr") if c in scores.columns]
        composites = pd.concat(
            [scores[id_cols].reset_index(drop=True), composites.reset_index(drop=True)],
            axis=1,
        )
        comp_path = out_dir / "composites.csv"
        composites.to_csv(comp_path, index=False)
        log.info("Wrote composites → %s", comp_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
