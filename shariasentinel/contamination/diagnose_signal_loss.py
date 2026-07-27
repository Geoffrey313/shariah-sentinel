"""Signal-loss diagnostic for the contamination study.

Runs all four mechanisms (M1–M4) at rho=0.05 and computes AUC at every
layer of the pipeline:

  Layer 1 — Raw financial variables (dlttq, ratio_debt_adj, implied CoD)
  Layer 2 — Individual z-scores after full detector pipeline (z1..z57)
  Layer 3 — Composite scores (z_plus, z_mahalanobis_sq, t_iut)

The goal is to find where the signal is absorbed: if Layer 1 has high AUC
but Layer 3 has AUC ≈ 0.5, the detectors or the composition kill the signal.
If Layer 1 itself has low AUC, the mechanism does not produce detectable
changes in the raw data.

Usage::

    python -m server.contamination.diagnose_signal_loss
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from shariasentinel.scoring.config import AnalysisSettings
from shariasentinel.scoring.compute_zscores import _run_detector, _DETECTOR_DISPATCH
from shariasentinel.scoring.composites import (
    truncated_sum,
    mahalanobis_squared,
    iut_statistic,
)
from shariasentinel.scoring.phase4_composites import _build_weights, _load_sigma

from .config import ContaminationSettings
from .evaluate import compute_auc
from .h0_generator import generate_h0
from .mechanisms import MECHANISM_DISPATCH

log = logging.getLogger(__name__)

RHO = 0.05
MECHANISMS = ["M1_threshold", "M2_smoothing", "M3_benford", "M4_inconsistency"]

# ── Raw-variable extractors ──────────────────────────────────────────────────

def _raw_variable_aucs(benchmark: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """AUC on raw financial variables before any detection.

    Variables chosen because each mechanism directly perturbs them:
      - dlttq          : long-term debt — modified by M4 (and M1 via ratio)
      - ratio_debt_adj : dlttq / atq — target of M1 threshold pinning
      - implied_cod    : xintq / dlttq — the inconsistency M4 creates
      - atq            : total assets — denominator for Sharia ratios
      - xintq          : interest expense — unchanged by M4 (creates CoD spike)
    """
    results: dict[str, float] = {}
    df = benchmark.copy()

    # dlttq — higher after M4 reduction is reversed: contaminated rows have
    # *lower* dlttq, so AUC > 0.5 means LOWER dlttq → more contaminated.
    # We flip sign so "high score = contaminated".
    if "dlttq" in df.columns:
        vals = df["dlttq"].to_numpy(dtype=float)
        # M4 reduces dlttq, so use -dlttq (lower debt = suspicious)
        results["raw_neg_dlttq"] = compute_auc(-vals, labels)

    # ratio_debt_adj = dlttq / atq — M1 pins this just below 0.33
    if "dlttq" in df.columns and "atq" in df.columns:
        dlttq = df["dlttq"].to_numpy(dtype=float)
        atq = df["atq"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(atq > 0, dlttq / atq, np.nan)
        results["raw_ratio_debt_adj"] = compute_auc(ratio, labels)

    # implied cost of debt = xintq / dlttq — M4 inflates this (debt reduced,
    # interest unchanged)
    if "xintq" in df.columns and "dlttq" in df.columns:
        xintq = df["xintq"].to_numpy(dtype=float)
        dlttq = df["dlttq"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cod = np.where(dlttq > 0, xintq / dlttq, np.nan)
        results["raw_implied_cod"] = compute_auc(cod, labels)

    # atq — M2/M3 can affect smoothed or rounded monetary values
    if "atq" in df.columns:
        atq = df["atq"].to_numpy(dtype=float)
        results["raw_atq"] = compute_auc(atq, labels)

    # xintq — interest expense (unchanged by M4; useful baseline)
    if "xintq" in df.columns:
        xintq = df["xintq"].to_numpy(dtype=float)
        results["raw_xintq"] = compute_auc(xintq, labels)

    return results


# ── Z-score layer ─────────────────────────────────────────────────────────────

def _compute_zscores(
    benchmark: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Run all detectors and apply merge rules; return z-score DataFrame."""
    active_raw = list(settings.zscores.include_detectors)
    result = pd.DataFrame(index=benchmark.index)

    for name in active_raw:
        log.debug("running detector %s", name)
        try:
            series = _run_detector(name, benchmark, settings)
            result[name] = series.reindex(benchmark.index).astype(float).values
        except Exception as exc:
            log.warning("detector %s failed: %s", name, exc)
            result[name] = np.nan

    # Apply merge rules
    for rule in settings.zscores.merge_rules:
        inputs_present = [c for c in rule.inputs if c in result.columns]
        if len(inputs_present) < 2:
            log.warning("merge rule %s: only %d inputs present, skipping",
                        rule.output_name, len(inputs_present))
            continue
        mat = result[inputs_present].to_numpy(dtype=float)
        if rule.method == "max":
            merged = np.nanmax(mat, axis=1)
        else:
            merged = np.nanmean(mat, axis=1)
        all_nan = np.isnan(mat).all(axis=1)
        merged = np.where(all_nan, np.nan, merged)

        # Optional empirical PIT on C (mirrors compute_zscores.py)
        if rule.empirical_pit_on_c and "_split" in benchmark.columns:
            from shariasentinel.detectors.pit import pit_empirical
            ref = merged[benchmark["_split"].to_numpy() == "C"]
            ref = ref[np.isfinite(ref)]
            if ref.size >= 30:
                finite = np.isfinite(merged)
                merged = merged.copy()
                merged[finite] = pit_empirical(merged[finite], ref)

        result[rule.output_name] = merged
        if rule.drop_inputs:
            result = result.drop(
                columns=[c for c in rule.inputs if c in result.columns]
            )

    return result


def _zscore_aucs(zscores: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """AUC for every individual detector z-score column."""
    results: dict[str, float] = {}
    for col in zscores.columns:
        vals = zscores[col].to_numpy(dtype=float)
        results[f"zscore_{col}"] = compute_auc(vals, labels)
    return results


# ── Composite layer ───────────────────────────────────────────────────────────

def _compute_composites(
    zscores: pd.DataFrame,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Compute the three headline composites on the z-score frame."""
    active = tuple(settings.zscores.active_detectors())
    active_present = [d for d in active if d in zscores.columns]
    z_matrix = zscores[active_present].to_numpy(dtype=float)

    comp = settings.composites
    composites = pd.DataFrame(index=zscores.index)
    composites["z_plus"] = truncated_sum(z_matrix, weights)
    composites["z_mahalanobis_sq"] = mahalanobis_squared(z_matrix, sigma)
    composites["t_iut"] = iut_statistic(
        z_matrix, min_active=comp.min_active_for_iut
    )
    return composites


def _composite_aucs(composites: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """AUC for every composite column."""
    results: dict[str, float] = {}
    for col in composites.columns:
        vals = composites[col].to_numpy(dtype=float)
        results[f"composite_{col}"] = compute_auc(vals, labels)
    return results


# ── Table formatting ──────────────────────────────────────────────────────────

def _print_table(rows: list[dict]) -> None:
    """Print a formatted table of AUC values by mechanism and layer."""
    if not rows:
        print("(no results)")
        return

    # Collect all metric keys in order of first appearance
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in ("mechanism",) and k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Group keys by layer prefix
    raw_keys = [k for k in all_keys if k.startswith("raw_")]
    zscore_keys = [k for k in all_keys if k.startswith("zscore_")]
    composite_keys = [k for k in all_keys if k.startswith("composite_")]

    def _auc_str(val: float) -> str:
        if not np.isfinite(val):
            return "  NaN "
        marker = " <--" if val >= 0.70 else ("    " if val >= 0.55 else "    ")
        return f"{val:.3f}{marker}"

    def _print_section(title: str, keys: list[str]) -> None:
        if not keys:
            return
        print(f"\n  {title}")
        print("  " + "-" * 72)
        # Header
        hdr = f"  {'Metric':<30}"
        for row in rows:
            hdr += f"  {row['mechanism'][:14]:>14}"
        print(hdr)
        print("  " + "-" * 72)
        for k in keys:
            label = k.split("_", 1)[1] if "_" in k else k
            line = f"  {label:<30}"
            for row in rows:
                v = row.get(k, float("nan"))
                line += f"  {_auc_str(v):>14}"
            print(line)

    print("\n" + "=" * 76)
    print("  SIGNAL-LOSS DIAGNOSTIC  —  AUC by layer (rho=0.05)")
    print("  AUC > 0.70 marked with '<--'; random chance = 0.500")
    print("=" * 76)
    _print_section("LAYER 1 — Raw financial variables", raw_keys)
    _print_section("LAYER 2 — Individual z-scores (post-PIT)", zscore_keys)
    _print_section("LAYER 3 — Composites", composite_keys)
    print("\n" + "=" * 76)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_diagnostic() -> None:
    """Run signal-loss diagnostic for all mechanisms at rho=0.05."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = AnalysisSettings()
    contam_settings = ContaminationSettings()

    # ── Locate panel ──
    panel_path = (
        settings.output_layout.root
        / Path(
            str(settings.output_layout.panel_relative).format(
                country=settings.output_layout.country_code_lower
            )
        )
    )
    split_path = (
        settings.output_layout.phase0_dir() / "panel_with_split.parquet"
    )

    if split_path.exists():
        print(f"Loading panel with split from {split_path}")
        panel = pd.read_parquet(split_path)
    elif panel_path.exists():
        print(f"Loading panel from {panel_path}")
        panel = pd.read_parquet(panel_path)
        if "_split" not in panel.columns:
            raise RuntimeError(
                "_split column missing. Run Phase 0 first or use the "
                "panel_with_split.parquet output."
            )
    else:
        raise FileNotFoundError(
            f"Panel not found at {panel_path} or {split_path}. "
            "Check your output directory."
        )

    print(f"Panel loaded: {len(panel):,} rows, "
          f"{(panel['_split'] == 'C').sum():,} in split C")

    # ── Load sigma + weights ──
    active = tuple(settings.zscores.active_detectors())
    sigma_path = settings.output_layout.phase2_dir() / "cov_ledoit_wolf.parquet"
    sigma = _load_sigma(sigma_path, active)
    weights = _build_weights(active, settings)

    print(f"Active detectors: {active}")
    print(f"Running rho={RHO} for mechanisms: {MECHANISMS}\n")

    rng = np.random.default_rng(contam_settings.random_seed)
    rows: list[dict] = []

    for mech_name in MECHANISMS:
        if mech_name not in MECHANISM_DISPATCH:
            log.warning("Unknown mechanism %s, skipping.", mech_name)
            continue

        mech_fn = MECHANISM_DISPATCH[mech_name]
        print(f"Running {mech_name} ...", end="", flush=True)

        # 1. Generate H0 (panel mode: preserves firm/temporal structure)
        h0 = generate_h0(panel, contam_settings, mode="panel")

        # 2. Apply mechanism
        benchmark = mech_fn(h0, RHO, contam_settings, rng)
        labels = benchmark["y"].to_numpy(dtype=int)
        n_contam = int(labels.sum())
        n_total = len(benchmark)
        print(f" n={n_total:,}, contaminated={n_contam} ({n_contam/n_total:.1%})")

        result_row: dict = {"mechanism": mech_name}

        # ── Layer 1: raw variables ──
        raw_aucs = _raw_variable_aucs(benchmark, labels)
        result_row.update(raw_aucs)

        # ── Layer 2: individual z-scores ──
        zscores = _compute_zscores(benchmark, settings)
        z_aucs = _zscore_aucs(zscores, labels)
        result_row.update(z_aucs)

        # ── Layer 3: composites ──
        composites = _compute_composites(zscores, sigma, weights, settings)
        comp_aucs = _composite_aucs(composites, labels)
        result_row.update(comp_aucs)

        rows.append(result_row)

    # ── Print summary table ──
    _print_table(rows)

    # ── Additional per-mechanism detail ──
    print("\n  DETAILED BREAKDOWN (one row per mechanism × metric)")
    print("  " + "-" * 72)
    fmt = f"  {'Mechanism':<22} {'Layer':<12} {'Metric':<28} {'AUC':>6}"
    print(fmt)
    print("  " + "-" * 72)
    for row in rows:
        mech = row["mechanism"]
        for k, v in row.items():
            if k == "mechanism":
                continue
            layer = (
                "raw_var"
                if k.startswith("raw_")
                else "z_score"
                if k.startswith("zscore_")
                else "composite"
            )
            label = k.split("_", 1)[1] if "_" in k else k
            flag = " <-- HIGH" if isinstance(v, float) and v >= 0.70 else ""
            auc_str = f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "  NaN"
            print(f"  {mech:<22} {layer:<12} {label:<28} {auc_str:>6}{flag}")
        print()

    print("\nDiagnosis complete.")
    print(
        "Hypothesis check:\n"
        "  - Layer 1 AUC >> 0.5  → mechanism changes raw data detectably\n"
        "  - Layer 2 AUC >> 0.5  → detectors see the change\n"
        "  - Layer 3 AUC ≈ 0.5   → composition / PIT absorbs the signal\n"
        "If Layer 1 is already low, the mechanism itself is too subtle at rho=0.05."
    )


if __name__ == "__main__":
    run_diagnostic()
