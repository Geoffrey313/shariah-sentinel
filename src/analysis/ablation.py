"""Reproduce the detector-ablation table (data/scores/mys/qualitative/ablation_results.csv).

Runs the full pipeline under the full model and four ablations:

  A1  no annual cash-flow recovery -> rebuild the panel with ``oancfq`` (the
                                  quarterly operating cash flow that
                                  ``derive_oancfq`` recovers by YTD decumulation
                                  of the annual ``oancfy``) set to NaN; it feeds
                                  the coherence (z5) and M-score (z3) detectors.
                                  This is the annual→quarterly recovery that
                                  actually feeds the canonical scores; seasonal
                                  disaggregation is a coverage diagnostic, not
                                  scored.
  A2  no z5/z7 merge      -> zscores.merge_rules = ()
  A3  no Monte-Carlo Benford -> detector_preconditions.benford.calibration_mode
                                = 'chi2_asymptotic'
  A4  no empirical PIT    -> the z57 merge rule with empirical_pit_on_c = False

Each run reuses the reconstructed ``mys`` panel (A1 uses a NaN-``oancfq``
variant) but writes to its own scores directory by overriding
``scores_relative`` (and ``panel_relative`` for A1). Six metrics are then read
back from each run's phase outputs and assembled into ablation_results.csv.

Run via ``python reproduce.py ablation --country mys``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.config import AnalysisSettings
from src.analysis.pipeline import run_full_analysis

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data"


def _base(country: str) -> AnalysisSettings:
    s = AnalysisSettings()
    ol = s.output_layout.model_copy(update={"country_code_lower": country, "root": str(ROOT)})
    return s.model_copy(update={"output_layout": ol})


def _scores_override(base: AnalysisSettings, rel: str) -> AnalysisSettings:
    return base.model_copy(update={
        "output_layout": base.output_layout.model_copy(update={"scores_relative": Path(rel)}),
    })


def _build_a1_panel(base: AnalysisSettings, country: str) -> str:
    """Write a NaN-oancfq panel variant for the no-annual-disaggregation run."""
    src = ROOT / str(base.output_layout.panel_relative).format(country=country)
    panel = pd.read_parquet(src)
    panel["oancfq"] = float("nan")
    dest_rel = f"panel/{country}_abl_a1/compustat_quarterly.parquet"
    dest = ROOT / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(dest, index=False)
    return dest_rel


def _configs(base: AnalysisSettings, country: str):
    full_rule = base.zscores.merge_rules[0]
    no_pit_rule = full_rule.model_copy(update={"empirical_pit_on_c": False})
    a1_panel_rel = _build_a1_panel(base, country)
    a1 = base.model_copy(update={"output_layout": base.output_layout.model_copy(update={
        "panel_relative": Path(a1_panel_rel),
        "scores_relative": Path(f"scores/{country}_abl_a1")})})
    return [
        ("FULL MODEL", f"scores/{country}", base),  # reuse the full pipeline run
        ("A1: no annual cash-flow recovery", f"scores/{country}_abl_a1", a1),
        ("A2: no z5/z7 merge", f"scores/{country}_abl_a2",
         _scores_override(base, f"scores/{country}_abl_a2").model_copy(update={
             "zscores": base.zscores.model_copy(update={"merge_rules": ()})})),
        ("A3: no MC Benford", f"scores/{country}_abl_a3",
         _scores_override(base, f"scores/{country}_abl_a3").model_copy(update={
             "detector_preconditions": base.detector_preconditions.model_copy(update={
                 "benford": base.detector_preconditions.benford.model_copy(update={
                     "calibration_mode": "chi2_asymptotic"})})})),
        ("A4: no empirical PIT", f"scores/{country}_abl_a4",
         _scores_override(base, f"scores/{country}_abl_a4").model_copy(update={
             "zscores": base.zscores.model_copy(update={"merge_rules": (no_pit_rule,)})})),
    ]


def _metrics(scores_rel: str) -> dict:
    d = ROOT / scores_rel
    comp = pd.read_parquet(d / "phase4_composites" / "composites_panel.parquet")
    zmah_finite = int(pd.to_numeric(comp["z_mahalanobis_sq"], errors="coerce").notna().sum())

    tec = pd.read_csv(d / "phase5_injection" / "theoretical_empirical_comparison.csv")
    row = tec[(tec["alpha"] == 0.05) & (tec["delta"] == 2.0)].iloc[0]
    tiut_power = float(row["empirical_t_iut_power"])
    tiut_gap = float(row["empirical_t_iut_power"] - row["theoretical_t_iut_power"])

    fdr = json.loads((d / "phase7_fdr" / "phase7_fdr.json").read_text())["firm_level_discoveries"]
    fdr_zmah = int(fdr["p_z_mahalanobis_sq"]["q<=0.05"])
    fdr_tiut = int(fdr["p_t_iut"]["q<=0.05"])

    integ = pd.read_csv(d / "phase6_robustness" / "integrity_sensitivity.csv")
    row_t = integ[integ["composite"] == "t_iut"]
    tiut_jac = float(row_t["jaccard_top_n"].iloc[0]) if len(row_t) else float("nan")

    return {
        "zmah_finite": zmah_finite,
        "tiut_power_d2": round(tiut_power, 3),
        "tiut_gap_d2": round(tiut_gap, 3),
        "fdr_zmah_q05": fdr_zmah,
        "fdr_tiut_q05": fdr_tiut,
        "tiut_jaccard": round(tiut_jac, 3),
    }


def run_ablation_study(country: str = "mys") -> pd.DataFrame:
    base = _base(country)
    rows = []
    for name, rel, settings in _configs(base, country):
        if name != "FULL MODEL":
            print(f"=== running {name} -> {rel} ===", flush=True)
            run_full_analysis(settings=settings)
        m = _metrics(rel)
        m["model"] = name
        rows.append(m)
        print(name, m, flush=True)

    cols = ["model", "zmah_finite", "tiut_power_d2", "tiut_gap_d2",
            "fdr_zmah_q05", "fdr_tiut_q05", "tiut_jaccard"]
    out = pd.DataFrame(rows)[cols]
    dest = ROOT / "scores" / country / "qualitative" / "ablation_results.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print("\nwrote", dest)
    print(out.to_string(index=False))
    return out
