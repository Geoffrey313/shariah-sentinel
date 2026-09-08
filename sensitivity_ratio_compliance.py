"""Ratio-cap exclusion sensitivity: retaining the ratio-cap-violating authority rows in C.

Scope. This is a SCOPED SENSITIVITY, never a replacement of the canonical run or
the 324-firm headline. The canonical pipeline excludes from the reference sample C
the ~13% of SAC-labelled rows whose Compustat-recomputed ratios exceed the
authority caps (ratio_debt_adj/ratio_cash_adj > 0.33, ratio_income > 0.05), a
data-source divergence between Compustat and the authority filings. Here we set
require_ratio_compliance=False so those rows are RETAINED in C, rebuild the whole
pipeline from the raw panel, and report the effect beside the canonical run.

Decomposition of the retained rows (pre-registered ex ante). Classify each
retained ratio-violating row by its maximum RELATIVE exceedance across the
violated caps, r = max_j (ratio_j - cap_j) / cap_j:
  * r  > 0.25  -> large source discrepancy / likely filing-data mismatch;
  * r <= 0.25  -> candidate statistical anomaly among borderline exceedances.
The 0.25 cut is fixed before seeing the scored result; robustness at 0.10 and 0.50
is reported so the split does not hinge on the exact threshold. When several caps
are violated on a row, the maximum relative exceedance classifies the row.

Requires the licensed panels under outputs/ (not self-contained).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import AnalysisSettings
from src.analysis.benchmark import _score_panel
from src.analysis.fdr import run_phase7
from src.analysis.composite_scoring import compute_verdict, VERDICT_P_COLS
from src.common.methodology import screening_thresholds_for_panel
from src.analysis.reference_sample import REASON_RATIO_VIOLATION, SPLIT_LABEL_INCLUDED


def _max_rel_exceedance(df: pd.DataFrame, caps: dict[str, float]) -> np.ndarray:
    """max_j (ratio_j - cap_j)/cap_j over the screened caps (NaN ratios ignored)."""
    rel = np.full(len(df), -np.inf, dtype=float)
    for col, cap in caps.items():
        if col in df.columns:
            e = ((pd.to_numeric(df[col], errors="coerce") - cap) / cap).to_numpy(dtype=float)
            e = np.where(np.isfinite(e), e, -np.inf)
            rel = np.maximum(rel, e)
    return rel


def _run(raw: pd.DataFrame, settings: AnalysisSettings, require_ratio: bool):
    s = settings.model_copy(update={"reference_sample":
        settings.reference_sample.model_copy(update={"require_ratio_compliance": require_ratio})})
    ctx = _score_panel(raw, s)
    p7 = run_phase7(panel=ctx.panel, settings=s, composites=ctx.composites, write_outputs=False)
    verdict = compute_verdict(ctx.composites, VERDICT_P_COLS,
                              s.figures.red_threshold, s.figures.amber_threshold)
    return ctx, p7, verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="mys")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = AnalysisSettings()
    settings = base.model_copy(update={"output_layout":
        base.output_layout.model_copy(update={"country_code_lower": args.country.lower()})})
    schema = settings.panel_schema
    fc, qc = schema.firm_id, schema.quarter
    raw = pd.read_parquet(settings.output_layout.panel_path())
    caps = screening_thresholds_for_panel(raw)

    t0 = time.time()
    print("[canonical] require_ratio_compliance=True ...", flush=True)
    ctx_c, p7_c, v_c = _run(raw, settings, True)
    print(f"[canonical] done [{time.time()-t0:.0f}s]", flush=True)
    print("[sensitivity] require_ratio_compliance=False (retain violators) ...", flush=True)
    ctx_s, p7_s, v_s = _run(raw, settings, False)
    print(f"[sensitivity] done [{time.time()-t0:.0f}s]", flush=True)

    nC = int((ctx_c.panel["_split"] == SPLIT_LABEL_INCLUDED).sum())
    nCp = int((ctx_s.panel["_split"] == SPLIT_LABEL_INCLUDED).sum())
    hl_c = p7_c.summary.get("firm_level_headline", {})
    hl_s = p7_s.summary.get("firm_level_headline", {})

    def keyed(ctx, v):
        d = ctx.composites[[fc, qc]].copy()
        d["v"] = np.asarray(v)
        return d.set_index([fc, qc])["v"]

    kc, ks = keyed(ctx_c, v_c), keyed(ctx_s, v_s)
    common = kc.index.intersection(ks.index)
    a, b = kc.loc[common], ks.loc[common]
    red_c, red_s = (a == "RED"), (b == "RED")
    flips = {
        "n_common_rows": int(len(common)),
        "red_canonical": int(red_c.sum()),
        "red_sensitivity": int(red_s.sum()),
        "red_to_notred": int((red_c & ~red_s).sum()),
        "notred_to_red": int((~red_c & red_s).sum()),
    }

    # retained ratio-violating rows (identified from the canonical split reason)
    viol = ctx_c.panel[ctx_c.panel.get("_split_reason") == REASON_RATIO_VIOLATION].copy()
    rel = _max_rel_exceedance(viol, caps)
    viol_keys = pd.MultiIndex.from_frame(viol[[fc, qc]])
    vs_viol = ks.reindex(viol_keys).to_numpy()
    red_viol = (vs_viol == "RED")
    dec: dict = {
        "n_retained_violating_rows": int(len(viol)),
        "verdict_under_sensitivity": {k: int((vs_viol == k).sum())
                                      for k in ["RED", "AMBER", "GREEN", "NO_DATA"]},
        "rel_exceedance_quantiles": {q: round(float(np.nanpercentile(rel[np.isfinite(rel)], q)), 3)
                                     for q in (10, 25, 50, 75, 90)},
    }
    for thr in (0.10, 0.25, 0.50):
        disc = rel > thr
        dec[f"threshold_{int(thr*100)}pct"] = {
            "n_large_source_discrepancy": int(disc.sum()),
            "n_candidate_statistical_anomaly": int((~disc).sum()),
            "red_among_large_source_discrepancy": int((red_viol & disc).sum()),
            "red_among_candidate_statistical_anomaly": int((red_viol & ~disc).sum()),
        }

    result = {
        "scope": ("scoped sensitivity: retaining the ratio-cap-violating authority rows in C "
                  "(require_ratio_compliance=False); reported BESIDE the canonical run, never a "
                  "replacement of the canonical pipeline or the 324-firm headline"),
        "caps": {k: float(v) for k, v in caps.items()},
        "reference_sample_size": {"canonical_C": nC, "sensitivity_C": nCp, "retained": nCp - nC},
        "firm_headline_canonical": {"q<=0.01": hl_c.get("primary_claim", {}).get("q<=0.01"),
                                    "q<=0.05": hl_c.get("secondary", {}).get("q<=0.05")},
        "firm_headline_sensitivity": {"q<=0.01": hl_s.get("primary_claim", {}).get("q<=0.01"),
                                      "q<=0.05": hl_s.get("secondary", {}).get("q<=0.05")},
        "row_verdict_flips": flips,
        "retained_row_decomposition": dec,
        "decomposition_rule": ("max relative exceedance r=max_j (ratio_j-cap_j)/cap_j; r>0.25 = large "
                               "source discrepancy / likely filing-data mismatch; r<=0.25 = candidate "
                               "statistical anomaly; pre-registered, robustness at 0.10 and 0.50"),
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out) if args.out else Path(
        f"outputs/scores/{args.country}/sensitivity/ratio_compliance_sensitivity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
