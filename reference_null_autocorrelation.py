"""Reference-sample estimate of the null serial dependence, the rho the firm-level
calibration sits above.

Scope. The firm-level exceedance test draws its null from a length-K uniform sequence
with an AR(1) Gaussian copula of autocorrelation ``exceedance_rho_cal`` (0.30), chosen
above what the reference sample shows. The manuscript quotes that reference estimate,
but no estimator produced it: only the calibration value appeared in the outputs. This
script estimates it on the same scale as the null it calibrates.

Two scales, both emitted, because they are not comparable. On the copula scale each
firm's quarterly unanimity p-values are mapped to normal scores e_t = Phi^-1(p_t), the
latent series the AR(1) copula generates, and rho is their pooled lag-1 correlation
within a firm; that is the scale rho_cal lives on. On the exceedance scale the same
lag-1 correlation is taken on the indicators 1{p_t <= tau} over the tau grid the
firm statistic counts, which is the quantity the manuscript quotes.

It changes no verdict and no calibration: rho_cal stays where it is, and this only
publishes the quantity it is set above.

Requires the licensed panel (not self-contained); the p-values come from the
reference bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.config import AnalysisSettings
from src.analysis.reference_sample import run_phase0, SPLIT_LABEL_INCLUDED

CLIP = 1e-6


def _lag1_pairs(df: pd.DataFrame, firm: str, quarter: str, p_col: str, transform) -> np.ndarray:
    """Consecutive within-firm (x_t, x_{t+1}) pairs on C, x = transform(p)."""
    pairs = []
    for _, g in df.sort_values([firm, quarter]).groupby(firm, observed=True):
        p = pd.to_numeric(g[p_col], errors="coerce").to_numpy(dtype=float)
        e = transform(p)
        ok = np.isfinite(e)
        e = e[ok]
        if len(e) >= 2:
            pairs.append(np.column_stack([e[:-1], e[1:]]))
    return np.vstack(pairs) if pairs else np.empty((0, 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="mys")
    ap.add_argument("--composite", default="p_t_iut")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = AnalysisSettings()
    settings = base.model_copy(update={"output_layout": base.output_layout.model_copy(
        update={"country_code_lower": args.country.lower(),
                "root": str(Path(__file__).resolve().parent / "data")})})
    schema = settings.panel_schema
    firm, quarter = schema.firm_id, schema.quarter

    raw = pd.read_parquet(settings.output_layout.panel_path())
    panel = run_phase0(panel=raw, settings=settings, write_outputs=False).panel
    composites = pd.read_parquet(
        settings.output_layout.phase4_dir() / "composites_panel.parquet")

    # The bundle is de-identified, so firm ids do not join across the two frames;
    # they are row-aligned instead, which the quarter column verifies.
    panel = panel.reset_index(drop=True)
    comp = composites.reset_index(drop=True)
    if len(panel) != len(comp) or not (
            panel[quarter].astype(str).values == comp[quarter].astype(str).values).all():
        raise ValueError("panel and composites are not row-aligned on the quarter column.")
    ref = comp[(panel["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()].copy()

    def _rho(transform) -> float:
        P = _lag1_pairs(ref, firm, quarter, args.composite, transform)
        return float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]) if len(P) > 1 else float("nan")

    normal = lambda p: norm.ppf(np.clip(p, CLIP, 1.0 - CLIP))
    pairs = _lag1_pairs(ref, firm, quarter, args.composite, normal)
    rho = _rho(normal)
    taus = tuple(float(x) for x in settings.fdr.exceedance_taus)
    rho_exceedance = {f"tau={t:g}": round(_rho(lambda p, th=t: (p <= th).astype(float)), 4)
                      for t in taus}

    result = {
        "scope": ("Reference-sample lag-1 autocorrelation of the within-firm quarterly "
                  "p-values on the normal-score scale of the AR(1) copula null. "
                  "Published only; the calibration rho is not changed."),
        "composite": args.composite,
        "reference_sample_rows": int(len(ref)),
        "firms_with_two_or_more_quarters": int(ref.groupby(firm, observed=True).size().ge(2).sum()),
        "consecutive_pairs": int(len(pairs)),
        "rho_hat_copula": round(rho, 4),
        "rho_hat_exceedance_indicators": rho_exceedance,
        "rho_calibration": float(settings.fdr.exceedance_rho_cal),
        "_note": ("rho_calibration parameterises the AR(1) copula, so it compares with "
                  "rho_hat_copula; the indicator autocorrelations are a different scale "
                  "and are reported for the tau grid the firm statistic counts."),
    }

    out = Path(args.out) if args.out else (
        settings.output_layout.phase7_dir() / "reference_null_autocorrelation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
