"""Regenerate the paper's figure-data CSVs from this repo's reproduced outputs.

The `paper-repro` twin of the product's `scripts/dump_paper_figure_data.py`: same
data reductions (power curves, IUT gap, integrity Jaccard, AnoShift FPR, U-shape
fit, near-red composition), but self-contained — `src.*` imports, the
`data/` root, and the handful of `client.figures` helpers inlined here so no
plotting stack is needed. Pure read + reshape; recomputes nothing.

Run AFTER `reproduce.py pipeline` (+ `benchmark`, and `family3-cf` for the near-red
figure) so the phase outputs exist:
    python dump_paper_figure_data.py
Writes CSVs to `figures/data/`; each figure is emitted independently and skips with
a note if its input is not present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.common.config import AnalysisSettings  # noqa: E402

OUT = REPO / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# ── Inlined from client.figures / client.style / scripts.family3_cf_paper_outputs ──
_P_COLS = ["p_z_plus", "p_z_plus_renorm", "p_z_mahalanobis_sq", "p_t_iut"]
COMPOSITE_LABELS = {
    "z_plus": r"$Z^+$",
    "z_plus_renorm": r"$Z^+_{\mathrm{renorm}}$",
    "z_mahalanobis_sq": r"$Z^2_{\mathrm{Mah}}$",
    "t_iut": r"$T_{\mathrm{IUT}}$",
    "breadth": r"$B_{\mathcal{A}}$",
}
VAR_LABELS = {
    "atq": "Assets", "dlcq": "Short-term debt", "cheq": "Cash", "revtq": "Revenue",
    "dlttq": "Long-term debt", "ltq": "Total liabilities", "niq": "Net income",
    "ibq": "Income before extra.", "oibdpq": "Operating income (EBITDA)",
    "nopiq": "Non-operating income", "iditq": "Interest & related income",
    "xintq": "Interest expense", "oancfq": "Operating cash flow", "actq": "Current assets",
    "lctq": "Current liabilities", "rectq": "Receivables", "invtq": "Inventory",
    "xsgaq": "SG&A expense", "ppentq": "Net PP&E",
}


def _settings():
    s = AnalysisSettings()
    return s.model_copy(update={"output_layout": s.output_layout.model_copy(
        update={"country_code_lower": "mys", "root": str(REPO / "data")})})


def _verdict(composites: pd.DataFrame, red_thr: float, amber_thr: float) -> pd.Series:
    present = [c for c in _P_COLS if c in composites.columns]
    min_p = composites[present].min(axis=1)
    v = pd.Series("GREEN", index=composites.index)
    v[min_p < amber_thr] = "AMBER"
    v[min_p < red_thr] = "RED"
    return v


def _cf_mean_share(csv: Path) -> pd.Series:
    """Mean per-variable share of the standardised perturbation over successful flips."""
    df = pd.read_csv(csv)
    raw = [c[len("delta_"):] for c in df.columns if c.startswith("delta_")]
    succ = df[df["success"] == True].copy()  # noqa: E712
    shares = pd.DataFrame(index=succ.index, columns=raw, dtype=float)
    for col in raw:
        d = pd.to_numeric(succ.get(f"delta_{col}"), errors="coerce")
        b = pd.to_numeric(succ.get(f"baseline_{col}"), errors="coerce")
        shares[col] = (d / np.maximum(b.abs(), 1.0)).abs()
    row_tot = shares.sum(axis=1).replace(0, np.nan)
    return (shares.div(row_tot, axis=0).mean(axis=0) * 100).sort_values(ascending=False)


def main() -> None:
    s = _settings()
    o = s.output_layout
    alpha = s.injection.primary_alpha_for_mde
    red_thr, amber_thr = s.figures.red_threshold, s.figures.amber_threshold
    written = []

    def _skip(name: str, why: str) -> None:
        print(f"  [skip] {name}: {why}")

    # 1. power curves (one file per archetype) ------------------------------
    pc_path = o.phase5_dir() / "power_curves.csv"
    if pc_path.exists():
        pc = pd.read_csv(pc_path)
        pc = pc[(pc["alpha"] == alpha) & (pc["composite"].isin(COMPOSITE_LABELS))]
        for arch in sorted(pc["archetype"].unique()):
            g = pc[pc["archetype"] == arch]
            w = g.pivot_table(index="delta", columns="composite", values="detection_rate").reset_index()
            w = w[["delta"] + [c for c in COMPOSITE_LABELS if c in w.columns]]
            w.to_csv(OUT / f"power_{arch}.csv", index=False)
            written.append(f"power_{arch}.csv")
    else:
        _skip("power_*", f"missing {pc_path.name} (run: reproduce.py pipeline)")

    # 2. IUT gap ------------------------------------------------------------
    ig_path = o.phase5_dir() / "theoretical_empirical_comparison.csv"
    if ig_path.exists():
        ig = pd.read_csv(ig_path)
        ig = ig[ig["alpha"] == alpha].sort_values("delta")[
            ["delta", "empirical_t_iut_power", "theoretical_t_iut_power"]]
        ig["gap_pp"] = (ig["empirical_t_iut_power"] - ig["theoretical_t_iut_power"]) * 100
        ig.to_csv(OUT / "iut_gap.csv", index=False)
        written.append("iut_gap.csv")
    else:
        _skip("iut_gap", f"missing {ig_path.name}")

    # 3. integrity Jaccard --------------------------------------------------
    ij_path = o.phase6_dir() / "integrity_sensitivity.csv"
    if ij_path.exists():
        ij = pd.read_csv(ij_path).sort_values("jaccard_top_n")
        ij["label"] = [COMPOSITE_LABELS.get(c, c) for c in ij["composite"]]
        ij[["composite", "label", "jaccard_top_n"]].to_csv(OUT / "integrity_jaccard.csv", index=False)
        written.append("integrity_jaccard.csv")
    else:
        _skip("integrity_jaccard", f"missing {ij_path.name}")

    # 4. AnoShift FPR -------------------------------------------------------
    an_path = o.robustness_benchmark_dir() / "family4_anoshift.csv"
    if an_path.exists():
        an = pd.read_csv(an_path)
        an = an[an["kind"] == "fpr"]
        piv = an.pivot_table(index="entity_name", columns="split", values="fpr", aggfunc="mean")
        piv = piv.loc[[m for m in COMPOSITE_LABELS if m in piv.index]]
        piv.to_csv(OUT / "anoshift_fpr.csv")
        written.append("anoshift_fpr.csv")
    else:
        _skip("anoshift_fpr", f"missing {an_path.name} (run: reproduce.py benchmark)")

    # 5. near-red variable composition (needs a generated counterfactual) ---
    cfs = [p for p in o.robustness_benchmark_dir().glob("family3_cf_*top*.csv") if "top3" not in p.name]
    if cfs:
        ms = _cf_mean_share(max(cfs, key=lambda p: p.stat().st_mtime)).head(6)
        pd.DataFrame({"variable": [VAR_LABELS.get(c, c) for c in ms.index],
                      "share_pct": ms.values}).to_csv(OUT / "near_red_composition.csv", index=False)
        written.append("near_red_composition.csv")
    else:
        _skip("near_red_composition", "no family3_cf_*.csv (run: reproduce.py family3-cf)")

    # 6. U-shape (compliant [0,0.33] / non-compliant (0.33,1.0]) ------------
    comp_path = o.phase4_dir() / "composites_panel.parquet"
    p0_path = o.phase0_dir() / "panel_with_split.parquet"
    if comp_path.exists() and p0_path.exists():
        comp = pd.read_parquet(comp_path)
        present = [c for c in _P_COLS if c in comp.columns]
        comp["is_red"] = (comp[present].min(axis=1) < red_thr).astype(float)
        ratio_df = pd.read_parquet(p0_path, columns=["gvkey", "datacqtr", "ratio_debt_adj"])
        comp = comp.merge(ratio_df, on=["gvkey", "datacqtr"], how="left")
        fit_rows = []
        for name, lo, hi in [("compliant", 0.0, 0.33), ("noncompliant", 0.33, 1.0)]:
            v = comp[(comp["ratio_debt_adj"] >= lo) & (comp["ratio_debt_adj"] <= hi)].copy()
            if len(v) < 100:
                _skip(f"ushape_{name}", f"only {len(v)} rows")
                continue
            v["bin"] = pd.cut(v["ratio_debt_adj"], bins=20)
            b = v.groupby("bin", observed=True).agg(
                mid=("ratio_debt_adj", "median"), red_rate=("is_red", "mean"),
                n=("is_red", "count")).reset_index()
            b = b[b["n"] >= 20].sort_values("mid")
            b["red_pct"] = b["red_rate"] * 100
            b[["mid", "red_pct", "n"]].to_csv(OUT / f"ushape_{name}_points.csv", index=False)
            x, y, w = b["mid"].values, b["red_pct"].values, b["n"].values
            coeffs = np.polyfit(x, y, deg=2, w=np.sqrt(w))
            xs = np.linspace(x.min(), x.max(), 60)
            pd.DataFrame({"x": xs, "y": np.polyval(coeffs, xs)}).to_csv(
                OUT / f"ushape_{name}_fit.csv", index=False)
            x_min = -coeffs[1] / (2 * coeffs[0])
            r2 = 1 - np.sum(w * (y - np.polyval(coeffs, x)) ** 2) / np.sum(w * (y - y.mean()) ** 2)
            fit_rows.append({"panel": name, "r2": round(float(r2), 3),
                             "x_min": round(float(x_min), 3),
                             "y_min": round(float(np.polyval(coeffs, x_min)), 2),
                             "red_lo": round(float(y.min()), 1), "red_hi": round(float(y.max()), 1),
                             "n_bins": len(b)})
            written.append(f"ushape_{name}_points.csv")
        if fit_rows:
            pd.DataFrame(fit_rows).to_csv(OUT / "ushape_fit_summary.csv", index=False)
            written.append("ushape_fit_summary.csv")
            print("  ushape fit:", fit_rows)
    else:
        _skip("ushape_*", "missing composites_panel / panel_with_split")

    print(f"\nDONE. {len(written)} CSVs in {OUT}")


if __name__ == "__main__":
    main()
