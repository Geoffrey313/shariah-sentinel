"""Collect the paper's reproduced numbers from this repo into ONE manifest.

Reads the reconstructed pipeline outputs under ``data/scores/<country>/`` and
centralises the canonical artifacts (phase JSONs, the ``paper_ready`` tables, the
qualitative cross-sections, the ablation table, family3) plus the per-country
reference/FDR headline for the jurisdictions table. A reader can diff the manifest
against the published one to confirm their reproduction matches.

This is a pure read + centralise step; it does not recompute anything. Sections
that this minimal repo generates on demand rather than shipping (the counterfactual
``family3_cf_*`` and the end-to-end contamination table) show ``_missing`` until the
matching ``reproduce.py`` command is run; the figure-derived reductions (U-shape,
IUT gap, integrity Jaccard, power matrix) are not covered here.

Entry point (mys anchor + 5-country table):
    python reproduce.py numbers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.common.config import AnalysisSettings  # noqa: E402

# A fresh reproduction writes here; the committed numbers_manifest.json is the
# reference to diff against, so a data-less run never clobbers it.
OUT = REPO / "numbers_manifest.local.json"

# Anchor panel (all heavy studies) + the five jurisdictions of the cross-country table.
ANCHOR = "mys"
JURISDICTIONS = ["mys", "idn", "pak", "sau", "uae"]

# Compustat detector code -> manuscript symbol (post z5/z7 merge = z57).
DETECTOR_SYMBOL = {
    "z1": "z_benford", "z2": "z_zipf", "z3": "z_mscore", "z4": "z_threshold",
    "z6": "z_temporal", "z8": "z_cod", "z57": "z_fused",
}


def _layout(country: str):
    s = AnalysisSettings()
    return s.output_layout.model_copy(
        update={"country_code_lower": country, "root": str(REPO / "data")})


def _scores_dir(country: str) -> Path:
    """Base ``outputs/scores/<country>`` folder (parent of every phase dir)."""
    return _layout(country).phase0_dir().parent


def _rel(path: Path) -> str:
    """Repo-relative POSIX path, so the manifest carries no machine paths."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path):
    if not path.exists():
        return {"_missing": _rel(path)}
    return json.loads(path.read_text())


def _load_csv(path: Path):
    """CSV -> list of row dicts, or a missing marker."""
    if not path.exists():
        return {"_missing": _rel(path)}
    return pd.read_csv(path).to_dict(orient="records")


REPRODUCE_COMMANDS = [
    "python reproduce.py pipeline   --country mys",
    "python reproduce.py benchmark  --country mys",
    "python reproduce.py family3-cf --country mys --rows 500 --near-boundary",
    "python reproduce.py family3-pgd --country mys --rows 5",
    "python reproduce.py ablation   --country mys",
    "python reproduce.py pipeline   --country {idn,pak,sau,uae}",
    "python reproduce.py figure-data",
    "python reproduce.py numbers",
]


def provenance() -> dict:
    """Honest, stable metadata for the reference manifest: how it is reproduced
    and how randomness is fixed. No timestamps, machine paths, or run-specific
    state, so the manifest keeps a stable digest across machines and re-runs."""
    return {
        "reproduced_from": (
            "The transformed panels are obtained on request (see data/README.md), "
            "placed under data/, and processed by the commands below. This manifest "
            "is the reference output to diff a fresh reproduction against."
        ),
        "seeds_note": "All randomness is seeded in AnalysisSettings "
        "(monte_carlo_seed / bootstrap.random_seed / injection.random_seed / "
        "robustness_benchmark.random_seed), so the run is deterministic.",
        "reproduce_commands": REPRODUCE_COMMANDS,
    }


def reference_and_fdr(country: str) -> dict:
    """Reference-sample size, panel shape, and firm-level FDR for one country."""
    lay = _layout(country)
    p0 = _load_json(lay.phase0_dir() / "reference_sample.json")
    p7 = _load_json(lay.phase7_dir() / "phase7_fdr.json")
    size = None
    if isinstance(p0, dict):
        size = p0.get("global", {}).get("size") if "global" in p0 else p0.get("c_rows")
    return {
        "reference_sample_size": size,
        "phase0": p0,
        "firm_quarters": p7.get("row_count") if isinstance(p7, dict) else None,
        "firms": p7.get("firm_count") if isinstance(p7, dict) else None,
        # Firm-level HEADLINE: the valid episodic-exceedance test.
        "fdr_firm_level_headline": p7.get("firm_level_headline") if isinstance(p7, dict) else None,
        "fdr_firm_level_panel": p7.get("firm_level_panel") if isinstance(p7, dict) else None,
        # min-p per-composite is kept only as a diagnostic (anti-conservative), never the headline.
        "fdr_firm_level_minp_diagnostic": (
            p7.get("firm_level_discoveries") if isinstance(p7, dict) else None
        ),
        "fdr_row_level": p7.get("row_level_discoveries") if isinstance(p7, dict) else None,
    }


def calibration(country: str) -> dict:
    """Per-detector null moments + KS verdict, plus dependence (lambda, K_eff)."""
    lay = _layout(country)
    nc = _load_json(lay.phase1_dir() / "null_calibration.json")
    dep = _load_json(lay.phase2_dir() / "dependence.json")
    per = {}
    if isinstance(nc, dict):
        for code, block in (nc.get("per_detector") or {}).items():
            per[DETECTOR_SYMBOL.get(code, code)] = {
                "code": code,
                "moments": block.get("moments"),
                "ks_pvalue": (block.get("tests") or {}).get("ks_pvalue"),
                "verdict": block.get("verdict"),
                "forced_bootstrap": block.get("forced_bootstrap"),
            }
    return {
        "c_rows": nc.get("c_rows") if isinstance(nc, dict) else None,
        "per_detector": per,
        "ledoit_wolf_shrinkage": dep.get("ledoit_wolf_shrinkage") if isinstance(dep, dict) else None,
        "complete_rows": dep.get("complete_rows") if isinstance(dep, dict) else None,
        "k_eff": (dep.get("pca") or {}).get("k_eff") if isinstance(dep, dict) else None,
    }


def qualitative(country: str) -> dict:
    """Cross-sections (sector / size / leverage), persistence buckets, chronic firms."""
    q = _scores_dir(country) / "qualitative"
    return {
        "verdict_by_sector": _load_csv(q / "verdict_by_sector.csv"),
        "verdict_by_firm_size": _load_csv(q / "verdict_by_firm_size.csv"),
        "verdict_by_debt_level": _load_csv(q / "verdict_by_debt_level.csv"),
        "verdict_persistence": _load_csv(q / "verdict_persistence.csv"),
        "chronic_red_firms_n": _n_rows(q / "chronic_red_firms.csv"),
        "ablation_results": _load_csv(q / "ablation_results.csv"),
    }


def _n_rows(path: Path):
    return len(pd.read_csv(path)) if path.exists() else None


def contamination_and_family3(country: str) -> dict:
    """Canonical paper_ready contamination + family3 tables, and the CF summary."""
    rb = _layout(country).robustness_benchmark_dir()
    pr = rb / "paper_ready"
    return {
        "contamination_auc_zmah": _load_csv(pr / "contamination_auc_table.csv"),
        "family3_evasion_summary": _load_csv(pr / "family3_summary_table.csv"),
        "robustness_summary": _load_json(rb / "robustness_benchmark_summary.json"),
        "counterfactual": _counterfactual_summary(rb),
    }


def _cf_stat(changed: pd.Series, mask: pd.Series) -> dict:
    """Count, rate, and edit-cost of the rows selected by ``mask``."""
    n = int(mask.sum())
    sub = changed[mask]
    return {
        "n": n,
        "rate": float(mask.mean()),
        "median_vars": float(sub.median()) if n else None,
        "share_ge5_vars_pct": float((sub >= 5).mean() * 100.0) if n else None,
    }


def _counterfactual_summary(rb: Path) -> dict:
    """Counterfactual cost, separating TWO distinct notions of success:

    - ``targeted_score``: the optimised composite ($Z^+_{renorm}$) crosses its own
      green threshold. This is what the optimiser's ``success`` flag records.
    - ``global_green``: the row's FULL tri-state verdict actually becomes GREEN
      (``final_status``). This is the honest "flag erased" event, and it is both
      rarer and costlier than crossing the targeted score alone.
    """
    hits = [p for p in rb.glob("family3_cf_*top*.csv") if "top3" not in p.name]
    if not hits:
        return {"_missing": "family3_cf_*top*.csv"}
    freshest = max(hits, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(freshest)
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    changed = (df[delta_cols].fillna(0).abs() > 1e-9).sum(axis=1)
    out = {"source": freshest.name, "n_rows": int(len(df))}
    if "success" in df.columns:
        out["targeted_score"] = _cf_stat(changed, df["success"] == 1)
    if "final_status" in df.columns:
        green = df["final_status"].astype(str).str.upper().eq("GREEN")
        out["global_green"] = _cf_stat(changed, green)
        out["final_status_counts"] = {
            str(k): int(v) for k, v in df["final_status"].value_counts().items()
        }
    if "baseline_status" in df.columns:
        out["baseline_status_counts"] = {
            str(k): int(v) for k, v in df["baseline_status"].value_counts().items()
        }
    return out


def sector_false_positive(country: str) -> dict:
    fp = _load_csv(_layout(country).phase6_dir() / "sector_false_positive.csv")
    return {"rows": fp}


def build() -> dict:
    anchor = {
        "reference_and_fdr": reference_and_fdr(ANCHOR),
        "calibration": calibration(ANCHOR),
        "qualitative": qualitative(ANCHOR),
        "contamination_and_family3": contamination_and_family3(ANCHOR),
        "sector_false_positive": sector_false_positive(ANCHOR),
    }
    jur = {c: reference_and_fdr(c) for c in JURISDICTIONS}
    totals = {
        "firms": sum((jur[c]["firms"] or 0) for c in JURISDICTIONS),
        "firm_quarters": sum((jur[c]["firm_quarters"] or 0) for c in JURISDICTIONS),
    }
    return {
        "anchor_country": ANCHOR,
        "anchor": anchor,
        "jurisdictions": jur,
        "cross_country_totals": totals,
        "detector_symbol_map": DETECTOR_SYMBOL,
        "provenance": provenance(),
    }


def main() -> None:
    manifest = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {OUT}")
    # Echo the load-bearing headline numbers for a quick eyeball.
    rf = manifest["anchor"]["reference_and_fdr"]
    cal = manifest["anchor"]["calibration"]
    hl = rf.get("fdr_firm_level_headline") or {}
    dg = rf.get("fdr_firm_level_minp_diagnostic") or {}
    print(f"  |C|={rf['reference_sample_size']}  firms={rf['firms']}  fq={rf['firm_quarters']}")
    print(f"  lambda={cal['ledoit_wolf_shrinkage']}  complete_rows={cal['complete_rows']}  K_eff={cal['k_eff']}")
    if hl:
        print(f"  firm HEADLINE (exceedance) q<=0.01={hl.get('primary_claim', {}).get('q<=0.01')}  "
              f"q<=0.05={hl.get('secondary', {}).get('q<=0.05')}  [{hl.get('composite')}]")
    if "p_t_iut" in dg:
        print(f"  [min-p DIAGNOSTIC, not headline] t_iut q<=0.01={dg['p_t_iut'].get('q<=0.01')}  "
              f"z_mah q<=0.01={dg['p_z_mahalanobis_sq'].get('q<=0.01')}")
    print(f"  cross-country: {manifest['cross_country_totals']}")


if __name__ == "__main__":
    main()
