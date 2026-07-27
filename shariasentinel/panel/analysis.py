"""
panel_analysis.py
─────────────────
Descriptive statistics, missingness analysis, anomaly detection and
consistency checks on the final SAC-anchored panel.

Outputs are written by `analyze.py` to `data_clean/reports/panel_analysis/`:
  - summary.json   : top-level metrics, machine-readable
  - *.csv          : long-form tables for human inspection
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Column groups ────────────────────────────────────────────────────────────
RATIO_COLS = [
    "ratio_debt", "ratio_debt_adj", "ratio_debt_upper",
    "ratio_cash", "ratio_cash_adj", "ratio_cash_strict", "ratio_cash_alt",
    "ratio_income", "ratio_income_idit", "ratio_income_cashadj",
    "ratio_income_idit_z0", "ratio_income_nopi",
]
INPUT_COLS = [
    "atq", "dlttq", "dlcq", "ltq", "cheq", "chq", "chsq",
    "revtq", "iditq", "nopiq", "xintq",
    "niq", "oibdpq", "ibq", "oancfy", "oancfq", "ppentq", "dpq", "xsgaq", "cogsq", "xrdq", "gpq",
    "actq", "lctq", "rectq", "invtq",
]
MACRO_COLS = ["sukuk_ratio_t", "islamic_cash_ratio_t"]
KEY_RATIOS = ["ratio_debt_adj", "ratio_cash_adj", "ratio_income"]

RATIO_THRESHOLDS = {
    "ratio_debt": 0.33, "ratio_debt_adj": 0.33, "ratio_debt_upper": 0.33,
    "ratio_cash": 0.33, "ratio_cash_adj": 0.33,
    "ratio_cash_strict": 0.33, "ratio_cash_alt": 0.33,
    "ratio_income": 0.05, "ratio_income_idit": 0.05,
    "ratio_income_cashadj": 0.05, "ratio_income_idit_z0": 0.05,
    "ratio_income_nopi": 0.05,
}

SANITY_BOUNDS = {
    "ratio_debt": 5.0,
    "ratio_debt_adj": 5.0,
    "ratio_cash": 1.0,
    "ratio_cash_adj": 1.0,
    "ratio_income": 2.0,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _add_year(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if "datadate" in df.columns:
        df["_year"] = pd.to_datetime(df["datadate"], errors="coerce").dt.year
    elif "datacqtr" in df.columns:
        df["_year"] = pd.to_numeric(
            df["datacqtr"].astype(str).str[:4], errors="coerce"
        )
    return df


def _nan_pivot(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """For each value col, compute n / n_missing / pct_missing per group."""
    if not value_cols:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    sizes = g.size()
    rows = []
    for c in value_cols:
        n_miss = g[c].apply(lambda s: int(s.isna().sum()))
        sub = pd.DataFrame({
            "n": sizes.values,
            "n_missing": n_miss.values,
        }, index=sizes.index).reset_index()
        sub.insert(0, "column", c)
        sub["pct_missing"] = (100 * sub["n_missing"] / sub["n"].clip(lower=1)).round(2)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _describe(s: pd.Series, threshold: float | None = None) -> dict | None:
    s = s.dropna()
    if len(s) == 0:
        return None
    out = {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "p10": round(float(s.quantile(0.10)), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "p75": round(float(s.quantile(0.75)), 4),
        "p90": round(float(s.quantile(0.90)), 4),
        "p99": round(float(s.quantile(0.99)), 4),
        "max": round(float(s.max()), 4),
    }
    if threshold is not None:
        n_above = int((s > threshold).sum())
        out["n_above_threshold"] = n_above
        out["pct_above_threshold"] = round(100 * n_above / len(s), 2)
    return out


# ── 1. Missingness ───────────────────────────────────────────────────────────
def analyze_missingness(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = _add_year(panel)
    cols = _present(df, RATIO_COLS + INPUT_COLS + MACRO_COLS)

    overall = pd.DataFrame({
        "column": cols,
        "n": len(df),
        "n_missing": [int(df[c].isna().sum()) for c in cols],
    })
    overall["pct_missing"] = (100 * overall["n_missing"] / overall["n"].clip(lower=1)).round(2)
    overall = overall.sort_values("pct_missing", ascending=False).reset_index(drop=True)

    out = {"overall": overall}

    if "_year" in df.columns:
        by_year = _nan_pivot(df, ["_year"], cols).rename(columns={"_year": "year"})
        out["by_year"] = by_year

    if "fqtr" in df.columns:
        out["by_fqtr"] = _nan_pivot(df, ["fqtr"], cols)

    if "sector" in df.columns:
        out["by_sector"] = _nan_pivot(df, ["sector"], cols)

    if "_synthetic_row" in df.columns:
        out["real_vs_synthetic"] = _nan_pivot(df, ["_synthetic_row"], cols)

    return out


# ── 2. Distributions ─────────────────────────────────────────────────────────
def analyze_distributions(
    panel: pd.DataFrame,
    exclude_synthetic: bool = True,
) -> dict[str, pd.DataFrame]:
    df = _add_year(panel)
    if exclude_synthetic and "_synthetic_row" in df.columns:
        df = df[~df["_synthetic_row"].fillna(False).astype(bool)].copy()

    cols = _present(df, RATIO_COLS)
    col_order = [
        "ratio", "n", "mean", "median", "std", "min",
        "p10", "p25", "p75", "p90", "p99", "max",
        "n_above_threshold", "pct_above_threshold",
    ]

    # overall
    rows = []
    for c in cols:
        d = _describe(df[c], RATIO_THRESHOLDS.get(c))
        if d is not None:
            d["ratio"] = c
            rows.append(d)
    overall = pd.DataFrame(rows)
    if not overall.empty:
        overall = overall[[c for c in col_order if c in overall.columns]]

    # by year
    by_year_rows = []
    if "_year" in df.columns:
        for c in cols:
            for year, sub in df.groupby("_year", dropna=True):
                d = _describe(sub[c], RATIO_THRESHOLDS.get(c))
                if d is not None:
                    d["ratio"] = c
                    d["year"] = int(year)
                    by_year_rows.append(d)
    by_year = pd.DataFrame(by_year_rows)
    if not by_year.empty:
        by_year = by_year[
            ["ratio", "year"] + [c for c in col_order if c not in ("ratio",) and c in by_year.columns]
        ]

    # by sector
    by_sector_rows = []
    if "sector" in df.columns:
        for c in cols:
            for sector, sub in df.groupby("sector", dropna=False):
                d = _describe(sub[c], RATIO_THRESHOLDS.get(c))
                if d is not None:
                    d["ratio"] = c
                    d["sector"] = sector if pd.notna(sector) else "(unknown)"
                    by_sector_rows.append(d)
    by_sector = pd.DataFrame(by_sector_rows)
    if not by_sector.empty:
        by_sector = by_sector[
            ["ratio", "sector"] + [c for c in col_order if c not in ("ratio",) and c in by_sector.columns]
        ]

    return {
        "overall": overall,
        "by_year": by_year,
        "by_sector": by_sector,
    }


# ── 3. Anomalies ─────────────────────────────────────────────────────────────
def detect_anomalies(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returns full (untruncated) anomaly DataFrames. Caller decides truncation."""
    df = _add_year(panel)

    def _gv(row):
        v = row.get("gvkey")
        return int(v) if pd.notna(v) else None

    out: dict[str, pd.DataFrame] = {}

    # 1. Extreme ratios (> p99.9 per column)
    # Sort by severity-relative magnitude (value / threshold) so no single
    # large-scale ratio monopolises the truncated output.
    extreme = []
    for c in _present(df, RATIO_COLS):
        s = df[c].dropna()
        if len(s) < 1000:
            continue
        threshold = float(s.quantile(0.999))
        sub = df[df[c] > threshold]
        for _, row in sub.iterrows():
            extreme.append({
                "gvkey": _gv(row),
                "conm": row.get("conm"),
                "datacqtr": row.get("datacqtr"),
                "ratio": c,
                "value": round(float(row[c]), 4),
                "threshold_p999": round(threshold, 4),
                "severity_ratio": round(float(row[c]) / threshold, 4) if threshold > 0 else None,
            })
    extreme_df = pd.DataFrame(extreme)
    if not extreme_df.empty:
        extreme_df = extreme_df.sort_values("severity_ratio", ascending=False).reset_index(drop=True)
    out["extreme_ratios"] = extreme_df

    # 2. Negative inputs (these inputs should never be negative)
    # Sort by |value| so every offending column is represented in the head-N
    # truncation (prevents one high-volume column from hiding the others).
    neg = []
    for c in _present(df, ["atq", "revtq", "cheq", "ltq", "chq", "chsq"]):
        sub = df[df[c] < 0]
        for _, row in sub.iterrows():
            neg.append({
                "gvkey": _gv(row),
                "conm": row.get("conm"),
                "datacqtr": row.get("datacqtr"),
                "column": c,
                "value": round(float(row[c]), 4),
                "abs_value": round(abs(float(row[c])), 4),
            })
    neg_df = pd.DataFrame(neg)
    if not neg_df.empty:
        neg_df = neg_df.sort_values("abs_value", ascending=False).reset_index(drop=True)
    out["negative_inputs"] = neg_df

    # 3. Sanity bounds (impossibly large ratios)
    sanity = []
    for c, bound in SANITY_BOUNDS.items():
        if c not in df.columns:
            continue
        sub = df[df[c] > bound]
        for _, row in sub.iterrows():
            sanity.append({
                "gvkey": _gv(row),
                "conm": row.get("conm"),
                "datacqtr": row.get("datacqtr"),
                "column": c,
                "value": round(float(row[c]), 4),
                "rule": f"{c} > {bound}",
            })
    sanity_df = pd.DataFrame(sanity)
    if not sanity_df.empty:
        sanity_df = sanity_df.sort_values("value", ascending=False).reset_index(drop=True)
    out["sanity_bounds"] = sanity_df

    # 4. Stale forward-fill (any *_lag > 4 → carried more than 1 year)
    stale = []
    lag_cols = [c for c in df.columns if c.endswith("_lag")]
    for c in lag_cols:
        sub = df[df[c] > 4]
        for _, row in sub.iterrows():
            stale.append({
                "gvkey": _gv(row),
                "conm": row.get("conm"),
                "datacqtr": row.get("datacqtr"),
                "column": c[:-4],  # strip "_lag"
                "lag_quarters": int(row[c]) if pd.notna(row[c]) else None,
            })
    stale_df = pd.DataFrame(stale)
    if not stale_df.empty:
        stale_df = stale_df.sort_values("lag_quarters", ascending=False).reset_index(drop=True)
    out["stale_ffill"] = stale_df

    # 5. YoY jumps on key ratios (>50pp absolute change in firm-year median)
    jumps = []
    if "gvkey" in df.columns and "_year" in df.columns:
        for c in _present(df, KEY_RATIOS):
            firm_year = (
                df.dropna(subset=[c, "_year"])
                  .groupby(["gvkey", "_year"])[c].median()
                  .reset_index()
                  .sort_values(["gvkey", "_year"])
            )
            firm_year["prev_value"] = firm_year.groupby("gvkey")[c].shift(1)
            firm_year["prev_year"]  = firm_year.groupby("gvkey")["_year"].shift(1)
            firm_year["delta"] = firm_year[c] - firm_year["prev_value"]
            big = firm_year.dropna(subset=["prev_value"])
            big = big[big["delta"].abs() > 0.50]
            for _, row in big.iterrows():
                jumps.append({
                    "gvkey": int(row["gvkey"]),
                    "ratio": c,
                    "year_from": int(row["prev_year"]),
                    "year_to": int(row["_year"]),
                    "value_from": round(float(row["prev_value"]), 4),
                    "value_to": round(float(row[c]), 4),
                    "delta_pp": round(float(row["delta"]) * 100, 2),
                })
    jumps_df = pd.DataFrame(jumps)
    if not jumps_df.empty:
        jumps_df = jumps_df.reindex(
            jumps_df["delta_pp"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    out["yoy_jumps"] = jumps_df

    return out


# ── 4. Consistency checks ────────────────────────────────────────────────────
def _add_check(
    checks: list,
    examples: list,
    df: pd.DataFrame,
    name: str,
    rule: str,
    required_cols: list[str],
    viol_fn,
    sample_cols: list[str],
    max_examples: int,
) -> None:
    if not set(required_cols).issubset(df.columns):
        return
    valid = df[required_cols].dropna()
    if valid.empty:
        return
    viol_mask = viol_fn(valid)
    n_eval = int(len(valid))
    n_viol = int(viol_mask.sum())
    pct = round(100 * n_viol / max(n_eval, 1), 4)
    checks.append({
        "check": name,
        "rule": rule,
        "n_eval": n_eval,
        "n_violations": n_viol,
        "pct_violations": pct,
        "status": "pass" if n_viol == 0 else "fail",
    })
    if n_viol > 0:
        idx = valid[viol_mask].index[:max_examples]
        ex_cols = [c for c in sample_cols if c in df.columns]
        ex = df.loc[idx, ex_cols].copy()
        ex.insert(0, "check", name)
        examples.append(ex)


def run_consistency_checks(
    panel: pd.DataFrame,
    max_examples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks: list[dict] = []
    examples: list[pd.DataFrame] = []
    eps = 1e-9

    _add_check(
        checks, examples, panel,
        "debt_adj_le_debt", "ratio_debt_adj <= ratio_debt",
        ["ratio_debt", "ratio_debt_adj"],
        lambda d: d["ratio_debt_adj"] > d["ratio_debt"] + eps,
        ["gvkey", "datacqtr", "ratio_debt", "ratio_debt_adj"],
        max_examples,
    )

    _add_check(
        checks, examples, panel,
        "cash_adj_le_cash", "ratio_cash_adj <= ratio_cash",
        ["ratio_cash", "ratio_cash_adj"],
        lambda d: d["ratio_cash_adj"] > d["ratio_cash"] + eps,
        ["gvkey", "datacqtr", "ratio_cash", "ratio_cash_adj"],
        max_examples,
    )

    _add_check(
        checks, examples, panel,
        "debt_upper_ge_debt", "ratio_debt_upper >= ratio_debt",
        ["ratio_debt", "ratio_debt_upper"],
        lambda d: d["ratio_debt_upper"] < d["ratio_debt"] - eps,
        ["gvkey", "datacqtr", "ratio_debt", "ratio_debt_upper"],
        max_examples,
    )

    _add_check(
        checks, examples, panel,
        "cheq_le_atq", "cheq <= atq",
        ["cheq", "atq"],
        lambda d: d["cheq"] > d["atq"] + eps,
        ["gvkey", "datacqtr", "cheq", "atq"],
        max_examples,
    )

    _add_check(
        checks, examples, panel,
        "debt_le_ltq", "dlttq + dlcq <= ltq",
        ["dlttq", "dlcq", "ltq"],
        lambda d: (d["dlttq"] + d["dlcq"]) > d["ltq"] + eps,
        ["gvkey", "datacqtr", "dlttq", "dlcq", "ltq"],
        max_examples,
    )

    # fqtr integrity
    if "fqtr" in panel.columns:
        valid = panel[panel["fqtr"].notna()]
        viol = valid[~valid["fqtr"].isin([1, 2, 3, 4])]
        n_eval = len(valid)
        n_viol = len(viol)
        checks.append({
            "check": "fqtr_in_range",
            "rule": "fqtr in {1,2,3,4}",
            "n_eval": n_eval,
            "n_violations": n_viol,
            "pct_violations": round(100 * n_viol / max(n_eval, 1), 4),
            "status": "pass" if n_viol == 0 else "fail",
        })
        if n_viol > 0:
            ex = viol[["gvkey", "datacqtr", "fqtr"]].head(max_examples).copy()
            ex.insert(0, "check", "fqtr_in_range")
            examples.append(ex)

    # Note: a `year(datadate) == year(datacqtr)` check was intentionally removed.
    # Compustat's `datacqtr` is assigned to the calendar quarter where most of the
    # fiscal period falls, not the quarter containing the close date. For firms
    # with non-December fiscal year-ends (e.g. Jan 31 close → datacqtr = prior Q4),
    # calendar years legitimately disagree. Both columns come directly from
    # Compustat; the pipeline does not derive one from the other.

    checks_df = pd.DataFrame(checks)
    examples_df = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()
    return checks_df, examples_df


# ── 5. Coverage ──────────────────────────────────────────────────────────────
def analyze_coverage(panel: pd.DataFrame) -> dict:
    df = _add_year(panel)
    out: dict = {}

    if "_year" in df.columns and "gvkey" in df.columns:
        agg = df.groupby("_year").agg(
            n_rows=("gvkey", "size"),
            n_firms=("gvkey", "nunique"),
        ).reset_index().rename(columns={"_year": "year"})
        if "_synthetic_row" in df.columns:
            syn = (
                df.groupby("_year")["_synthetic_row"]
                  .apply(lambda s: round(100 * s.fillna(False).astype(bool).mean(), 2))
                  .reset_index()
                  .rename(columns={"_year": "year", "_synthetic_row": "pct_synthetic"})
            )
            agg = agg.merge(syn, on="year", how="left")
        out["by_year"] = agg

    if "sector" in df.columns and "gvkey" in df.columns:
        agg = df.groupby("sector", dropna=False).agg(
            n_rows=("gvkey", "size"),
            n_firms=("gvkey", "nunique"),
        ).reset_index()
        if "_year" in df.columns:
            yr = df.groupby("sector", dropna=False)["_year"].agg(["min", "max"]).reset_index()
            yr.columns = ["sector", "year_min", "year_max"]
            agg = agg.merge(yr, on="sector", how="left")
        out["by_sector"] = agg

    macro_gaps: dict[str, list[int]] = {}
    for c in _present(df, MACRO_COLS):
        if "_year" in df.columns:
            yearly = df.groupby("_year")[c].apply(lambda s: s.notna().any())
            missing_years = [int(y) for y, ok in yearly.items() if not ok and pd.notna(y)]
            macro_gaps[c] = sorted(missing_years)
        else:
            macro_gaps[c] = []
    out["macro_gaps"] = macro_gaps

    return out


# ── 6. Summary builder ───────────────────────────────────────────────────────
def build_summary(
    panel: pd.DataFrame,
    missingness: dict,
    anomalies_full: dict,
    consistency_checks: pd.DataFrame,
    coverage: dict,
) -> dict:
    df_y = _add_year(panel)

    if "_synthetic_row" in panel.columns:
        synth = panel["_synthetic_row"].fillna(False).astype(bool)
    else:
        synth = pd.Series(False, index=panel.index)

    n_real = int((~synth).sum())
    n_synth = int(synth.sum())

    year_min = int(df_y["_year"].min()) if "_year" in df_y.columns and df_y["_year"].notna().any() else None
    year_max = int(df_y["_year"].max()) if "_year" in df_y.columns and df_y["_year"].notna().any() else None

    top_missing = (
        missingness["overall"].head(5).to_dict("records")
        if "overall" in missingness and not missingness["overall"].empty
        else []
    )

    anomaly_counts = {k: int(len(v)) for k, v in anomalies_full.items()}

    if not consistency_checks.empty:
        cc_pass = int((consistency_checks["status"] == "pass").sum())
        cc_fail = int((consistency_checks["status"] == "fail").sum())
        cc_details = consistency_checks.to_dict("records")
    else:
        cc_pass = cc_fail = 0
        cc_details = []

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel_shape": {
            "n_rows": int(len(panel)),
            "n_firms": int(panel["gvkey"].nunique()) if "gvkey" in panel.columns else None,
            "n_real_rows": n_real,
            "n_synthetic_rows": n_synth,
            "year_range": [year_min, year_max],
        },
        "top_missing_columns": top_missing,
        "anomaly_counts": anomaly_counts,
        "consistency_checks": {
            "passed": cc_pass,
            "failed": cc_fail,
            "details": cc_details,
        },
        "macro_gaps": coverage.get("macro_gaps", {}),
    }


# ── 7. Orchestrator ──────────────────────────────────────────────────────────
def _truncate(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df is None or df.empty or max_rows <= 0:
        return df if df is not None else pd.DataFrame()
    return df.head(max_rows).copy() if len(df) > max_rows else df


def run_full_analysis(panel: pd.DataFrame, max_anomalies: int = 200) -> dict:
    log.info("panel analysis: missingness")
    missingness = analyze_missingness(panel)

    log.info("panel analysis: distributions")
    distributions = analyze_distributions(panel, exclude_synthetic=True)

    log.info("panel analysis: anomalies")
    anomalies_full = detect_anomalies(panel)
    anomalies_truncated = {k: _truncate(v, max_anomalies) for k, v in anomalies_full.items()}

    log.info("panel analysis: consistency checks")
    consistency_checks, consistency_examples = run_consistency_checks(panel)

    log.info("panel analysis: coverage")
    coverage = analyze_coverage(panel)

    log.info("panel analysis: summary")
    summary = build_summary(panel, missingness, anomalies_full, consistency_checks, coverage)

    return {
        "summary": summary,
        "missingness": missingness,
        "distributions": distributions,
        "anomalies": anomalies_truncated,
        "consistency_checks": consistency_checks,
        "consistency_examples": consistency_examples,
        "coverage": coverage,
    }


# ── 8. I/O ───────────────────────────────────────────────────────────────────
def _json_safe(obj):
    """Recursively convert numpy/pandas/NaN values to JSON-safe Python types."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


def write_analysis_outputs(results: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(results["summary"]), f, indent=2, ensure_ascii=False)
    log.info("wrote %s", summary_path)

    # Long tables → CSV (prefix per section)
    section_prefix = {
        "missingness": "nan",
        "distributions": "ratio_stats",
    }
    for section, prefix in section_prefix.items():
        for name, df in results.get(section, {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = out_dir / f"{prefix}_{name}.csv"
                df.to_csv(path, index=False)
                log.info("wrote %s (%d rows)", path, len(df))

    # Anomalies
    for name, df in results.get("anomalies", {}).items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            path = out_dir / f"anomalies_{name}.csv"
            df.to_csv(path, index=False)
            log.info("wrote %s (%d rows)", path, len(df))

    # Consistency
    cc = results.get("consistency_checks", pd.DataFrame())
    if isinstance(cc, pd.DataFrame) and not cc.empty:
        cc.to_csv(out_dir / "consistency_checks.csv", index=False)
        log.info("wrote %s (%d rows)", out_dir / "consistency_checks.csv", len(cc))

    ce = results.get("consistency_examples", pd.DataFrame())
    if isinstance(ce, pd.DataFrame) and not ce.empty:
        ce.to_csv(out_dir / "consistency_violations_examples.csv", index=False)
        log.info("wrote %s (%d rows)", out_dir / "consistency_violations_examples.csv", len(ce))

    # Coverage
    for name, val in results.get("coverage", {}).items():
        if isinstance(val, pd.DataFrame) and not val.empty:
            path = out_dir / f"coverage_{name}.csv"
            val.to_csv(path, index=False)
            log.info("wrote %s (%d rows)", path, len(val))
        # macro_gaps is a dict — already in summary.json
