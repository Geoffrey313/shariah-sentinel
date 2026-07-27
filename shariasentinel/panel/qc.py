
import json
from typing import Dict

import pandas as pd
import re
from .mapping import build_mapping_compustat, intersection_stats, _name_keywords

def _name_match_score(name_sc: str, name_comp: str) -> float:
    """Jaccard score on significant company-name keywords."""
    s1 = _name_keywords(name_sc)
    s2 = _name_keywords(name_comp)

    if not s1 or not s2:
        return 0.0
    if s1 <= s2 or s2 <= s1:
        return 100.0
    return round(100 * len(s1 & s2) / len(s1 | s2), 2)

def build_qc_report(
    df_sc: pd.DataFrame,
    sc_reports: list[dict],
    df_comp: pd.DataFrame,
    comp_report: dict,
    panel: pd.DataFrame,
    df_comp_pre_pydantic=None
) -> Dict:
    mapping = build_mapping_compustat(df_comp)


    # ── SC Malaysia ───────────────────────────────────────────────────────────
    sc_section = {
        "sc_files":              len(sc_reports),
        "sc_rows_raw":           sum(r["raw_rows"]           for r in sc_reports),
        "sc_rows_valid":         sum(r["valid_rows"]         for r in sc_reports),
        "sc_duplicates_removed": sum(r.get("duplicates_removed", 0) for r in sc_reports),
        "sc_rows_final":         int(len(df_sc)),
        "sc_validation_errors":  sum(r.get("validation_errors_n", 0) for r in sc_reports),
        "sc_unique_stock_code":  int(df_sc["stock_code"].nunique())
                                 if "stock_code" in df_sc.columns else 0,
        "sc_unique_quarter":     int(df_sc["quarter"].nunique())
                                 if "quarter" in df_sc.columns else 0,
        "sc_quarter_format":     "YYYY-Qn",
        "sc_quarter_range": (
            [str(df_sc["quarter"].min()), str(df_sc["quarter"].max())]
            if "quarter" in df_sc.columns and not df_sc.empty else []
        ),
    }

    # ── Compustat ─────────────────────────────────────────────────────────────
    comp_section = {
        "compustat_rows_raw":           int(comp_report.get("raw_rows", 0)),
        "compustat_rows_after_country":  int(comp_report.get("after_country_filter", 0)),
        "compustat_duplicates_removed":  int(comp_report.get("duplicates_removed", 0)),
        "compustat_rows_valid":          int(len(df_comp)),
        "compustat_validation_errors":   int(comp_report.get("validation_errors_n", 0)),
        "compustat_unique_gvkey":        int(df_comp["gvkey"].nunique())
                                         if "gvkey" in df_comp.columns else 0,
        "compustat_unique_quarter":      int(df_comp["datacqtr"].nunique())
                                         if "datacqtr" in df_comp.columns else 0,
        "compustat_quarter_format":      "YYYY-Qn",
        "compustat_quarter_range": (
            [str(df_comp["datacqtr"].min()), str(df_comp["datacqtr"].max())]
            if "datacqtr" in df_comp.columns and not df_comp.empty else []
        ),
        # Currency
        "compustat_non_myr_obs":        int(comp_report.get("non_local_currency_obs", 0)),
        "compustat_non_myr_currencies": comp_report.get("non_local_currencies", []),
    }

    # ── Mapping / intersection ────────────────────────────────────────────────
    inter = intersection_stats(df_sc, mapping, df_comp_pre_pydantic)

    mapping_section = {
        "mapping_total_gvkey":            int(len(mapping)),
        # strict 3:7 extraction
        "mapping_isin_3_7_ok":            int((mapping["status_3_7"] == "ok").sum()),
        "mapping_isin_3_7_status":        mapping["status_3_7"].value_counts().to_dict(),
        # flexible extraction (recommended)
        "mapping_isin_flex_ok":           int((mapping["status_flex"] == "ok").sum()),
        "mapping_isin_flex_status":       mapping["status_flex"].value_counts().to_dict(),
        **inter,
    }
    # if "sc_codes_not_in_compustat_diagnosed" in inter:
    #     mapping_section["sc_missing_diagnosed"] = inter.pop("sc_codes_not_in_compustat_diagnosed")

    # ── Panel ─────────────────────────────────────────────────────────────────
    panel_section: Dict = {}
    if not panel.empty:
        panel_section = {
            "panel_rows":           int(len(panel)),
            "panel_unique_gvkey":   int(panel["gvkey"].nunique()),
            "panel_unique_quarter": int(
                panel["datacqtr"].nunique() if "datacqtr" in panel.columns
                else panel["datadate"].nunique()
            ),
            "panel_quarter_format": "YYYY-Qn",
            "panel_is_shariah_1":   int((panel["is_shariah"] == 1).sum())
                                    if "is_shariah" in panel.columns else 0,
            "panel_is_shariah_0":   int((panel["is_shariah"] == 0).sum())
                                    if "is_shariah" in panel.columns else 0,
        }
    else:
        panel_section = {"panel_rows": 0}

    # Coverage of financial variables in the panel
    ratio_vars = [
    "atq", "dlttq", "dlcq", "ltq",
    "cheq", "chq", "chsq",
    "revtq", "xintq", "iditq", "nopiq",
    "cogsq", "xrdq",
    "sukuk_ratio_t", "islamic_cash_ratio_t",
]
    var_coverage = {}
    for v in ratio_vars:
        if v in panel.columns:
            n_nonnull = int(panel[v].notna().sum())
            pct = round(100 * n_nonnull / max(len(panel), 1), 1)
            var_coverage[v] = {"n_nonnull": n_nonnull, "pct": pct}
    panel_section["financial_vars_coverage"] = var_coverage




    if not panel.empty and "is_shariah" in panel.columns:
        # Most recent SC name per stock_code
        sc_names = (
            df_sc.sort_values("date_effective")
            .drop_duplicates("stock_code", keep="last")[["stock_code", "company_name"]]
        )
        # Bridge gvkey -> stock_code via bursa_code_flex
        mapping_names = mapping[["gvkey", "bursa_code_flex"]].copy()
        mapping_names["stock_code"] = (
            mapping_names["bursa_code_flex"]
            .dropna()
            .apply(lambda x: int(x) if str(x).isdigit() else None)
        )
        mapping_names = mapping_names.dropna(subset=["stock_code"])
        mapping_names["stock_code"] = mapping_names["stock_code"].astype(int)

        panel_shariah = panel[panel["is_shariah"] == 1][["gvkey", "conm"]].drop_duplicates("gvkey")
        panel_shariah = panel_shariah.merge(mapping_names[["gvkey","stock_code"]], on="gvkey", how="left")
        panel_shariah = panel_shariah.merge(sc_names, on="stock_code", how="left")

        panel_shariah["name_match_score"] = panel_shariah.apply(
            lambda r: _name_match_score(
                str(r.get("company_name", "")),
                str(r.get("conm", ""))
            ), axis=1
        )

        low_scores = panel_shariah[panel_shariah["name_match_score"] < 50].sort_values("name_match_score")

        panel_section["name_match_score_mean"]     = round(float(panel_shariah["name_match_score"].mean()), 2)
        panel_section["name_match_score_below_50"] = int(len(low_scores))
        panel_section["name_match_low_detail"] = low_scores[
            ["gvkey", "conm", "company_name", "name_match_score"]
        ].to_dict(orient="records")


    # ── Assemble ──────────────────────────────────────────────────────────────
    qc = {
        **sc_section,
        **comp_section,
        **mapping_section,
        **panel_section,
        # Per-file detail (useful for debugging)
        "sc_files_detail": sc_reports,
    }

    return qc


def write_qc_report(qc: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, ensure_ascii=False)
