"""Panel construction orchestrator.

This is the full pipeline logic extracted from the original
``panel_creation/main.py``. It chains all panel construction steps
in the correct order with the correct arguments.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from shariasentinel.panel.cleaning import load_sc_folder, load_compustat_csv
from shariasentinel.panel.build import (
    build_panel_sac_anchored, build_panel_compliance_anchored,
    build_merge_diagnostics,
    expand_to_full_quarter_grid, apply_forward_fill, build_ffill_report,
    apply_firm_level_zero_fill, add_balance_sheet_ffill_flag,
    add_clean_sample_flag,
)
from shariasentinel.panel.gics import gics_sector_name
from shariasentinel.panel.methodology import get_policy
from shariasentinel.panel.qc import build_qc_report, write_qc_report
from shariasentinel.panel.ratios import (
    compute_shariah_ratios, compute_ratio_stats, attach_macro_connectors,
)
from shariasentinel.panel.mapping import build_mapping_compustat

log = logging.getLogger(__name__)

# Columns that are forward-filled within each firm
FFILL_COLS = [
    "atq",    # total assets — denominator for all ratios
    "dlttq",  # long-term debt
    "dlcq",   # current portion of debt
    "ltq",    # total liabilities (upper bound)
    "cheq",   # cash & short-term investments
    "chq",    # strict cash
    "chsq",   # alternative cash
    "revtq",  # total revenue
    "iditq",  # interest & dividend income (ratio_income)
    "nopiq",  # non-operating income (broad proxy)
    "cogsq",  # cost of goods sold — needed for ABN_PROD when available
    "xrdq",   # R&D expense — needed for full ABN_DISX when available
    "ibq",    # income before extraordinary items — M-score inputs
    "rectq",  # receivables — M-score AQI inputs
]

# Columns zeroed for firms that never report them
ZERO_FILL_COLS = ["dlttq", "dlcq", "nopiq"]


def run_panel_pipeline(
    country_code: str = "MYS",
    currency: str | None = None,
    sc_dir: str | None = None,
    analyze: bool = False,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full panel construction pipeline.

    Routes by country policy: ``MYS`` keeps the historical SAC-anchored
    path untouched (byte-identical output); other countries (``UAE``) go
    through the gvkey-anchored generic path.

    Args:
        country_code: Country key (e.g. "MYS", "UAE").
        currency: Local reporting currency; ``None`` resolves from the
            country policy ("MYR", "AED").
        sc_dir: Path to SC Malaysia list folder (MYS only).
        analyze: If True, run descriptive analysis after building.
        out_dir: Output directory. If None, uses config defaults.

    Returns:
        The completed panel as a DataFrame.
    """
    from shariasentinel import config as cfg

    policy = get_policy(country_code)
    currency = currency or policy.currency
    if policy.country_key != "MYS":
        return _run_generic_panel_pipeline(
            policy, currency=currency, analyze=analyze, out_dir=out_dir,
        )

    _sc_dir = sc_dir or str(cfg.raw_sc_dir())
    _raw_csv = str(cfg.raw_compustat_csv(country_code))
    _out = out_dir or cfg.panel_dir(country_code)
    _reports = _out / "reports"

    _out.mkdir(parents=True, exist_ok=True)
    _reports.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & clean raw data ──────────────────────────────────────────
    df_sc, sc_reports = load_sc_folder(_sc_dir)
    df_comp, comp_report, df_comp_pre_pydantic = load_compustat_csv(
        _raw_csv, country_code=country_code, currency=currency,
    )

    # ── 2. Build main panel: SAC-anchored quarterly ───────────────────────
    panel = build_panel_sac_anchored(df_comp=df_comp, df_sc=df_sc)

    # ── 3. Merge diagnostics ─────────────────────────────────────────────
    firm_diag, _event_diag, _merge_summary = build_merge_diagnostics(
        df_sc=df_sc,
        df_comp=df_comp,
        df_comp_pre_pydantic=df_comp_pre_pydantic,
    )

    # ── 4a. Expand quarterly grid ─────────────────────────────────────────
    panel = expand_to_full_quarter_grid(panel)

    # ── 4b. Forward-fill ──────────────────────────────────────────────────
    panel = apply_forward_fill(panel, cols=FFILL_COLS)

    # ── 4c. Forward-fill diagnostics ──────────────────────────────────────
    ffill_summary, ffill_firms = build_ffill_report(panel, cols=FFILL_COLS)
    ffill_summary.to_csv(_reports / "ffill_report_summary.csv", index=False)
    ffill_firms.to_csv(_reports / "ffill_report_firms.csv", index=False)
    log.info("ffill diagnostics → %s", _reports)

    # ── 4d. Firm-level NaN→0 ──────────────────────────────────────────────
    panel = apply_firm_level_zero_fill(panel, cols=ZERO_FILL_COLS)

    # ── 4e. Flag rows with ffilled balance-sheet inputs ───────────────────
    panel = add_balance_sheet_ffill_flag(panel)

    # ── 4f. Macro connectors & Shariah ratios ─────────────────────────────
    panel = attach_macro_connectors(panel)
    panel = compute_shariah_ratios(panel)

    # ── 4g. Clean-sample flag ─────────────────────────────────────────────
    panel = add_clean_sample_flag(panel)

    # ── 5. Add SC Malaysia sector ─────────────────────────────────────────
    mapping = build_mapping_compustat(df_comp)
    sector_sc = (
        (
            df_sc.sort_values("date_effective")
            if "date_effective" in df_sc.columns
            else df_sc
        )[["stock_code", "sector"]]
        .assign(stock_code_z4=lambda x: x["stock_code"].astype(str).str.zfill(4))
        .drop_duplicates("stock_code_z4", keep="last")
        .rename(columns={"stock_code_z4": "bursa_code_flex"})
    )
    sector_ref = (
        mapping[["gvkey", "bursa_code_flex"]]
        .dropna(subset=["bursa_code_flex"])
        .drop_duplicates("gvkey")
        .merge(sector_sc[["bursa_code_flex", "sector"]], on="bursa_code_flex", how="left")
    )
    panel = panel.drop(columns=["sector"], errors="ignore")
    panel = panel.merge(sector_ref[["gvkey", "sector"]], on="gvkey", how="left")

    sec_cov = panel["sector"].notna().mean()
    log.info("Sector coverage: %.1f%%", 100 * sec_cov)

    # ── 6. Save panel ────────────────────────────────────────────────────
    parquet_path = _out / "compustat_quarterly.parquet"
    csv_path = _out / "compustat_quarterly.csv"
    panel.to_parquet(parquet_path, index=False)
    panel.to_csv(csv_path, index=False)

    firm_diag.to_parquet(_reports / "sc_merge_loss_firm_level.parquet", index=False)
    firm_diag.to_csv(_reports / "sc_merge_loss_firm_level.csv", index=False)

    # ── 7. QC report ─────────────────────────────────────────────────────
    panel_for_qc = panel.copy()
    if "sac_shariah" in panel_for_qc.columns and "is_shariah" not in panel_for_qc.columns:
        panel_for_qc["is_shariah"] = panel_for_qc["sac_shariah"]
    elif "is_shariah" not in panel_for_qc.columns:
        panel_for_qc["is_shariah"] = 0

    qc = build_qc_report(
        df_sc, sc_reports, df_comp, comp_report,
        panel_for_qc,
        df_comp_pre_pydantic=df_comp_pre_pydantic,
    )
    ratio_stats = compute_ratio_stats(panel_for_qc[panel_for_qc["is_shariah"] == 1])
    qc.update(ratio_stats)

    if not firm_diag.empty:
        lost_only = firm_diag[firm_diag["matched_in_compustat"] == 0].copy()
        if "diagnostic" in lost_only.columns:
            qc["sc_merge_loss_by_reason"] = (
                lost_only["diagnostic"].fillna("unknown").value_counts().to_dict()
            )

    write_qc_report(qc, str(_reports / "qc_report.json"))

    # ── 8. Optional analysis ─────────────────────────────────────────────
    if analyze:
        from shariasentinel.panel.analysis import run_full_analysis, write_analysis_outputs
        log.info("Running panel analysis…")
        results = run_full_analysis(panel)
        write_analysis_outputs(results, _reports / "panel_analysis")

    # ── 9. Summary ───────────────────────────────────────────────────────
    sac1 = int((panel["sac_shariah"] == 1).sum())
    sac0 = int((panel["sac_shariah"] == 0).sum())
    log.info("Panel: %d rows, %d firms", len(panel), panel["gvkey"].nunique())
    log.info("sac_shariah=1: %d | sac_shariah=0: %d", sac1, sac0)
    log.info("Panel → %s", parquet_path)

    return panel


def _run_generic_panel_pipeline(
    policy,
    currency: str,
    analyze: bool = False,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Country-generic panel build (UAE path) — gvkey-anchored.

    Differences vs the MYS path, all deliberate:

    - compliance events come from ``load_compliance_list`` + the name
      crosswalk (``uae_dfm_gvkey_crosswalk.csv``), anchored by gvkey;
    - the full Compustat universe is kept (firms without an authority
      verdict carry ``compliance_shariah = NaN``, never 0);
    - ``sector`` comes from Compustat ``gsector`` (GICS) since the DFM
      list has no sector column;
    - macro connectors use the country subfolder with nearest-year fill;
    - QC is a compact JSON (the Bursa merge diagnostics do not apply).
    """
    import json

    from shariasentinel import config as cfg
    from shariasentinel.panel.compliance_sources import load_compliance_list
    from shariasentinel.panel.uae_crosswalk import load_crosswalk

    _raw_csv = str(cfg.raw_compustat_csv(policy.country_key))
    _out = out_dir or cfg.panel_dir(policy.country_key)
    _reports = _out / "reports"
    _out.mkdir(parents=True, exist_ok=True)
    _reports.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & clean raw data ──────────────────────────────────────────
    df_sc, sc_reports = load_compliance_list(policy)
    df_comp, comp_report, _df_comp_pre = load_compustat_csv(
        _raw_csv, country_code=policy.compustat_fic, currency=currency,
    )

    # ── 2. Crosswalk → gvkey-level compliance events ─────────────────────
    xwalk = load_crosswalk(policy=policy)
    matched = xwalk[xwalk["gvkey"].notna()][["stock_code", "gvkey"]]
    events = df_sc.merge(matched, on="stock_code", how="inner")
    events = events.rename(columns={"shariah_status": "compliant"})
    n_secs, n_matched = df_sc["stock_code"].nunique(), matched["stock_code"].nunique()
    log.info(
        "compliance events: %d/%d %s securities matched to gvkeys → %d events",
        n_matched, n_secs, policy.authority_label, len(events),
    )

    # ── 3. Build gvkey-anchored panel ─────────────────────────────────────
    panel = build_panel_compliance_anchored(
        df_comp=df_comp, df_events=events, policy=policy,
    )

    # ── 4. Grid expansion, ffill, zero-fill, integrity flags ─────────────
    panel = expand_to_full_quarter_grid(panel)
    panel = apply_forward_fill(panel, cols=FFILL_COLS)

    ffill_summary, ffill_firms = build_ffill_report(panel, cols=FFILL_COLS)
    ffill_summary.to_csv(_reports / "ffill_report_summary.csv", index=False)
    ffill_firms.to_csv(_reports / "ffill_report_firms.csv", index=False)

    panel = apply_firm_level_zero_fill(panel, cols=ZERO_FILL_COLS)
    panel = add_balance_sheet_ffill_flag(panel)

    # methodology_key must survive grid expansion for threshold routing.
    panel["methodology_key"] = policy.methodology_key

    # ── 5. Macro connectors & ratios (policy thresholds) ─────────────────
    panel = attach_macro_connectors(panel, country_key=policy.country_key)
    panel = compute_shariah_ratios(panel, policy=policy)
    panel = add_clean_sample_flag(panel)

    # ── 6. Sector from Compustat GICS ─────────────────────────────────────
    raw_sectors = pd.read_csv(
        _raw_csv, dtype=str, low_memory=False, usecols=["gvkey", "gsector"],
    )
    gv = raw_sectors["gvkey"].astype(str).str.strip()
    raw_sectors = raw_sectors[gv.str.fullmatch(r"\d+") & (gv.str.strip("0") != "")]
    raw_sectors["gvkey"] = raw_sectors["gvkey"].astype(int)
    sector_map = (
        raw_sectors.dropna(subset=["gsector"])
        .drop_duplicates("gvkey")
        .assign(sector=lambda x: x["gsector"].map(gics_sector_name))
        [["gvkey", "sector"]]
    )
    panel = panel.drop(columns=["sector"], errors="ignore")
    panel = panel.merge(sector_map, on="gvkey", how="left")
    log.info("Sector coverage (GICS): %.1f%%", 100 * panel["sector"].notna().mean())

    # ── 7. Save panel + compact QC ────────────────────────────────────────
    parquet_path = _out / "compustat_quarterly.parquet"
    panel.to_parquet(parquet_path, index=False)
    panel.to_csv(_out / "compustat_quarterly.csv", index=False)

    n1 = int((panel["compliance_shariah"] == 1).sum())
    n0 = int((panel["compliance_shariah"] == 0).sum())
    nna = int(panel["compliance_shariah"].isna().sum())
    qc = {
        "country": policy.country_key,
        "methodology_key": policy.methodology_key,
        "authority": policy.authority_label,
        "compliance_list": sc_reports,
        "compustat": comp_report,
        "crosswalk": {
            "securities": int(n_secs),
            "matched_to_gvkey": int(n_matched),
            "events_after_join": int(len(events)),
        },
        "panel": {
            "rows": int(len(panel)),
            "firms": int(panel["gvkey"].nunique()),
            "compliance_shariah_1": n1,
            "compliance_shariah_0": n0,
            "compliance_shariah_na": nna,
            "sector_coverage_pct": round(100 * float(panel["sector"].notna().mean()), 1),
        },
    }
    panel_for_qc = panel.copy()
    panel_for_qc["is_shariah"] = (panel_for_qc["compliance_shariah"] == 1).astype(int)
    qc.update(compute_ratio_stats(panel_for_qc[panel_for_qc["is_shariah"] == 1]))
    (_reports / "qc_report.json").write_text(
        json.dumps(qc, indent=2, default=str), encoding="utf-8",
    )

    # ── 8. Optional analysis ─────────────────────────────────────────────
    if analyze:
        from shariasentinel.panel.analysis import run_full_analysis, write_analysis_outputs
        log.info("Running panel analysis…")
        results = run_full_analysis(panel)
        write_analysis_outputs(results, _reports / "panel_analysis")

    log.info("Panel: %d rows, %d firms", len(panel), panel["gvkey"].nunique())
    log.info("compliance_shariah 1: %d | 0: %d | NA: %d", n1, n0, nna)
    log.info("Panel → %s", parquet_path)
    return panel
