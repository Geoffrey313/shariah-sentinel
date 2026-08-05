"""Qualitative cross-sectional analysis.

Computes verdict breakdowns by sector, firm size, debt level, and
persistence. Outputs go to ``outputs/scores/<country>/qualitative/``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)

_P_COLS = ("p_z_plus", "p_z_plus_renorm", "p_z_mahalanobis_sq", "p_t_iut")


def _add_verdict(df: pd.DataFrame, red: float = 0.01) -> pd.DataFrame:
    present = [c for c in _P_COLS if c in df.columns]
    df = df.copy()
    df["min_p"] = df[present].min(axis=1)
    df["verdict"] = "GREEN"
    df.loc[df["min_p"] < 0.05, "verdict"] = "AMBER"
    df.loc[df["min_p"] < red, "verdict"] = "RED"
    df.loc[df["min_p"].isna(), "verdict"] = "NO_DATA"
    df["is_red"] = (df["verdict"] == "RED").astype(int)
    return df


def run_qualitative(
    panel: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run all cross-sectional analyses and return dataframes."""
    settings = settings or AnalysisSettings()
    schema = settings.panel_schema

    from .phase0_reference_sample import resolve_label_column

    label_col = resolve_label_column(panel, settings)
    df = composites.merge(
        panel[[schema.firm_id, schema.quarter, "conm", "sector",
               label_col, "atq"]],
        on=[schema.firm_id, schema.quarter], how="left",
    )

    # Add ratio_debt_adj from panel if available
    panel_path = settings.output_layout.panel_path()
    if not panel_path.exists():
        base = settings.output_layout.country_code_lower.split("_ex_")[0]
        panel_path = settings.output_layout.root / Path(
            str(settings.output_layout.panel_relative).format(country=base)
        )
    if panel_path.exists():
        ratio_df = pd.read_parquet(panel_path, columns=["gvkey", "datacqtr", "ratio_debt_adj"])
        df = df.merge(ratio_df, on=["gvkey", "datacqtr"], how="left")

    df = _add_verdict(df)
    results = {}

    # 1. Verdict by sector
    sec = df.groupby("sector", dropna=False).agg(
        n_rows=("is_red", "size"),
        n_firms=(schema.firm_id, "nunique"),
        n_red=("is_red", "sum"),
    ).reset_index()
    sec["red_share"] = sec["n_red"] / sec["n_rows"]
    sec = sec.sort_values("red_share", ascending=False).reset_index(drop=True)
    results["verdict_by_sector"] = sec

    # 2. Verdict by firm size
    df["atq_quartile"] = pd.qcut(
        df["atq"].clip(lower=0), q=4,
        labels=["Q1 small", "Q2", "Q3", "Q4 large"],
        duplicates="drop",
    )
    size = df.groupby("atq_quartile", dropna=False, observed=True).agg(
        n_rows=("is_red", "size"),
        median_atq=("atq", "median"),
        n_red=("is_red", "sum"),
    ).reset_index()
    size["red_share"] = size["n_red"] / size["n_rows"]
    results["verdict_by_firm_size"] = size

    # 3. Verdict by debt level
    if "ratio_debt_adj" in df.columns:
        debt_valid = df[df["ratio_debt_adj"].notna() & (df["ratio_debt_adj"] >= 0)].copy()
        if len(debt_valid) > 100:
            debt_valid["debt_quartile"] = pd.qcut(
                debt_valid["ratio_debt_adj"], q=4,
                labels=["Q1 low debt", "Q2", "Q3", "Q4 high debt"],
                duplicates="drop",
            )
            debt = debt_valid.groupby("debt_quartile", dropna=False, observed=True).agg(
                n_rows=("is_red", "size"),
                median_debt_ratio=("ratio_debt_adj", "median"),
                n_red=("is_red", "sum"),
            ).reset_index()
            debt["red_share"] = debt["n_red"] / debt["n_rows"]
            results["verdict_by_debt_level"] = debt

    # 4. Verdict persistence
    firm = df.groupby([schema.firm_id, "conm", "sector"], dropna=False).agg(
        n_quarters=("is_red", "size"),
        n_red=("is_red", "sum"),
    ).reset_index()
    firm["red_share"] = firm["n_red"] / firm["n_quarters"]
    firm["persistence"] = pd.cut(
        firm["red_share"],
        bins=[-0.01, 0.01, 0.10, 0.30, 1.01],
        labels=["never", "episodic (<10%)", "intermittent (10-30%)", "chronic (>30%)"],
    )
    persist = firm.groupby("persistence", observed=True).agg(
        n_firms=(schema.firm_id, "count"),
        mean_red_share=("red_share", "mean"),
    ).reset_index()
    results["verdict_persistence"] = persist

    chronic = firm[firm["persistence"] == "chronic (>30%)"].sort_values(
        "red_share", ascending=False
    )
    results["chronic_red_firms"] = chronic

    # 5. Ablation results (if available from prior runs)
    ablation_path = settings.output_layout.phase5_dir().parent / "qualitative" / "ablation_results.csv"
    if ablation_path.exists():
        results["ablation_results"] = pd.read_csv(ablation_path)

    if write_outputs:
        out_dir = (
            settings.output_layout.root
            / Path(str(settings.output_layout.scores_relative).format(
                country=settings.output_layout.country_code_lower
            ))
            / "qualitative"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in results.items():
            path = out_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
        log.info("qualitative: wrote %d files to %s", len(results), out_dir)

    return results
