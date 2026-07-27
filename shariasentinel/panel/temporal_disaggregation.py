"""Temporal disaggregation — annual-to-quarterly imputation.

For each ``(firm, quarter, variable)`` cell where the quarterly value is
NaN but the annual total exists, estimate the quarterly value using the
sector's empirical quarterly share of the annual total:

    x_q_hat = x_annual * s_q^sector

where ``s_q^sector`` is the median share ``x_q / x_annual`` observed on
peer firms in the same sector that report both quarterly and annually.

Flow variables (income-statement / cash-flow) are disaggregated by
seasonal share. Stock variables (balance-sheet) are not disaggregated
by this module — forward-fill handles them, because the annual value
is a year-end snapshot and intra-year interpolation is less principled.

Every imputed cell receives ``<col>_source = "annual_disaggregated"``
so downstream detectors and composites can filter or sensitivity-test.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from shariasentinel.common.schemas import ANNUAL_TO_QUARTERLY_MAP

log = logging.getLogger(__name__)

DEFAULT_MIN_TEMPLATE_FIRMS: int = 5


def build_seasonal_template(
    df_quarterly: pd.DataFrame,
    df_annual: pd.DataFrame,
    sector_col: str = "sector",
    variables: list[str] | None = None,
    min_template_firms: int = DEFAULT_MIN_TEMPLATE_FIRMS,
) -> pd.DataFrame:
    """Compute the sector × fiscal-quarter seasonal share template.

    For each ``(sector, fqtr, variable)`` cell, the seasonal share is
    the median of ``x_q / x_annual`` across firms that report both
    quarterly and annually in the same fiscal year. Shares are then
    normalised so that the four fiscal quarters sum to 1 within each
    ``(sector, variable)`` group.

    Args:
        df_quarterly: Quarterly panel with ``gvkey``, ``fyearq``,
            ``fqtr``, ``sector_col`` and the quarterly financial columns.
        df_annual: Annual frame with ``gvkey``, ``fyear`` and the annual
            financial columns.
        sector_col: Panel column for sector stratification.
        variables: Subset of annual column names to build templates for.
            Defaults to all flow variables in the mapping.
        min_template_firms: Minimum firms per ``(sector, fqtr, variable)``
            required to emit a template row; below this the cell falls
            back to the next level (super-sector → market → uniform).

    Returns:
        A long-form DataFrame with columns ``[sector_col, fqtr, variable,
        share, n_firms, level]`` where ``level`` is one of ``"sector"``,
        ``"market"``, or ``"uniform"``.
    """
    if variables is None:
        variables = [
            ann for ann, info in ANNUAL_TO_QUARTERLY_MAP.items()
            if info["type"] == "flow"
        ]

    if "fyearq" not in df_quarterly.columns or "fqtr" not in df_quarterly.columns:
        log.warning("build_seasonal_template: fyearq/fqtr missing; returning empty.")
        return pd.DataFrame(
            columns=[sector_col, "fqtr", "variable", "share", "n_firms", "level"]
        )

    merged = df_quarterly[
        ["gvkey", "fyearq", "fqtr", sector_col]
        + [ANNUAL_TO_QUARTERLY_MAP[v]["quarterly"] for v in variables
           if ANNUAL_TO_QUARTERLY_MAP[v]["quarterly"] in df_quarterly.columns]
    ].merge(
        df_annual[["gvkey", "fyear"] + [v for v in variables if v in df_annual.columns]],
        left_on=["gvkey", "fyearq"],
        right_on=["gvkey", "fyear"],
        how="inner",
    )

    rows: list[dict] = []
    for ann_col in variables:
        info = ANNUAL_TO_QUARTERLY_MAP.get(ann_col)
        if info is None or info["type"] != "flow":
            continue
        qtr_col = info["quarterly"]
        if qtr_col not in merged.columns or ann_col not in merged.columns:
            continue

        both_finite = merged[qtr_col].notna() & merged[ann_col].notna() & (merged[ann_col].abs() > 1e-10)
        sub = merged.loc[both_finite].copy()
        sub["_share"] = sub[qtr_col] / sub[ann_col]

        for sector_val, grp in sub.groupby(sector_col, dropna=True):
            for fqtr_val, qgrp in grp.groupby("fqtr", dropna=True):
                n = int(qgrp["gvkey"].nunique())
                if n >= min_template_firms:
                    rows.append({
                        sector_col: sector_val,
                        "fqtr": int(fqtr_val),
                        "variable": ann_col,
                        "share": float(qgrp["_share"].median()),
                        "n_firms": n,
                        "level": "sector",
                    })

        for fqtr_val, qgrp in sub.groupby("fqtr", dropna=True):
            n = int(qgrp["gvkey"].nunique())
            if n >= min_template_firms:
                rows.append({
                    sector_col: "__market__",
                    "fqtr": int(fqtr_val),
                    "variable": ann_col,
                    "share": float(qgrp["_share"].median()),
                    "n_firms": n,
                    "level": "market",
                })

        for fqtr_val in [1, 2, 3, 4]:
            rows.append({
                sector_col: "__uniform__",
                "fqtr": fqtr_val,
                "variable": ann_col,
                "share": 0.25,
                "n_firms": 0,
                "level": "uniform",
            })

    template = pd.DataFrame(rows)
    if template.empty:
        return template

    for level in ["sector", "market"]:
        level_mask = template["level"] == level
        if not level_mask.any():
            continue
        for (grp_key, var), idx in template.loc[level_mask].groupby(
            [sector_col, "variable"]
        ).groups.items():
            total = template.loc[idx, "share"].sum()
            if total > 0 and np.isfinite(total):
                template.loc[idx, "share"] = template.loc[idx, "share"] / total

    log.info(
        "build_seasonal_template: %d template rows (%d sector, %d market, %d uniform) for %d variables.",
        len(template),
        int((template["level"] == "sector").sum()),
        int((template["level"] == "market").sum()),
        int((template["level"] == "uniform").sum()),
        len(variables),
    )
    return template


def _lookup_share(
    template: pd.DataFrame,
    sector_val: str,
    fqtr: int,
    variable: str,
    sector_col: str,
) -> tuple[float, str]:
    """Find the best-available seasonal share via the fallback chain."""
    for level, key in [
        ("sector", sector_val),
        ("market", "__market__"),
        ("uniform", "__uniform__"),
    ]:
        match = template[
            (template[sector_col] == key)
            & (template["fqtr"] == fqtr)
            & (template["variable"] == variable)
            & (template["level"] == level)
        ]
        if not match.empty:
            return float(match.iloc[0]["share"]), level
    return 0.25, "uniform_fallback"


def disaggregate_annual_to_quarterly(
    df_quarterly: pd.DataFrame,
    df_annual: pd.DataFrame,
    template: pd.DataFrame,
    variables: list[str] | None = None,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Fill quarterly NaN cells using annual totals and seasonal shares.

    Only **flow variables** are disaggregated; stock variables are
    skipped (handled by forward-fill downstream). Every imputed cell
    gets ``<qtr_col>_source = "annual_disaggregated"``.

    Args:
        df_quarterly: Quarterly panel (modified in place).
        df_annual: Annual frame with ``gvkey``, ``fyear``.
        template: Output of :func:`build_seasonal_template`.
        variables: Annual column names to process. Defaults to all flow
            variables in the mapping.
        sector_col: Sector column name in the quarterly panel.

    Returns:
        The quarterly panel with NaN cells filled where annual data
        permitted, plus ``<col>_source`` columns.
    """
    if variables is None:
        variables = [
            ann for ann, info in ANNUAL_TO_QUARTERLY_MAP.items()
            if info["type"] == "flow"
        ]

    panel = df_quarterly.copy()

    annual_lookup = df_annual.set_index(["gvkey", "fyear"])

    filled_counts: dict[str, int] = {}
    for ann_col in variables:
        info = ANNUAL_TO_QUARTERLY_MAP.get(ann_col)
        if info is None or info["type"] != "flow":
            continue
        qtr_col = info["quarterly"]
        if qtr_col not in panel.columns:
            continue
        if ann_col not in annual_lookup.columns:
            continue

        source_col = f"{qtr_col}_source"
        if source_col not in panel.columns:
            panel[source_col] = "quarterly_observed"
            panel.loc[panel[qtr_col].isna(), source_col] = "missing"

        missing_mask = panel[qtr_col].isna()
        if not missing_mask.any():
            filled_counts[qtr_col] = 0
            continue

        filled = 0
        for idx in panel.index[missing_mask]:
            gvkey = panel.at[idx, "gvkey"]
            fyear = panel.at[idx, "fyearq"] if "fyearq" in panel.columns else None
            fqtr = panel.at[idx, "fqtr"] if "fqtr" in panel.columns else None
            sector = panel.at[idx, sector_col] if sector_col in panel.columns else None

            if fyear is None or fqtr is None or not np.isfinite(fyear) or not np.isfinite(fqtr):
                continue

            key = (int(gvkey), int(fyear))
            if key not in annual_lookup.index:
                continue

            ann_val = annual_lookup.at[key, ann_col] if key in annual_lookup.index else np.nan
            if not np.isfinite(ann_val):
                continue

            share, level = _lookup_share(
                template, str(sector), int(fqtr), ann_col, sector_col,
            )
            panel.at[idx, qtr_col] = ann_val * share
            panel.at[idx, source_col] = "annual_disaggregated"
            filled += 1

        filled_counts[qtr_col] = filled
        log.info(
            "disaggregate: %s → %s: filled %d quarterly NaN cells from annual data.",
            ann_col, qtr_col, filled,
        )

    total = sum(filled_counts.values())
    log.info("disaggregate: total cells filled = %d across %d variables.", total, len(filled_counts))
    return panel


def add_source_flags_for_existing(
    panel: pd.DataFrame,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Ensure ``<col>_source`` columns exist for all tracked variables.

    Called at the start of the imputation pipeline so that even the
    ``STANDARD`` strategy emits source flags (all set to
    ``"quarterly_observed"`` or ``"missing"``).
    """
    if variables is None:
        variables = [
            info["quarterly"]
            for info in ANNUAL_TO_QUARTERLY_MAP.values()
        ]
    for qtr_col in variables:
        source_col = f"{qtr_col}_source"
        if source_col in panel.columns or qtr_col not in panel.columns:
            continue
        panel[source_col] = "quarterly_observed"
        panel.loc[panel[qtr_col].isna(), source_col] = "missing"
    return panel
