"""Load and filter the Compustat Global annual CSV.

Parallel to ``cleaning.load_compustat_csv`` but for the annual file
``comp_global_daily_annual.csv``. Output is keyed on ``(gvkey, fyear)``
so the temporal-disaggregation engine can join it to the quarterly panel
on ``(gvkey, fyearq)``.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from shariasentinel.common.schemas import AnnualCompustatRecord, ANNUAL_TO_QUARTERLY_MAP

log = logging.getLogger(__name__)

_ANNUAL_COLUMNS = (
    ["gvkey", "datadate", "conm", "fic", "fyear", "curcd"]
    + list(ANNUAL_TO_QUARTERLY_MAP.keys())
)


def load_annual_compustat(
    path: str,
    country_codes: list[str] | None = None,
    currency: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load the annual Compustat CSV filtered to the target countries.

    Args:
        path: Path to ``comp_global_daily_annual.csv``.
        country_codes: ISO-3 codes to keep (e.g. ``["MYS"]``). When
            ``None`` every country is kept.
        currency: If set, filter to rows with ``curcd == currency``.

    Returns:
        ``(df_annual, report)`` where ``df_annual`` is the cleaned frame
        keyed on ``(gvkey, fyear)`` and ``report`` is a diagnostic dict.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"annual CSV not found: {csv_path}")

    present_cols = set(
        pd.read_csv(csv_path, nrows=0).columns.str.strip().str.lower()
    )
    use_cols = [c for c in _ANNUAL_COLUMNS if c in present_cols]
    missing_cols = [c for c in _ANNUAL_COLUMNS if c not in present_cols]
    if missing_cols:
        log.warning("annual_loading: columns absent from CSV: %s", missing_cols)

    df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    initial_rows = len(df)

    if country_codes:
        df = df[df["fic"].isin(country_codes)].copy()
    if currency and "curcd" in df.columns:
        df = df[df["curcd"] == currency].copy()

    df["gvkey"] = pd.to_numeric(df["gvkey"], errors="coerce")
    df = df.dropna(subset=["gvkey"])
    df["gvkey"] = df["gvkey"].astype(int)

    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce").dt.date

    if "fyear" in df.columns:
        df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce")
        df = df.dropna(subset=["fyear"])
        df["fyear"] = df["fyear"].astype(int)
    elif "datadate" in df.columns:
        df["fyear"] = pd.to_datetime(df["datadate"]).dt.year

    for col in ANNUAL_TO_QUARTERLY_MAP:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["gvkey", "fyear", "datadate"]).drop_duplicates(
        subset=["gvkey", "fyear"], keep="last",
    )

    report = {
        "csv_path": str(csv_path),
        "initial_rows": initial_rows,
        "after_country_filter": len(df),
        "country_codes": country_codes,
        "n_firms": int(df["gvkey"].nunique()),
        "fyear_range": (int(df["fyear"].min()), int(df["fyear"].max()))
        if not df.empty else (None, None),
        "missing_csv_columns": missing_cols,
        "coverage": {
            col: int(df[col].notna().sum())
            for col in ANNUAL_TO_QUARTERLY_MAP
            if col in df.columns
        },
    }

    log.info(
        "annual_loading: %d rows, %d firms, fyear %s–%s from %s",
        len(df), report["n_firms"],
        report["fyear_range"][0], report["fyear_range"][1], csv_path.name,
    )
    return df, report
