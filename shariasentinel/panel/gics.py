"""GICS sector code → name mapping.

Used by country panel builds whose authority list carries no sector label
(the DFM list, unlike SC Malaysia's, has none) — the panel's ``sector``
column then comes from Compustat ``gsector``. Names follow the MSCI/S&P
GICS taxonomy (2023 revision).
"""
from __future__ import annotations

GICS_SECTOR_NAMES: dict[str, str] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}


def gics_sector_name(code: object) -> str | None:
    """Resolve a raw ``gsector`` cell ('40', 40, 40.0) to its GICS name.

    Returns ``None`` for missing/unknown codes so callers can decide the
    fallback (the panel keeps NaN and QC reports the coverage).
    """
    if code is None:
        return None
    s = str(code).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return GICS_SECTOR_NAMES.get(s)
