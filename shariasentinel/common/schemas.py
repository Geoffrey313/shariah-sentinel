from __future__ import annotations

import re
from datetime import date
from typing import Optional

# Pydantic v2 fallback v1
try:
    from pydantic import BaseModel, field_validator, ValidationError
    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, validator, ValidationError  # type: ignore
    PYDANTIC_V2 = False


QUARTER_STD_RE = re.compile(r"^\d{4}-Q[1-4]$")
QUARTER_YYYQ_RE = re.compile(r"^\d{4}Q[1-4]$")


class ShariahRecord(BaseModel):
    """One row = one firm x one SC list (later expanded to quarterly)."""

    source: str
    date_effective: date
    quarter: str              # standard format: YYYY-Qn
    market: str
    sector: str
    stock_code: int           # Bursa stock code (int)
    company_name: str
    shariah_status: str

    if PYDANTIC_V2:
        @field_validator("quarter")
        @classmethod
        def _q(cls, v: str) -> str:
            if not QUARTER_STD_RE.match(str(v)):
                raise ValueError(f"quarter invalide: {v!r} (attendu YYYY-Qn)")
            return v

        @field_validator("shariah_status")
        @classmethod
        def _s(cls, v: str) -> str:
            allowed = {"Shariah-Compliant", "Non-Shariah-Compliant"}
            if v not in allowed:
                raise ValueError(f"shariah_status invalide: {v!r}")
            return v
    else:  # pragma: no cover
        @validator("quarter")
        def _q(cls, v):
            if not QUARTER_STD_RE.match(str(v)):
                raise ValueError(f"quarter invalide: {v!r} (attendu YYYY-Qn)")
            return v

        @validator("shariah_status")
        def _s(cls, v):
            allowed = {"Shariah-Compliant", "Non-Shariah-Compliant"}
            if v not in allowed:
                raise ValueError(f"shariah_status invalide: {v!r}")
            return v


class CompustatRecord(BaseModel):
    """One row = one firm (gvkey) x one Compustat quarter."""

    gvkey: int
    datadate: date
    datacqtr: str             # standard format: YYYY-Qn
    conm: str
    isin: Optional[str] = None
    sedol: Optional[str] = None
    fic: str
    curcdq: Optional[str] = None

    # financials (optional)
    dlttq: Optional[float] = None   # Long-term debt
    dlcq:  Optional[float] = None   # Debt in current liabilities
    atq:   Optional[float] = None   # Total assets
    cheq:  Optional[float] = None   # Cash and short-term investments   
    dd1q:  Optional[float] = None   # Long-term debt due in 1 year
    dlsq:  Optional[float] = None   # Short-term borrowings
    notesq: Optional[float] = None  # Notes payable
    rectrq: Optional[float] = None  # Receivables - trade
    revtq:  Optional[float] = None  # Revenue
    xintdy: Optional[float] = None  # Interest paid (annual, si trimestriel absent)
    niq:    Optional[float] = None  # Net income (utile pour stats descriptives)
    oibdpq: Optional[float] = None  # Operating income before D&A
    fqtr: Optional[int] = None


    ltq:    Optional[float] = None
    chq:    Optional[float] = None
    chsq:   Optional[float] = None
    iditq:  Optional[float] = None
    nopiq:  Optional[float] = None
    fyearq: Optional[int] = None   # fiscal year, if present in the CSV

    actq:   Optional[float] = None  # Current assets (total)
    lctq:   Optional[float] = None  # Current liabilities (total)
    rectq:  Optional[float] = None  # Receivables (total)
    invtq:  Optional[float] = None  # Inventories (total)
    ibq:    Optional[float] = None  # Income before extraordinary items (net income proxy)
    oancfy: Optional[float] = None  # Operating cash flow YTD cumulative
    oancfq: Optional[float] = None  # Operating cash flow quarterly (derived)
    ppentq: Optional[float] = None  # Net property, plant & equipment
    dpq:    Optional[float] = None  # Depreciation
    xsgaq:  Optional[float] = None  # SG&A expenses
    cogsq:  Optional[float] = None  # Cost of goods sold
    xrdq:   Optional[float] = None  # R&D expense
    gpq:    Optional[float] = None  # Gross profit
    ltmibq: Optional[float] = None  # Total liabilities + minority interest (proxy)
    xintq:  Optional[float] = None  # Quarterly interest expense


    if PYDANTIC_V2:
        @field_validator("datacqtr")
        @classmethod
        def _q(cls, v: str) -> str:
            if not QUARTER_STD_RE.match(str(v)):
                raise ValueError(f"datacqtr invalide: {v!r} (attendu YYYY-Qn)")
            return v
    else:  # pragma: no cover
        @validator("datacqtr")
        def _q(cls, v):
            if not QUARTER_STD_RE.match(str(v)):
                raise ValueError(f"datacqtr invalide: {v!r} (attendu YYYY-Qn)")
            return v


# ─── Annual-to-quarterly column mapping ────────────────────────────────────
# Maps each annual Compustat column to its quarterly equivalent and
# classifies it as a flow (income-statement / cash-flow) or stock
# (balance-sheet) variable. Flow variables are disaggregated by
# seasonal share; stock variables use point-in-time logic.
ANNUAL_TO_QUARTERLY_MAP: dict[str, dict] = {
    # flow variables — annual = sum of 4 quarters
    "idit":  {"quarterly": "iditq",  "type": "flow"},
    "xint":  {"quarterly": "xintq",  "type": "flow"},
    "nopi":  {"quarterly": "nopiq",  "type": "flow"},
    "oibdp": {"quarterly": "oibdpq", "type": "flow"},
    "revt":  {"quarterly": "revtq",  "type": "flow"},
    "nicon": {"quarterly": "niq",    "type": "flow"},
    # stock variables — annual = year-end snapshot
    "at":    {"quarterly": "atq",    "type": "stock"},
    "dltt":  {"quarterly": "dlttq",  "type": "stock"},
    "dlc":   {"quarterly": "dlcq",   "type": "stock"},
    "che":   {"quarterly": "cheq",   "type": "stock"},
    "lt":    {"quarterly": "ltq",    "type": "stock"},
    "act":   {"quarterly": "actq",   "type": "stock"},
    "lct":   {"quarterly": "lctq",   "type": "stock"},
    "rect":  {"quarterly": "rectq",  "type": "stock"},
    "invt":  {"quarterly": "invtq",  "type": "stock"},
}


class AnnualCompustatRecord(BaseModel):
    """One row = one firm (gvkey) x one Compustat fiscal year."""

    gvkey: int
    datadate: date
    conm: str
    fic: str
    fyear: Optional[int] = None
    curcd: Optional[str] = None

    # flow variables (annual = sum of 4 quarters)
    idit:  Optional[float] = None
    xint:  Optional[float] = None
    nopi:  Optional[float] = None
    oibdp: Optional[float] = None
    revt:  Optional[float] = None
    nicon: Optional[float] = None

    # stock variables (annual = year-end snapshot)
    at:    Optional[float] = None
    dltt:  Optional[float] = None
    dlc:   Optional[float] = None
    che:   Optional[float] = None
    lt:    Optional[float] = None
    act:   Optional[float] = None
    lct:   Optional[float] = None
    rect:  Optional[float] = None
    invt:  Optional[float] = None
