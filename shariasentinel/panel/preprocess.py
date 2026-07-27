from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional, Callable, Any

import pandas as pd

log = logging.getLogger(__name__)


# -----------------------------
# Robust coercers
# -----------------------------
def coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is None")
    if isinstance(value, float):
        if pd.isna(value):
            raise ValueError(f"{field_name} is NaN")
        return int(value)
    s = str(value).strip().lstrip("'\"").strip()
    s = re.sub(r"\.0+$", "", s)
    if not re.fullmatch(r"\d+", s):
        raise ValueError(f"{field_name} not int-like: {value!r} -> {s!r}")
    return int(s)


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return None if pd.isna(value) else value
    s = str(value).strip().lstrip("'\"").strip()
    if s in ("", ".", "NA", "NaN", "nan", "#N/A", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def coerce_date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.to_datetime(str(value), errors="coerce").date()
    except Exception:
        return None


def coerce_str(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s else None


# -----------------------------
# Quarter helpers
# -----------------------------
def parse_quarter(q_str: Any) -> Optional[str]:
    """Return quarter as 'YYYY-Qn' (standard)."""
    if q_str is None or (isinstance(q_str, float) and pd.isna(q_str)):
        return None
    s = str(q_str).strip()

    if re.fullmatch(r"\d{4}-Q[1-4]", s):
        return s
    m = re.fullmatch(r"Q([1-4])-(\d{4})", s, flags=re.I)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"
    m = re.fullmatch(r"(\d{4})Q([1-4])", s)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    m = re.fullmatch(r"(\d{4})-([1-4])", s)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    return None


def quarter_to_yyyq(q_std: str) -> str:
    """'YYYY-Qn' -> 'YYYYQn' (string for display / joins)."""
    return q_std.replace("-", "")


# -----------------------------
# Generic column preprocessing
# -----------------------------
_DTYPE_HANDLERS: dict[str, Callable[..., Any]] = {
    "int": coerce_int,
    "float": coerce_float,
    "str": coerce_str,
    "date": coerce_date,
}


def preprocess_column_types(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Convert per-column using robust coercers. Non-convertibles -> NA/None."""
    df = df.copy()
    issues: dict[str, int] = {}

    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        raw_null = int(df[col].isna().sum())
        handler = _DTYPE_HANDLERS[dtype]

        if dtype == "int":
            def safe(v, col=col):
                try:
                    return coerce_int(v, col)
                except Exception:
                    return pd.NA
            df[col] = df[col].apply(safe)
        else:
            df[col] = df[col].apply(handler)

        new_null = int(df[col].isna().sum())
        if new_null > raw_null:
            issues[col] = new_null - raw_null

    if issues:
        log.warning("Values not convertible -> NA: %s", issues)

    return df
