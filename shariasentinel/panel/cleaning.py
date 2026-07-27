import glob
import logging
import os
from typing import Tuple, List, Dict

import pandas as pd
from tqdm import tqdm

from .preprocess import preprocess_column_types, parse_quarter
from shariasentinel.common.schemas import ShariahRecord, CompustatRecord, PYDANTIC_V2

try:
    from pydantic import ValidationError
except ImportError:
    ValidationError = Exception

try:
    from pydantic import TypeAdapter
except ImportError:
    TypeAdapter = None

log = logging.getLogger(__name__)


# ─── SC Malaysia ──────────────────────────────────────────────────────────────

_SC_SCHEMA = {
    "Stock_Code":     "int",
    "Date_Effective": "date",
    "Quarter":        "str",
    "Market":         "str",
    "Sector":         "str",
    "Company_Name":   "str",
    "Shariah_Status": "str",
    "Source":         "str",
}
_SC_RENAME = {
    "Stock_Code":     "stock_code",
    "Date_Effective": "date_effective",
    "Quarter":        "quarter",
    "Market":         "market",
    "Sector":         "sector",
    "Company_Name":   "company_name",
    "Shariah_Status": "shariah_status",
    "Source":         "source",
}
_SC_FINAL = [
    "source", "date_effective", "quarter", "market",
    "sector", "stock_code", "company_name", "shariah_status",
]


def _validate_rows(df: pd.DataFrame, model, label: str, chunk_size: int = 5000) -> Tuple[pd.DataFrame, int]:
    valid = []
    n_err = 0

    records = []
    for row in df.to_dict(orient="records"):
        d = {
            k: v for k, v in row.items()
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        }
        records.append(d)

    if PYDANTIC_V2 and TypeAdapter is not None:
        from typing import List
        adapter = TypeAdapter(List[model])

        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            try:
                validated = adapter.validate_python(chunk)
                valid.extend([rec.model_dump() for rec in validated])
            except Exception:
                for d in chunk:
                    try:
                        rec = model(**d)
                        valid.append(rec.model_dump())
                    except Exception:
                        n_err += 1
    else:
        for d in records:
            try:
                rec = model(**d)
                valid.append(rec.dict() if not PYDANTIC_V2 else rec.model_dump())
            except Exception:
                n_err += 1

    return pd.DataFrame(valid), n_err


def load_sc_folder(sc_dir: str) -> Tuple[pd.DataFrame, List[Dict]]:
    files = sorted(glob.glob(os.path.join(sc_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No SC files in {sc_dir}")

    all_df  = []
    reports = []

    for f in tqdm(files, desc="SC cleaning"):
        raw = pd.read_csv(f, dtype=str)
        rep = {"file": os.path.basename(f), "raw_rows": len(raw)}

        df = preprocess_column_types(raw, _SC_SCHEMA)

        if "Quarter" in df.columns:
            df["Quarter"] = df["Quarter"].apply(parse_quarter)

        if "Source" not in df.columns or df.get("Source", pd.Series()).isna().all():
            df["Source"] = "SC Malaysia"

        df = df.rename(columns={k: v for k, v in _SC_RENAME.items() if k in df.columns})
        df = df[[c for c in _SC_FINAL if c in df.columns]].copy()

        df_valid, n_err = _validate_rows(df, ShariahRecord, f"SC:{rep['file']}")
        rep["valid_rows"]      = len(df_valid)
        rep["validation_errors_n"] = n_err

        # Track duplicates explicitly before dropping them
        if not df_valid.empty:
            n_before = len(df_valid)
            df_valid = df_valid.drop_duplicates(
                subset=["stock_code", "quarter"], keep="first"
            )
            rep["rows_after_dedup"]   = len(df_valid)
            rep["duplicates_removed"] = n_before - len(df_valid)
        else:
            rep["rows_after_dedup"]   = 0
            rep["duplicates_removed"] = 0

        all_df.append(df_valid)
        reports.append(rep)

    out = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
    return out, reports


# ─── Compustat ────────────────────────────────────────────────────────────────

_COMP_SCHEMA = {
    "gvkey":   "int",
    "datadate": "date",
    "datacqtr": "str",
    "conm":    "str",
    "isin":    "str",
    "sedol":   "str",
    "fic":     "str",
    "curcdq":  "str",

    "atq":     "float",
    "dlttq":   "float",
    "dlcq":    "float",
    "ltq":     "float",

    "cheq":    "float",
    "chq":     "float",
    "chsq":    "float",

    "revtq":   "float",
    "xintq":   "float",
    "iditq":   "float",
    "nopiq":   "float",
    "niq":     "float",
    "oibdpq":  "float",

    "fqtr":    "int",
    "fyearq":  "int",

    "actq":    "float",
    "lctq":    "float",
    "rectq":   "float",
    "invtq":   "float",
    "ibq":     "float",
    "oancfy":  "float",
    "ppentq":  "float",
    "dpq":     "float",
    "xsgaq":   "float",
    "cogsq":   "float",
    "xrdq":    "float",
    "gpq":     "float",
    "ltmibq":  "float",
}

def load_compustat_csv(
    path: str,
    country_code: str = "MYS",
    currency: str = "MYR",
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    df0  = pd.read_csv(path, dtype=str, low_memory=False)
    rep  = {"raw_rows": len(df0)}

    # Reject rows whose gvkey is non-numeric or all-zero BEFORE any cast.
    # The repaired UAE vendor file carries 43 column-shifted rows whose
    # gvkey cell holds the company NAME ("FALCON ENERGY MATERIALS PLC");
    # Pydantic used to drop them silently — make the drop explicit so a
    # future schema shift is visible in the report. MYS has zero such
    # rows, so this is a no-op there.
    gv = df0["gvkey"].astype(str).str.strip()
    bad_gvkey = ~gv.str.fullmatch(r"\d+") | (gv.str.strip("0") == "")
    n_bad = int(bad_gvkey.sum())
    if n_bad:
        log.warning(
            "load_compustat_csv: dropped %d row(s) with invalid gvkey "
            "(non-numeric or all-zero), e.g. %r.",
            n_bad, df0.loc[bad_gvkey, "gvkey"].iloc[0],
        )
        df0 = df0[~bad_gvkey].copy()
    rep["invalid_gvkey_rows_dropped"] = n_bad

    avail = [c for c in _COMP_SCHEMA if c in df0.columns]
    df    = df0[avail].copy()
    df    = preprocess_column_types(df, {k: v for k, v in _COMP_SCHEMA.items() if k in df.columns})

    if "fic" in df.columns:
        df = df[df["fic"] == country_code].copy()
    rep["after_country_filter"] = len(df)

    if "curcdq" in df.columns:
        non_local = df[df["curcdq"] != currency]
        rep["non_local_currency_obs"]  = int(len(non_local))
        rep["non_local_currencies"]    = sorted(non_local["curcdq"].dropna().unique().tolist())
    else:
        rep["non_local_currency_obs"]  = 0
        rep["non_local_currencies"]    = []

    if "datacqtr" in df.columns:
        df["datacqtr"] = df["datacqtr"].apply(parse_quarter)

    n_before = len(df)
    df = (df.sort_values("datadate")
            .drop_duplicates(subset=["gvkey", "datacqtr"], keep="last"))
    rep["duplicates_removed"] = n_before - len(df)

    # Snapshot before Pydantic validation — used for full-universe intersection stats
    df_pre_pydantic = df.copy()

    df_valid, n_err = _validate_rows(df, CompustatRecord, "Compustat")
    rep["valid_rows"]          = len(df_valid)
    rep["validation_errors_n"] = n_err

    return df_valid, rep, df_pre_pydantic
