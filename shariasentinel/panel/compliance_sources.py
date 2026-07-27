"""Country-routed loading of Sharia compliance lists.

``load_compliance_list`` is the single entry point the orchestrator uses:

- ``MYS`` → :func:`server.panel.cleaning.load_sc_folder` — the historical
  SC Malaysia "wide" folder of quarterly CSVs, untouched.
- ``UAE`` → :func:`load_dfm_long` — one tidy-long CSV
  (``date, quarter, source, stock_code, company_name, shariah_status,
  source_file``) published positive-only: every row IS a compliant
  security; absence means unknown, never non-compliant.

A schema adapter alone is NOT enough to reuse the Bursa anchoring
(``build_panel_sac_anchored`` int-casts stock codes, joins through
ISIN-derived 4-digit Bursa codes and matches the literal
``"Shariah-Compliant"``); the UAE path therefore feeds
``build_panel_compliance_anchored`` (gvkey-level anchoring) through the
name-based crosswalk instead — see ``server/panel/uae_crosswalk.py``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from shariasentinel import config as cfg
from shariasentinel.panel.cleaning import load_sc_folder
from shariasentinel.panel.methodology import CountryCompliancePolicy

log = logging.getLogger(__name__)

# Canonical output columns of every compliance loader (superset of what the
# anchoring consumes). ``shariah_status`` semantics stay authority-native:
# SC Malaysia uses the literal "Shariah-Compliant"; DFM uses int 1.
DFM_EXPECTED_COLUMNS: tuple[str, ...] = (
    "date", "quarter", "source", "stock_code", "company_name",
    "shariah_status", "source_file",
)


def load_dfm_long(path: str | os.PathLike[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Parse the DFM tidy-long compliance list.

    Returns a frame with columns ``source, date_effective, quarter,
    stock_code, company_name, shariah_status`` (one row per security ×
    review date, deduplicated) plus a report list shaped like the SC
    loader's so downstream QC can aggregate either.
    """
    raw = pd.read_csv(path, dtype=str)
    rep: Dict = {"file": os.path.basename(str(path)), "raw_rows": len(raw)}

    missing = [c for c in DFM_EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"DFM list {path} is missing expected columns {missing}; "
            f"got {list(raw.columns)}."
        )

    df = raw.copy()
    df["date_effective"] = pd.to_datetime(df["date"], errors="coerce")
    quarter_num = pd.to_numeric(df["quarter"], errors="coerce")
    df["quarter"] = (
        df["date_effective"].dt.year.astype("Int64").astype(str)
        + "-Q"
        + quarter_num.astype("Int64").astype(str)
    )
    # Upper-case: the source file mixes cases for the same security
    # ("Gulfnav" vs "GULFNAV") — 65 raw codes are really 64 securities.
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["shariah_status"] = pd.to_numeric(df["shariah_status"], errors="coerce")
    df["source"] = df["source"].fillna("DFM")

    n_bad = int(
        df["date_effective"].isna().sum()
        + (df["stock_code"] == "").sum()
        + df["shariah_status"].isna().sum()
    )
    df = df[
        df["date_effective"].notna()
        & (df["stock_code"] != "")
        & df["shariah_status"].notna()
    ].copy()
    rep["valid_rows"] = len(df)
    rep["validation_errors_n"] = n_bad

    n_before = len(df)
    df = df.drop_duplicates(subset=["stock_code", "quarter"], keep="first")
    rep["rows_after_dedup"] = len(df)
    rep["duplicates_removed"] = n_before - len(df)

    # Positive-only sanity: DFM publishes compliant securities exclusively.
    n_nonpositive = int((df["shariah_status"] != 1).sum())
    if n_nonpositive:
        log.warning(
            "load_dfm_long: %d rows with shariah_status != 1 in a "
            "positive-only list — check the source file.", n_nonpositive,
        )
    rep["nonpositive_status_rows"] = n_nonpositive

    out = df[
        ["source", "date_effective", "quarter", "stock_code",
         "company_name", "shariah_status"]
    ].sort_values(["date_effective", "stock_code"]).reset_index(drop=True)

    log.info(
        "load_dfm_long: %d rows, %d securities, %s → %s",
        len(out), out["stock_code"].nunique(),
        out["quarter"].min(), out["quarter"].max(),
    )
    return out, [rep]


def _split_camel_case(s: str) -> str:
    """"BankAljaziraJSC" → "Bank Aljazira JSC" so the name matcher can tokenize."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)      # aB → a B
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)     # ABc → A Bc
    s = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", s)       # a1 → a 1
    s = s.replace("&", " & ")
    return re.sub(r"\s+", " ", s).strip()


def load_boubyan_long(path: str | os.PathLike[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Parse a Boubyan Capital GCC positive list (Saudi ``boubyan_sau``).

    Different schema from DFM/QSE: capitalized headers + BOM
    (``Source, Date_Effective, Quarter, Company_Name, Ticker,
    Shariah_Status``), a ``2021Q1`` quarter string, and a textual status
    (``Shariah-Compliant``) rather than int 1. Normalized to the same
    canonical output as :func:`load_dfm_long`. Positive-only: every row is a
    compliant security.
    """
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    rep: Dict = {"file": os.path.basename(str(path)), "raw_rows": len(raw)}
    cols = {c.lower().strip(): c for c in raw.columns}

    def _col(name: str) -> pd.Series:
        src = cols.get(name)
        if src is None:
            raise ValueError(
                f"Boubyan list {path} missing column {name!r}; "
                f"got {list(raw.columns)}."
            )
        return raw[src]

    df = pd.DataFrame(index=raw.index)
    df["date_effective"] = pd.to_datetime(_col("date_effective"), errors="coerce")
    # "2021Q1" → "2021-Q1" (match the DFM quarter format).
    df["quarter"] = (
        _col("quarter").astype(str).str.strip().str.upper().str.replace("Q", "-Q", regex=False)
    )
    df["stock_code"] = _col("ticker").astype(str).str.strip().str.upper()
    # Boubyan names are camelCase-concatenated ("BankAljaziraJSC") — split
    # into space-separated tokens so the name crosswalk can match them
    # against the spaced Compustat conm ("BANK ALJAZIRA").
    df["company_name"] = _col("company_name").astype(str).str.strip().map(_split_camel_case)
    status = _col("shariah_status").astype(str).str.strip().str.lower()
    recognized = status.isin(["shariah-compliant", "compliant", "1", "1.0"])
    df["shariah_status"] = 1  # positive-only list: recognized rows are compliant
    df["_status_recognized"] = recognized
    df["source"] = (
        raw[cols["source"]].fillna("Boubyan") if "source" in cols else "Boubyan"
    )

    # Positive-only source: an UNRECOGNIZED status is invalid data, not an
    # implicit non-compliant label — drop it (loudly) rather than emit a 0.
    n_unrecognized = int((~recognized).sum())
    if n_unrecognized:
        bad_values = sorted(status[~recognized].unique())[:5]
        log.warning(
            "load_boubyan_long: dropped %d row(s) with unrecognized "
            "shariah_status in a positive-only list (e.g. %s).",
            n_unrecognized, bad_values,
        )
    n_bad = int(
        df["date_effective"].isna().sum()
        + (df["stock_code"] == "").sum()
        + n_unrecognized
    )
    df = df[
        df["date_effective"].notna()
        & (df["stock_code"] != "")
        & df["_status_recognized"]
    ].drop(columns=["_status_recognized"]).copy()
    rep["valid_rows"] = len(df)
    rep["validation_errors_n"] = n_bad

    n_before = len(df)
    df = df.drop_duplicates(subset=["stock_code", "quarter"], keep="first")
    rep["rows_after_dedup"] = len(df)
    rep["duplicates_removed"] = n_before - len(df)
    rep["nonpositive_status_rows"] = int((df["shariah_status"] != 1).sum())

    out = df[
        ["source", "date_effective", "quarter", "stock_code",
         "company_name", "shariah_status"]
    ].sort_values(["date_effective", "stock_code"]).reset_index(drop=True)
    log.info(
        "load_boubyan_long: %d rows, %d securities, %s → %s",
        len(out), out["stock_code"].nunique(),
        out["quarter"].min(), out["quarter"].max(),
    )
    return out, [rep]


_PSX_COMPLIANT = {"compliant"}
_PSX_NONCOMPLIANT = {"non-compliant", "nc by nature"}  # explicit non-compliant


def load_psx_kmi_long(path: str | os.PathLike[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Parse the PSX / KMI (Pakistan) list — a NEGATIVE-determinable list.

    Unlike the positive-only Gulf lists, PSX publishes both compliant and
    non-compliant verdicts, so ``shariah_status`` is tri-state (1/0/NA). The
    authoritative verdict is in ``final_shariah_status`` (the plain
    ``shariah_status`` column is polluted with ratio values):

      Compliant → 1 · Non-Compliant / NC by Nature → 0 · No Opinion / N/A → drop

    ``date`` + numeric ``quarter`` mirror the DFM layout.
    """
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    rep: Dict = {"file": os.path.basename(str(path)), "raw_rows": len(raw)}
    required = ["date", "quarter", "stock_code", "company_name", "final_shariah_status"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"PSX/KMI list {path} missing columns {missing}; got {list(raw.columns)}."
        )

    df = pd.DataFrame(index=raw.index)
    df["date_effective"] = pd.to_datetime(raw["date"], errors="coerce")
    qn = pd.to_numeric(raw["quarter"], errors="coerce")
    df["quarter"] = (
        df["date_effective"].dt.year.astype("Int64").astype(str)
        + "-Q" + qn.astype("Int64").astype(str)
    )
    df["stock_code"] = raw["stock_code"].astype(str).str.strip().str.upper()
    df["company_name"] = raw["company_name"].astype(str).str.strip()

    status = raw["final_shariah_status"].astype(str).str.strip().str.lower()
    val = pd.Series(pd.NA, index=raw.index, dtype="Float64")
    val = val.mask(status.isin(_PSX_COMPLIANT), 1.0)
    val = val.mask(status.isin(_PSX_NONCOMPLIANT), 0.0)
    df["shariah_status"] = val
    df["source"] = raw["source"].fillna("PSX") if "source" in raw.columns else "PSX"

    n_no_verdict = int(df["shariah_status"].isna().sum())  # No Opinion / N/A
    n_bad = int(df["date_effective"].isna().sum() + (df["stock_code"] == "").sum())
    df = df[
        df["date_effective"].notna()
        & (df["stock_code"] != "")
        & df["shariah_status"].notna()
    ].copy()
    df["shariah_status"] = df["shariah_status"].astype(int)
    rep["valid_rows"] = len(df)
    rep["dropped_no_verdict"] = n_no_verdict
    rep["validation_errors_n"] = n_bad

    n_before = len(df)
    df = df.drop_duplicates(subset=["stock_code", "quarter"], keep="first")
    rep["rows_after_dedup"] = len(df)
    rep["duplicates_removed"] = n_before - len(df)

    out = df[
        ["source", "date_effective", "quarter", "stock_code",
         "company_name", "shariah_status"]
    ].sort_values(["date_effective", "stock_code"]).reset_index(drop=True)
    log.info(
        "load_psx_kmi_long: %d rows, %d securities (%d compliant, %d non-compliant), "
        "%s → %s; dropped %d no-verdict.",
        len(out), out["stock_code"].nunique(),
        int((out["shariah_status"] == 1).sum()), int((out["shariah_status"] == 0).sum()),
        out["quarter"].min(), out["quarter"].max(), n_no_verdict,
    )
    return out, [rep]


def load_boubyan_kwt_long(path: str | os.PathLike[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Parse the Boubyan Kuwait positive list (``boubyan_kwt``).

    Lowercase tidy-long like DFM, but keyed by ``ticker`` (not ``stock_code``)
    and with a letter-prefixed ``quarter`` (``Q4``, no year) — the year comes
    from ``date``. Positive-only (every row is compliant).
    """
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    rep: Dict = {"file": os.path.basename(str(path)), "raw_rows": len(raw)}
    required = ["date", "quarter", "ticker", "company_name", "shariah_status"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Boubyan KWT list {path} missing columns {missing}; got {list(raw.columns)}."
        )

    df = pd.DataFrame(index=raw.index)
    df["date_effective"] = pd.to_datetime(raw["date"], errors="coerce")
    # "Q4" → "YYYY-Q4" (year from the effective date).
    qnum = raw["quarter"].astype(str).str.strip().str.upper().str.replace("Q", "", regex=False)
    df["quarter"] = df["date_effective"].dt.year.astype("Int64").astype(str) + "-Q" + qnum
    df["stock_code"] = raw["ticker"].astype(str).str.strip().str.upper()
    df["company_name"] = raw["company_name"].astype(str).str.strip()
    status = raw["shariah_status"].astype(str).str.strip().str.lower()
    recognized = status.isin(["shariah-compliant", "compliant", "1", "1.0"])
    df["shariah_status"] = 1
    df["_rec"] = recognized
    df["source"] = raw["source"].fillna("Boubyan") if "source" in raw.columns else "Boubyan"

    n_unrecognized = int((~recognized).sum())
    if n_unrecognized:
        log.warning(
            "load_boubyan_kwt_long: dropped %d row(s) with unrecognized "
            "shariah_status in a positive-only list.", n_unrecognized,
        )
    df = df[
        df["date_effective"].notna() & (df["stock_code"] != "") & df["_rec"]
    ].drop(columns=["_rec"]).copy()
    n_before = len(df)
    df = df.drop_duplicates(subset=["stock_code", "quarter"], keep="first")
    rep["valid_rows"] = len(df)
    rep["duplicates_removed"] = n_before - len(df)

    out = df[
        ["source", "date_effective", "quarter", "stock_code",
         "company_name", "shariah_status"]
    ].sort_values(["date_effective", "stock_code"]).reset_index(drop=True)
    log.info(
        "load_boubyan_kwt_long: %d rows, %d securities, %s → %s",
        len(out), out["stock_code"].nunique(),
        out["quarter"].min(), out["quarter"].max(),
    )
    return out, [rep]


def load_compliance_list(
    policy: CountryCompliancePolicy,
    sc_dir: str | None = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Load the authority compliance list for ``policy``'s country."""
    if policy.country_key == "MYS":
        return load_sc_folder(sc_dir or str(cfg.raw_sc_dir()))
    if policy.country_key == "UAE":
        return load_dfm_long(cfg.raw_dfm_uae_csv())
    if policy.country_key == "QAT":
        # QSE list is the same tidy-long, positive-only shape as DFM
        # (date, quarter, stock_code, company_name, shariah_status=1) plus
        # extra columns the loader ignores — reuse the DFM reader directly.
        return load_dfm_long(cfg.raw_qse_qat_csv())
    if policy.country_key == "SAU":
        return load_boubyan_long(cfg.raw_boubyan_sau_csv())
    if policy.country_key == "PAK":
        return load_psx_kmi_long(cfg.raw_psx_kmi_pak_csv())
    if policy.country_key == "KWT":
        return load_boubyan_kwt_long(cfg.raw_boubyan_kwt_csv())
    if policy.country_key == "IDN":
        # OJK DES list is the same tidy-long, positive-only shape as DFM
        # (date, quarter, stock_code, company_name, shariah_status=1, with
        # extra columns the loader ignores) — reuse the DFM reader.
        return load_dfm_long(cfg.raw_des_idn_csv())
    raise KeyError(
        f"No compliance-list loader wired for country {policy.country_key!r}."
    )
