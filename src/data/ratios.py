"""SC Malaysia SAC Shariah ratios: two balance-sheet ratios and one income ratio.

Official source:
  SC Malaysia SAC — Shariah-Compliant Securities Screening Methodology
  https://www.sc.com.my/development/icm/shariah-compliant-securities/
      shariah-compliant-securities-screening-methodology

Official ratios (verified against source):
  1. Interest-bearing debt / Total assets  < 33%
  2. Conventional cash / Total assets      < 33%
  3. Non-permissible income / Group total income < 5%  (Compustat proxy)

Variables excluded from scope:
  - ltq, lctq  : too broad (include Islamic and non-interest-bearing debt)
                 used only as a diagnostic upper bound
  - rectq, rectrq : receivables ratio absent from SC Malaysia SAC methodology
  - chsq : kept in schema but often redundant with cheq

Documented limitations:
  - Compustat does not distinguish Islamic vs. conventional financing
    -> ratios 1 and 2 are conservative proxies (possible overestimation)
    -> Islamic correction applied via sukuk_ratio_t and islamic_cash_ratio_t

  - Ratio 3 — Non-permissible income (R4 SAC):
    R4 SAC = Non-Permissible Income RECEIVED / Group total income  (threshold: 5%)
    SAC resolution 281st meeting (July 2024): denominator = total income only.
    SAC resolution 288th meeting (February 2025): single 5% benchmark, 20% removed.

    -> ratio_income = iditq / revtq  (SAC-compliant proxy <- MAIN)
       iditq = interest & dividend income RECEIVED — closest match to R4 SAC.
    -> ratio_income_cashadj = iditq x (1 - islamic_cash_ratio_t) / revtq
       robustness specification: adjusts iditq for estimated Islamic share.
    -> iditq NaN->0 available via ratio_income_idit_z0 (conservative assumption).
    -> xintq (interest expense PAID) retained as raw financial variable
       but no longer used in any ratio.

  - revtq = total revenue, proxy for SAC "Group total income"
    (SAC definition also includes other income and share of profit)

Islamic corrections (sukuk_ratio_t, islamic_cash_ratio_t):
  - Must be present in the input panel (prior merge required).
  - If absent, adjusted ratios will be NaN, and a warning is logged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.methodology import CountryCompliancePolicy, policy_for_panel

log = logging.getLogger(__name__)

# This file lives at <repo>/src/panel/ratios.py, so the repo root
# is parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MACRO_DIR = _PROJECT_ROOT / "data" / "raw" / "macro_connectors"


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division: NaN if the denominator is <= 0 or NaN."""
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den.notna() & (den > 0) & num.notna()
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def _combine_missing_as_nan(*series: pd.Series) -> pd.Series:
    """
    Strict combination:
    - if any component is NaN on a row -> return NaN
    - otherwise return the sum
    """
    df_tmp = pd.concat(series, axis=1)
    any_null = df_tmp.isna().any(axis=1)
    result = df_tmp.sum(axis=1)
    result.loc[any_null] = np.nan
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Islamic macro connectors
# ──────────────────────────────────────────────────────────────────────────────

def _coerce_year_column(tbl: pd.DataFrame, country_key: str, label: str) -> pd.DataFrame:
    """Coerce ``year`` to int, DROPPING rows whose year is not a plain integer.

    A vendor macro file may carry an interim label like ``2019-H1`` (a
    half-year sukuk row); the old ``to_numeric(...).astype(int)`` raised ``IntCastingNaNError``
    on those. We drop such rows (with a warning) rather than crash the panel —
    the annual series still covers the panel via the nearest-year fill.
    """
    year = pd.to_numeric(tbl["year"], errors="coerce")
    bad = year.isna()
    if bad.any():
        dropped = tbl.loc[bad, "year"].astype(str).tolist()
        log.warning(
            "%s connector [%s]: dropped %d row(s) with a non-integer year %s.",
            label, country_key, int(bad.sum()), dropped,
        )
    out = tbl.loc[~bad].copy()
    out["year"] = year[~bad].astype(int)
    return out


def build_sukuk_connector(country_key: str = "MYS") -> pd.DataFrame:
    """
    Load the annual sukuk share in corporate debt for a country.

    - ``MYS`` (default): ``macro_connectors/sukuk_annual.csv`` — flat file,
      behaviour unchanged. Key columns: year, R_t, sukuk_ratio_t (alias of
      R_t), sukuk_status, sukuk_source.
    - ``UAE``: ``macro_connectors/uae/sukuk_annual.csv`` — the observed
      ``R_t`` (USD-denominated DCM shares) exists for 3 years only; the
      file ships ``R_t_operational_fill`` for every year, used as the
      operational fallback and marked ``R_T_OPERATIONAL_FILL`` in
      ``sukuk_status``. Output is normalized to the canonical columns.
    """
    if country_key == "MYS":
        path = _MACRO_DIR / "sukuk_annual.csv"
        tbl = pd.read_csv(path, dtype={"sukuk_status": str, "sukuk_source": str})
        tbl["year"] = tbl["year"].astype(int)
        tbl["sukuk_ratio_t"] = tbl["R_t"]
        return tbl.sort_values("year").reset_index(drop=True)

    path = _MACRO_DIR / country_key.lower() / "sukuk_annual.csv"
    if not path.exists():
        log.warning(
            "no sukuk connector for %s at %s — sukuk_ratio_t (→ ratio_debt_adj) "
            "will be NaN.", country_key, path,
        )
        return pd.DataFrame(
            columns=["year", "R_t", "sukuk_ratio_t", "sukuk_status", "sukuk_source"]
        )
    tbl = pd.read_csv(path)
    tbl = _coerce_year_column(tbl, country_key, "sukuk")
    r_t = pd.to_numeric(tbl.get("R_t"), errors="coerce")
    # Some country files (e.g. Saudi) ship no operational-fill column — default
    # to an all-NaN Series so ``.notna()`` below stays a vector op, not a scalar.
    fill = pd.to_numeric(
        tbl.get("R_t_operational_fill", pd.Series(np.nan, index=tbl.index)),
        errors="coerce",
    )
    tbl["R_t"] = r_t
    tbl["sukuk_ratio_t"] = r_t.fillna(fill)
    status = tbl.get("sukuk_status", pd.Series(pd.NA, index=tbl.index)).astype("string")
    status = status.mask(status.str.upper().isin(["NA", "NAN"]), pd.NA)
    tbl["sukuk_status"] = status.mask(r_t.isna() & fill.notna(), "R_T_OPERATIONAL_FILL")
    source = tbl.get("sukuk_source", pd.Series(pd.NA, index=tbl.index)).astype("string")
    tbl["sukuk_source"] = source.mask(source.str.upper().isin(["NA", "NAN"]), pd.NA)
    return (
        tbl[["year", "R_t", "sukuk_ratio_t", "sukuk_status", "sukuk_source"]]
        .sort_values("year")
        .reset_index(drop=True)
    )


def build_islamic_cash_connector(country_key: str = "MYS") -> pd.DataFrame:
    """
    Load the annual Islamic share of deposits/cash for a country.

    - ``MYS`` (default): flat file, behaviour unchanged. Key columns:
      year, islamic_cash_ratio_t, cash_ratio_status, cash_source.
    - ``UAE``: ``macro_connectors/uae/islamic_cash_annual.csv`` —
      normalized to the same canonical columns.
    """
    if country_key == "MYS":
        path = _MACRO_DIR / "islamic_cash_annual.csv"
        tbl = pd.read_csv(path, dtype={"cash_ratio_status": str, "cash_source": str})
        tbl["year"] = tbl["year"].astype(int)
        return tbl.sort_values("year").reset_index(drop=True)

    path = _MACRO_DIR / country_key.lower() / "islamic_cash_annual.csv"
    if not path.exists():
        log.warning(
            "no islamic-cash connector for %s at %s — islamic_cash_ratio_t "
            "(→ ratio_cash_adj / ratio_income_cashadj) will be NaN.",
            country_key, path,
        )
        return pd.DataFrame(
            columns=["year", "islamic_cash_ratio_t", "cash_ratio_status", "cash_source"]
        )
    tbl = pd.read_csv(path, dtype={"cash_ratio_status": str, "cash_source": str})
    tbl = _coerce_year_column(tbl, country_key, "islamic-cash")
    tbl["islamic_cash_ratio_t"] = pd.to_numeric(
        tbl["islamic_cash_ratio_t"], errors="coerce"
    )
    return (
        tbl[["year", "islamic_cash_ratio_t", "cash_ratio_status", "cash_source"]]
        .sort_values("year")
        .reset_index(drop=True)
    )


def build_macro_connectors(country_key: str = "MYS") -> pd.DataFrame:
    """
    Merge both annual connectors by year for the given country.
    """
    sukuk = build_sukuk_connector(country_key)
    cash = build_islamic_cash_connector(country_key)

    macro = sukuk.merge(cash, on="year", how="outer").sort_values("year").reset_index(drop=True)
    return macro


def attach_macro_connectors(
    panel: pd.DataFrame,
    country_key: str = "MYS",
) -> pd.DataFrame:
    """
    Attach annual Islamic macro connectors to the panel:
      - R_t
      - sukuk_ratio_t
      - islamic_cash_ratio_t
      - associated status/source columns

    The merge key 'year' is inferred from:
      1. existing 'fyearq' column
      2. otherwise 'year' column
      3. otherwise datadate
      4. otherwise date_effective

    If connector columns already exist, they are dropped first to avoid _x / _y suffixes.

    Non-MYS countries use a **nearest-year operational fill**: panel years
    outside the connector's observed span borrow the closest year's value
    (UAE islamic-cash covers 2015–2018 only, while the panel spans
    2012–2025 — a hard year-equality merge would void ``ratio_cash_adj``
    for most rows). Filled rows are marked ``NEAREST_YEAR_FILL(<year>)``
    in the status columns and the borrowed year is exposed as
    ``macro_year_source`` so the approximation stays auditable.
    """
    df = panel.copy()

    if "fyearq" in df.columns and df["fyearq"].notna().any():
        df["year"] = pd.to_numeric(df["fyearq"], errors="coerce")
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    elif "datadate" in df.columns:
        df["year"] = pd.to_datetime(df["datadate"], errors="coerce").dt.year
    elif "date_effective" in df.columns:
        df["year"] = pd.to_datetime(df["date_effective"], errors="coerce").dt.year
    else:
        raise ValueError("Could not infer merge year from fyearq, year, datadate, or date_effective.")
    
    connector_cols = [
        "R_t",
        "sukuk_ratio_t",
        "sukuk_status",
        "sukuk_source",
        "sukuk_outstanding_Bn_MYR",
        "bonds_outstanding_Bn_MYR",
        "total_corporate_debt_Bn_MYR",
        "islamic_cash_ratio_t",
        "cash_ratio_status",
        "cash_source",
        "islamic_deposits_ia_Bn_MYR",
        "macro_year_source",
    ]
    cols_to_drop = [c for c in connector_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if country_key == "MYS":
        # Historical exact-year merge — byte-identical MYS behaviour.
        macro = build_macro_connectors(country_key)
        df = df.merge(macro, on="year", how="left")
        df = df.drop(columns=["year"], errors="ignore")
        return df

    # Non-MYS: nearest-year operational fill, PER CONNECTOR — each
    # connector borrows from its own observed years (UAE sukuk spans
    # 2012–2025 but islamic-cash only 2015–2018; a joint nearest-row
    # merge would leave the cash ratio NaN outside its span).
    connectors: list[tuple[str, pd.DataFrame, str, str]] = [
        (
            "sukuk",
            build_sukuk_connector(country_key),
            "sukuk_ratio_t",
            "sukuk_status",
        ),
        (
            "islamic_cash",
            build_islamic_cash_connector(country_key),
            "islamic_cash_ratio_t",
            "cash_ratio_status",
        ),
    ]

    df["_macro_row_order"] = np.arange(len(df))
    has_year = df["year"].notna()
    left = df.loc[has_year].copy()
    left["year"] = left["year"].astype(int)
    left = left.sort_values("year", kind="mergesort")
    rest = df.loc[~has_year].copy()

    for name, tbl, value_col, status_col in connectors:
        src_year_col = f"_{name}_year_source"
        tbl = tbl[tbl[value_col].notna()].sort_values("year")
        tbl = tbl.rename(columns={"year": src_year_col})
        if tbl.empty:
            # No macro connector for this country (missing/empty file).
            # ZERO-fill the Islamic share rather than leaving it NaN: with a
            # 0 share the Islamic-adjusted ratio collapses to the RAW
            # (conventional) ratio — the conservative screening choice — so
            # flag_debt / flag_cash stay computable instead of being disabled
            # entirely. Marked with an auditable status so the degrade is
            # visible. (merge_asof is skipped: its empty object-dtype key
            # would raise.)
            for frame in (left, rest):
                for col in tbl.columns:
                    if col != src_year_col and col not in frame.columns:
                        frame[col] = np.nan
                frame[value_col] = 0.0
                frame[status_col] = "NO_CONNECTOR_ZERO_FILL"
            log.warning(
                "attach_macro_connectors[%s]: %s connector absent — %s "
                "zero-filled; adjusted ratio degrades to raw (conservative).",
                country_key, name, value_col,
            )
            continue
        if not rest.empty:
            # Rows with no inferable fiscal/calendar year cannot use nearest-year
            # matching. Keep the conservative behavior from the missing-connector
            # path: zero-fill the Islamic share so adjusted ratios collapse to raw
            # ratios and flags remain computable, with an explicit status marker.
            rest[value_col] = 0.0
            rest[status_col] = "NO_YEAR_ZERO_FILL"
        left = pd.merge_asof(
            left,
            tbl,
            left_on="year",
            right_on=src_year_col,
            direction="nearest",
        )
        borrowed = left[src_year_col].notna() & (left[src_year_col] != left["year"])
        n_borrowed = int(borrowed.sum())
        if n_borrowed and status_col in left.columns:
            marker = (
                "NEAREST_YEAR_FILL("
                + left[src_year_col].astype("Int64").astype(str)
                + ")"
            )
            orig = left[status_col].astype("string")
            combined = (orig.fillna("") + "|" + marker).str.lstrip("|")
            left[status_col] = orig.where(~borrowed, combined)
        log.info(
            "attach_macro_connectors[%s]: %s — %d/%d rows borrowed the "
            "nearest observed year (observed span %d–%d).",
            country_key, name, n_borrowed, len(left),
            int(tbl[src_year_col].min()), int(tbl[src_year_col].max()),
        )
        left = left.drop(columns=[src_year_col], errors="ignore")

    out = pd.concat([left, rest], ignore_index=False, sort=False)
    out = out.sort_values("_macro_row_order").reset_index(drop=True)
    out = out.drop(columns=["year", "_macro_row_order"], errors="ignore")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shariah ratio computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_shariah_ratios(
    panel: pd.DataFrame,
    *,
    policy: CountryCompliancePolicy | None = None,
    log_coverage: bool = True,
    warn_on_missing_connectors: bool = True,
) -> pd.DataFrame:
    """
    Add authority Shariah ratio and flag columns to the panel.

    ``policy`` selects the flag thresholds (SAC 33/33/5, DFM 30/30/10…).
    When ``None`` it resolves from the panel's ``methodology_key`` column
    — absent on MYS panels, which therefore keep the SAC defaults and
    stay byte-identical. The ratio *definitions* are methodology-agnostic;
    only the flag cutoffs move.

    New columns:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Ratio 1 — Interest-bearing debt / Total assets                          │
    │   ratio_debt         (dlttq + dlcq) / atq              [raw]           │
    │   ratio_debt_adj     (dlttq*(1-R_t) + dlcq) / atq      [adjusted]      │
    │   ratio_debt_upper   ltq / atq                         [upper bound]   │
    │                                                                         │
    │ Ratio 2 — Conventional cash / Total assets                              │
    │   ratio_cash         cheq / atq                        [raw]           │
    │   ratio_cash_adj     cheq*(1-islamic_cash_ratio_t)/atq [adjusted]      │
    │   ratio_cash_strict  chq / atq                         [strict]        │
    │   ratio_cash_alt     chsq / atq                        [alt.]          │
    │                                                                         │
    │ Ratio 3 — Non-permissible income / Group total income                   │
    │   ratio_income           iditq / revtq                 [SAC-compliant <- MAIN] │
    │   ratio_income_idit      iditq / revtq                 [alias ratio_income]    │
    │   ratio_income_cashadj   iditq*(1-cash_t)/revtq        [robustness]           │
    │   ratio_income_idit_z0   iditq_z0 / revtq              [NaN->0]               │
    │   ratio_income_nopi      nopiq_pos / revtq             [broad proxy]          │
    │                                                                         │
    │ Main flags (based on ratio_debt_adj, ratio_cash_adj, ratio_income,      │
    │ thresholds per the resolved country policy)                             │
    │   flag_debt  flag_cash  flag_income  flag_any  n_flags                  │
    │                                                                         │
    │ Alternative income flags                                                │
    │   flag_income_idit  flag_income_cashadj  flag_income_idit_z0            │
    │   flag_income_nopi                                                      │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    df = panel.copy()

    atq   = pd.to_numeric(df.get("atq",   pd.Series(np.nan, index=df.index)), errors="coerce")
    dlttq = pd.to_numeric(df.get("dlttq", pd.Series(np.nan, index=df.index)), errors="coerce")
    dlcq  = pd.to_numeric(df.get("dlcq",  pd.Series(np.nan, index=df.index)), errors="coerce")
    ltq   = pd.to_numeric(df.get("ltq",   pd.Series(np.nan, index=df.index)), errors="coerce")
    cheq  = pd.to_numeric(df.get("cheq",  pd.Series(np.nan, index=df.index)), errors="coerce")
    chq   = pd.to_numeric(df.get("chq",   pd.Series(np.nan, index=df.index)), errors="coerce")
    chsq  = pd.to_numeric(df.get("chsq",  pd.Series(np.nan, index=df.index)), errors="coerce")
    revtq = pd.to_numeric(df.get("revtq", pd.Series(np.nan, index=df.index)), errors="coerce")
    iditq = pd.to_numeric(df.get("iditq", pd.Series(np.nan, index=df.index)), errors="coerce")
    nopiq = pd.to_numeric(df.get("nopiq", pd.Series(np.nan, index=df.index)), errors="coerce")

    # ── Islamic macro ratios ────────────────────────────────────────────────
    sukuk_ratio = pd.to_numeric(
        df.get("sukuk_ratio_t", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    islamic_cash_ratio = pd.to_numeric(
        df.get("islamic_cash_ratio_t", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )

    # Policy gate: when the country's sukuk series is not trusted for the debt
    # screen (country-total / non-corporate proxy), force the
    # sukuk share to 0 so ``ratio_debt_adj`` degrades to the RAW (conservative)
    # ratio. The cash adjustment is untouched. Resolve the policy up front so
    # this holds regardless of whether it was passed or read from the panel.
    _pol = policy if policy is not None else policy_for_panel(df)
    if not getattr(_pol, "apply_sukuk_adjustment", True):
        sukuk_ratio = pd.Series(0.0, index=df.index)
        # Keep the loaded proxy visible for audit, but stamp every row so the
        # panel self-documents that it was NOT applied to the debt screen —
        # avoids the "sukuk_ratio_t=0.23 yet ratio_debt_adj==ratio_debt"
        # confusion (Codex follow-up).
        if "sukuk_status" in df.columns:
            _marker = "LOADED_NOT_APPLIED_TO_DEBT_SCREEN"
            _base = df["sukuk_status"].astype("string").fillna("")
            _needs = ~_base.str.contains(_marker, na=False)
            df.loc[_needs, "sukuk_status"] = (
                (_base[_needs] + "|" + _marker).str.lstrip("|")
            )
        if log_coverage:
            log.info(
                "compute_shariah_ratios[%s]: sukuk adjustment disabled by "
                "policy (non-corporate proxy) — debt uses the raw ratio; "
                "sukuk_ratio_t kept for audit.",
                getattr(_pol, "country_key", "?"),
            )

    if warn_on_missing_connectors and sukuk_ratio.isna().all():
        log.warning("sukuk_ratio_t missing from panel — ratio_debt_adj will be NaN.")
    if warn_on_missing_connectors and islamic_cash_ratio.isna().all():
        log.warning("islamic_cash_ratio_t missing from panel — ratio_cash_adj and ratio_income_cashadj will be NaN.")

    # ── Ratio 1: Interest-bearing debt / atq ────────────────────────────────
    debt_principal = _combine_missing_as_nan(dlttq, dlcq)
    df["ratio_debt"] = _safe_div(debt_principal, atq)

    # Islamic adjustment: subtract estimated sukuk share from dlttq.
    # dlcq is not adjusted (short-term debt — sukuk ST not distinguishable).
    dlttq_adj = dlttq * (1 - sukuk_ratio)
    debt_adj = _combine_missing_as_nan(dlttq_adj, dlcq)
    df["ratio_debt_adj"] = _safe_div(debt_adj, atq)

    # Diagnostic upper bound
    df["ratio_debt_upper"] = _safe_div(ltq, atq)

    # ── Ratio 2: Conventional cash / atq ────────────────────────────────────
    df["ratio_cash"] = _safe_div(cheq, atq)

    # Islamic adjustment: subtract estimated Islamic deposits share
    cheq_adj = cheq * (1 - islamic_cash_ratio)
    df["ratio_cash_adj"] = _safe_div(cheq_adj, atq)

    # Alternative specifications
    df["ratio_cash_strict"] = _safe_div(chq, atq)
    df["ratio_cash_alt"]    = _safe_div(chsq, atq)

    # ── Ratio 3: Non-permissible income / Group total income ─────────────────
    # SAC-compliant proxy: iditq = interest & dividend income RECEIVED.
    # Closest match to "conventional interest income received" targeted by
    # SAC resolutions (2024-2025).
    df["ratio_income_idit"] = _safe_div(iditq, revtq)

    # ratio_income = canonical alias -> points to the SAC-compliant proxy (iditq)
    df["ratio_income"] = df["ratio_income_idit"]

    # Macro-adjusted proxy: for non-financials, iditq mainly comes from
    # cash placements; the conventional share is approximated by (1 - islamic_cash_ratio_t).
    iditq_cashadj = iditq * (1 - islamic_cash_ratio)
    df["ratio_income_cashadj"] = _safe_div(iditq_cashadj, revtq)

    # Variant: NaN replaced by 0 before computation.
    # Assumption: if iditq is missing, interest income received ≈ 0.
    iditq_z0 = iditq.fillna(0)
    df["iditq_z0_flag"] = iditq.isna().astype(int)  # 1 = value imputed to 0
    df["ratio_income_idit_z0"] = _safe_div(iditq_z0, revtq)

    # Broad proxy: nopiq clipped to 0 if negative.
    nopiq_pos = nopiq.clip(lower=0)
    df["ratio_income_nopi"] = _safe_div(nopiq_pos, revtq)

    # ── Main flags ──────────────────────────────────────────────────────────
    # Based on the authority's active adjusted-ratio screens. When a regime has
    # no cash screen (Indonesia DES), keep the diagnostic column present but
    # mark it NaN so it cannot contribute to flag_any / n_flags.
    def _flag(ratio: pd.Series, threshold: float) -> pd.Series:
        out = pd.Series(np.nan, index=ratio.index, dtype=float)
        valid = ratio.notna()
        out.loc[valid] = (ratio.loc[valid] > threshold).astype(int)
        return out

    active_policy = policy or policy_for_panel(df)
    df["flag_debt"]   = _flag(df["ratio_debt_adj"], active_policy.threshold_debt)
    if active_policy.cash_screen_enabled:
        df["flag_cash"] = _flag(df["ratio_cash_adj"], active_policy.threshold_cash)
    else:
        df["flag_cash"] = pd.Series(np.nan, index=df.index, dtype=float)
    df["flag_income"] = _flag(df["ratio_income"],   active_policy.threshold_income)

    flags = df[["flag_debt", "flag_cash", "flag_income"]]
    df["flag_any"] = flags.max(axis=1)
    df["n_flags"]  = flags.sum(axis=1, min_count=1)

    # ── Alternative income flags ─────────────────────────────────────────────
    df["flag_income_idit"]     = _flag(df["ratio_income_idit"],     active_policy.threshold_income)
    df["flag_income_cashadj"]  = _flag(df["ratio_income_cashadj"],  active_policy.threshold_income)
    df["flag_income_idit_z0"]  = _flag(df["ratio_income_idit_z0"],  active_policy.threshold_income)
    df["flag_income_nopi"]     = _flag(df["ratio_income_nopi"],     active_policy.threshold_income)

    # ── Coverage log ────────────────────────────────────────────────────────
    if log_coverage:
        n = len(df)
        coverage_cols = [
            ("ratio_debt",            "dlttq+dlcq / atq            [raw]"),
            ("ratio_debt_adj",        "dlttq_adj+dlcq / atq        [adjusted]"),
            ("ratio_debt_upper",      "ltq / atq                   [upper bound]"),
            ("ratio_cash",            "cheq / atq                  [raw]"),
            ("ratio_cash_adj",        "cheq_adj / atq              [adjusted]"),
            ("ratio_cash_strict",     "chq / atq                   [strict]"),
            ("ratio_cash_alt",        "chsq / atq                  [alt.]"),
            ("ratio_income",          "iditq / revtq               [SAC-compliant <- MAIN]"),
            ("ratio_income_idit",     "iditq / revtq               [alias ratio_income]"),
            ("ratio_income_cashadj",  "iditq_cashadj / revtq       [robustness]"),
            ("ratio_income_idit_z0",  "iditq_z0 / revtq            [NaN->0]"),
            ("ratio_income_nopi",     "nopiq_pos / revtq           [broad proxy]"),
        ]
        for col, label in coverage_cols:
            n_ok = int(df[col].notna().sum())
            log.info("Coverage %-40s: %d/%d (%.1f%%)", label, n_ok, n, 100 * n_ok / max(n, 1))

    return df


def compute_ratio_stats(panel: pd.DataFrame) -> dict:
    """
    Stats descriptives sur les ratios Shariah — bruts, ajustés et alternatifs.
    Retourne un dict à merger dans build_qc_report().
    """
    ratio_cols = [
        "ratio_debt",
        "ratio_debt_adj",
        "ratio_debt_upper",
        "ratio_cash",
        "ratio_cash_adj",
        "ratio_cash_strict",
        "ratio_cash_alt",
        "ratio_income",
        "ratio_income_idit",
        "ratio_income_cashadj",
        "ratio_income_idit_z0",
        "ratio_income_nopi",
    ]
    flag_cols = [
        "flag_debt",
        "flag_cash",
        "flag_income",
        "flag_income_idit",
        "flag_income_cashadj",
        "flag_income_idit_z0",
        "flag_income_nopi",
        "flag_any",
    ]

    out: dict = {"shariah_ratios": {}}

    for col in ratio_cols:
        if col not in panel.columns:
            continue
        s = panel[col].dropna()
        if len(s) == 0:
            continue
        out["shariah_ratios"][col] = {
            "n_valid": int(len(s)),
            "n_missing": int(panel[col].isna().sum()),
            "pct_valid": round(100 * len(s) / max(len(panel), 1), 1),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
        }

    for col in flag_cols:
        if col not in panel.columns:
            continue
        f = panel[col].dropna()
        out["shariah_ratios"][f"{col}_n_noncompliant"] = int((f == 1).sum())
        out["shariah_ratios"][f"{col}_pct_noncompliant"] = round(
            100 * (f == 1).sum() / max(len(f), 1),
            1,
        )

    # ── Table comparatif income (résumé lisible) ────────────────────────────
    income_proxies = {
        "iditq (revenu reçu) ← PRINCIPAL": "ratio_income",
        "iditq cash-adjusted (robustesse)": "ratio_income_cashadj",
        "iditq NaN→0 (conservateur)": "ratio_income_idit_z0",
        "nopiq clippé (diagnostique)": "ratio_income_nopi",
    }

    comparatif = {}
    for label, col in income_proxies.items():
        flag_col = {
            "ratio_income":          "flag_income",
            "ratio_income_cashadj":  "flag_income_cashadj",
            "ratio_income_idit_z0":  "flag_income_idit_z0",
            "ratio_income_nopi":     "flag_income_nopi",
        }.get(col, col)

        if col not in panel.columns:
            continue

        s = panel[col].dropna()
        f = panel[flag_col].dropna() if flag_col in panel.columns else pd.Series(dtype=float)
        comparatif[label] = {
            "couverture_pct": round(100 * len(s) / max(len(panel), 1), 1),
            "mediane": round(float(s.median()), 4) if len(s) else None,
            "pct_flag": round(100 * (f == 1).sum() / max(len(f), 1), 1) if len(f) else None,
        }

    out["income_proxy_comparatif"] = comparatif
    return out
