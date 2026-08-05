"""Sharia M-score detector with enriched Beneish-style components.

This detector implements a stronger version of the Working Paper's ``j=3``
detector while staying honest about what is still missing from the active
panel.

Two signal tracks are combined:

1. **Beneish partial** — observable Beneish-style components currently
   available on the panel (DSRI, AQI, SGAI, TATA), aggregated with the
   published Beneish coefficients.
2. **Shariah-proxy mean** — Sharia-ratio dynamics and mismatch proxies that
   remain useful even when some Beneish inputs are absent.

The final raw statistic is the row-wise mean of these two sub-scores, then an
empirical PIT on split ``C`` maps the statistic to the common anomaly-oriented
z-scale. The Phase 3 validity mask may still enforce sector-aware reference
requirements even though the current PIT implementation remains global on ``C``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.engine.pit import pit_empirical
from src.common.ratio_inputs import get_canonical_sharia_ratios

log = logging.getLogger(__name__)

REQUIRED_COLS: list[str] = ["gvkey", "datacqtr"]
MIN_REF_SIZE: int = 30
EPS_PREVIOUS_RATIO: float = 1e-8
EPS_LEVEL_RATIO: float = 1e-10

# Beneish (1999) coefficients for the observable subset. The intercept is
# omitted because empirical PIT calibration removes location/scale.
BENEISH_COEF: dict[str, float] = {
    "dsri_index": 0.920,
    "aqi_index": 0.404,
    "sgai_index": -0.172,
    "tata_level": 4.679,
}


def _ratio_index(r_t: float, r_prev: float) -> float:
    if not (np.isfinite(r_t) and np.isfinite(r_prev)):
        return np.nan
    if abs(r_prev) < EPS_PREVIOUS_RATIO:
        return np.nan
    return float(r_t / r_prev)


def _safe_level_mismatch(current: float, alternate: float) -> float:
    if not (np.isfinite(current) and np.isfinite(alternate)):
        return np.nan
    scale = max(abs(current), EPS_LEVEL_RATIO)
    return float(abs(current - alternate) / scale)


def _compute_mscore_components_firm(df_firm: pd.DataFrame) -> pd.DataFrame:
    n = len(df_firm)
    components = pd.DataFrame(
        {
            "income_index_proxy": np.full(n, np.nan, dtype=float),
            "debt_index_proxy": np.full(n, np.nan, dtype=float),
            "debt_shift_proxy": np.full(n, np.nan, dtype=float),
            "cash_income_mismatch_proxy": np.full(n, np.nan, dtype=float),
            "dsri_index": np.full(n, np.nan, dtype=float),
            "aqi_index": np.full(n, np.nan, dtype=float),
            "sgai_index": np.full(n, np.nan, dtype=float),
            "tata_level": np.full(n, np.nan, dtype=float),
        },
        index=df_firm.index,
    )

    debt = df_firm["ratio_debt_adj"].to_numpy(dtype=float) if "ratio_debt_adj" in df_firm.columns else None
    income = df_firm["ratio_income"].to_numpy(dtype=float) if "ratio_income" in df_firm.columns else None
    income_cashadj = (
        df_firm["ratio_income_cashadj"].to_numpy(dtype=float)
        if "ratio_income_cashadj" in df_firm.columns
        else None
    )

    rectq = df_firm["rectq"].to_numpy(dtype=float) if "rectq" in df_firm.columns else None
    revtq = df_firm["revtq"].to_numpy(dtype=float) if "revtq" in df_firm.columns else None
    ppentq = df_firm["ppentq"].to_numpy(dtype=float) if "ppentq" in df_firm.columns else None
    actq = df_firm["actq"].to_numpy(dtype=float) if "actq" in df_firm.columns else None
    atq = df_firm["atq"].to_numpy(dtype=float) if "atq" in df_firm.columns else None
    xsgaq = df_firm["xsgaq"].to_numpy(dtype=float) if "xsgaq" in df_firm.columns else None
    ibq = df_firm["ibq"].to_numpy(dtype=float) if "ibq" in df_firm.columns else None
    oancfq = df_firm["oancfq"].to_numpy(dtype=float) if "oancfq" in df_firm.columns else None

    for t in range(n):
        if ibq is not None and oancfq is not None and atq is not None:
            if (
                np.isfinite(ibq[t])
                and np.isfinite(oancfq[t])
                and np.isfinite(atq[t])
                and abs(atq[t]) > EPS_LEVEL_RATIO
            ):
                components.iat[t, 7] = (ibq[t] - oancfq[t]) / atq[t]

        if t == 0:
            continue

        if income is not None:
            components.iat[t, 0] = _ratio_index(income[t], income[t - 1])
        if debt is not None:
            components.iat[t, 1] = _ratio_index(debt[t], debt[t - 1])
            if np.isfinite(debt[t]) and np.isfinite(debt[t - 1]):
                components.iat[t, 2] = float(debt[t] - debt[t - 1])
        if income is not None and income_cashadj is not None:
            components.iat[t, 3] = _safe_level_mismatch(income[t], income_cashadj[t])

        if rectq is not None and revtq is not None:
            if (
                np.isfinite(rectq[t])
                and np.isfinite(revtq[t])
                and np.isfinite(rectq[t - 1])
                and np.isfinite(revtq[t - 1])
                and abs(revtq[t]) > EPS_PREVIOUS_RATIO
                and abs(revtq[t - 1]) > EPS_PREVIOUS_RATIO
            ):
                curr = rectq[t] / revtq[t]
                prev = rectq[t - 1] / revtq[t - 1]
                components.iat[t, 4] = _ratio_index(curr, prev)

        if ppentq is not None and actq is not None and atq is not None:
            vals = [ppentq[t], actq[t], atq[t], ppentq[t - 1], actq[t - 1], atq[t - 1]]
            if all(np.isfinite(v) for v in vals):
                if abs(atq[t]) > EPS_LEVEL_RATIO and abs(atq[t - 1]) > EPS_LEVEL_RATIO:
                    curr = 1.0 - (ppentq[t] + actq[t]) / atq[t]
                    prev = 1.0 - (ppentq[t - 1] + actq[t - 1]) / atq[t - 1]
                    components.iat[t, 5] = _ratio_index(curr, prev)

        if xsgaq is not None and revtq is not None:
            if (
                np.isfinite(xsgaq[t])
                and np.isfinite(revtq[t])
                and np.isfinite(xsgaq[t - 1])
                and np.isfinite(revtq[t - 1])
                and abs(revtq[t]) > EPS_PREVIOUS_RATIO
                and abs(revtq[t - 1]) > EPS_PREVIOUS_RATIO
            ):
                curr = xsgaq[t] / revtq[t]
                prev = xsgaq[t - 1] / revtq[t - 1]
                components.iat[t, 6] = _ratio_index(curr, prev)

    return components


def _aggregate_reduced_mscore(components_firm: pd.DataFrame) -> pd.Series:
    beneish_terms = pd.DataFrame(index=components_firm.index, dtype=float)
    for col, coef in BENEISH_COEF.items():
        if col in components_firm.columns:
            beneish_terms[col] = coef * components_firm[col]
    m_beneish = beneish_terms.mean(axis=1, skipna=True)

    sharia_terms = pd.DataFrame(index=components_firm.index, dtype=float)
    for col in ("income_index_proxy", "debt_index_proxy"):
        if col in components_firm.columns:
            sharia_terms[col] = components_firm[col]
    if "debt_shift_proxy" in components_firm.columns:
        sharia_terms["debt_shift_proxy"] = components_firm["debt_shift_proxy"].clip(lower=0.0)
    if "cash_income_mismatch_proxy" in components_firm.columns:
        sharia_terms["cash_income_mismatch_proxy"] = components_firm["cash_income_mismatch_proxy"]
    m_sharia = sharia_terms.mean(axis=1, skipna=True)

    combined = pd.concat([m_beneish.rename("beneish"), m_sharia.rename("sharia")], axis=1)
    return combined.mean(axis=1, skipna=True)


def _compute_mscore_raw_firm(df_firm: pd.DataFrame) -> pd.Series:
    """Raw (pre-calibration) reduced M-score for one firm's time series.

    This is the reduced M-score that ``detect_mscore`` aggregates per firm
    before the empirical-PIT calibration step. Exposed so incremental rescoring
    (the Family-3 counterfactual search) can recompute the raw z3 statistic on a
    single firm's rows without rerunning the whole detector.
    """
    return _aggregate_reduced_mscore(_compute_mscore_components_firm(df_firm))


def _compute_mscore_components_vectorized(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of looping ``_compute_mscore_components_firm`` per firm.

    ``df_sorted`` must already be sorted by ``(gvkey, datacqtr)``. Uses
    ``groupby(gvkey).shift(1)`` for the quarter-over-quarter components
    instead of a per-row Python loop inside a per-firm Python loop (see
    profiling notes). ``tata_level`` and ``cash_income_mismatch_proxy`` are
    both row-wise (no shift), but only ``tata_level`` is valid on a firm's
    first row -- the original loop computes ``cash_income_mismatch_proxy``
    *after* its ``if t == 0: continue`` guard, so that one is forced to NaN
    on the first row of every firm despite not needing ``t-1`` data.
    Verified bit-identical (component-by-component and after aggregation)
    against the original loop on the full real MYS panel.
    """
    firm_id = df_sorted["gvkey"]

    def _get(col: str) -> np.ndarray | None:
        return df_sorted[col].to_numpy(dtype=float) if col in df_sorted.columns else None

    def _shift1(arr: np.ndarray) -> np.ndarray:
        return pd.Series(arr, index=df_sorted.index).groupby(firm_id, sort=False).shift(1).to_numpy(dtype=float)

    def _ratio_index_vec(r_t: np.ndarray, r_prev: np.ndarray) -> np.ndarray:
        valid = np.isfinite(r_t) & np.isfinite(r_prev) & (np.abs(r_prev) >= EPS_PREVIOUS_RATIO)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(valid, r_t / r_prev, np.nan)

    is_first_row = (firm_id.groupby(firm_id, sort=False).cumcount() == 0).to_numpy()

    debt = _get("ratio_debt_adj")
    income = _get("ratio_income")
    income_cashadj = _get("ratio_income_cashadj")
    rectq = _get("rectq")
    revtq = _get("revtq")
    ppentq = _get("ppentq")
    actq = _get("actq")
    atq = _get("atq")
    xsgaq = _get("xsgaq")
    ibq = _get("ibq")
    oancfq = _get("oancfq")

    components = pd.DataFrame(index=df_sorted.index)
    components["income_index_proxy"] = np.nan
    components["debt_index_proxy"] = np.nan
    components["debt_shift_proxy"] = np.nan
    components["cash_income_mismatch_proxy"] = np.nan
    components["dsri_index"] = np.nan
    components["aqi_index"] = np.nan
    components["sgai_index"] = np.nan
    components["tata_level"] = np.nan

    if ibq is not None and oancfq is not None and atq is not None:
        valid_tata = np.isfinite(ibq) & np.isfinite(oancfq) & np.isfinite(atq) & (np.abs(atq) > EPS_LEVEL_RATIO)
        with np.errstate(invalid="ignore", divide="ignore"):
            components["tata_level"] = np.where(valid_tata, (ibq - oancfq) / atq, np.nan)

    if income is not None:
        components["income_index_proxy"] = _ratio_index_vec(income, _shift1(income))

    if debt is not None:
        debt_prev = _shift1(debt)
        components["debt_index_proxy"] = _ratio_index_vec(debt, debt_prev)
        valid_shift = np.isfinite(debt) & np.isfinite(debt_prev)
        components["debt_shift_proxy"] = np.where(valid_shift, debt - debt_prev, np.nan)

    if income is not None and income_cashadj is not None:
        valid_mismatch = np.isfinite(income) & np.isfinite(income_cashadj) & ~is_first_row
        scale = np.maximum(np.abs(income), EPS_LEVEL_RATIO)
        with np.errstate(invalid="ignore"):
            components["cash_income_mismatch_proxy"] = np.where(
                valid_mismatch, np.abs(income - income_cashadj) / scale, np.nan
            )

    if rectq is not None and revtq is not None:
        rectq_prev = _shift1(rectq)
        revtq_prev = _shift1(revtq)
        valid = (
            np.isfinite(rectq) & np.isfinite(revtq) & np.isfinite(rectq_prev) & np.isfinite(revtq_prev)
            & (np.abs(revtq) > EPS_PREVIOUS_RATIO) & (np.abs(revtq_prev) > EPS_PREVIOUS_RATIO)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            curr = np.where(valid, rectq / revtq, np.nan)
            prev = np.where(valid, rectq_prev / revtq_prev, np.nan)
        components["dsri_index"] = _ratio_index_vec(curr, prev)

    if ppentq is not None and actq is not None and atq is not None:
        ppentq_prev = _shift1(ppentq)
        actq_prev = _shift1(actq)
        atq_prev = _shift1(atq)
        valid = (
            np.isfinite(ppentq) & np.isfinite(actq) & np.isfinite(atq)
            & np.isfinite(ppentq_prev) & np.isfinite(actq_prev) & np.isfinite(atq_prev)
            & (np.abs(atq) > EPS_LEVEL_RATIO) & (np.abs(atq_prev) > EPS_LEVEL_RATIO)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            curr = np.where(valid, 1.0 - (ppentq + actq) / atq, np.nan)
            prev = np.where(valid, 1.0 - (ppentq_prev + actq_prev) / atq_prev, np.nan)
        components["aqi_index"] = _ratio_index_vec(curr, prev)

    if xsgaq is not None and revtq is not None:
        revtq_prev = _shift1(revtq)
        xsgaq_prev = _shift1(xsgaq)
        valid = (
            np.isfinite(xsgaq) & np.isfinite(revtq) & np.isfinite(xsgaq_prev) & np.isfinite(revtq_prev)
            & (np.abs(revtq) > EPS_PREVIOUS_RATIO) & (np.abs(revtq_prev) > EPS_PREVIOUS_RATIO)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            curr = np.where(valid, xsgaq / revtq, np.nan)
            prev = np.where(valid, xsgaq_prev / revtq_prev, np.nan)
        components["sgai_index"] = _ratio_index_vec(curr, prev)

    return components


def _compute_mscore_raw_vectorized(df_sorted: pd.DataFrame) -> pd.Series:
    """Vectorized equivalent of looping ``_compute_mscore_raw_firm`` per firm.

    ``df_sorted`` must already be sorted by ``(gvkey, datacqtr)`` with the
    canonical ratio columns attached (matching ``detect_mscore``'s own
    setup). ~290x faster on the full real panel, verified bit-identical
    (see profiling notes).
    """
    return _aggregate_reduced_mscore(_compute_mscore_components_vectorized(df_sorted))


def detect_mscore(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        log.warning("detect_mscore: missing required columns %s; returning NaN.", missing)
        return pd.Series(np.nan, index=df.index)

    ratios = get_canonical_sharia_ratios(df)
    if ratios.empty:
        log.warning("detect_mscore: no canonical Sharia ratios available; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    df_sorted = df.sort_values(["gvkey", "datacqtr"]).copy()
    for col in ratios.columns:
        df_sorted[col] = ratios.loc[df_sorted.index, col].values
    if "ratio_income_cashadj" in df.columns:
        df_sorted["ratio_income_cashadj"] = pd.to_numeric(
            df.loc[df_sorted.index, "ratio_income_cashadj"],
            errors="coerce",
        ).values

    ms_all = _compute_mscore_raw_vectorized(df_sorted)

    mask_cal = (df_sorted["_split"] == "C") if "_split" in df_sorted.columns else pd.Series(True, index=df_sorted.index)
    ref = ms_all[mask_cal].dropna().values

    if len(ref) < MIN_REF_SIZE:
        log.warning(
            "detect_mscore: calibration split C too small (%d < %d); returning NaN.",
            len(ref),
            MIN_REF_SIZE,
        )
        return pd.Series(np.nan, index=df.index)

    z = pd.Series(pit_empirical(ms_all.values, ref), index=df_sorted.index)
    return z.reindex(df.index)
