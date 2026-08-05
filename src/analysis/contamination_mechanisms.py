"""Contamination mechanisms M1–M4.

Each mechanism takes a clean DataFrame, selects **firms** to contaminate,
modifies their raw financial variables across **all quarters**, and
returns the contaminated DataFrame with ``y = 1`` labels. Clean rows
are left unchanged (``y = 0``).

All modifications are on the RAW Compustat fields, not on z-scores
or composites. The full detector pipeline must be re-run on the
contaminated data to produce z-scores and composites.

Firm-level contamination preserves temporal ordering, which is required
for detectors like z6 (temporal consistency) and z57 (peer-relative)
to detect the manipulation signal.
"""
from __future__ import annotations

import logging
from math import floor

import numpy as np
import pandas as pd

from src.analysis.contamination_config import ContaminationSettings

log = logging.getLogger(__name__)


def _select_contamination_firms(
    df: pd.DataFrame,
    rho: float,
    rng: np.random.Generator,
    firm_col: str = "gvkey",
) -> set:
    """Return a set of firm IDs to contaminate.

    Selects a fraction ``rho`` of firms (not rows), so all quarters
    of a selected firm are contaminated together.
    """
    if firm_col not in df.columns:
        # Fallback: treat each row as a "firm"
        n_contam = max(1, floor(len(df) * rho))
        return set(rng.choice(df.index, size=n_contam, replace=False))

    firms = df[firm_col].unique()
    n_firms = max(1, floor(len(firms) * rho))
    selected = rng.choice(firms, size=n_firms, replace=False)
    return set(selected)


# ─────────────────────────────────────────────────────────────────────
# M1 — Threshold optimisation
# ─────────────────────────────────────────────────────────────────────
def apply_m1(
    df: pd.DataFrame,
    rho: float,
    settings: ContaminationSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pin a Sharia ratio just below the SAC cap for entire firms.

    For each contaminated firm, every quarter's closest-to-threshold
    ratio is set to land in [tau - epsilon, tau), where
    epsilon = band_fraction * tau.
    """
    out = df.copy()
    m1 = settings.m1
    firm_col = "gvkey"

    contam_firms = _select_contamination_firms(out, rho, rng, firm_col)

    if firm_col in out.columns:
        contam_mask = out[firm_col].isin(contam_firms)
    else:
        contam_mask = out.index.isin(contam_firms)

    contam_idx = out.index[contam_mask]

    for i in contam_idx:
        best_gap = float("inf")
        best_ratio = None
        for ratio_name, spec in settings.ratio_definitions.items():
            num_col, den_col = spec["num"], spec["den"]
            tau = float(spec["threshold"])
            if num_col not in out.columns or den_col not in out.columns:
                continue
            num = out.at[i, num_col]
            den = out.at[i, den_col]
            if not (np.isfinite(num) and np.isfinite(den) and den > 0):
                continue
            current_ratio = num / den
            gap = abs(tau - current_ratio)
            if gap < best_gap:
                best_gap = gap
                best_ratio = (num_col, den_col, tau)

        if best_ratio is None:
            continue

        num_col, den_col, tau = best_ratio
        den = out.at[i, den_col]
        epsilon = m1.band_fraction * tau
        target_ratio = tau - rng.uniform(0, epsilon)
        out.at[i, num_col] = target_ratio * den
        out.at[i, "y"] = 1

    n_contam = (out["y"] == 1).sum()
    n_firms = len(contam_firms)
    log.info("M1 threshold: contaminated %d rows across %d firms (rho=%.2f)",
             n_contam, n_firms, rho)
    return out


# ─────────────────────────────────────────────────────────────────────
# M2 — Temporal smoothing
# ─────────────────────────────────────────────────────────────────────
def apply_m2(
    df: pd.DataFrame,
    rho: float,
    settings: ContaminationSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Replace quarter values with a moving average within each firm.

    Selects firms for contamination so the smoothing operates on a
    continuous time series.
    """
    out = df.copy()
    m2 = settings.m2
    firm_col = "gvkey"

    if firm_col not in out.columns or "datacqtr" not in out.columns:
        log.warning("M2: missing gvkey/datacqtr; skipping.")
        return out

    contam_firms = _select_contamination_firms(out, rho, rng, firm_col)

    monetary = [c for c in settings.monetary_columns if c in out.columns]
    for firm in contam_firms:
        mask = out[firm_col] == firm
        firm_idx = out.index[mask]
        if len(firm_idx) < m2.window:
            continue
        firm_data = out.loc[firm_idx].sort_values("datacqtr")
        for col in monetary:
            vals = firm_data[col].to_numpy(dtype=float)
            if np.isnan(vals).all():
                continue
            smoothed = pd.Series(vals).rolling(
                window=m2.window, min_periods=1
            ).mean().to_numpy()
            out.loc[firm_data.index, col] = smoothed
        out.loc[firm_data.index, "y"] = 1

    n_contam = (out["y"] == 1).sum()
    log.info("M2 smoothing: contaminated %d rows across %d firms (rho=%.2f)",
             n_contam, len(contam_firms), rho)
    return out


# ─────────────────────────────────────────────────────────────────────
# M3 — Benford distortion (round-number preference)
# ─────────────────────────────────────────────────────────────────────
def apply_m3(
    df: pd.DataFrame,
    rho: float,
    settings: ContaminationSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Snap monetary figures to round multiples for entire firms."""
    out = df.copy()
    m3 = settings.m3
    firm_col = "gvkey"

    contam_firms = _select_contamination_firms(out, rho, rng, firm_col)

    if firm_col in out.columns:
        contam_mask = out[firm_col].isin(contam_firms)
    else:
        contam_mask = out.index.isin(contam_firms)

    contam_idx = out.index[contam_mask]
    monetary = [c for c in settings.monetary_columns if c in out.columns]
    magnitude = 10 ** m3.rounding_magnitude
    n_fields = max(1, floor(len(monetary) * m3.fraction_of_fields))

    for i in contam_idx:
        fields = rng.choice(monetary, size=n_fields, replace=False)
        for col in fields:
            val = out.at[i, col]
            if not np.isfinite(val) or val == 0:
                continue
            rounded = np.round(val / magnitude) * magnitude
            if rounded == 0:
                rounded = magnitude * np.sign(val)
            out.at[i, col] = rounded
        out.at[i, "y"] = 1

    n_contam = (out["y"] == 1).sum()
    log.info("M3 benford: contaminated %d rows across %d firms (rho=%.2f)",
             n_contam, len(contam_firms), rho)
    return out


# ─────────────────────────────────────────────────────────────────────
# M4 — Cross-statement inconsistency
# ─────────────────────────────────────────────────────────────────────
def apply_m4(
    df: pd.DataFrame,
    rho: float,
    settings: ContaminationSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reduce debt without reducing interest expense for entire firms.

    Creates a deterministic inconsistency in the implied cost of
    debt (R1 = xintq / dlttq), which z5 / z57 are designed to catch.
    """
    out = df.copy()
    m4 = settings.m4
    firm_col = "gvkey"

    contam_firms = _select_contamination_firms(out, rho, rng, firm_col)

    if firm_col in out.columns:
        contam_mask = out[firm_col].isin(contam_firms)
    else:
        contam_mask = out.index.isin(contam_firms)

    contam_idx = out.index[contam_mask]

    debt_col = "dlttq"
    interest_col = "xintq"
    if debt_col not in out.columns or interest_col not in out.columns:
        log.warning("M4: missing %s or %s; skipping.", debt_col, interest_col)
        return out

    for i in contam_idx:
        debt = out.at[i, debt_col]
        if not np.isfinite(debt) or debt <= 0:
            continue
        out.at[i, debt_col] = debt * (1 - m4.debt_reduction)
        # Interest expense left unchanged — creates the inconsistency
        out.at[i, "y"] = 1

    n_contam = (out["y"] == 1).sum()
    log.info("M4 inconsistency: contaminated %d rows across %d firms (rho=%.2f)",
             n_contam, len(contam_firms), rho)
    return out


# ─────────────────────────────────────────────────────────────────────
# M5 — Cost-of-debt break (targets z8)
# ─────────────────────────────────────────────────────────────────────
def apply_m5(
    df: pd.DataFrame,
    rho: float,
    settings: ContaminationSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reduce debt in only 1-2 quarters per firm, creating a CoD spike.

    Unlike M4 which reduces debt in ALL quarters (level shift), M5
    perturbs only a few quarters per firm. This creates the exact
    break pattern that z8 (cost-of-debt break) is designed to catch.
    """
    out = df.copy()
    m5 = settings.m5
    firm_col = "gvkey"

    contam_firms = _select_contamination_firms(out, rho, rng, firm_col)

    debt_col = "dlttq"
    if debt_col not in out.columns:
        log.warning("M5: missing %s; skipping.", debt_col)
        return out

    for firm in contam_firms:
        if firm_col in out.columns:
            mask = out[firm_col] == firm
        else:
            mask = out.index == firm
        firm_idx = out.index[mask]
        if len(firm_idx) == 0:
            continue

        # Pick n_quarters random quarters from this firm
        n_pick = min(m5.n_quarters, len(firm_idx))
        target_idx = rng.choice(firm_idx, size=n_pick, replace=False)

        for i in target_idx:
            debt = out.at[i, debt_col]
            if not np.isfinite(debt) or debt <= 0:
                continue
            out.at[i, debt_col] = debt * (1 - m5.debt_reduction)
            # Interest expense left unchanged — creates the CoD spike
            out.at[i, "y"] = 1

    n_contam = (out["y"] == 1).sum()
    log.info("M5 cod_break: contaminated %d rows across %d firms (rho=%.2f)",
             n_contam, len(contam_firms), rho)
    return out


# ─────────────────────────────────────────────────────────────────────
# M6 — Seasonal manipulation (targets z9)
# ─────────────────────────────────────────────────────────────────────
def apply_m6(
    df: pd.DataFrame,
    rho: float,
    settings: ContaminationSettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reduce debt only at fiscal year-end, leave interim unchanged.

    Creates the seasonal pattern that z9 (review-date gap) is
    designed to detect: ratios are systematically lower at year-end
    (when the SAC reviews) than during interim quarters.
    """
    out = df.copy()
    m6 = settings.m6
    firm_col = "gvkey"

    if "fqtr" not in out.columns:
        log.warning("M6: missing fqtr; skipping.")
        return out

    contam_firms = _select_contamination_firms(out, rho, rng, firm_col)

    debt_col = "dlttq"
    if debt_col not in out.columns:
        log.warning("M6: missing %s; skipping.", debt_col)
        return out

    for firm in contam_firms:
        if firm_col in out.columns:
            mask = (out[firm_col] == firm) & (out["fqtr"] == 4)
        else:
            mask = (out.index == firm) & (out["fqtr"] == 4)
        ye_idx = out.index[mask]

        for i in ye_idx:
            debt = out.at[i, debt_col]
            if not np.isfinite(debt) or debt <= 0:
                continue
            out.at[i, debt_col] = debt * (1 - m6.debt_reduction)
            out.at[i, "y"] = 1

        # Also mark interim quarters as contaminated (for AUC labeling)
        # because the firm IS being manipulated — z9 should flag it
        if firm_col in out.columns:
            all_mask = out[firm_col] == firm
        else:
            all_mask = out.index == firm
        out.loc[all_mask, "y"] = 1

    n_contam = (out["y"] == 1).sum()
    n_ye_modified = sum(
        1 for firm in contam_firms
        for i in out.index[
            (out.get(firm_col, pd.Series()) == firm) & (out["fqtr"] == 4)
        ]
        if np.isfinite(out.at[i, debt_col])
    ) if firm_col in out.columns else 0
    log.info("M6 seasonal: contaminated %d rows (%d YE modified) across %d firms (rho=%.2f)",
             n_contam, n_ye_modified, len(contam_firms), rho)
    return out


# ─────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────
MECHANISM_DISPATCH = {
    "M1_threshold": apply_m1,
    "M2_smoothing": apply_m2,
    "M3_benford": apply_m3,
    "M4_inconsistency": apply_m4,
    "M5_cod_break": apply_m5,
    "M6_seasonal": apply_m6,
}
