"""Cross-statement coherence detector.

This module implements the Working Paper ``j=5`` detector with the relation-
standardization-aggregation structure preserved.

Implemented relations:
- core: implied borrowing cost and its quarter-over-quarter change
- core: operating cash flow to revenue and TATA-style accruals
- proxy: canonical Sharia ratios already computed upstream

Still missing from the current panel / implementation:
- purification-specific relations
- segment-level coherence
- investment-security split relations

Each relation is standardized on split ``C`` and, where possible, conditioned
on sector. Small sector references fall back to the global calibration sample
or a shrinkage blend, following the Phase 3 / D5 configuration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.engine.pit import pit_chi2
from src.common.ratio_inputs import get_canonical_sharia_ratios

if TYPE_CHECKING:
    from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)

SECTOR_CANDIDATES: tuple[str, ...] = ("sector", "gsector", "ggroup", "gind")
MIN_GLOBAL_REFERENCE: int = 30
MIN_SHRINK_REFERENCE: int = 8
EPS: float = 1e-9


@dataclass(frozen=True)
class CoherenceRelationSpec:
    """One named relation used by the reduced D5 detector."""

    name: str
    wp_relation: str
    description: str
    kind: str
    required_columns: tuple[str, ...]
    builder: Callable[[pd.DataFrame], pd.Series]


def _select_sector_column(df: pd.DataFrame) -> str | None:
    """Return the first usable sector column, if available."""
    for column in SECTOR_CANDIDATES:
        if column in df.columns and df[column].notna().any():
            return column
    return None


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division that returns ``NaN`` on invalid denominators."""
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = num.notna() & den.notna() & (den.abs() > EPS)
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series if the column exists, otherwise ``NaN``."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _total_debt(df: pd.DataFrame) -> pd.Series:
    """Return total debt, preferring debt components over broad liabilities."""
    dlttq = _safe_numeric(df, "dlttq")
    dlcq = _safe_numeric(df, "dlcq")
    total_debt = dlttq + dlcq
    if "ltq" in df.columns:
        ltq = _safe_numeric(df, "ltq")
        total_debt = total_debt.where(total_debt.notna(), ltq)
    return total_debt


def _sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a panel by firm and quarter when the required keys are present."""
    if "gvkey" not in df.columns or "datacqtr" not in df.columns:
        return df.copy()
    return df.sort_values(["gvkey", "datacqtr"]).copy()


def _firm_delta_ratio(
    df: pd.DataFrame,
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Compute a firm-level ratio of quarter-over-quarter deltas."""
    if "gvkey" not in df.columns or "datacqtr" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    ordered = _sort_panel(df)
    num = numerator.loc[ordered.index]
    den = denominator.loc[ordered.index]
    delta_num = num.groupby(ordered["gvkey"], sort=False).diff()
    delta_den = den.groupby(ordered["gvkey"], sort=False).diff()
    ratio = _safe_divide(delta_num, delta_den)
    return ratio.reindex(df.index)


def _relation_interest_to_debt(df: pd.DataFrame) -> pd.Series:
    """Core relation: implied borrowing cost."""
    xintq = _safe_numeric(df, "xintq")
    return _safe_divide(xintq, _total_debt(df))


def _relation_delta_interest_to_debt(df: pd.DataFrame) -> pd.Series:
    """Core relation: quarter-over-quarter coherence of interest vs debt."""
    xintq = _safe_numeric(df, "xintq")
    return _firm_delta_ratio(df, xintq, _total_debt(df))


def _relation_income_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy relation: non-permissible income ratio."""
    ratios = get_canonical_sharia_ratios(df)
    return ratios["ratio_income"] if "ratio_income" in ratios.columns else pd.Series(np.nan, index=df.index, dtype=float)


def _relation_debt_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy relation: adjusted debt ratio."""
    ratios = get_canonical_sharia_ratios(df)
    return ratios["ratio_debt_adj"] if "ratio_debt_adj" in ratios.columns else pd.Series(np.nan, index=df.index, dtype=float)


def _relation_cash_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy relation: adjusted cash ratio."""
    ratios = get_canonical_sharia_ratios(df)
    return ratios["ratio_cash_adj"] if "ratio_cash_adj" in ratios.columns else pd.Series(np.nan, index=df.index, dtype=float)


def _relation_earnings_gap(df: pd.DataFrame) -> pd.Series:
    """Proxy relation: gap between net income and operating income."""
    niq = _safe_numeric(df, "niq")
    oibdpq = _safe_numeric(df, "oibdpq")
    atq = _safe_numeric(df, "atq")
    gap = _safe_divide(niq - oibdpq, atq)
    return gap


def _relation_cfo_to_revenue(df: pd.DataFrame) -> pd.Series:
    """R2: quarterly operating cash flow divided by revenue."""
    oancfq = _safe_numeric(df, "oancfq")
    revtq = _safe_numeric(df, "revtq")
    return _safe_divide(oancfq, revtq)


def _relation_tata_accrual(df: pd.DataFrame) -> pd.Series:
    """R5 exact: total accruals to total assets = (net income − CFO) / assets."""
    ibq = _safe_numeric(df, "ibq")
    oancfq = _safe_numeric(df, "oancfq")
    atq = _safe_numeric(df, "atq")
    return _safe_divide(ibq - oancfq, atq)


def _relation_working_capital_accrual_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy relation for the accrual residual when CFO is unavailable."""
    if "gvkey" not in df.columns or "datacqtr" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    ordered = _sort_panel(df)
    actq = _safe_numeric(ordered, "actq")
    cheq = _safe_numeric(ordered, "cheq")
    lctq = _safe_numeric(ordered, "lctq")
    dlcq = _safe_numeric(ordered, "dlcq")
    atq = _safe_numeric(ordered, "atq")

    non_cash_wc = (actq - cheq) - (lctq - dlcq)
    delta_non_cash_wc = non_cash_wc.groupby(ordered["gvkey"], sort=False).diff()
    return _safe_divide(delta_non_cash_wc, atq).reindex(df.index)


RELATION_SPECS: tuple[CoherenceRelationSpec, ...] = (
    CoherenceRelationSpec(
        name="cfo_to_revenue",
        wp_relation="R2",
        description="Operating cash flow (quarterly) divided by revenue.",
        kind="core",
        required_columns=("oancfq", "revtq"),
        builder=_relation_cfo_to_revenue,
    ),
    CoherenceRelationSpec(
        name="tata_accrual",
        wp_relation="R5",
        description="Total accruals to total assets: (net income − operating CFO) / total assets.",
        kind="core",
        required_columns=("ibq", "oancfq", "atq"),
        builder=_relation_tata_accrual,
    ),
    CoherenceRelationSpec(
        name="interest_to_debt",
        wp_relation="R1",
        description="Implied borrowing cost: interest expense divided by total debt.",
        kind="core",
        required_columns=("xintq", "dlttq", "dlcq"),
        builder=_relation_interest_to_debt,
    ),
    CoherenceRelationSpec(
        name="delta_interest_to_debt",
        wp_relation="R4",
        description="Quarter-over-quarter change in interest expense divided by change in total debt.",
        kind="core",
        required_columns=("gvkey", "datacqtr", "xintq", "dlttq", "dlcq"),
        builder=_relation_delta_interest_to_debt,
    ),
    CoherenceRelationSpec(
        name="working_capital_accrual_proxy",
        wp_relation="R5_proxy",
        description="Non-cash working-capital accrual proxy scaled by total assets.",
        kind="core_proxy",
        required_columns=("gvkey", "datacqtr", "actq", "cheq", "lctq", "dlcq", "atq"),
        builder=_relation_working_capital_accrual_proxy,
    ),
    CoherenceRelationSpec(
        name="debt_ratio_proxy",
        wp_relation="coverage_proxy",
        description="Canonical adjusted debt ratio used as a Sharia coherence proxy.",
        kind="proxy",
        required_columns=("dlttq", "dlcq", "atq"),
        builder=_relation_debt_proxy,
    ),
    CoherenceRelationSpec(
        name="cash_ratio_proxy",
        wp_relation="coverage_proxy",
        description="Canonical adjusted cash ratio used as a Sharia coherence proxy.",
        kind="proxy",
        required_columns=("cheq", "atq"),
        builder=_relation_cash_proxy,
    ),
    CoherenceRelationSpec(
        name="income_ratio_proxy",
        wp_relation="coverage_proxy",
        description="Canonical non-permissible income ratio used as a Sharia coherence proxy.",
        kind="proxy",
        required_columns=("iditq", "revtq"),
        builder=_relation_income_proxy,
    ),
    CoherenceRelationSpec(
        name="earnings_gap_proxy",
        wp_relation="cross_statement_proxy",
        description="Simplified earnings-gap proxy based on net income minus operating income.",
        kind="proxy",
        required_columns=("niq", "oibdpq", "atq"),
        builder=_relation_earnings_gap,
    ),
)

MISSING_WP_RELATIONS: tuple[dict[str, object], ...] = (
    {
        "wp_relation": "R3",
        "name": "non_operating_return",
        "description": "Non-operating income divided by non-core asset base.",
        "missing_columns": ("non_operating_assets", "operating_assets"),
        "reason": "The active panel does not expose a clean non-core asset denominator.",
    },
)


def describe_cross_statement_relation_status(df: pd.DataFrame) -> pd.DataFrame:
    """Describe active and still-missing D5 relations on the current panel."""
    rows: list[dict[str, object]] = []
    available_columns = set(df.columns)

    for spec in RELATION_SPECS:
        missing_columns = [column for column in spec.required_columns if column not in available_columns]
        rows.append(
            {
                "wp_relation": spec.wp_relation,
                "relation_name": spec.name,
                "status": spec.kind,
                "implemented": len(missing_columns) == 0,
                "description": spec.description,
                "missing_columns": ", ".join(missing_columns),
                "reason": "",
            }
        )

    for item in MISSING_WP_RELATIONS:
        missing_columns = [column for column in item["missing_columns"] if column not in available_columns]
        rows.append(
            {
                "wp_relation": item["wp_relation"],
                "relation_name": item["name"],
                "status": "missing_exact_wp_relation",
                "implemented": False,
                "description": item["description"],
                "missing_columns": ", ".join(missing_columns) if missing_columns else ", ".join(item["missing_columns"]),
                "reason": item["reason"],
            }
        )

    return pd.DataFrame(rows)


def compute_cross_statement_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the raw named relations used by the reduced D5 detector.

    New Working Paper relations should be added through ``RELATION_SPECS`` so
    the detector remains modular and the documentation of what is active versus
    still missing stays explicit.
    """
    relation_frame = pd.DataFrame(index=df.index)
    for spec in RELATION_SPECS:
        relation_frame[spec.name] = spec.builder(df)

    # Avoid double-counting accrual information: when the exact CFO-based
    # accrual relation is available, suppress the working-capital proxy on the
    # same rows. The proxy remains valuable only as a fallback when ``oancfq``
    # is unavailable.
    if {
        "tata_accrual",
        "working_capital_accrual_proxy",
    }.issubset(relation_frame.columns):
        exact_available = relation_frame["tata_accrual"].notna()
        relation_frame.loc[exact_available, "working_capital_accrual_proxy"] = np.nan

    return relation_frame


def _estimate_moments(reference: pd.Series, min_size: int) -> tuple[float, float] | None:
    """Estimate mean and standard deviation on one reference sample."""
    ref = reference.dropna()
    if len(ref) < min_size:
        return None
    mu = float(ref.mean())
    sigma = float(ref.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= EPS:
        return None
    return mu, sigma


def _resolve_sector_moments(
    sector_ref: pd.Series,
    global_moments: tuple[float, float] | None,
    min_sector_reference: int,
) -> tuple[float, float] | None:
    """Resolve the best calibration moments for one sector bucket."""
    sector_strong = _estimate_moments(sector_ref, min_sector_reference)
    if sector_strong is not None:
        return sector_strong

    if global_moments is None:
        return None

    sector_shrink = _estimate_moments(sector_ref, MIN_SHRINK_REFERENCE)
    if sector_shrink is None:
        return global_moments

    global_mu, global_sigma = global_moments
    sector_mu, sector_sigma = sector_shrink
    shrink = len(sector_ref.dropna()) / float(min_sector_reference)
    mu = shrink * sector_mu + (1.0 - shrink) * global_mu
    sigma_sq = shrink * sector_sigma**2 + (1.0 - shrink) * global_sigma**2
    sigma = float(np.sqrt(max(sigma_sq, EPS)))
    return (mu, sigma) if np.isfinite(sigma) and sigma > EPS else global_moments


def _calibrate_relation(
    raw: pd.Series,
    df: pd.DataFrame,
    mask_cal: pd.Series,
    sector_col: str | None,
    min_sector_reference: int,
) -> pd.Series:
    """Standardize one relation on the calibration sample."""
    z = pd.Series(np.nan, index=df.index, dtype=float)
    global_moments = _estimate_moments(raw.loc[mask_cal], MIN_GLOBAL_REFERENCE)

    if sector_col is None:
        if global_moments is None:
            return z
        global_mu, global_sigma = global_moments
        valid = raw.notna()
        z.loc[valid] = (raw.loc[valid] - global_mu) / global_sigma
        return z

    sector_series = df[sector_col]
    for sector_value, sector_idx in df.groupby(sector_col, dropna=True).groups.items():
        sector_mask = mask_cal & (sector_series == sector_value)
        sector_ref = raw.loc[sector_mask].dropna()
        sector_moments = _resolve_sector_moments(
            sector_ref,
            global_moments,
            min_sector_reference=min_sector_reference,
        )
        if sector_moments is None:
            continue
        mu, sigma = sector_moments

        idx = pd.Index(sector_idx)
        valid = raw.loc[idx].notna()
        valid_idx = idx[valid.to_numpy()]
        z.loc[valid_idx] = (raw.loc[valid_idx] - mu) / sigma

    # Fill any remaining rows with the global calibration.
    if global_moments is not None:
        global_mu, global_sigma = global_moments
        remaining = z.isna() & raw.notna()
        if remaining.any():
            z.loc[remaining] = (raw.loc[remaining] - global_mu) / global_sigma
    return z


def detect_cross_statement_coherence(
    df: pd.DataFrame,
    settings: "AnalysisSettings | None" = None,
) -> pd.Series:
    """Compute the reduced ``z5`` cross-statement coherence detector.

    Args:
        df: Input dataframe containing the financial statement columns used by
            the named coherence relations. ``gvkey`` and ``datacqtr`` are used
            only for the delta-based relation; the detector still works without
            them, but that relation will then be omitted.

    Returns:
        A series aligned on ``df.index``. Rows with too little calibration
        support, or without any usable relation, remain ``NaN``.

    Notes:
        This is a reduced-WP implementation. It preserves the original
        relation-standardization-aggregation structure while using only the
        cross-statement relations that are currently observable on the panel.
    """
    if settings is None:
        from src.common.config import AnalysisSettings
        settings = AnalysisSettings()

    min_sector_reference = (
        settings.detector_preconditions.cross_statement.min_sector_reference
    )

    relation_frame = compute_cross_statement_relations(df)
    if relation_frame.empty:
        log.warning("detect_cross_statement_coherence: no relations available; returning NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    status = describe_cross_statement_relation_status(df)
    missing_exact = status.loc[
        status["status"].eq("missing_exact_wp_relation"), "wp_relation"
    ].tolist()
    if missing_exact:
        log.info(
            "detect_cross_statement_coherence: exact WP relations still missing on the active panel: %s.",
            missing_exact,
        )

    mask_cal = df["_split"] == "C" if "_split" in df.columns else pd.Series(True, index=df.index)
    sector_col = _select_sector_column(df)
    if sector_col is None:
        log.info(
            "detect_cross_statement_coherence: no sector column available; using the global C reference."
        )
    else:
        log.info(
            "detect_cross_statement_coherence: conditioning on %s when sector calibration is sufficiently populated.",
            sector_col,
        )

    z_rel = pd.DataFrame(index=df.index)
    used_relations: list[str] = []
    for spec in RELATION_SPECS:
        raw = relation_frame[spec.name]
        z_raw = _calibrate_relation(
            raw,
            df,
            mask_cal,
            sector_col,
            min_sector_reference=min_sector_reference,
        )
        if z_raw.notna().any():
            z_rel[spec.name] = z_raw
            used_relations.append(spec.name)

    if not used_relations:
        log.warning("detect_cross_statement_coherence: calibration failed for all relations.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    z_values = z_rel.to_numpy(dtype=float)
    valid_mask = np.isfinite(z_values)
    q_eff = valid_mask.sum(axis=1)
    squared = np.where(valid_mask, z_values, 0.0) ** 2
    t5 = np.where(q_eff > 0, squared.sum(axis=1), np.nan)

    z5 = np.full(len(df), np.nan, dtype=float)
    for q in sorted(set(q_eff[q_eff > 0])):
        mask = q_eff == q
        if not mask.any():
            continue
        z5[mask] = pit_chi2(t5[mask], degrees_of_freedom=int(q))

    log.debug(
        "detect_cross_statement_coherence: used relations=%s, sector_col=%s.",
        used_relations,
        sector_col or "None",
    )
    return pd.Series(z5, index=df.index, dtype=float)


def detect_joint_zscore(df: pd.DataFrame) -> pd.Series:
    """Backward-compatible alias for ``detect_cross_statement_coherence``."""
    return detect_cross_statement_coherence(df)
