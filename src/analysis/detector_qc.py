"""Phase 3 — detector QC + validity-condition enforcement.

Implements the Sprint 1 second deliverable of
``docs/scores/analysis_plan_v2.md``: check the row-level validity
preconditions listed in framework §9.2 per detector z₁..z₇ and emit a
``(row, detector)`` boolean mask so downstream phases mark invalid rows as
``NaN`` rather than running a detector on insufficient data.

The module works on the output of Phase 0 — specifically, it expects the
``_split`` column so that sector/peer reference sizes are measured on ``C``
only, not on the full (potentially non-honest) panel.

Preconditions implemented (framework §9.2):

- z₁ Benford         — pooled ``N_fig ≥ min_figures`` over the rolling window.
- z₂ Zipf            — same pooled window must reach ``min_points`` positive
  magnitudes.
- z₃ M-Score         — the row's sector has ``|C_s| ≥ min_reference_size``.
- z₄ Threshold       — quarters per firm available up to the row satisfy the
  §9.2 D4 power formula.
- z₅ Cross-stmt      — ``|C_s| ≥ m q`` on the row's sector.
- z₆ Temporal        — ``Q_hist`` past quarters exist for this firm.
- z₇ Peer            — peer group of the row satisfies ``N_peer ≥ m p``.

The mask is data-only: the detector modules themselves still apply their own
guards, but Phase 3 gives downstream reporting a single canonical answer to
"is detector j valid on row i?".
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)


# Column names in the emitted validity-mask parquet. Exposed as module
# constants so callers can filter without guessing the spelling.
COL_Z1: str = "z1_valid"
COL_Z2: str = "z2_valid"
COL_Z3: str = "z3_valid"
COL_Z4: str = "z4_valid"
COL_Z5: str = "z5_valid"
COL_Z6: str = "z6_valid"
COL_Z7: str = "z7_valid"
COL_Z8: str = "z8_valid"
COL_Z9: str = "z9_valid"


@dataclass(frozen=True)
class DetectorQCOutcome:
    """Return value of :func:`run_phase3`."""

    mask: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _count_positive_figures_per_row(
    panel: pd.DataFrame,
    value_columns: tuple[str, ...],
) -> pd.Series:
    """Number of finite, strictly positive monetary figures available per row.

    Benford (§9.2 D1) and Zipf (§9.2 D2) pool figures over a rolling firm
    window. The per-row count is the elementary building block; the rolling
    sum is computed separately per detector.
    """
    present = [c for c in value_columns if c in panel.columns]
    if not present:
        return pd.Series(0, index=panel.index, dtype="int64")
    counts = pd.Series(0, index=panel.index, dtype="int64")
    for col in present:
        vals = pd.to_numeric(panel[col], errors="coerce")
        counts = counts + ((vals.notna()) & (vals > 0)).astype("int64")
    return counts


def _rolling_group_sum(
    values: pd.Series,
    group_key: pd.Series,
    order_key: pd.Series,
    window: int,
) -> pd.Series:
    """Right-aligned rolling sum of ``values`` within each group.

    Pure pandas — no for-loop over firms — so large panels stay affordable.
    """
    aligned = pd.DataFrame({"v": values, "g": group_key, "o": order_key}).sort_values(
        ["g", "o"], kind="stable"
    )
    rolled = (
        aligned.groupby("g", sort=False)["v"]
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return rolled.reindex(aligned.index).reindex(values.index).astype("int64")


def _quarter_position_within_firm(
    firm_id: pd.Series,
    quarter: pd.Series,
) -> pd.Series:
    """0-indexed rank of a row among its firm's sorted quarters.

    Used by z₄ (history-length precondition) and z₆ (moving-window
    precondition).
    """
    order = pd.DataFrame({"firm": firm_id, "quarter": quarter}).sort_values(
        ["firm", "quarter"], kind="stable"
    )
    order["pos"] = order.groupby("firm", sort=False).cumcount()
    return order["pos"].reindex(firm_id.index).astype("int64")


# ─────────────────────────────────────────────────────────────────────────────
# Per-detector precondition builders
# ─────────────────────────────────────────────────────────────────────────────
def _mask_benford(panel: pd.DataFrame, settings: AnalysisSettings) -> pd.Series:
    """``z1_valid`` — pooled Benford figures ≥ the mode-specific floor.

    Under ``chi2_asymptotic`` the floor is ``min_figures`` (the classic
    ``E_d ≥ 5`` rule). Under ``monte_carlo`` / ``auto`` the floor drops to
    ``min_figures_mc`` since MC calibration is valid below the asymptotic
    ``χ²`` threshold.
    """
    s = settings.panel_schema
    p = settings.detector_preconditions.benford
    effective_floor = (
        p.min_figures if p.calibration_mode == "chi2_asymptotic" else p.min_figures_mc
    )
    per_row = _count_positive_figures_per_row(
        panel, settings.detector_preconditions.benford_value_columns
    )
    pooled = _rolling_group_sum(
        values=per_row,
        group_key=panel[s.firm_id],
        order_key=panel[s.quarter],
        window=p.pooling_window_quarters,
    )
    return (pooled >= effective_floor).rename(COL_Z1)


def _mask_zipf(panel: pd.DataFrame, settings: AnalysisSettings) -> pd.Series:
    """``z2_valid`` — pooled window reaches the OLS minimum point count."""
    s = settings.panel_schema
    p = settings.detector_preconditions.zipf
    per_row = _count_positive_figures_per_row(
        panel, settings.detector_preconditions.benford_value_columns
    )
    pooled = _rolling_group_sum(
        values=per_row,
        group_key=panel[s.firm_id],
        order_key=panel[s.quarter],
        window=p.pooling_window_quarters,
    )
    return (pooled >= p.min_points).rename(COL_Z2)


def _mask_mscore(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    sector_c_counts: pd.Series,
) -> pd.Series:
    """``z3_valid`` — sector of row has ``|C_s| ≥ min_reference_size``."""
    s = settings.panel_schema
    p = settings.detector_preconditions.mscore
    if not p.require_same_sector_reference:
        # Fall back to the global ``|C|`` check; every row is valid iff ``C``
        # satisfies the floor (the caller enforces that before Phase 3).
        return pd.Series(True, index=panel.index, name=COL_Z3)
    sector_counts = panel[s.sector].map(sector_c_counts).fillna(0).astype("int64")
    return (sector_counts >= p.min_reference_size).rename(COL_Z3)


def _min_history_for_threshold(
    preconds: AnalysisSettings,
) -> int:
    """Compute ``Q_min`` from the §9.2 D4 power formula.

    ``Q_min ≈ (z_{1-α} + z_{1-β})^2 / (3 (1 - δ_m)^2)``, subject to the
    configured absolute floor.
    """
    t = preconds.detector_preconditions.threshold
    z_alpha = norm.ppf(1.0 - t.significance_alpha)
    z_beta = norm.ppf(t.power_target)
    numer = (z_alpha + z_beta) ** 2
    denom = 3.0 * (1.0 - t.detectable_relative_shift) ** 2
    q_min = int(math.ceil(numer / denom))
    return max(q_min, t.absolute_floor)


def _mask_threshold(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    firm_position: pd.Series,
) -> pd.Series:
    """``z4_valid`` — at least ``Q_min`` prior quarters on the firm."""
    q_min = _min_history_for_threshold(settings)
    # ``firm_position`` is zero-indexed; row ``pos`` sees ``pos + 1`` quarters.
    quarters_seen = firm_position + 1
    return (quarters_seen >= q_min).rename(COL_Z4)


def _mask_cross_statement(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    sector_c_counts: pd.Series,
) -> pd.Series:
    """``z5_valid`` — sector ``|C_s|`` satisfies ``m q`` floor."""
    s = settings.panel_schema
    floor = settings.detector_preconditions.cross_statement.min_sector_reference
    sector_counts = panel[s.sector].map(sector_c_counts).fillna(0).astype("int64")
    return (sector_counts >= floor).rename(COL_Z5)


def _mask_temporal(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    firm_position: pd.Series,
) -> pd.Series:
    """``z6_valid`` — firm already has ``min_history_quarters`` past rows.

    The moving mean/std of Δm_i needs that many samples before the current
    row can be scored.
    """
    q_hist = settings.detector_preconditions.temporal.min_history_quarters
    quarters_seen = firm_position + 1
    return (quarters_seen > q_hist).rename(COL_Z6)


def _mask_peer(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    peer_c_counts: pd.Series,
) -> pd.Series:
    """``z7_valid`` — peer group ``N_peer`` satisfies ``m p`` floor."""
    r = settings.reference_sample
    floor = settings.detector_preconditions.peer.min_peer_size
    peer_counts = panel[r.peer_group_column].map(peer_c_counts).fillna(0).astype("int64")
    return (peer_counts >= floor).rename(COL_Z7)


def _mask_cost_of_debt(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    firm_position: pd.Series,
) -> pd.Series:
    """``z8_valid`` — firm has debt, interest data, and enough history.

    Requires observed long-term and short-term debt, positive total debt,
    xintq non-null, and at least MIN_HIST prior quarters of CoD data.
    """
    min_hist = 4  # matches d8_cost_of_debt.MIN_HIST
    dlttq = pd.to_numeric(panel.get("dlttq", pd.Series(np.nan, index=panel.index)), errors="coerce")
    dlcq = pd.to_numeric(panel.get("dlcq", pd.Series(np.nan, index=panel.index)), errors="coerce")
    has_debt = dlttq.notna() & dlcq.notna() & ((dlttq + dlcq) > 0)
    has_xintq = panel.get("xintq", pd.Series(dtype=float)).notna()
    has_history = (firm_position + 1) > min_hist
    return (has_debt & has_xintq & has_history).rename(COL_Z8)


def _mask_seasonal_gap(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    firm_position: pd.Series,
) -> pd.Series:
    """``z9_valid`` — firm has enough year-end and interim quarters.

    Requires at least 4 year-end (fqtr=4) quarters and 4 interim
    quarters with ratio data.
    """
    min_quarters = 8  # need 4 YE + 4 interim minimum
    has_ratios = panel.get("ratio_debt_adj", pd.Series(dtype=float)).notna()
    has_fqtr = panel.get("fqtr", pd.Series(dtype=float)).notna()
    has_history = (firm_position + 1) >= min_quarters
    return (has_ratios & has_fqtr & has_history).rename(COL_Z9)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def _coverage_table(mask: pd.DataFrame) -> pd.DataFrame:
    """Per-detector fraction of rows that satisfy the precondition."""
    rows = []
    for col in mask.columns:
        valid = int(mask[col].sum())
        total = int(len(mask))
        rows.append(
            {
                "detector": col,
                "n_valid": valid,
                "n_total": total,
                "share_valid": (valid / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _require_phase0_split(panel: pd.DataFrame) -> None:
    """Fail fast if Phase 0 hasn't been applied (``_split`` missing)."""
    if "_split" not in panel.columns:
        raise KeyError(
            "phase3: panel lacks the `_split` column — run Phase 0 first "
            "(or pass the panel returned by `run_phase0`)."
        )


def run_phase3(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    write_outputs: bool = True,
) -> DetectorQCOutcome:
    """Enforce per-detector preconditions and emit a validity mask.

    Args:
        panel: Output of :func:`run_phase0` — must contain the ``_split``
            column produced there and the schema columns listed in
            ``settings.panel_schema``.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        write_outputs: If ``True``, persist the mask parquet and JSON
            summary under ``settings.output_layout.phase3_dir()``.

    Returns:
        A :class:`DetectorQCOutcome` wrapping the ``(gvkey, datacqtr,
        z1_valid..z7_valid)`` parquet, a per-detector coverage summary and
        the resolved paths.

    Raises:
        KeyError: If the panel lacks ``_split`` (Phase 0 not run) or a
            required schema column.
    """
    settings = settings or AnalysisSettings()
    schema = settings.panel_schema
    _require_phase0_split(panel)

    in_c = panel["_split"] == SPLIT_LABEL_INCLUDED
    sector_c_counts = panel.loc[in_c, schema.sector].value_counts(dropna=False)
    peer_c_counts = panel.loc[in_c, settings.reference_sample.peer_group_column].value_counts(dropna=False)
    firm_position = _quarter_position_within_firm(panel[schema.firm_id], panel[schema.quarter])

    mask_cols = {
        COL_Z1: _mask_benford(panel, settings),
        COL_Z2: _mask_zipf(panel, settings),
        COL_Z3: _mask_mscore(panel, settings, sector_c_counts),
        COL_Z4: _mask_threshold(panel, settings, firm_position),
        COL_Z5: _mask_cross_statement(panel, settings, sector_c_counts),
        COL_Z6: _mask_temporal(panel, settings, firm_position),
        COL_Z7: _mask_peer(panel, settings, peer_c_counts),
        COL_Z8: _mask_cost_of_debt(panel, settings, firm_position),
        COL_Z9: _mask_seasonal_gap(panel, settings, firm_position),
    }

    mask = pd.DataFrame(mask_cols, index=panel.index)
    mask.insert(0, schema.quarter, panel[schema.quarter].values)
    mask.insert(0, schema.firm_id, panel[schema.firm_id].values)

    coverage = _coverage_table(mask.drop(columns=[schema.firm_id, schema.quarter]))

    summary: dict = {
        "panel_rows": int(len(panel)),
        "c_rows": int(in_c.sum()),
        "thresholds": {
            "benford_min_figures": settings.detector_preconditions.benford.min_figures,
            "benford_pooling_window": settings.detector_preconditions.benford.pooling_window_quarters,
            "zipf_min_points": settings.detector_preconditions.zipf.min_points,
            "zipf_pooling_window": settings.detector_preconditions.zipf.pooling_window_quarters,
            "mscore_min_reference_size": settings.detector_preconditions.mscore.min_reference_size,
            "threshold_q_min": _min_history_for_threshold(settings),
            "threshold_alpha": settings.detector_preconditions.threshold.significance_alpha,
            "threshold_power_target": settings.detector_preconditions.threshold.power_target,
            "threshold_delta_m": settings.detector_preconditions.threshold.detectable_relative_shift,
            "cross_statement_min_sector": settings.detector_preconditions.cross_statement.min_sector_reference,
            "temporal_min_history": settings.detector_preconditions.temporal.min_history_quarters,
            "peer_min_size": settings.detector_preconditions.peer.min_peer_size,
        },
        "coverage": coverage.to_dict(orient="records"),
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase3_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["mask_parquet"] = out_dir / "detector_validity_mask.parquet"
        paths["json"] = out_dir / "detector_qc.json"
        paths["coverage_csv"] = out_dir / "detector_coverage.csv"

        mask.to_parquet(paths["mask_parquet"], index=False)
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        coverage.to_csv(paths["coverage_csv"], index=False)
        log.info("phase3: wrote %d deliverables to %s", len(paths), out_dir)

    return DetectorQCOutcome(mask=mask, summary=summary, paths=paths)
