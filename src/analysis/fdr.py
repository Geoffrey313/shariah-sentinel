"""Phase 7 — Benjamini-Hochberg FDR correction.

Framework §7 downgraded this to a finishing touch: ``T_IUT`` already
produces p-values in the ``10^{-9}`` range under independence when it
rejects, so FDR is only meaningful for the other composites. Two
corrections are reported side-by-side:

- **Row-level BH** — classic Benjamini-Hochberg on the full column of
  row-level p-values. The output ``q_<composite>`` column is the usual
  adjusted p-value.
- **Firm-level BH** — collapse each firm's row-level p-values into a
  single firm-level p (the minimum across the firm's rows, per
  :attr:`FDRSettings.firm_aggregation`), then BH-correct the firm-level
  vector. This prevents a single persistently-flagged firm from counting
  as many independent discoveries.

Deliverables (under ``outputs/scores/<country>/phase7_fdr/``):

- ``fdr_row_level.parquet`` — per-row raw p + BH q per composite.
- ``fdr_firm_level.parquet`` — one row per firm with the aggregated p
  and its BH q, per composite.
- ``phase7_fdr.json`` — count of discoveries at each ``q_level``,
  before and after firm aggregation, per composite.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Phase7Outcome:
    """Return value of :func:`run_phase7`."""

    row_level: pd.DataFrame
    firm_level: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return the BH-adjusted q-values for a 1-D p-value array.

    NaN p-values are kept as NaN in the output so downstream users can
    filter them. The computation follows Benjamini & Hochberg 1995 in
    its textbook form: sort, compute ``q_i = p_i · m / i``, enforce the
    monotone non-increasing cumulative min from the right, unsort.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    vals = p[finite]
    m = vals.size
    order = np.argsort(vals, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)
    raw_q = vals[order] * m / ranks
    # Right-to-left cumulative minimum enforces the step-up rule.
    adjusted_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    unsort = np.empty(m, dtype=int)
    unsort[order] = np.arange(m, dtype=int)
    finite_q = np.clip(adjusted_sorted[unsort], 0.0, 1.0)
    q[finite] = finite_q
    return q


def _firm_aggregate(
    df: pd.DataFrame,
    p_col: str,
    firm_col: str,
    method: str,
) -> pd.Series:
    """Collapse per-row p-values into one p-value per firm."""
    if method == "min_pvalue":
        return df.groupby(firm_col, observed=True)[p_col].min()
    raise ValueError(f"unknown firm_aggregation {method!r}.")


# M4 — alternative firm-level aggregations, reported as a SENSITIVITY only; the
# headline stays ``min_pvalue``/BH. ``sidak`` corrects the min-p for the firm's
# within-firm multiplicity; ``fisher`` is the omnibus over all the firm's
# quarters; ``cauchy`` is the dependence-robust ACAT combination.
_SENSITIVITY_METHODS: tuple[str, ...] = ("sidak", "fisher", "cauchy")


def _firm_aggregate_variant(
    df: pd.DataFrame,
    p_col: str,
    firm_col: str,
    method: str,
) -> pd.Series:
    """One firm-level p-value per firm under an alternative combination."""
    from scipy import stats

    eps = 1e-12

    def _agg(p: pd.Series) -> float:
        vals = np.clip(p.to_numpy(dtype=float), eps, 1.0 - eps)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n == 0:
            return np.nan
        if method == "sidak":
            return float(1.0 - (1.0 - vals.min()) ** n)
        if method == "fisher":
            return float(stats.chi2.sf(-2.0 * np.log(vals).sum(), 2 * n))
        if method == "cauchy":
            return float(0.5 - np.arctan(np.mean(np.tan((0.5 - vals) * np.pi))) / np.pi)
        raise ValueError(f"unknown sensitivity method {method!r}.")

    return df.groupby(firm_col, observed=True)[p_col].apply(_agg)


def _discovery_counts(
    q_values: pd.Series,
    q_levels: tuple[float, ...],
) -> dict:
    """Number of discoveries at each BH threshold."""
    finite = q_values.dropna()
    return {
        f"q<={q:.3g}": int((finite <= q).sum()) for q in q_levels
    }


def run_phase7(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    composites: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> Phase7Outcome:
    """Apply BH FDR at row and firm level for every configured composite.

    Args:
        panel: Output of Phase 0 — only the schema columns are read, not
            the detector values.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        composites: Phase 4 output. When ``None`` the parquet is loaded
            from ``settings.output_layout.phase4_dir()``.
        write_outputs: If ``True``, persist parquet + JSON under
            ``settings.output_layout.phase7_dir()``.
    """
    settings = settings or AnalysisSettings()
    schema = settings.panel_schema
    fdr = settings.fdr

    if composites is None:
        path = settings.output_layout.phase4_dir() / "composites_panel.parquet"
        composites = pd.read_parquet(path)

    # ── Row-level BH ─────────────────────────────────────────────────────
    row_level = composites[[schema.firm_id, schema.quarter]].copy()
    for p_col in fdr.composites:
        if p_col not in composites.columns:
            log.warning("phase7: composite %r absent; skipping.", p_col)
            continue
        row_level[p_col] = composites[p_col].values
        row_level[f"q_{p_col}"] = _benjamini_hochberg(composites[p_col].to_numpy(dtype=float))

    # ── Firm-level BH ────────────────────────────────────────────────────
    firm_blocks: dict[str, pd.Series] = {}
    for p_col in fdr.composites:
        if p_col not in composites.columns:
            continue
        agg = _firm_aggregate(
            composites, p_col=p_col, firm_col=schema.firm_id,
            method=fdr.firm_aggregation,
        )
        firm_blocks[p_col] = agg

    firm_level = pd.DataFrame(firm_blocks).reset_index()
    for p_col in firm_blocks:
        firm_level[f"q_{p_col}"] = _benjamini_hochberg(
            firm_level[p_col].to_numpy(dtype=float)
        )

    # ── Summary ─────────────────────────────────────────────────────────
    # Row-level BH is often saturated by the bootstrap p-value floor:
    # with m rows and min p = 1/(B+1), the best possible adjusted q is
    # m / (B + 1). Report this ceiling so the reader doesn't mis-read
    # "0 discoveries" as a substantive result.
    bootstrap_B = settings.bootstrap.n_replicates
    min_p_floor = 1.0 / (bootstrap_B + 1.0)
    row_level_q_floor = min_p_floor * len(row_level) / max(len(row_level), 1)
    # Equivalent: m * min_p / 1 = m / (B+1).
    row_level_q_ceiling_best = len(row_level) * min_p_floor

    summary: dict = {
        "q_levels": list(fdr.q_levels),
        "firm_aggregation": fdr.firm_aggregation,
        "row_level_discoveries": {
            p_col: _discovery_counts(row_level[f"q_{p_col}"], fdr.q_levels)
            for p_col in fdr.composites if f"q_{p_col}" in row_level.columns
        },
        "firm_level_discoveries": {
            p_col: _discovery_counts(firm_level[f"q_{p_col}"], fdr.q_levels)
            for p_col in fdr.composites if f"q_{p_col}" in firm_level.columns
        },
        "row_count": int(len(row_level)),
        "firm_count": int(len(firm_level)),
        "bootstrap_pvalue_floor": float(min_p_floor),
        "row_level_best_achievable_q": float(row_level_q_ceiling_best),
        "row_level_note": (
            "Row-level BH saturates at q_min = m / (B+1) = "
            f"{row_level_q_ceiling_best:.3f}. With B={bootstrap_B}, m={len(row_level)}, "
            "no row can achieve q ≤ 0.05 purely from the bootstrap floor. "
            "Raise B or rely on firm-level aggregation."
        ),
    }

    # M4 — firm-aggregation sensitivity. The headline above stays min-p/BH; here
    # we re-aggregate each composite with conservative/omnibus combinations so a
    # reader can see how much the very-low-q counts depend on the aggregator.
    sensitivity: dict = {}
    for method in _SENSITIVITY_METHODS:
        per_comp: dict = {}
        for p_col in fdr.composites:
            if p_col not in composites.columns:
                continue
            fp = _firm_aggregate_variant(
                composites, p_col=p_col, firm_col=schema.firm_id, method=method
            )
            q = _benjamini_hochberg(fp.to_numpy(dtype=float))
            per_comp[p_col] = _discovery_counts(pd.Series(q), fdr.q_levels)
        sensitivity[method] = per_comp
    summary["firm_level_discoveries_sensitivity"] = sensitivity

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase7_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["row_level"] = out_dir / "fdr_row_level.parquet"
        paths["firm_level"] = out_dir / "fdr_firm_level.parquet"
        paths["json"] = out_dir / "phase7_fdr.json"
        row_level.to_parquet(paths["row_level"], index=False)
        firm_level.to_parquet(paths["firm_level"], index=False)
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("phase7: wrote %d deliverables to %s", len(paths), out_dir)

    return Phase7Outcome(
        row_level=row_level,
        firm_level=firm_level,
        summary=summary,
        paths=paths,
    )
