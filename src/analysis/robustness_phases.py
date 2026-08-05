"""Phase 6 — robustness and external validation.

Four diagnostics, each addressing a distinct failure mode of the battery,
computed on the outputs of Phase 4:

1. **Delisting proxy** — firms whose last observed quarter is well before
   the panel end are a proxy for "delisted". Compare their composites in
   the trailing N quarters against their own earlier history. If the
   detectors catch genuine issues, flagged rates should rise near
   delisting. Framework §6 plan, operationalised via the heuristic
   (no delisting flag exists on the panel).
2. **Sector false-positive rate** — RED rate per sector on the honest
   ``C`` sample. A single sector capturing a disproportionate share of
   the "SAC-compliant but flagged RED" cohort points to a business-model
   bias. The Sprint 4 interpretation doc flagged Financial Services and
   Islamic insurance as candidates — this table scales the check.
3. **Integrity sensitivity** — rank composites once on the full panel
   and once on ``bs_any_ffilled == 0``. Large rank shifts indicate the
   composite relies on forward-filled balance-sheet values.
4. **Composite stability** — per-firm lag-1 autocorrelation of each
   composite. Near-zero means noisy quarter-to-quarter; near-one means a
   firm is persistently flagged (sector artefact or chronic issue).

Deliverables (under ``outputs/scores/<country>/phase6_robustness/``):

- ``delisting_proxy.csv`` — per-firm mean composite in early vs late
  periods, plus the delta used for ranking.
- ``sector_false_positive.csv`` — RED share per sector on the ``C``
  sample with sector-level row counts.
- ``integrity_sensitivity.csv`` — Spearman rank correlation between the
  full-panel and strict-integrity rankings, per composite.
- ``composite_stability.csv`` — per-firm lag-1 autocorrelation per
  composite, aggregated into a mean and median.
- ``phase6_robustness.json`` — summary of all four diagnostics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED
from src.analysis.composite_scoring import (
    COL_BREADTH,
    COL_P_T_IUT, COL_P_Z_MAHALANOBIS, COL_P_Z_PLUS, COL_P_Z_PLUS_ORTH,
    COL_P_Z_PLUS_RENORM, COL_P_Z_PLUS_SOFTMAX,
    COL_T_IUT, COL_Z_MAHALANOBIS, COL_Z_PLUS, COL_Z_PLUS_ORTH,
    COL_Z_PLUS_RENORM, COL_Z_PLUS_SOFTMAX,
)

log = logging.getLogger(__name__)


COMPOSITE_VALUE_COLS: tuple[str, ...] = (
    COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_BREADTH, COL_Z_MAHALANOBIS, COL_T_IUT,
    COL_Z_PLUS_SOFTMAX, COL_Z_PLUS_ORTH,
)
COMPOSITE_P_COLS: tuple[str, ...] = (
    COL_P_Z_PLUS, COL_P_Z_PLUS_RENORM, COL_P_Z_MAHALANOBIS, COL_P_T_IUT,
    COL_P_Z_PLUS_SOFTMAX, COL_P_Z_PLUS_ORTH,
)
# The RED verdict is defined on the four primary composites only (the
# softmax/orthogonal variants are reporting extensions, not part of the
# headline verdict). The sector-level RED rate must use the same definition
# as the flag rate and the FDR so the three are mutually consistent.
VERDICT_P_COLS: tuple[str, ...] = (
    COL_P_Z_PLUS, COL_P_Z_PLUS_RENORM, COL_P_Z_MAHALANOBIS, COL_P_T_IUT,
)
# ``p_breadth`` is intentionally excluded: breadth only takes a small discrete
# set of values, so its bootstrap p-values are not very informative for the
# continuous-style robustness checks used in Phase 6.


@dataclass(frozen=True)
class Phase6Outcome:
    """Return value of :func:`run_phase6`."""

    delisting: pd.DataFrame
    sector_fp: pd.DataFrame
    integrity: pd.DataFrame
    stability: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Delisting proxy
# ─────────────────────────────────────────────────────────────────────────────
def _firm_last_quarter(panel: pd.DataFrame, schema) -> pd.Series:
    """Last datacqtr per firm (lexicographic on ``YYYY-Qn``)."""
    return panel.groupby(schema.firm_id, observed=True)[schema.quarter].max()


def _delisting_cohort(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
) -> set:
    """Firms whose last observed quarter is in the bottom ``cutoff_share``.

    Proxy for delisting — no authoritative flag exists. A firm that stops
    reporting well before the panel ends is more likely to have exited.
    """
    schema = settings.panel_schema
    last_q = _firm_last_quarter(panel, schema)
    ranks = last_q.rank(method="min")  # smaller = earlier last quarter
    cutoff = ranks.quantile(settings.robustness.delisting_cutoff_share)
    return set(last_q.index[ranks <= cutoff])


def _delisting_trajectory(
    panel: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Early-vs-late composite means per firm in the delisting cohort.

    For each firm in the proxy cohort, split its observed quarters into
    the trailing ``delisting_trailing_quarters`` (late) and everything
    before (early). Compare composite means; the delta is the quantity of
    interest — manipulation should raise the late composite.
    """
    schema = settings.panel_schema
    cohort = _delisting_cohort(panel, settings)
    if not cohort:
        return pd.DataFrame()

    merged = composites.merge(
        panel[[schema.firm_id, schema.quarter]], on=[schema.firm_id, schema.quarter], how="left",
    )
    merged = merged[merged[schema.firm_id].isin(cohort)].copy()
    merged = merged.sort_values([schema.firm_id, schema.quarter])

    trailing = settings.robustness.delisting_trailing_quarters
    rows: list[dict] = []
    for firm_id, grp in merged.groupby(schema.firm_id, observed=True):
        if len(grp) <= trailing:
            # Not enough quarters to meaningfully split.
            continue
        late = grp.tail(trailing)
        early = grp.iloc[:-trailing]
        row = {
            schema.firm_id: firm_id,
            "n_early": int(len(early)),
            "n_late": int(len(late)),
            "last_quarter": grp[schema.quarter].iloc[-1],
        }
        for col in COMPOSITE_VALUE_COLS:
            early_mean = float(np.nanmean(early[col])) if col in early else np.nan
            late_mean = float(np.nanmean(late[col])) if col in late else np.nan
            row[f"{col}_early_mean"] = early_mean
            row[f"{col}_late_mean"] = late_mean
            row[f"{col}_delta"] = late_mean - early_mean
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sector false-positive rate on C
# ─────────────────────────────────────────────────────────────────────────────
def _sector_false_positive(
    panel: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Per-sector RED rate on ``C``.

    A RED row satisfies ``min(p across composites) < red_threshold``. The
    table reports the share of RED rows per sector on the honest sample —
    high shares signal a business-model bias rather than a manipulation
    pattern.
    """
    schema = settings.panel_schema
    rob = settings.robustness

    merged = composites.merge(
        panel[[schema.firm_id, schema.quarter, schema.sector, "_split"]],
        on=[schema.firm_id, schema.quarter],
        how="left",
    )
    c_rows = merged[merged["_split"] == SPLIT_LABEL_INCLUDED].copy()
    if c_rows.empty:
        return pd.DataFrame()

    c_rows["min_p"] = c_rows[list(VERDICT_P_COLS)].min(axis=1)
    c_rows["is_red"] = c_rows["min_p"] < rob.red_threshold

    grouped = (
        c_rows.groupby(schema.sector, dropna=False)
        .agg(n_rows=("is_red", "size"), n_red=("is_red", "sum"))
        .reset_index()
    )
    grouped["red_share"] = grouped["n_red"] / grouped["n_rows"]
    # Separate "large enough to matter" sectors from noise. 30 matches the
    # framework §9.2 D3 reference-sample floor used throughout the pipeline.
    grouped["is_n_sufficient"] = grouped["n_rows"] >= 30
    grouped = grouped.sort_values(
        ["is_n_sufficient", "red_share"], ascending=[False, False],
    ).reset_index(drop=True)
    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# 3. Integrity sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def _integrity_sensitivity(
    panel: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Agreement between the full-panel top-N and the strict-integrity top-N.

    Naïve rank correlation on the shared rows is trivially 1 — the two
    rankings use the same values. The meaningful question is whether the
    *top-N RED firms* identified on the full panel remain at the top when
    ffilled rows are excluded. A large drop means the composite is
    amplifying ffill artefacts.

    The table reports:
    - ``n_strict_rows`` — rows with ``bs_any_ffilled == 0`` and a finite
      composite value.
    - ``jaccard_top100`` — share of firm-quarters shared between the
      full-panel top-100 and the strict-subset top-100. 1.0 = strict
      filter doesn't change the ranking; <0.5 = high sensitivity.
    - ``share_of_full_top100_in_strict`` — fraction of the full-panel
      top-100 rows that survive the strict filter (the top ranker rows
      already had no ffill).
    """
    schema = settings.panel_schema
    col = settings.robustness.strict_integrity_column
    if col not in panel.columns:
        log.warning("phase6: %r absent from panel; skipping integrity check.", col)
        return pd.DataFrame()

    merged = composites.merge(
        panel[[schema.firm_id, schema.quarter, col]],
        on=[schema.firm_id, schema.quarter],
        how="left",
    )
    strict_mask = merged[col].fillna(1).astype(int) == 0
    top_n = 100

    rows = []
    for comp_col in COMPOSITE_VALUE_COLS:
        full_ranked = merged[comp_col].dropna()
        strict_ranked = merged.loc[strict_mask, comp_col].dropna()
        if full_ranked.empty or strict_ranked.empty:
            continue
        top_full = set(full_ranked.nlargest(top_n).index)
        top_strict = set(strict_ranked.nlargest(top_n).index)
        inter = len(top_full & top_strict)
        union = len(top_full | top_strict)
        survive = len(top_full & set(strict_ranked.index))
        rows.append(
            {
                "composite": comp_col,
                "n_strict_rows": int(strict_ranked.size),
                "jaccard_top_n": (inter / union) if union else 0.0,
                "share_of_full_top_n_in_strict": (survive / top_n) if top_n else 0.0,
                "top_n": top_n,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Composite stability (lag-1 autocorrelation per firm)
# ─────────────────────────────────────────────────────────────────────────────
def _composite_stability(
    panel: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Per-firm lag-1 autocorrelation of each composite, then aggregated.

    High autocorrelation = persistent chronic signal (plausibly a sector
    or business-model artefact rather than episodic manipulation). Low
    autocorrelation = quarter-to-quarter noise.
    """
    schema = settings.panel_schema
    rob = settings.robustness

    merged = composites.merge(
        panel[[schema.firm_id, schema.quarter]],
        on=[schema.firm_id, schema.quarter],
        how="left",
    ).sort_values([schema.firm_id, schema.quarter])

    rows: list[dict] = []
    for firm_id, grp in merged.groupby(schema.firm_id, observed=True):
        if len(grp) < rob.stability_min_quarters:
            continue
        row = {schema.firm_id: firm_id, "n_quarters": int(len(grp))}
        for col in COMPOSITE_VALUE_COLS:
            series = pd.to_numeric(grp[col], errors="coerce")
            if series.notna().sum() < rob.stability_min_quarters:
                row[f"{col}_lag1"] = float("nan")
                continue
            lag1 = series.autocorr(lag=1)
            row[f"{col}_lag1"] = float(lag1) if np.isfinite(lag1) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _stability_summary(stability: pd.DataFrame) -> dict:
    """Mean / median lag-1 autocorrelation per composite across firms."""
    out: dict = {}
    for col in COMPOSITE_VALUE_COLS:
        target = f"{col}_lag1"
        if target not in stability.columns:
            continue
        series = stability[target].dropna()
        if series.empty:
            continue
        out[col] = {
            "n_firms": int(series.size),
            "mean_lag1": float(series.mean()),
            "median_lag1": float(series.median()),
            "share_above_half": float((series > 0.5).mean()),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_phase6(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    composites: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> Phase6Outcome:
    """Run all four robustness diagnostics on the Phase 4 output.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0`.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        composites: Per-row composites (output of Phase 4). When ``None``
            the parquet at ``settings.output_layout.phase4_dir() /
            'composites_panel.parquet'`` is loaded.
        write_outputs: If ``True``, persist CSVs + JSON under
            ``settings.output_layout.phase6_dir()``.
    """
    settings = settings or AnalysisSettings()
    if composites is None:
        cpath = settings.output_layout.phase4_dir() / "composites_panel.parquet"
        composites = pd.read_parquet(cpath)

    delisting = _delisting_trajectory(panel, composites, settings)
    sector_fp = _sector_false_positive(panel, composites, settings)
    integrity = _integrity_sensitivity(panel, composites, settings)
    stability = _composite_stability(panel, composites, settings)

    summary: dict = {
        "delisting_cohort_size": int(delisting.shape[0]),
        "delisting_mean_delta_z_mahalanobis_sq": float(
            delisting[f"{COL_Z_MAHALANOBIS}_delta"].mean()
        ) if f"{COL_Z_MAHALANOBIS}_delta" in delisting.columns and not delisting.empty else None,
        "sector_top_red": (
            sector_fp[sector_fp["is_n_sufficient"]].head(5).to_dict(orient="records")
            if not sector_fp.empty else []
        ),
        "integrity_sensitivity": integrity.to_dict(orient="records") if not integrity.empty else [],
        "stability_summary": _stability_summary(stability),
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase6_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["delisting"] = out_dir / "delisting_proxy.csv"
        paths["sector_fp"] = out_dir / "sector_false_positive.csv"
        paths["integrity"] = out_dir / "integrity_sensitivity.csv"
        paths["stability"] = out_dir / "composite_stability.csv"
        paths["json"] = out_dir / "phase6_robustness.json"
        delisting.to_csv(paths["delisting"], index=False)
        sector_fp.to_csv(paths["sector_fp"], index=False)
        integrity.to_csv(paths["integrity"], index=False)
        stability.to_csv(paths["stability"], index=False)
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("phase6: wrote %d deliverables to %s", len(paths), out_dir)

    return Phase6Outcome(
        delisting=delisting,
        sector_fp=sector_fp,
        integrity=integrity,
        stability=stability,
        summary=summary,
        paths=paths,
    )
