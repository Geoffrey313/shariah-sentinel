"""Per-detector z-score computation and cache.

Phase 1 (null calibration) and Phase 2 (dependence) both consume the seven
detector z-scores on the panel. Running every detector on every phase is
wasteful — this module runs them once and caches the result as a parquet.

Contract:

- Input is the panel returned by :func:`phase0_reference_sample.run_phase0`
  — i.e. the panel augmented with ``_split``. Detectors consult ``_split``
  to distinguish the honest calibration sample ``C`` from the rest.
- Output is a dataframe with ``(gvkey, datacqtr, z1, …, z7)`` columns. The
  detector order is driven by :attr:`ZScoreSettings.include_detectors` so
  downstream Σ̂ matrices, PCA axes and QQ-plot decks all use the same
  ordering.
- The cache is **not** masked by the Phase 3 validity parquet. Downstream
  phases combine ``zscores`` and ``mask`` columns themselves; keeping the
  two artefacts independent lets a reviewer study detector output before
  the preconditions, or compare pass rates between config variants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.pit import pit_empirical

from src.engine import (
    detect_benford,
    detect_cross_statement_coherence,
    detect_mscore,
    detect_peer,
    detect_temporal,
    detect_threshold_proximity,
    detect_zipf,
)
from src.engine.cost_of_debt import detect_cost_of_debt
from src.engine.seasonal_gap import detect_seasonal_gap

from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZScoreOutcome:
    """Return value of :func:`compute_zscores`."""

    zscores: pd.DataFrame
    path: Path | None


_DETECTOR_DISPATCH: dict[str, str] = {
    "z1": "benford",
    "z2": "zipf",
    "z3": "mscore",
    "z4": "threshold",
    "z5": "cross_statement",
    "z6": "temporal",
    "z7": "peer",
    "z8": "cost_of_debt",
    "z9": "seasonal_gap",
}


def _run_detector(name: str, panel: pd.DataFrame, settings: AnalysisSettings) -> pd.Series:
    """Dispatch to the correct detector function.

    Kept as a single switch so that adding a new detector or swapping the
    implementation of one touches exactly one place. ``panel`` already
    carries ``_split`` (Phase 0) so each detector restricts calibration
    internally.
    """
    if name == "z1":
        precond = settings.detector_preconditions.benford
        return detect_benford(
            panel,
            value_columns=settings.detector_preconditions.benford_value_columns,
            min_obs=precond.min_figures,
            min_obs_mc=precond.min_figures_mc,
            pooling_window_quarters=precond.pooling_window_quarters,
            calibration_mode=precond.calibration_mode,
            chi2_minimum_figures=precond.chi2_minimum_figures,
            monte_carlo_replicates=precond.monte_carlo_replicates,
            monte_carlo_seed=precond.monte_carlo_seed,
        )
    if name == "z2":
        precond = settings.detector_preconditions.zipf
        return detect_zipf(
            panel,
            value_columns=settings.detector_preconditions.benford_value_columns,
            min_points=precond.min_points,
            pooling_window_quarters=precond.pooling_window_quarters,
        )
    if name == "z3":
        return detect_mscore(panel)
    if name == "z4":
        return detect_threshold_proximity(panel)
    if name == "z5":
        return detect_cross_statement_coherence(panel, settings=settings)
    if name == "z6":
        return detect_temporal(panel, mode=settings.zscores.temporal_mode)
    if name == "z7":
        return detect_peer(panel)
    if name == "z8":
        return detect_cost_of_debt(panel)
    if name == "z9":
        return detect_seasonal_gap(panel)
    raise KeyError(f"unknown detector {name!r}; known: {tuple(_DETECTOR_DISPATCH)}")


def compute_zscores(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    write_outputs: bool = True,
) -> ZScoreOutcome:
    """Run detectors z1..z9 on the panel and cache the result.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0` — must
            contain ``_split``, ``gvkey``, ``datacqtr`` and the schema
            columns the detectors expect.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        write_outputs: If ``True``, persist the parquet under
            ``settings.output_layout.zscores_dir()``.

    Returns:
        A :class:`ZScoreOutcome` wrapping the z-score dataframe and — when
        written — the resolved cache path.

    Raises:
        KeyError: If ``_split`` is missing or the panel lacks schema columns.
    """
    settings = settings or AnalysisSettings()
    schema = settings.panel_schema

    if "_split" not in panel.columns:
        raise KeyError(
            "compute_zscores: panel lacks `_split` — run Phase 0 first."
        )
    for col in (schema.firm_id, schema.quarter):
        if col not in panel.columns:
            raise KeyError(f"compute_zscores: panel lacks schema column {col!r}.")

    result = pd.DataFrame(
        {
            schema.firm_id: panel[schema.firm_id].values,
            schema.quarter: panel[schema.quarter].values,
        },
        index=panel.index,
    )

    det_list = list(settings.zscores.include_detectors)
    # Only show per-detector progress for the main pipeline run (large panel).
    # Suppress for benchmark inner calls (write_outputs=False) to avoid bar spam.
    _show_progress = write_outputs
    if _show_progress:
        try:
            from tqdm import tqdm
            det_iter = tqdm(det_list, desc="detectors", unit="det", leave=False)
        except ImportError:
            det_iter = det_list
            _show_progress = False
    else:
        det_iter = det_list

    for name in det_iter:
        if _show_progress:
            det_iter.set_postfix(det=name)
        else:
            log.info("compute_zscores: running detector %s", name)
        series = _run_detector(name, panel, settings)
        result[name] = series.reindex(panel.index).astype(float).values

    # Apply merges — output columns appended, inputs optionally dropped.
    dropped_inputs: set[str] = set()
    for rule in settings.zscores.merge_rules:
        missing = [i for i in rule.inputs if i not in result.columns]
        if missing:
            log.warning(
                "compute_zscores: merge %r skipped, missing inputs %s.",
                rule.output_name, missing,
            )
            continue
        input_matrix = result.loc[:, list(rule.inputs)].to_numpy(dtype=float)
        if rule.method == "max":
            merged = np.nanmax(input_matrix, axis=1)
        elif rule.method == "mean":
            merged = np.nanmean(input_matrix, axis=1)
        else:  # pragma: no cover — guarded by pydantic validator
            raise ValueError(f"unknown merge method {rule.method!r}.")
        # ``np.nan{max,mean}`` of an all-NaN row raises a RuntimeWarning and
        # returns NaN — which is the intended "no input finite" behaviour.
        all_nan = np.isnan(input_matrix).all(axis=1)
        merged = np.where(all_nan, np.nan, merged)

        # Optional empirical PIT on C — same shape-calibration as z1/z4.
        # ``max`` of correlated z-scores is right-skewed; PITing against C
        # re-centres honest firms to mean ≈ 0 so the composite bootstrap
        # reads injections as real departures rather than baseline drift.
        if rule.empirical_pit_on_c and "_split" in panel.columns:
            ref = merged[panel["_split"].to_numpy() == "C"]
            ref = ref[np.isfinite(ref)]
            if ref.size >= 30:
                finite = np.isfinite(merged)
                merged = merged.copy()
                merged[finite] = pit_empirical(merged[finite], ref)

        result[rule.output_name] = merged
        log.info(
            "compute_zscores: merged %s → %s via %s (%d/%d finite, pit_on_c=%s)",
            list(rule.inputs), rule.output_name, rule.method,
            int(np.isfinite(merged).sum()), len(merged), rule.empirical_pit_on_c,
        )
        if rule.drop_inputs:
            dropped_inputs.update(rule.inputs)

    if dropped_inputs:
        result = result.drop(columns=[c for c in dropped_inputs if c in result.columns])

    path: Path | None = None
    if write_outputs:
        out_dir = settings.output_layout.zscores_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / settings.zscores.filename
        result.to_parquet(path, index=False)
        log.info("compute_zscores: cached %d rows × %d detectors → %s",
                 len(result), len(settings.zscores.include_detectors), path)

    return ZScoreOutcome(zscores=result, path=path)


def load_zscores(settings: AnalysisSettings | None = None) -> pd.DataFrame:
    """Read the cached z-score parquet back into memory."""
    settings = settings or AnalysisSettings()
    path = settings.output_layout.zscores_dir() / settings.zscores.filename
    if not path.exists():
        raise FileNotFoundError(
            f"zscores cache missing at {path}; run `compute_zscores` first."
        )
    return pd.read_parquet(path)
