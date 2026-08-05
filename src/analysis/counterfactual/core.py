"""Core engine for canonical Family 3 counterfactual robustness.

This module contains the exact counterfactual loss, working-set/snapshot logic,
and the orchestration for the three public modes:
- global_per_row
- global_cohort
- local_firm_single
"""


from src.analysis.counterfactual.utils import *
from src.analysis.counterfactual.torch import *


import numpy as np
import pandas as pd



def _coerce_float(value: object) -> float:
    """Best-effort scalar coercion used by the Family 3 loss helpers."""
    return float(pd.to_numeric(value, errors="coerce"))


def family3_target_score_kind(score_name: str) -> str:
    """Classify the target score using the registered composite namespace.

    Composite scores must be declared in ``COMPOSITE_VALUE_COLUMNS``. Falling
    back to a string-prefix heuristic would silently misclassify future
    composites such as ``z_*`` names.
    """
    return "composite" if score_name in COMPOSITE_VALUE_COLUMNS else "base"


def family3_status_from_p(pvalue: float, settings) -> str:
    rb = settings.robustness_benchmark
    if not np.isfinite(pvalue):
        return "UNKNOWN"
    if pvalue < rb.family3_p_red:
        return "RED"
    if pvalue < rb.family3_p_orange:
        return "ORANGE"
    return "GREEN"


def family3_target_status_from_z(z_value: float, settings) -> str:
    rb = settings.robustness_benchmark
    if not np.isfinite(z_value):
        return "UNKNOWN"
    if z_value >= rb.family3_z_target_red:
        return "RED"
    if z_value >= rb.family3_z_target_green:
        return "ORANGE"
    return "GREEN"


def family3_z_equivalent_from_p(pvalue: float) -> float:
    from scipy.stats import norm

    p = float(np.clip(pvalue, 1e-12, 1.0 - 1e-12))
    return float(norm.isf(p))


def family3_raw_score_value(
    row_index: object,
    score_name: str,
    snapshot: Family3Snapshot,
) -> float:
    if score_name in snapshot.raw_zscores.columns:
        return _coerce_float(snapshot.raw_zscores.at[row_index, score_name])
    if score_name in snapshot.composites.columns:
        return _coerce_float(snapshot.composites.at[row_index, score_name])
    return float("nan")


def family3_score_z_value(
    row_index: object,
    score_name: str,
    snapshot: Family3Snapshot,
) -> float:
    if score_name in snapshot.raw_zscores.columns:
        return _coerce_float(snapshot.raw_zscores.at[row_index, score_name])
    p_col = f"p_{score_name}"
    if p_col in snapshot.composites.columns:
        pvalue = _coerce_float(snapshot.composites.at[row_index, p_col])
        if np.isfinite(pvalue):
            return family3_z_equivalent_from_p(pvalue)
    if score_name in snapshot.composites.columns:
        raw_value = _coerce_float(snapshot.composites.at[row_index, score_name])
        if np.isfinite(raw_value):
            return raw_value
    if score_name in snapshot.merged_zscores.columns:
        return _coerce_float(snapshot.merged_zscores.at[row_index, score_name])
    return float("nan")


def family3_primary_pvalue(
    row_index: object,
    snapshot: Family3Snapshot,
    settings,
) -> float:
    primary = settings.robustness_benchmark.primary_composite
    if primary not in snapshot.composites.columns:
        return float("nan")
    return _coerce_float(snapshot.composites.at[row_index, primary])


def family3_firm_target_pvalue(
    firm_id: object,
    score_name: str,
    snapshot: Family3Snapshot,
    settings,
) -> float:
    """Return one firm-level target p-value using the configured aggregation.

    The current Family 3 firm-level prototype supports only the working-paper
    default: collapse all row-level target p-values of a firm with
    `min_pvalue`, i.e. keep the most anomalous quarter.
    """

    schema = settings.panel_schema
    firm_mask = snapshot.panel[schema.firm_id].astype(str) == str(firm_id)
    if not bool(firm_mask.any()):
        return float("nan")
    p_col = f"p_{score_name}"
    if p_col not in snapshot.composites.columns:
        return float("nan")
    target_p = pd.to_numeric(snapshot.composites.loc[firm_mask, p_col], errors="coerce")
    if target_p.notna().any():
        return float(target_p.min(skipna=True))
    return float("nan")


def family3_firm_score_z_value(
    firm_id: object,
    score_name: str,
    snapshot: Family3Snapshot,
    settings,
) -> float:
    """Return one firm-level target score on the common z-equivalent scale."""

    pvalue = family3_firm_target_pvalue(
        firm_id=firm_id,
        score_name=score_name,
        snapshot=snapshot,
        settings=settings,
    )
    if np.isfinite(pvalue):
        return family3_z_equivalent_from_p(pvalue)

    schema = settings.panel_schema
    firm_mask = snapshot.panel[schema.firm_id].astype(str) == str(firm_id)
    if not bool(firm_mask.any()):
        return float("nan")
    row_scores = np.array(
        [family3_score_z_value(idx, score_name, snapshot) for idx in snapshot.panel.index[firm_mask]],
        dtype=float,
    )
    if np.isfinite(row_scores).any():
        return float(np.nanmax(row_scores))
    return float("nan")


def family3_flag_pvalues(snapshot: Family3Snapshot) -> pd.Series:
    p_cols = [col for col in VERDICT_P_COLUMNS if col in snapshot.composites.columns]
    if not p_cols:
        return pd.Series(np.nan, index=snapshot.composites.index, dtype=float)
    work = snapshot.composites.loc[:, p_cols].apply(pd.to_numeric, errors="coerce")
    return work.min(axis=1, skipna=True)


def family3_flag_pvalue(
    row_index: object,
    snapshot: Family3Snapshot,
) -> float:
    values = family3_flag_pvalues(snapshot)
    if row_index not in values.index:
        return float("nan")
    return _coerce_float(values.loc[row_index])


def family3_normalized_delta(
    candidate_x: np.ndarray,
    baseline_x: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.divide(
        np.asarray(candidate_x, dtype=float) - np.asarray(baseline_x, dtype=float),
        scale,
        out=np.zeros_like(scale, dtype=float),
        where=scale > 1e-12,
    )




def family3_score_loss_term(
    *,
    target_score_z: float,
    direction: str,
    settings,
) -> tuple[float, float]:
    rb = settings.robustness_benchmark
    threshold = rb.family3_z_target_green if direction == "to_green" else rb.family3_z_target_red
    if direction == "to_green":
        margin = target_score_z - threshold
    else:
        margin = threshold - target_score_z
    margin = float(margin)
    loss_name = getattr(rb, "family3_loss_name", "hinge")
    if loss_name == "hinge_squared":
        hinge = max(0.0, margin)
        loss_score = hinge * hinge
    elif loss_name == "hinge":
        loss_score = max(0.0, margin)
    elif loss_name == "mse":
        loss_score = margin * margin
    else:
        raise ValueError(f"Unsupported Family 3 loss {loss_name!r}.")
    return float(loss_score), float(threshold)


def family3_exact_loss(
    *,
    target_score_z: float,
    direction: str,
    delta_norm: np.ndarray,
    settings,
) -> tuple[float, float, float, float, float]:
    rb = settings.robustness_benchmark
    loss_score, threshold = family3_score_loss_term(
        target_score_z=target_score_z,
        direction=direction,
        settings=settings,
    )
    loss_l1 = rb.family3_lambda_l1 * float(np.abs(delta_norm).sum())
    loss_l2 = rb.family3_lambda_l2 * float(np.square(delta_norm).sum())
    loss_regularizer = float(loss_l1 + loss_l2)
    return float(loss_score + loss_regularizer), float(loss_score), float(loss_l1), float(loss_l2), float(threshold)


from dataclasses import dataclass
import logging
import time

import numpy as np
import pandas as pd

from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RatioCache:
    panel_with_ratios: pd.DataFrame
    canonical_ratios: pd.DataFrame
    ratio_columns: tuple[str, ...]


@dataclass(frozen=True)
class DetectorCache:
    raw_zscores: pd.DataFrame
    z3_raw_values: np.ndarray | None = None
    z3_reference_values: np.ndarray | None = None
    z5_context: object | None = None
    z4_reference_values: np.ndarray | None = None
    z4_firm_raw: dict[str, float] | None = None
    z4_reference_firms: frozenset[str] | None = None
    z7_context: object | None = None
    z8_raw_values: np.ndarray | None = None
    z8_reference_values: np.ndarray | None = None
    z9_reference_values: np.ndarray | None = None
    z1_mc_null_cache: dict[int, np.ndarray] | None = None
    z1_reference_values: np.ndarray | None = None
    z1_settings: dict[str, object] | None = None
    z2_sigma0_sq: float | None = None
    z2_settings: dict[str, object] | None = None


@dataclass(frozen=True)
class CompositeCache:
    merged_zscores: pd.DataFrame
    composites: pd.DataFrame


@dataclass(frozen=True)
class SnapshotCache:
    ratio: RatioCache
    detector: DetectorCache
    composite: CompositeCache


@dataclass(frozen=True)
class WorkingSet:
    panel: pd.DataFrame
    scratch_panel: pd.DataFrame
    scratch_panel_with_ratios: pd.DataFrame
    scratch_raw_zscores: pd.DataFrame
    scratch_merged_zscores: pd.DataFrame
    scratch_composites: pd.DataFrame
    row_indices: tuple[object, ...]
    columns: tuple[str, ...]
    baseline_matrix: np.ndarray
    scales: np.ndarray
    snapshot_cache: SnapshotCache


@dataclass(frozen=True)
class WorkingSetTemplate:
    panel: pd.DataFrame
    snapshot_cache: SnapshotCache


def _family3_with_split_columns(panel: pd.DataFrame, base_ctx: ScoreContext) -> pd.DataFrame:
    split_cols = [c for c in ("_split", "_split_reason") if c in base_ctx.panel.columns]
    missing = [c for c in split_cols if c not in panel.columns]
    if not missing:
        return panel
    panel_with_split = panel.copy()
    aligned = base_ctx.panel.reindex(panel_with_split.index)
    for col in missing:
        panel_with_split[col] = aligned[col]
    return panel_with_split


def family3_raw_zscore_settings(settings):
    return settings.model_copy(
        update={
            "zscores": settings.zscores.model_copy(update={"merge_rules": tuple()}),
        }
    )


def _rowwise_nanreduce(matrix: np.ndarray, *, method: str) -> np.ndarray:
    """Reduce rows without emitting NumPy all-NaN warnings.

    Family 3 intentionally preserves NaN rows when every detector input is
    missing. The helper keeps that behavior while avoiding noisy runtime
    warnings inside long optimization loops.
    """
    if method == "max":
        all_nan = np.isnan(matrix).all(axis=1)
        reduced = np.where(all_nan, np.nan, np.where(np.isnan(matrix), -np.inf, matrix).max(axis=1))
        return reduced.astype(float, copy=False)
    if method == "mean":
        counts = np.isfinite(matrix).sum(axis=1)
        sums = np.nansum(matrix, axis=1)
        reduced = np.full(matrix.shape[0], np.nan, dtype=float)
        np.divide(sums, counts, out=reduced, where=counts > 0)
        return reduced
    raise ValueError(f"unknown merge method {method!r}.")


def family3_build_snapshot_cache(
    panel: pd.DataFrame,
    settings,
    *,
    base_ctx: ScoreContext,
    changed_index: pd.Index | None = None,
    global_refs: "GlobalDetectorReferences | None" = None,
) -> SnapshotCache:
    from src.data.ratios import compute_shariah_ratios
    from src.common.ratio_inputs import get_canonical_sharia_ratios

    panel_for_snapshot = _family3_with_split_columns(panel, base_ctx)
    panel_with_ratios = compute_shariah_ratios(
        panel_for_snapshot,
        log_coverage=False,
        warn_on_missing_connectors=False,
    )
    canonical_ratios = get_canonical_sharia_ratios(panel_with_ratios)
    ratio_columns = tuple(
        col for col in panel_with_ratios.columns
        if col not in panel.columns or col.startswith("ratio_") or col.startswith("flag_") or col == "n_flags"
    )
    (
        z3_raw_values,
        z3_reference_values,
        z5_context,
        z4_reference_values,
        z4_firm_raw,
        z4_reference_firms,
        z7_context,
        z8_raw_values,
        z8_reference_values,
        z9_reference_values,
        z1_mc_null_cache,
        z1_reference_values,
        z1_settings,
        z2_sigma0_sq,
        z2_settings,
    ) = _family3_build_detector_incremental_refs(
        panel_with_ratios, canonical_ratios, settings, changed_index=changed_index, global_refs=global_refs,
    )
    # This function only ever runs on still-unperturbed (baseline) data --
    # it has a single caller (`family3_build_working_template`), invoked once
    # per firm before any optimisation epoch touches the panel. Re-deriving
    # `merged_zscores`/`composites` from `reindexed_raw_zscores` (itself just
    # a reindex of `base_ctx.raw_zscores`, not the freshly recomputed values
    # above) therefore always reproduces `base_ctx.zscores`/`base_ctx.composites`
    # verbatim -- confirmed bit-identical empirically on the real MYS panel,
    # including firms whose rows straddle the C/non-C split boundary. Reusing
    # `base_ctx` directly skips a full merge-rule pass and the expensive
    # `orthogonal_softmax` composite recomputation on the whole working panel
    # (~7-9k rows), which was previously redone on every row of `global_per_row`
    # (see profiling notes).
    reindexed_raw_zscores = base_ctx.raw_zscores.reindex(panel.index)
    merged_zscores = base_ctx.zscores.reindex(panel.index)
    composites = base_ctx.composites.reindex(panel.index)
    return SnapshotCache(
        ratio=RatioCache(panel_with_ratios=panel_with_ratios, canonical_ratios=canonical_ratios, ratio_columns=ratio_columns),
        detector=DetectorCache(
            raw_zscores=reindexed_raw_zscores.copy(),
            z3_raw_values=z3_raw_values,
            z3_reference_values=z3_reference_values,
            z5_context=z5_context,
            z4_reference_values=z4_reference_values,
            z4_firm_raw=z4_firm_raw,
            z4_reference_firms=z4_reference_firms,
            z7_context=z7_context,
            z8_raw_values=z8_raw_values,
            z8_reference_values=z8_reference_values,
            z9_reference_values=z9_reference_values,
            z1_mc_null_cache=z1_mc_null_cache,
            z1_reference_values=z1_reference_values,
            z1_settings=z1_settings,
            z2_sigma0_sq=z2_sigma0_sq,
            z2_settings=z2_settings,
        ),
        composite=CompositeCache(merged_zscores=merged_zscores, composites=composites),
    )


def family3_build_working_set(
    panel: pd.DataFrame,
    target_indices: list[object],
    columns: tuple[str, ...],
    settings,
    *,
    base_ctx: ScoreContext,
    preserve_global_context: bool = False,
    template: WorkingSetTemplate | None = None,
    global_refs: "GlobalDetectorReferences | None" = None,
    scale_floor: np.ndarray | None = None,
) -> WorkingSet:
    if template is None:
        template = family3_build_working_template(
            panel,
            target_indices,
            settings,
            base_ctx=base_ctx,
            preserve_global_context=preserve_global_context,
            global_refs=global_refs,
        )
    working_panel = template.panel
    baseline_frame = (
        working_panel.loc[target_indices, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    baseline_matrix = baseline_frame.to_numpy(dtype=float, copy=True)
    floor_vec = scale_floor if scale_floor is not None else np.ones(len(columns), dtype=float)
    scales = np.maximum(np.abs(baseline_matrix), floor_vec)
    return WorkingSet(
        panel=working_panel,
        scratch_panel=working_panel.copy(),
        scratch_panel_with_ratios=template.snapshot_cache.ratio.panel_with_ratios.copy(),
        scratch_raw_zscores=template.snapshot_cache.detector.raw_zscores.copy(),
        scratch_merged_zscores=template.snapshot_cache.composite.merged_zscores.copy(),
        scratch_composites=template.snapshot_cache.composite.composites.copy(),
        row_indices=tuple(target_indices),
        columns=tuple(columns),
        baseline_matrix=baseline_matrix,
        scales=scales,
        snapshot_cache=template.snapshot_cache,
    )


def family3_build_working_template(
    panel: pd.DataFrame,
    target_indices: list[object],
    settings,
    *,
    base_ctx: ScoreContext,
    preserve_global_context: bool = False,
    global_refs: "GlobalDetectorReferences | None" = None,
) -> WorkingSetTemplate:
    panel_for_working = _family3_with_split_columns(panel, base_ctx)
    working_panel = family3_working_panel(
        panel_for_working,
        target_indices,
        settings,
        preserve_global_context=preserve_global_context,
    )
    snapshot_cache = family3_build_snapshot_cache(
        working_panel, settings, base_ctx=base_ctx, changed_index=pd.Index(target_indices), global_refs=global_refs,
    )
    return WorkingSetTemplate(panel=working_panel, snapshot_cache=snapshot_cache)


def family3_detector_dependencies(settings) -> dict[str, set[str] | None]:
    """Return coarse raw-column dependencies for incremental detector updates.

    ``None`` means "treat this detector as always impacted". We use that
    fallback for Benford/Zipf when the configured value columns are empty so we
    never silently skip ``z1``/``z2`` recomputation because of an incomplete
    configuration payload.
    """
    ratio_driver_columns = {
        "atq",
        "dlttq",
        "dlcq",
        "cheq",
        "iditq",
        "revtq",
    }
    benford_columns = set(settings.detector_preconditions.benford_value_columns)
    benford_dependencies: set[str] | None = benford_columns or None
    return {
        "z1": benford_dependencies,
        "z2": benford_dependencies,
        "z3": {"rectq", "revtq", "ppentq", "actq", "atq", "xsgaq", "ibq", "oancfq"},
        "z4": set(ratio_driver_columns),
        "z5": {
            "niq",
            "oibdpq",
            "oancfq",
            "revtq",
            "ibq",
            "atq",
            "xintq",
            "dlttq",
            "dlcq",
            "actq",
            "cheq",
            "lctq",
            "iditq",
        },
        "z6": set(ratio_driver_columns),
        "z7": set(ratio_driver_columns),
        "z8": {"xintq", "dlttq", "dlcq"},
        "z9": set(ratio_driver_columns),
    }


def _family3_build_detector_incremental_refs(
    panel_with_ratios: pd.DataFrame,
    canonical_ratios: pd.DataFrame,
    settings,
    *,
    changed_index: pd.Index | None = None,
    force_build_z1z2: bool = False,
    global_refs: "GlobalDetectorReferences | None" = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    object | None,
    np.ndarray | None,
    object | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[int, np.ndarray] | None,
    np.ndarray | None,
    dict[str, object] | None,
    float | None,
    dict[str, object] | None,
]:
    from src.engine.benford import (
        _MIN_EMPIRICAL_PIT_REF as D1_MIN_EMPIRICAL_PIT_REF,
        detect_benford,
    )
    from src.engine.zipf import (
        MIN_REFERENCE_FIRMS as D2_MIN_REFERENCE_FIRMS,
        _collect_row_sizes_matrix,
        _estimate_null_scale,
        _rolling_zipf_fits_fast,
    )
    from src.engine.mscore import _compute_mscore_raw_vectorized, MIN_REF_SIZE as D3_MIN_REF_SIZE
    from src.engine.coherence import (
        MIN_GLOBAL_REFERENCE as D5_MIN_GLOBAL_REFERENCE,
        RELATION_SPECS,
        _estimate_moments,
        _resolve_sector_moments,
        _select_sector_column,
        compute_cross_statement_relations,
    )
    from src.engine.proximity import _firm_raw_statistics, _get_ratio_matrix, _reference_firms
    from src.engine.peer import _get_global_reference
    from src.engine.cost_of_debt import _implied_cost_of_debt, _t8_vectorized, MIN_REF_FIRMS
    from src.engine.seasonal_gap import _seasonal_gap_vectorized, MIN_REF_FIRMS as MIN_REF_FIRMS_Z9
    from src.common.methodology import thresholds_for_panel

    thresholds = thresholds_for_panel(panel_with_ratios)

    def _z9_reference_from_firm_raw(firm_raw_z9: dict[str, float]) -> np.ndarray | None:
        if "_split" in panel_with_ratios.columns:
            c_firms = set(panel_with_ratios.loc[panel_with_ratios["_split"] == "C", "gvkey"].astype(str).unique())
        else:
            c_firms = set(firm_raw_z9.keys())
        ref_values_z9 = np.asarray(
            [v for k, v in firm_raw_z9.items() if k in c_firms and np.isfinite(v)],
            dtype=float,
        )
        return ref_values_z9 if ref_values_z9.size >= MIN_REF_FIRMS_Z9 else None

    z3_raw_values: np.ndarray | None = None
    z3_reference_values: np.ndarray | None = None
    z8_raw_values: np.ndarray | None = None
    z8_reference_values: np.ndarray | None = None
    z9_reference_values: np.ndarray | None = None
    z1_mc_null_cache: dict[int, np.ndarray] | None = None
    z1_reference_values: np.ndarray | None = None
    z1_settings: dict[str, object] | None = None
    z2_sigma0_sq: float | None = None
    z2_settings: dict[str, object] | None = None
    has_firm_quarter_cols = {"gvkey", "datacqtr"}.issubset(panel_with_ratios.columns)
    available_ratios_z9 = [c for c in thresholds if c in panel_with_ratios.columns]
    has_fqtr = "fqtr" in panel_with_ratios.columns
    z8_inputs_available = has_firm_quarter_cols and {"dlttq", "xintq"}.issubset(panel_with_ratios.columns)
    z9_inputs_available = bool(available_ratios_z9) and "gvkey" in panel_with_ratios.columns and has_fqtr
    benford_value_columns = [
        c for c in settings.detector_preconditions.benford_value_columns if c in panel_with_ratios.columns
    ]

    if z9_inputs_available:
        # z9 is fully vectorized (groupby.transform, no Python loop and no
        # sort needed -- it only takes medians of year-end vs interim
        # subsets, which don't care about row order). See profiling notes.
        t9_broadcast = _seasonal_gap_vectorized(panel_with_ratios, thresholds)
        firm_raw_z9 = t9_broadcast.groupby(panel_with_ratios["gvkey"].astype(str), sort=False).first().to_dict()
        z9_reference_values = _z9_reference_from_firm_raw(firm_raw_z9)

    if (
        has_firm_quarter_cols
        and benford_value_columns
        and (
            force_build_z1z2
            or (changed_index is not None and _family3_can_use_local_detector_patch(panel_with_ratios, changed_index))
        )
    ):
        # z1/z2 are per-firm rolling-window detectors like z3/z8/z9, but
        # their second-stage calibration needs extra state a single row
        # can't provide (a Monte-Carlo null per N_fig for z1's
        # "auto"/"monte_carlo" modes, a scalar null scale for z2).
        # `force_build_z1z2=True` is how `family3_build_global_detector_references`
        # (profiling notes) requests this once, on the full panel, for
        # the whole run; without a caller forcing it, this only builds when
        # `_family3_can_use_local_detector_patch` is true for the specific
        # changed row -- kept as a narrow per-row fallback, but no longer the
        # primary path now that the global build covers virtually every row.
        # Building it unconditionally in that old per-row path once
        # regressed real runs where nothing consumed it (0/1043 firms in the
        # MYS panel are whole-firm-disjoint from split C -- see profiling notes), which is exactly why the global, run-wide build
        # exists instead.
        benford_precond = settings.detector_preconditions.benford
        z1_mc_null_cache = {}
        z1_capture: dict[str, object] = {}
        detect_benford(
            panel_with_ratios,
            value_columns=benford_value_columns,
            min_obs=benford_precond.min_figures,
            min_obs_mc=benford_precond.min_figures_mc,
            pooling_window_quarters=benford_precond.pooling_window_quarters,
            calibration_mode=benford_precond.calibration_mode,
            chi2_minimum_figures=benford_precond.chi2_minimum_figures,
            monte_carlo_replicates=benford_precond.monte_carlo_replicates,
            monte_carlo_seed=benford_precond.monte_carlo_seed,
            mc_null_cache=z1_mc_null_cache,
            capture=z1_capture,
        )
        ref_values_z1 = z1_capture.get("ref_values")
        if isinstance(ref_values_z1, np.ndarray) and ref_values_z1.size >= D1_MIN_EMPIRICAL_PIT_REF:
            z1_reference_values = ref_values_z1
        z1_settings = {
            "value_columns": tuple(benford_value_columns),
            "min_obs": benford_precond.min_figures,
            "min_obs_mc": benford_precond.min_figures_mc,
            "pooling_window_quarters": benford_precond.pooling_window_quarters,
            "calibration_mode": benford_precond.calibration_mode,
            "chi2_minimum_figures": benford_precond.chi2_minimum_figures,
            "monte_carlo_replicates": benford_precond.monte_carlo_replicates,
            "monte_carlo_seed": benford_precond.monte_carlo_seed,
        }

        zipf_precond = settings.detector_preconditions.zipf
        row_sizes_matrix = _collect_row_sizes_matrix(panel_with_ratios, benford_value_columns)
        if row_sizes_matrix.size and np.isfinite(row_sizes_matrix).any():
            sigma2_all, _ = _rolling_zipf_fits_fast(
                df=panel_with_ratios,
                row_sizes_matrix=row_sizes_matrix,
                min_points=zipf_precond.min_points,
                pooling_window_quarters=zipf_precond.pooling_window_quarters,
            )
            mask_cal_z2 = (
                panel_with_ratios["_split"] == SPLIT_LABEL_INCLUDED
                if "_split" in panel_with_ratios.columns
                else pd.Series(True, index=panel_with_ratios.index)
            )
            z2_sigma0_sq = _estimate_null_scale(
                sigma2=sigma2_all,
                mask_calibration=mask_cal_z2,
                min_reference_firms=D2_MIN_REFERENCE_FIRMS,
            )
            if z2_sigma0_sq is not None and (not np.isfinite(z2_sigma0_sq) or z2_sigma0_sq <= 0):
                z2_sigma0_sq = None
            z2_settings = {
                "value_columns": tuple(benford_value_columns),
                "min_points": zipf_precond.min_points,
                "pooling_window_quarters": zipf_precond.pooling_window_quarters,
            }

    if has_firm_quarter_cols:
        # z3, z8, z9 each used to sort/group the panel by firm independently
        # (three separate `.sort_values`/`.groupby("gvkey")` passes over the
        # ~1000+ firms). Merged into one pass here: same grouping, same row
        # order within each firm, so results are unchanged -- only the
        # repeated pandas setup (sort, groupby bookkeeping, per-group `.loc`)
        # is what's eliminated (see profiling notes: profiling showed ~66% of
        # this function's time was in that generic overhead, not the detector
        # math itself).
        merged_sorted = panel_with_ratios.sort_values(["gvkey", "datacqtr"]).copy()
        for col in canonical_ratios.columns:
            merged_sorted[col] = canonical_ratios.loc[merged_sorted.index, col].values
        if "ratio_income_cashadj" in panel_with_ratios.columns:
            merged_sorted["ratio_income_cashadj"] = pd.to_numeric(
                panel_with_ratios.loc[merged_sorted.index, "ratio_income_cashadj"],
                errors="coerce",
            ).values
        if z8_inputs_available:
            merged_sorted["_cod"] = _implied_cost_of_debt(panel_with_ratios).loc[merged_sorted.index].values

        # z3/z8 no longer need a per-firm Python loop at all (see profiling notes): ~34x/~30x faster on the full panel, verified bit-identical.
        d3_raw = _compute_mscore_raw_vectorized(merged_sorted)
        t8_all = pd.Series(np.nan, index=merged_sorted.index, dtype=float)

        if z8_inputs_available:
            firm_codes_z8, _ = pd.factorize(merged_sorted["gvkey"], sort=False)
            t8_all[:] = _t8_vectorized(merged_sorted["_cod"].to_numpy(dtype=float), firm_codes_z8)

        z3_raw_values = d3_raw.reindex(panel_with_ratios.index).to_numpy(dtype=float, copy=True)
        if "_split" in merged_sorted.columns:
            d3_ref = d3_raw[merged_sorted["_split"].astype(str) == SPLIT_LABEL_INCLUDED].dropna().to_numpy(dtype=float)
            if d3_ref.size >= D3_MIN_REF_SIZE:
                z3_reference_values = d3_ref

        if z8_inputs_available:
            z8_raw_values = t8_all.reindex(panel_with_ratios.index).to_numpy(dtype=float, copy=True)
            c_mask = (
                panel_with_ratios["_split"].values == "C"
                if "_split" in panel_with_ratios.columns
                else np.ones(len(panel_with_ratios), dtype=bool)
            )
            ref_values = z8_raw_values[c_mask & np.isfinite(z8_raw_values)]
            if ref_values.size >= MIN_REF_FIRMS:
                z8_reference_values = ref_values

    z5_context: object | None = None
    relation_frame = compute_cross_statement_relations(panel_with_ratios)
    if not relation_frame.empty:
        mask_cal = panel_with_ratios["_split"] == "C" if "_split" in panel_with_ratios.columns else pd.Series(True, index=panel_with_ratios.index)
        sector_col = _select_sector_column(panel_with_ratios)
        min_sector_reference = settings.detector_preconditions.cross_statement.min_sector_reference
        relation_refs: dict[str, object] = {}
        used_relations: list[str] = []
        for spec in RELATION_SPECS:
            raw = relation_frame[spec.name]
            global_moments = _estimate_moments(raw.loc[mask_cal], D5_MIN_GLOBAL_REFERENCE)
            sector_moments: dict[object, tuple[float, float] | None] = {}
            if sector_col is not None:
                sector_series = panel_with_ratios[sector_col]
                for sector_value, sector_idx in panel_with_ratios.groupby(sector_col, dropna=True).groups.items():
                    sector_mask = mask_cal & (sector_series == sector_value)
                    sector_ref = raw.loc[sector_mask].dropna()
                    sector_moments[sector_value] = _resolve_sector_moments(
                        sector_ref,
                        global_moments,
                        min_sector_reference=min_sector_reference,
                    )
            if global_moments is not None or any(v is not None for v in sector_moments.values()):
                relation_refs[spec.name] = {
                    "global_moments": global_moments,
                    "sector_moments": sector_moments,
                }
                used_relations.append(spec.name)
        if used_relations:
            z5_context = {
                "sector_col": sector_col,
                "used_relations": tuple(used_relations),
                "relation_refs": relation_refs,
            }

    z4_reference_values: np.ndarray | None = None
    z4_firm_raw: dict[str, float] | None = None
    z4_reference_firms: frozenset[str] | None = None
    ratios = _get_ratio_matrix(panel_with_ratios, thresholds)
    if not ratios.empty and "gvkey" in panel_with_ratios.columns:
        firm_raw = _firm_raw_statistics(panel_with_ratios, ratios, thresholds)
        ref_firms = _reference_firms(panel_with_ratios)
        z4_firm_raw = {str(k): float(v) for k, v in firm_raw.items() if np.isfinite(v)}
        z4_reference_firms = frozenset(str(fid) for fid in ref_firms)
        z4_reference_values = np.asarray(
            [v for k, v in firm_raw.items() if k in ref_firms and np.isfinite(v)],
            dtype=float,
        )
        if z4_reference_values.size < 30:
            z4_reference_values = None

    z7_context = None
    peer_ratios = canonical_ratios
    if not peer_ratios.empty and "_split" in panel_with_ratios.columns:
        z7_context = _get_global_reference(panel_with_ratios, peer_ratios)

    if global_refs is not None:
        # profiling notes fix: references extracted from `panel_with_ratios` above are
        # built on a row-restricted working panel (split C ∪ the attacked
        # firm) -- any OTHER split-C firm whose own history isn't entirely
        # in C loses its non-C quarters in that restriction, truncating its
        # rolling-window/whole-history statistics and silently corrupting
        # these reference arrays (confirmed empirically: e.g. z8's reference
        # mean was 2.3 restricted vs 9.9 on the full panel). These
        # references depend only on `base_ctx`/the full panel, never on
        # which row is being attacked, so they're computed once per run on
        # the full panel (`family3_build_global_detector_references`) and
        # substituted here. Per-row RAW values above (z3_raw_values,
        # z8_raw_values, z4_firm_raw) are left untouched -- they're already
        # correct since the attacked firm's own full history is always
        # included in the restricted panel.
        z1_reference_values = global_refs.z1_reference_values
        z1_mc_null_cache = global_refs.z1_mc_null_cache
        z1_settings = global_refs.z1_settings if global_refs.z1_settings is not None else z1_settings
        z2_sigma0_sq = global_refs.z2_sigma0_sq
        z2_settings = global_refs.z2_settings if global_refs.z2_settings is not None else z2_settings
        z3_reference_values = global_refs.z3_reference_values
        z4_reference_values = global_refs.z4_reference_values
        z8_reference_values = global_refs.z8_reference_values
        z9_reference_values = global_refs.z9_reference_values
        z5_context = global_refs.z5_context
        z7_context = global_refs.z7_context

    return (
        z3_raw_values,
        z3_reference_values,
        z5_context,
        z4_reference_values,
        z4_firm_raw,
        z4_reference_firms,
        z7_context,
        z8_raw_values,
        z8_reference_values,
        z9_reference_values,
        z1_mc_null_cache,
        z1_reference_values,
        z1_settings,
        z2_sigma0_sq,
        z2_settings,
    )


@dataclass(frozen=True)
class GlobalDetectorReferences:
    """C-split reference statistics computed once per run on the FULL panel.

    See profiling notes: `_family3_build_detector_incremental_refs` normally
    computes these on a row-restricted working panel (split C ∪ the attacked
    firm), which truncates other split-C firms' history and produces wrong
    references. These are invariant across every row of a run (they never
    depend on which row is being attacked), so building them once here on the
    complete panel is both more correct AND cheaper than the previous
    per-row recomputation.
    """

    z1_reference_values: np.ndarray | None
    z1_mc_null_cache: dict[int, np.ndarray] | None
    z1_settings: dict[str, object] | None
    z2_sigma0_sq: float | None
    z2_settings: dict[str, object] | None
    z3_reference_values: np.ndarray | None
    z4_reference_values: np.ndarray | None
    z8_reference_values: np.ndarray | None
    z9_reference_values: np.ndarray | None
    z5_context: object | None
    z7_context: object | None


def family3_build_global_detector_references(
    panel: pd.DataFrame, settings, base_ctx: ScoreContext,
) -> GlobalDetectorReferences:
    """Build the run-wide, full-panel-correct detector references (profiling notes fix).

    Call once per run (see `local_firm_single`/`global_per_row`/
    `global_cohort`) and pass the result down to every
    `family3_build_snapshot_cache` call for that run.
    """
    from src.data.ratios import compute_shariah_ratios
    from src.common.ratio_inputs import get_canonical_sharia_ratios

    panel_for_full = _family3_with_split_columns(panel, base_ctx)
    panel_with_ratios_full = compute_shariah_ratios(
        panel_for_full, log_coverage=False, warn_on_missing_connectors=False,
    )
    canonical_ratios_full = get_canonical_sharia_ratios(panel_with_ratios_full)
    # force_build_z1z2=True (see profiling notes): the z1/z2 incremental patch
    # is no longer gated behind the old whole-firm-disjoint check, so this
    # Monte-Carlo Benford null / Zipf null-scale cache is worth building once
    # here. This stays unconditional even though z1/z2 are frozen by default
    # during optimization (`family3_frozen_detectors_during_optimization`):
    # the *final* per-row snapshot used for reporting (`local_firm_single`/
    # `global_per_row`/`global_cohort`'s `final_snapshot` calls) does NOT
    # pass `frozen_detectors` -- it deliberately reports the true z1/z2
    # value for the chosen delta, not the frozen baseline used only to guide
    # the search -- so the cache is read once per row regardless of the
    # optimization-time freeze setting.
    refs = _family3_build_detector_incremental_refs(
        panel_with_ratios_full, canonical_ratios_full, settings,
        changed_index=None, force_build_z1z2=True,
    )
    return GlobalDetectorReferences(
        z1_mc_null_cache=refs[10],
        z1_reference_values=refs[11],
        z1_settings=refs[12],
        z2_sigma0_sq=refs[13],
        z2_settings=refs[14],
        z3_reference_values=refs[1],
        z4_reference_values=refs[3],
        z8_reference_values=refs[8],
        z9_reference_values=refs[9],
        z5_context=refs[2],
        z7_context=refs[6],
    )


def _family3_can_use_local_detector_patch(panel: pd.DataFrame, changed_index: pd.Index) -> bool:
    if len(changed_index) == 0 or "_split" not in panel.columns or "gvkey" not in panel.columns:
        return False
    changed_split = panel.loc[changed_index, "_split"].astype(str)
    if (changed_split == SPLIT_LABEL_INCLUDED).any():
        return False
    changed_firms = set(panel.loc[changed_index, "gvkey"].astype(str))
    c_firms = set(panel.loc[panel["_split"].astype(str) == SPLIT_LABEL_INCLUDED, "gvkey"].astype(str))
    return changed_firms.isdisjoint(c_firms)


def _owned_reindexed_copy(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Return a writable reindexed frame detached from the cache source.

    The Family 3 snapshot path selectively patches a handful of rows in cached
    intermediate tables. Using ``copy(deep=False)`` here makes those writes
    vulnerable to aliasing back into the cache-owned frame under pandas'
    block-sharing semantics, so we materialize an owned copy before mutation.
    """

    return frame.reindex(index).copy()


def _family3_patch_supported_detectors(
    *,
    raw_zscores: pd.DataFrame,
    panel_with_ratios: pd.DataFrame,
    canonical_ratios: pd.DataFrame,
    changed_index: pd.Index,
    impacted_detectors: tuple[str, ...],
    detector_cache: DetectorCache,
) -> tuple[pd.DataFrame, set[str]]:
    """Patch detectors that can be updated safely from local row/firm context.

    This function mutates ``raw_zscores`` in place and returns the same frame
    together with the subset of detectors that still require a broader recompute.
    Callers should therefore pass an owned/writable frame, not a cache view.
    """

    from src.engine.benford import BENFORD_DF, _apply_monte_carlo_pit, _rolling_benford_chi2_fast, _row_digit_counts
    from src.engine.zipf import _collect_row_sizes_matrix, _rolling_zipf_fits_fast
    from src.engine.mscore import MIN_REF_SIZE as D3_MIN_REF_SIZE, _compute_mscore_raw_vectorized
    from src.engine.coherence import compute_cross_statement_relations
    from src.engine.temporal import _t6_diagonal_vectorized
    from src.engine.proximity import _firm_raw_statistics, _get_ratio_matrix
    from src.engine.peer import _resolve_reference_for_row, _score_peer_row
    from src.engine.cost_of_debt import MIN_REF_FIRMS, _implied_cost_of_debt, _t8_vectorized
    from src.engine.seasonal_gap import _seasonal_gap_vectorized
    from src.engine.pit import pit_chi2, pit_empirical
    from src.common.methodology import thresholds_for_panel

    thresholds = thresholds_for_panel(panel_with_ratios)
    patched = raw_zscores
    remaining = set(impacted_detectors)
    has_gvkey = "gvkey" in panel_with_ratios.columns
    changed_firms = set(panel_with_ratios.loc[changed_index, "gvkey"].astype(str).tolist()) if has_gvkey else set()
    # `.astype(str).isin(...)` on the full ~79k-row `gvkey` column was being
    # recomputed independently in every detector block below (z3/z5/z6/z8/z9)
    # -- 5-6x the same full-column string conversion per call to this
    # function, which happens on every optimization replay step. Computed
    # once here, on the native dtype (no per-row string conversion), and
    # reused everywhere (see profiling notes).
    if has_gvkey and changed_firms:
        changed_firms_native = set(panel_with_ratios.loc[changed_index, "gvkey"].tolist())
        changed_firms_mask = panel_with_ratios["gvkey"].isin(changed_firms_native)
    else:
        changed_firms_mask = pd.Series(False, index=panel_with_ratios.index)
    # z1/z2 (see profiling notes): the C-reference they calibrate against is
    # now a run-wide constant built once on the full, unperturbed panel
    # (`global_refs`, see profiling notes) -- unlike the old whole-firm-disjoint gate,
    # perturbing a firm that merely *has* other rows in split C is no longer
    # a special risk (that reference never gets rebuilt mid-run regardless of
    # which firm is touched -- see profiling notes for the residual, bounded staleness
    # this shares with z3/z4/z5/z7/z8/z9, which have never been gated at
    # all). The one thing that still must hold: the changed row itself must
    # not be a split-C member (patching it would need to also update the
    # reference it belongs to, which this mechanism never does).
    changed_rows_in_c = (
        bool((panel_with_ratios.loc[changed_index, "_split"].astype(str) == SPLIT_LABEL_INCLUDED).any())
        if "_split" in panel_with_ratios.columns
        else False
    )

    if "z3" in remaining and detector_cache.z3_raw_values is not None and changed_firms:
        target_panel = panel_with_ratios.loc[changed_firms_mask].copy()
        if {"gvkey", "datacqtr"}.issubset(target_panel.columns):
            for col in canonical_ratios.columns:
                if col not in target_panel.columns:
                    target_panel[col] = canonical_ratios.loc[target_panel.index, col].values
            if "ratio_income_cashadj" in panel_with_ratios.columns:
                target_panel["ratio_income_cashadj"] = pd.to_numeric(
                    panel_with_ratios.loc[target_panel.index, "ratio_income_cashadj"],
                    errors="coerce",
                ).values
            target_sorted = target_panel.sort_values(["gvkey", "datacqtr"]).copy()
            updated_raw = np.asarray(detector_cache.z3_raw_values, dtype=float).copy()
            # Vectorized across all changed firms at once instead of looping
            # per firm (see profiling notes: `_compute_mscore_raw_vectorized`
            # is the same already-verified bit-identical implementation used
            # for the full-panel case).
            raw_series = _compute_mscore_raw_vectorized(target_sorted)
            row_indices = target_sorted.index.to_numpy()
            row_positions = panel_with_ratios.index.get_indexer(row_indices)
            valid_positions = row_positions >= 0
            if valid_positions.any():
                updated_raw[row_positions[valid_positions]] = raw_series.to_numpy(dtype=float)[valid_positions]
            c_mask = (
                panel_with_ratios["_split"].astype(str).to_numpy() == SPLIT_LABEL_INCLUDED
                if "_split" in panel_with_ratios.columns
                else np.ones(len(panel_with_ratios), dtype=bool)
            )
            ref_values = updated_raw[c_mask & np.isfinite(updated_raw)]
            if ref_values.size >= D3_MIN_REF_SIZE:
                z3_all = np.full(len(updated_raw), np.nan, dtype=float)
                finite_mask = np.isfinite(updated_raw)
                if finite_mask.any():
                    z3_all[finite_mask] = pit_empirical(updated_raw[finite_mask], ref_values)
                patched["z3"] = pd.Series(z3_all, index=panel_with_ratios.index)
                remaining.discard("z3")

    if "z5" in remaining and detector_cache.z5_context is not None and changed_firms:
        target_panel = panel_with_ratios.loc[changed_firms_mask]
        relation_frame = compute_cross_statement_relations(target_panel)
        z5_context = detector_cache.z5_context
        sector_col = z5_context["sector_col"]
        used_relations = z5_context["used_relations"]
        relation_refs = z5_context["relation_refs"]
        z_rel = pd.DataFrame(index=target_panel.index)
        for relation_name in used_relations:
            if relation_name not in relation_frame.columns:
                continue
            raw = pd.to_numeric(relation_frame[relation_name], errors="coerce")
            rel_ref = relation_refs.get(relation_name, {})
            global_moments = rel_ref.get("global_moments")
            sector_moments = rel_ref.get("sector_moments", {})
            z_raw = pd.Series(np.nan, index=target_panel.index, dtype=float)
            if sector_col is None or sector_col not in target_panel.columns:
                if global_moments is not None:
                    mu, sigma = global_moments
                    valid = raw.notna()
                    z_raw.loc[valid] = (raw.loc[valid] - mu) / sigma
            else:
                sector_series = target_panel[sector_col]
                for sector_value, sector_idx in target_panel.groupby(sector_col, dropna=True).groups.items():
                    moments = sector_moments.get(sector_value)
                    if moments is None:
                        moments = global_moments
                    if moments is None:
                        continue
                    mu, sigma = moments
                    idx = pd.Index(sector_idx)
                    valid = raw.loc[idx].notna()
                    valid_idx = idx[valid.to_numpy()]
                    z_raw.loc[valid_idx] = (raw.loc[valid_idx] - mu) / sigma
                if global_moments is not None:
                    mu, sigma = global_moments
                    remaining_mask = z_raw.isna() & raw.notna()
                    if remaining_mask.any():
                        z_raw.loc[remaining_mask] = (raw.loc[remaining_mask] - mu) / sigma
            if z_raw.notna().any():
                z_rel[relation_name] = z_raw
        if not z_rel.empty:
            z_values = z_rel.to_numpy(dtype=float)
            valid_mask = np.isfinite(z_values)
            q_eff = valid_mask.sum(axis=1)
            squared = np.where(valid_mask, z_values, 0.0) ** 2
            t5 = np.where(q_eff > 0, squared.sum(axis=1), np.nan)
            z5 = np.full(len(target_panel), np.nan, dtype=float)
            for q in sorted(set(q_eff[q_eff > 0])):
                mask = q_eff == q
                if mask.any():
                    z5[mask] = pit_chi2(t5[mask], degrees_of_freedom=int(q))
            patched.loc[target_panel.index, "z5"] = pd.Series(z5, index=target_panel.index)
            remaining.discard("z5")

    if (
        "z4" in remaining
        and detector_cache.z4_reference_values is not None
        and detector_cache.z4_firm_raw is not None
        and detector_cache.z4_reference_firms is not None
        and changed_firms
    ):
        ratios = _get_ratio_matrix(panel_with_ratios, thresholds)
        target_panel = panel_with_ratios.loc[changed_firms_mask].copy()
        target_ratios = ratios.loc[target_panel.index]
        firm_raw = _firm_raw_statistics(target_panel, target_ratios, thresholds)
        firm_ids = [k for k, v in firm_raw.items() if np.isfinite(v)]
        if firm_ids:
            raw_vals = np.asarray([firm_raw[k] for k in firm_ids], dtype=float)
            updated_firm_raw = dict(detector_cache.z4_firm_raw)
            updated_firm_raw.update({str(k): float(v) for k, v in firm_raw.items() if np.isfinite(v)})
            reference_values = np.asarray(
                [
                    v
                    for k, v in updated_firm_raw.items()
                    if k in detector_cache.z4_reference_firms and np.isfinite(v)
                ],
                dtype=float,
            )
            if reference_values.size >= 30:
                z_vals = pit_empirical(raw_vals, reference_values)
            else:
                z_vals = pit_empirical(raw_vals, detector_cache.z4_reference_values)
            firm_z = pd.Series(z_vals, index=firm_ids)
            firm_mask = changed_firms_mask
            patched.loc[firm_mask, "z4"] = panel_with_ratios.loc[firm_mask, "gvkey"].astype(str).map(firm_z)
            remaining.discard("z4")

    if "z7" in remaining and detector_cache.z7_context is not None:
        peer_ratios = canonical_ratios
        n_dims = len(peer_ratios.columns)
        if n_dims > 0:
            z7_vals: dict[object, float] = {}
            for idx in changed_index:
                if idx not in peer_ratios.index:
                    continue
                row = peer_ratios.loc[idx].to_numpy(dtype=float)
                if not np.isfinite(row).all():
                    z7_vals[idx] = np.nan
                    continue
                resolved = _resolve_reference_for_row(idx, detector_cache.z7_context)
                if resolved is None:
                    z7_vals[idx] = np.nan
                    continue
                mu_ref, inv_ref, n_ref, _ = resolved
                z7_vals[idx] = _score_peer_row(row, mu_ref, inv_ref, n_ref, n_dims)
            if z7_vals:
                patched.loc[list(z7_vals.keys()), "z7"] = pd.Series(z7_vals)
                remaining.discard("z7")

    if "z6" in remaining and changed_firms:
        ratios = canonical_ratios
        if not ratios.empty and {"gvkey", "datacqtr"}.issubset(panel_with_ratios.columns):
            # Restrict to changed firms *before* sorting/grouping (mirrors the
            # z3/z8 patch blocks above) -- avoids building groupby bookkeeping
            # for all ~1000+ firms just to skip almost all of them. Keeping
            # every row of a changed firm (not just the edited row) preserves
            # the full rolling history the diagonal-mode computation needs,
            # so results are unchanged (see profiling notes).
            target_panel_z6 = panel_with_ratios.loc[changed_firms_mask]
            df_sorted = target_panel_z6[["gvkey", "datacqtr"]].copy()
            df_sorted["_row_index"] = target_panel_z6.index.to_numpy()
            df_sorted = df_sorted.sort_values(["gvkey", "datacqtr"])
            ratios_sorted = ratios.loc[df_sorted.index]
            # Vectorized across all changed firms at once (see profiling notes: `_t6_diagonal_vectorized` is the same
            # already-verified bit-identical implementation used for the
            # full-panel "diagonal" mode, which is the only mode active here).
            firm_codes_z6, _ = pd.factorize(df_sorted["gvkey"], sort=False)
            t6_all, dims_all = _t6_diagonal_vectorized(ratios_sorted.to_numpy(dtype=float), firm_codes_z6)
            z6_all = np.full(len(t6_all), np.nan, dtype=float)
            finite_mask = np.isfinite(t6_all)
            if finite_mask.any():
                for p_eff in sorted(set(dims_all[finite_mask])):
                    if p_eff <= 0:
                        continue
                    mask = finite_mask & (dims_all == p_eff)
                    if mask.any():
                        z6_all[mask] = pit_chi2(t6_all[mask], degrees_of_freedom=int(p_eff))
            row_indices = df_sorted["_row_index"].to_numpy()
            z6_updates = {row_idx: (float(z6_all[pos]) if np.isfinite(z6_all[pos]) else np.nan) for pos, row_idx in enumerate(row_indices)}
            if z6_updates:
                patched.loc[list(z6_updates.keys()), "z6"] = pd.Series(z6_updates)
                remaining.discard("z6")

    if "z8" in remaining and detector_cache.z8_raw_values is not None and changed_firms:
        updated_t8 = np.asarray(detector_cache.z8_raw_values, dtype=float).copy()
        target_panel = panel_with_ratios.loc[changed_firms_mask].copy()
        cod = _implied_cost_of_debt(target_panel)
        df_sorted = target_panel[["gvkey", "datacqtr"]].copy()
        df_sorted["_row_index"] = target_panel.index.to_numpy()
        df_sorted["_cod"] = cod.values
        df_sorted = df_sorted.sort_values(["gvkey", "datacqtr"])
        # Vectorized across all changed firms at once (see profiling notes: `_t8_vectorized` is the same already-verified
        # bit-identical implementation used for the full-panel case).
        firm_codes_z8_patch, _ = pd.factorize(df_sorted["gvkey"], sort=False)
        t8_all = _t8_vectorized(df_sorted["_cod"].to_numpy(dtype=float), firm_codes_z8_patch)
        row_indices = df_sorted["_row_index"].to_numpy()
        row_positions = panel_with_ratios.index.get_indexer(row_indices)
        valid_positions = row_positions >= 0
        if valid_positions.any():
            updated_t8[row_positions[valid_positions]] = t8_all[valid_positions]
        c_mask = (
            panel_with_ratios["_split"].astype(str).to_numpy() == SPLIT_LABEL_INCLUDED
            if "_split" in panel_with_ratios.columns
            else np.ones(len(panel_with_ratios), dtype=bool)
        )
        ref_values = updated_t8[c_mask & np.isfinite(updated_t8)]
        if ref_values.size >= MIN_REF_FIRMS:
            z8_all = np.full(len(updated_t8), np.nan, dtype=float)
            finite_mask = np.isfinite(updated_t8)
            if finite_mask.any():
                z8_all[finite_mask] = pit_empirical(updated_t8[finite_mask], ref_values)
            patched["z8"] = pd.Series(z8_all, index=panel_with_ratios.index)
            remaining.discard("z8")

    if not changed_rows_in_c and "z9" in remaining and detector_cache.z9_reference_values is not None and changed_firms:
        available_ratios = [c for c in thresholds if c in panel_with_ratios.columns]
        if available_ratios and {"gvkey", "fqtr"}.issubset(panel_with_ratios.columns):
            # Restrict to changed firms *before* computing (mirrors the
            # z3/z6/z8 patch blocks above) and use the already-vectorized,
            # bit-identical `_seasonal_gap_vectorized` (profiling notes) instead of
            # looping every firm in the panel just to skip almost all of
            # them (see profiling notes).
            target_panel_z9 = panel_with_ratios.loc[changed_firms_mask]
            t9_broadcast = _seasonal_gap_vectorized(target_panel_z9, thresholds)
            firm_raw_z9 = {
                str(gid): float(val)
                for gid, val in t9_broadcast.groupby(target_panel_z9["gvkey"].astype(str), sort=False).first().items()
                if np.isfinite(val)
            }
            if firm_raw_z9:
                firm_ids = list(firm_raw_z9.keys())
                raw_vals = np.asarray([firm_raw_z9[k] for k in firm_ids], dtype=float)
                z_vals = pit_empirical(raw_vals, detector_cache.z9_reference_values)
                firm_z = pd.Series(z_vals, index=firm_ids)
                firm_mask = changed_firms_mask
                patched.loc[firm_mask, "z9"] = panel_with_ratios.loc[firm_mask, "gvkey"].astype(str).map(firm_z)
                remaining.discard("z9")

    if (
        not changed_rows_in_c
        and "z1" in remaining
        and detector_cache.z1_mc_null_cache is not None
        and detector_cache.z1_reference_values is not None
        and detector_cache.z1_settings is not None
        and changed_firms
    ):
        z1_cfg = detector_cache.z1_settings
        target_panel = panel_with_ratios.loc[changed_firms_mask]
        row_counts, row_nfig = _row_digit_counts(target_panel, z1_cfg["value_columns"])
        if int(row_nfig.sum()) > 0:
            mode = z1_cfg["calibration_mode"]
            effective_floor = z1_cfg["min_obs_mc"] if mode in {"monte_carlo", "auto"} else z1_cfg["min_obs"]
            chi2_stat, nfig = _rolling_benford_chi2_fast(
                df=target_panel,
                row_counts=row_counts,
                row_nfig=row_nfig,
                min_obs=effective_floor,
                pooling_window_quarters=z1_cfg["pooling_window_quarters"],
            )
            finite = chi2_stat.notna()
            if finite.any():
                # Mirror detect_benford's exact per-mode masks (chi2 mask is
                # empty under "monte_carlo", mc mask is empty under
                # "chi2_asymptotic", so the same code path below covers all
                # three modes without duplicating the PIT logic per mode).
                if mode == "chi2_asymptotic":
                    chi2_mask_z1 = finite & (nfig >= z1_cfg["min_obs"])
                    mc_mask_z1 = pd.Series(False, index=target_panel.index)
                elif mode == "monte_carlo":
                    chi2_mask_z1 = pd.Series(False, index=target_panel.index)
                    mc_mask_z1 = finite & (nfig >= z1_cfg["min_obs_mc"])
                else:  # auto
                    chi2_mask_z1 = finite & (nfig >= z1_cfg["chi2_minimum_figures"])
                    mc_mask_z1 = finite & (nfig >= z1_cfg["min_obs_mc"]) & ~chi2_mask_z1

                # Only patch if every N_fig this firm's MC-eligible rows need
                # was already drawn during the snapshot-cache build -- a
                # brand-new N_fig here would need a fresh RNG draw that
                # can't be proven bit-identical to a full recompute, so fall
                # back to the slow-but-correct compute_zscores path instead
                # (see profiling notes and the profiling notes cautionary tale: never
                # guess when correctness can't be proven).
                can_patch_z1 = True
                if mc_mask_z1.any():
                    needed_n = {int(x) for x in np.unique(nfig[mc_mask_z1].to_numpy(dtype=int))}
                    if not needed_n.issubset(detector_cache.z1_mc_null_cache.keys()):
                        can_patch_z1 = False

                if can_patch_z1:
                    z1_new = pd.Series(np.nan, index=target_panel.index, dtype=float)
                    if chi2_mask_z1.any():
                        z1_new.loc[chi2_mask_z1] = pit_chi2(
                            chi2_stat.loc[chi2_mask_z1].to_numpy(dtype=float), degrees_of_freedom=BENFORD_DF,
                        )
                    if mc_mask_z1.any():
                        mc_z1 = _apply_monte_carlo_pit(
                            chi2=chi2_stat, nfig=nfig, mask=mc_mask_z1.to_numpy(dtype=bool),
                            n_replicates=z1_cfg["monte_carlo_replicates"], seed=z1_cfg["monte_carlo_seed"],
                            null_cache=detector_cache.z1_mc_null_cache,
                        )
                        z1_new = z1_new.where(~mc_mask_z1, mc_z1)
                    finite_z1 = np.isfinite(z1_new)
                    if finite_z1.any():
                        z1_new.loc[finite_z1] = pit_empirical(
                            z1_new.loc[finite_z1].to_numpy(dtype=float), detector_cache.z1_reference_values,
                        )
                    patched.loc[target_panel.index, "z1"] = z1_new
                    remaining.discard("z1")

    if (
        not changed_rows_in_c
        and "z2" in remaining
        and detector_cache.z2_sigma0_sq is not None
        and detector_cache.z2_settings is not None
        and changed_firms
    ):
        z2_cfg = detector_cache.z2_settings
        target_panel = panel_with_ratios.loc[changed_firms_mask]
        row_sizes_matrix_z2 = _collect_row_sizes_matrix(target_panel, z2_cfg["value_columns"])
        if row_sizes_matrix_z2.size and np.isfinite(row_sizes_matrix_z2).any():
            sigma2_new, df_resid_new = _rolling_zipf_fits_fast(
                df=target_panel,
                row_sizes_matrix=row_sizes_matrix_z2,
                min_points=z2_cfg["min_points"],
                pooling_window_quarters=z2_cfg["pooling_window_quarters"],
            )
            valid_z2 = np.isfinite(sigma2_new) & np.isfinite(df_resid_new) & (df_resid_new > 0)
            if valid_z2.any():
                z2_new = pd.Series(np.nan, index=target_panel.index, dtype=float)
                for dof in sorted(df_resid_new.loc[valid_z2].astype(int).unique()):
                    dof_mask = valid_z2 & (df_resid_new.astype("Int64") == dof)
                    if not dof_mask.any():
                        continue
                    chi2_stat_z2 = (df_resid_new.loc[dof_mask] * sigma2_new.loc[dof_mask]) / detector_cache.z2_sigma0_sq
                    z2_new.loc[dof_mask] = pit_chi2(chi2_stat_z2.to_numpy(dtype=float), degrees_of_freedom=int(dof))
                patched.loc[target_panel.index, "z2"] = z2_new
                remaining.discard("z2")

    return patched, remaining


def family3_impacted_detectors(settings, changed_columns: tuple[str, ...] | None, frozen_detectors: tuple[str, ...] | None = None) -> tuple[str, ...]:
    raw_settings = family3_raw_zscore_settings(settings)
    if not changed_columns:
        impacted = tuple(raw_settings.zscores.include_detectors)
    else:
        changed = set(changed_columns)
        dependencies = family3_detector_dependencies(settings)
        impacted = tuple(
            detector_name
            for detector_name in raw_settings.zscores.include_detectors
            if dependencies.get(detector_name) is None or changed.intersection(dependencies.get(detector_name, set()))
        )
    if frozen_detectors:
        frozen = set(frozen_detectors)
        impacted = tuple(det for det in impacted if det not in frozen)
    return impacted


def family3_merge_zscores(
    raw_zscores: pd.DataFrame,
    panel: pd.DataFrame,
    settings,
    *,
    changed_index: pd.Index | None = None,
    merged_cache: pd.DataFrame | None = None,
    merged_buffer: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from src.engine.pit import pit_empirical

    changed_index = pd.Index(changed_index) if changed_index is not None else pd.Index([])
    if merged_buffer is not None:
        merged = merged_buffer
        if len(changed_index) > 0 and merged_cache is not None:
            merged.loc[changed_index, :] = merged_cache.loc[changed_index, :].values
    else:
        merged = merged_cache.reindex(raw_zscores.index).copy() if merged_cache is not None else raw_zscores.copy()
    dropped_inputs: set[str] = set()
    for rule in settings.zscores.merge_rules:
        missing = [col for col in rule.inputs if col not in merged.columns]
        source_missing = [col for col in rule.inputs if col not in raw_zscores.columns]
        if missing and source_missing:
            continue
        can_do_row_local = (
            len(changed_index) > 0
            and merged_cache is not None
            and (
                not rule.empirical_pit_on_c
                or "_split" not in panel.columns
                or not (panel.loc[changed_index, "_split"].astype(str) == SPLIT_LABEL_INCLUDED).any()
            )
        )
        target_index = changed_index if can_do_row_local else raw_zscores.index
        input_matrix = raw_zscores.loc[target_index, list(rule.inputs)].to_numpy(dtype=float)
        reduced = _rowwise_nanreduce(input_matrix, method=rule.method)
        if rule.empirical_pit_on_c and "_split" in panel.columns:
            if can_do_row_local and rule.output_name in merged_cache.columns:
                ref = pd.to_numeric(
                    merged_cache.loc[
                        panel["_split"].astype(str) == SPLIT_LABEL_INCLUDED,
                        rule.output_name,
                    ],
                    errors="coerce",
                ).to_numpy(dtype=float)
            else:
                full_inputs = raw_zscores.loc[:, list(rule.inputs)].to_numpy(dtype=float)
                full_reduced = _rowwise_nanreduce(full_inputs, method=rule.method)
                ref = full_reduced[panel["_split"].to_numpy() == SPLIT_LABEL_INCLUDED]
            ref = ref[np.isfinite(ref)]
            if ref.size >= 30:
                finite = np.isfinite(reduced)
                reduced = reduced.copy()
                reduced[finite] = pit_empirical(reduced[finite], ref)
        if can_do_row_local:
            merged.loc[target_index, rule.output_name] = reduced
        else:
            merged[rule.output_name] = reduced
        if rule.drop_inputs:
            dropped_inputs.update(rule.inputs)
    if dropped_inputs:
        cols_to_drop = [col for col in dropped_inputs if col in merged.columns]
        if cols_to_drop:
            if merged_buffer is not None:
                merged.drop(columns=cols_to_drop, inplace=True)
            else:
                merged = merged.drop(columns=cols_to_drop)
    return merged


def family3_composites_from_zscores(
    panel: pd.DataFrame,
    merged_zscores: pd.DataFrame,
    settings,
    *,
    base_ctx: ScoreContext,
    changed_index: pd.Index | None = None,
    composites_cache: pd.DataFrame | None = None,
    composites_buffer: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from src.analysis.composite_scoring import _attach_pvalues, _composite_columns

    schema = settings.panel_schema
    changed_index = pd.Index(changed_index) if changed_index is not None else pd.Index([])
    if len(changed_index) > 0 and composites_cache is not None:
        composites = composites_buffer if composites_buffer is not None else composites_cache.reindex(panel.index).copy()
        composites.loc[changed_index, :] = composites_cache.loc[changed_index, :].values
        z_matrix = merged_zscores.loc[changed_index, list(base_ctx.active)].to_numpy(dtype=float)
        composite_cols = _composite_columns(z_matrix, base_ctx.sigma, base_ctx.weights, settings)
        pvalue_cols = _attach_pvalues(composite_cols, base_ctx.null)
        composites.loc[changed_index, schema.firm_id] = panel.loc[changed_index, schema.firm_id].values
        composites.loc[changed_index, schema.quarter] = panel.loc[changed_index, schema.quarter].values
        for name, arr in composite_cols.items():
            composites.loc[changed_index, name] = arr
        for name, arr in pvalue_cols.items():
            composites.loc[changed_index, name] = arr
        return composites
    z_matrix = merged_zscores.loc[:, list(base_ctx.active)].to_numpy(dtype=float)
    composite_cols = _composite_columns(z_matrix, base_ctx.sigma, base_ctx.weights, settings)
    pvalue_cols = _attach_pvalues(composite_cols, base_ctx.null)
    composites = pd.DataFrame({schema.firm_id: panel[schema.firm_id].values, schema.quarter: panel[schema.quarter].values}, index=panel.index)
    for name, arr in composite_cols.items():
        composites[name] = arr
    for name, arr in pvalue_cols.items():
        composites[name] = arr
    return composites


def family3_snapshot(
    panel: pd.DataFrame,
    settings,
    *,
    base_ctx: ScoreContext,
    changed_columns: tuple[str, ...] | None = None,
    frozen_detectors: tuple[str, ...] | None = None,
    changed_index: list[object] | tuple[object, ...] | pd.Index | None = None,
    snapshot_cache: SnapshotCache | None = None,
    ratio_buffer: pd.DataFrame | None = None,
    raw_zscores_buffer: pd.DataFrame | None = None,
    merged_zscores_buffer: pd.DataFrame | None = None,
    composites_buffer: pd.DataFrame | None = None,
    perf_stats: dict[str, object] | None = None,
) -> Family3Snapshot:
    from src.data.ratios import compute_shariah_ratios
    from src.engine.zscores import compute_zscores
    from src.common.ratio_inputs import get_canonical_sharia_ratios

    changed_index = pd.Index(changed_index) if changed_index is not None else pd.Index([])

    # Baseline fast path: nothing has been perturbed yet and no incremental
    # cache/buffers were supplied, so base_ctx's own zscores/composites are
    # already the correct answer for this exact panel. Without this shortcut,
    # `changed_columns=None` falls through to `family3_impacted_detectors`,
    # which treats "unspecified" as "assume every detector changed" and
    # recomputes the full 8-detector suite over the whole panel just to
    # reproduce numbers base_ctx already has (~130s on the real MYS panel,
    # paid once per Family 3 run regardless of how many rows are attacked).
    if (
        changed_columns is None
        and len(changed_index) == 0
        and snapshot_cache is None
        and ratio_buffer is None
        and raw_zscores_buffer is None
        and merged_zscores_buffer is None
        and composites_buffer is None
    ):
        if perf_stats is not None:
            perf_stats.clear()
            perf_stats["changed_rows"] = 0
            perf_stats["path"] = "baseline_reuse"
        return Family3Snapshot(
            panel=base_ctx.panel.copy(),
            raw_zscores=_owned_reindexed_copy(base_ctx.raw_zscores, base_ctx.panel.index),
            merged_zscores=_owned_reindexed_copy(base_ctx.zscores, base_ctx.panel.index),
            composites=_owned_reindexed_copy(base_ctx.composites, base_ctx.panel.index),
        )

    split_cols = [c for c in ("_split", "_split_reason") if c in base_ctx.panel.columns]
    panel_for_snapshot = _family3_with_split_columns(panel, base_ctx)

    if perf_stats is not None:
        perf_stats.clear()
        perf_stats["changed_rows"] = int(len(changed_index))
        perf_stats["path"] = "unknown"
        perf_stats["impacted_detectors"] = []
        perf_stats["partial_remaining_detectors"] = []
        perf_stats["compute_zscores_mode"] = "none"
    ratio_start = time.perf_counter()
    if snapshot_cache is not None and len(changed_index) > 0:
        if ratio_buffer is not None:
            panel_with_ratios = ratio_buffer
            panel_with_ratios.loc[changed_index, :] = snapshot_cache.ratio.panel_with_ratios.loc[changed_index, :].values
        else:
            panel_with_ratios = _owned_reindexed_copy(
                snapshot_cache.ratio.panel_with_ratios,
                panel_for_snapshot.index,
            )
        raw_patch_cols = [col for col in changed_columns or tuple() if col in panel_for_snapshot.columns]
        if raw_patch_cols:
            panel_with_ratios.loc[changed_index, raw_patch_cols] = panel_for_snapshot.loc[changed_index, raw_patch_cols]
        local_recompute = compute_shariah_ratios(
            panel_for_snapshot.loc[changed_index],
            log_coverage=False,
            warn_on_missing_connectors=False,
        )
        ratio_patch_cols = [col for col in snapshot_cache.ratio.ratio_columns if col in local_recompute.columns]
        if ratio_patch_cols:
            panel_with_ratios.loc[changed_index, ratio_patch_cols] = local_recompute.loc[changed_index, ratio_patch_cols]
    else:
        panel_with_ratios = compute_shariah_ratios(panel_for_snapshot, log_coverage=False, warn_on_missing_connectors=False)
    for col in split_cols:
        if col not in panel_with_ratios.columns:
            panel_with_ratios[col] = panel_for_snapshot[col]
    if snapshot_cache is not None and len(changed_index) > 0:
        canonical_ratios = _owned_reindexed_copy(
            snapshot_cache.ratio.canonical_ratios,
            panel_with_ratios.index,
        )
        changed_ratio_cols = [col for col in canonical_ratios.columns if col in panel_with_ratios.columns]
        if changed_ratio_cols:
            canonical_ratios.loc[changed_index, changed_ratio_cols] = panel_with_ratios.loc[changed_index, changed_ratio_cols].values
    else:
        canonical_ratios = get_canonical_sharia_ratios(panel_with_ratios)
    if perf_stats is not None:
        perf_stats["ratio_stage_seconds"] = time.perf_counter() - ratio_start
    raw_settings = family3_raw_zscore_settings(settings)
    impacted_detectors = family3_impacted_detectors(settings, changed_columns, frozen_detectors)
    if perf_stats is not None:
        perf_stats["impacted_detectors"] = list(impacted_detectors)
    zscore_start = time.perf_counter()
    if not impacted_detectors:
        if perf_stats is not None:
            perf_stats["path"] = "reuse_only"
        if snapshot_cache is not None:
            raw_zscores = (
                raw_zscores_buffer
                if raw_zscores_buffer is not None
                else _owned_reindexed_copy(snapshot_cache.detector.raw_zscores, panel_with_ratios.index)
            )
            merged_zscores = (
                merged_zscores_buffer
                if merged_zscores_buffer is not None
                else _owned_reindexed_copy(snapshot_cache.composite.merged_zscores, panel_with_ratios.index)
            )
            composites = (
                composites_buffer
                if composites_buffer is not None
                else _owned_reindexed_copy(snapshot_cache.composite.composites, panel_with_ratios.index)
            )
            if len(changed_index) > 0:
                raw_zscores.loc[changed_index, :] = snapshot_cache.detector.raw_zscores.loc[changed_index, :].values
                merged_zscores.loc[changed_index, :] = snapshot_cache.composite.merged_zscores.loc[changed_index, :].values
                composites.loc[changed_index, :] = snapshot_cache.composite.composites.loc[changed_index, :].values
            return Family3Snapshot(panel=panel_with_ratios, raw_zscores=raw_zscores, merged_zscores=merged_zscores, composites=composites)
        raw_zscores = _owned_reindexed_copy(base_ctx.raw_zscores, panel_with_ratios.index)
    elif snapshot_cache is not None and len(changed_index) > 0:
        if raw_zscores_buffer is not None:
            raw_zscores = raw_zscores_buffer
            raw_zscores.loc[changed_index, :] = snapshot_cache.detector.raw_zscores.loc[changed_index, :].values
        else:
            raw_zscores = _owned_reindexed_copy(snapshot_cache.detector.raw_zscores, panel_with_ratios.index)
        raw_zscores, remaining = _family3_patch_supported_detectors(
            raw_zscores=raw_zscores,
            panel_with_ratios=panel_with_ratios,
            canonical_ratios=canonical_ratios,
            changed_index=changed_index,
            impacted_detectors=impacted_detectors,
            detector_cache=snapshot_cache.detector,
        )
        impacted_detectors = tuple(sorted(remaining))
        if perf_stats is not None:
            # "local_patch": every impacted detector was patched incrementally
            # (see profiling notes -- z1/z2/z9 no longer need the changed
            # firm to be whole-firm-disjoint from split C, only the changed
            # row itself must not be a split-C member). "mixed_local_partial"
            # means at least one detector still needed the compute_zscores
            # fallback below (e.g. a brand-new Benford N_fig not covered by
            # the cached Monte-Carlo null, or the changed row is itself in C).
            perf_stats["path"] = "local_patch" if not impacted_detectors else "mixed_local_partial"
            perf_stats["partial_remaining_detectors"] = list(impacted_detectors)
        if impacted_detectors:
            if perf_stats is not None:
                perf_stats["compute_zscores_mode"] = "partial_after_local_patch"
            partial_settings = raw_settings.model_copy(
                update={"zscores": raw_settings.zscores.model_copy(update={"include_detectors": impacted_detectors})}
            )
            partial_raw = compute_zscores(panel=panel_with_ratios, settings=partial_settings, write_outputs=False).zscores
            for col in [c for c in partial_raw.columns if c.startswith("z")]:
                if len(changed_index) > 0:
                    raw_zscores.loc[changed_index, col] = partial_raw.loc[changed_index, col]
                else:
                    raw_zscores[col] = partial_raw[col].reindex(panel_with_ratios.index)
            schema = settings.panel_schema
            raw_zscores[schema.firm_id] = panel_with_ratios[schema.firm_id].values
            raw_zscores[schema.quarter] = panel_with_ratios[schema.quarter].values
    elif len(impacted_detectors) == len(raw_settings.zscores.include_detectors):
        if perf_stats is not None:
            perf_stats["path"] = "full_recompute"
            perf_stats["compute_zscores_mode"] = "full"
        full_raw = compute_zscores(panel=panel_with_ratios, settings=raw_settings, write_outputs=False).zscores
        if raw_zscores_buffer is not None:
            raw_zscores = raw_zscores_buffer
            raw_zscores.loc[:, :] = full_raw.loc[raw_zscores.index, raw_zscores.columns].values
        else:
            raw_zscores = full_raw
    else:
        if perf_stats is not None:
            perf_stats["path"] = "partial_recompute"
            perf_stats["compute_zscores_mode"] = "partial"
        log.info("family3_exact: partial detector recompute %s (changed_columns=%s)", impacted_detectors, tuple(changed_columns or ()))
        partial_settings = raw_settings.model_copy(
            update={"zscores": raw_settings.zscores.model_copy(update={"include_detectors": impacted_detectors})}
        )
        partial_raw = compute_zscores(panel=panel_with_ratios, settings=partial_settings, write_outputs=False).zscores
        if snapshot_cache is not None and raw_zscores_buffer is not None:
            raw_zscores = raw_zscores_buffer
            if len(changed_index) > 0:
                raw_zscores.loc[changed_index, :] = snapshot_cache.detector.raw_zscores.loc[changed_index, :].values
        else:
            raw_zscores = (
                _owned_reindexed_copy(snapshot_cache.detector.raw_zscores, panel_with_ratios.index)
                if snapshot_cache is not None
                else _owned_reindexed_copy(base_ctx.raw_zscores, panel_with_ratios.index)
            )
        for col in [c for c in partial_raw.columns if c.startswith("z")]:
            raw_zscores[col] = partial_raw[col].reindex(panel_with_ratios.index)
        schema = settings.panel_schema
        raw_zscores[schema.firm_id] = panel_with_ratios[schema.firm_id].values
        raw_zscores[schema.quarter] = panel_with_ratios[schema.quarter].values
    if perf_stats is not None:
        perf_stats["zscore_stage_seconds"] = time.perf_counter() - zscore_start
    merge_start = time.perf_counter()
    merged_zscores = family3_merge_zscores(
        raw_zscores,
        panel_with_ratios,
        settings,
        changed_index=changed_index,
        merged_cache=snapshot_cache.composite.merged_zscores if snapshot_cache is not None else None,
        merged_buffer=merged_zscores_buffer,
    )
    if perf_stats is not None:
        perf_stats["merge_stage_seconds"] = time.perf_counter() - merge_start
    composite_start = time.perf_counter()
    composites = family3_composites_from_zscores(
        panel_with_ratios,
        merged_zscores,
        settings,
        base_ctx=base_ctx,
        changed_index=changed_index,
        composites_cache=snapshot_cache.composite.composites if snapshot_cache is not None else None,
        composites_buffer=composites_buffer,
    )
    if perf_stats is not None:
        perf_stats["composite_stage_seconds"] = time.perf_counter() - composite_start
    return Family3Snapshot(panel=panel_with_ratios, raw_zscores=raw_zscores, merged_zscores=merged_zscores, composites=composites)


def family3_candidate_panel(base_panel: pd.DataFrame, row_indices: list[object], candidate_matrix: np.ndarray, columns: tuple[str, ...]) -> pd.DataFrame:
    updated = base_panel.copy()
    return family3_apply_candidate_inplace(updated, row_indices, candidate_matrix, columns)


def family3_apply_candidate_inplace(
    panel: pd.DataFrame,
    row_indices: list[object] | tuple[object, ...],
    candidate_matrix: np.ndarray,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    valid_cols = [col for col in columns if col in panel.columns]
    if valid_cols and row_indices:
        width = len(valid_cols)
        panel.loc[list(row_indices), valid_cols] = np.asarray(candidate_matrix, dtype=float)[:, :width]
    return panel


def family3_working_panel(
    panel: pd.DataFrame,
    target_indices: list[object],
    settings,
    *,
    preserve_global_context: bool = False,
) -> pd.DataFrame:
    """Return the minimal panel slice needed for one Family 3 optimisation task."""
    if preserve_global_context:
        return panel.copy()
    if not target_indices:
        return panel.copy()
    schema = settings.panel_schema
    target_firms = set(panel.loc[target_indices, schema.firm_id].astype(str).tolist())
    if "_split" not in panel.columns:
        mask_firms = panel[schema.firm_id].astype(str).isin(target_firms)
        return panel.loc[mask_firms].copy()
    mask_c = panel["_split"].astype(str) == SPLIT_LABEL_INCLUDED
    mask_firms = panel[schema.firm_id].astype(str).isin(target_firms)
    keep = mask_c | mask_firms
    return panel.loc[keep].copy()


from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.analysis.optimizers import (
    OptimizationEvaluation,
    run_optimizer,
)
try:
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional torch install
    _TORCH_AVAILABLE = False

    def run_cohort_torch_adam(*args, **kwargs):
        raise ImportError("optimizer_name='torch_adam' requires the optional torch stack to be installed.")

    def run_local_firm_torch_adam(*args, **kwargs):
        raise ImportError("optimizer_name='torch_adam' requires the optional torch stack to be installed.")

    def run_rowwise_torch_adam(*args, **kwargs):
        raise ImportError("optimizer_name='torch_adam' requires the optional torch stack to be installed.")

log = logging.getLogger(__name__)

def _utc_run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def _warn_if_benford_detectors_are_live(
    *,
    settings,
    mod_cols: tuple[str, ...],
) -> None:
    """Warn when Benford/Zipf may be recomputed on manipulated values."""
    benford_cols = set(getattr(settings.detector_preconditions, "benford_value_columns", ()))
    overlap = sorted(benford_cols & set(mod_cols))
    if not overlap:
        return
    frozen = set(settings.robustness_benchmark.family3_frozen_detectors_during_optimization)
    if "z1" in frozen:
        return
    if bool(getattr(settings.robustness_benchmark, "family3_strict_benford_freeze", False)):
        raise ValueError(
            "Family3 exact: Benford-linked columns overlap modifiable columns "
            f"{overlap} while z1 is not frozen. Add 'z1' to "
            "family3_frozen_detectors_during_optimization or disable "
            "family3_strict_benford_freeze explicitly."
        )
    log.warning(
        "Family3 exact: Benford-linked columns overlap modifiable columns %s while z1 is live. "
        "This makes Benford-based optimization targets methodologically suspect.",
        overlap,
    )


def _validate_torch_adam_request(
    *,
    direction: str,
    target_score_name: str,
    loss_name: str,
) -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "optimizer_name='torch_adam' requires the optional torch stack to be installed."
        )
    allowed_directions = {"to_green", "to_red"}
    if direction not in allowed_directions:
        raise ValueError(
            "optimizer_name='torch_adam' requires a supported Family 3 direction in "
            f"{sorted(allowed_directions)}, got {direction!r}."
        )
    allowed_target_scores = {
        "z_plus",
        "z_plus_renorm",
        "breadth",
        "z_mahalanobis_sq",
        "t_iut",
        "z_plus_softmax",
        "z_plus_orth",
    }
    if target_score_name not in allowed_target_scores:
        raise ValueError(
            "optimizer_name='torch_adam' requires a supported Family 3 target score in "
            f"{sorted(allowed_target_scores)}, got {target_score_name!r}."
        )
    allowed_loss_names = {"hinge_squared", "hinge", "mse"}
    if loss_name not in allowed_loss_names:
        raise ValueError(
            "optimizer_name='torch_adam' requires a supported Family 3 loss in "
            f"{sorted(allowed_loss_names)}, got {loss_name!r}."
        )


def append_epoch_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    batch.to_csv(path, mode="a", header=write_header, index=False)


def store_epoch_rows(
    batch_rows: list[dict[str, object]],
    *,
    epoch_rows: list[dict[str, object]] | None,
    epoch_writer: Callable[[list[dict[str, object]]], None] | None,
) -> None:
    if epoch_writer is not None and batch_rows:
        epoch_writer(batch_rows)
    if epoch_rows is not None and batch_rows:
        epoch_rows.extend(batch_rows)


def sample_orange_index(
    panel: pd.DataFrame,
    settings,
    *,
    base_ctx: ScoreContext,
    candidate_index: pd.Index | None = None,
) -> pd.Index | None:
    rb = settings.robustness_benchmark
    baseline_snapshot = family3_snapshot(panel, settings, base_ctx=base_ctx)
    flag_p = family3_flag_pvalues(baseline_snapshot)
    orange_mask = (flag_p >= rb.family3_p_red) & (flag_p < rb.family3_p_orange)
    candidate_rows = baseline_snapshot.panel.loc[orange_mask]
    if candidate_index is not None:
        return pd.Index(candidate_rows.index.intersection(candidate_index))
    if candidate_rows.empty:
        return pd.Index([])
    if rb.family3_max_rows > 0 and len(candidate_rows) > rb.family3_max_rows:
        rng = np.random.default_rng(rb.random_seed)
        sampled = rng.choice(candidate_rows.index.to_numpy(), size=rb.family3_max_rows, replace=False)
        return pd.Index(sampled)
    return pd.Index(candidate_rows.index)


def target_candidate_index(
    baseline_snapshot: Family3Snapshot,
    settings,
    *,
    target_score_name: str,
    direction: str,
    candidate_index: pd.Index | None = None,
    sample_seed: int | None = None,
    sample_limit: int | None = None,
) -> pd.Index:
    rb = settings.robustness_benchmark
    work_index = baseline_snapshot.panel.index
    if candidate_index is not None:
        work_index = work_index.intersection(candidate_index)
    if len(work_index) == 0:
        return pd.Index([])

    p_col = f"p_{target_score_name}"
    if p_col in baseline_snapshot.composites.columns:
        target_p = pd.to_numeric(baseline_snapshot.composites.loc[work_index, p_col], errors="coerce")
        if direction == "to_green":
            # For remediation runs, both RED and ORANGE rows are valid starting points.
            eligible_mask = target_p < rb.family3_p_orange
        else:
            # For degradation runs, keep only ORANGE rows and exclude already-RED cases.
            eligible_mask = (target_p >= rb.family3_p_red) & (target_p < rb.family3_p_orange)
        eligible_index = target_p.index[eligible_mask.fillna(False)]
    else:
        eligible_index = work_index
    if len(eligible_index) == 0:
        return pd.Index([])

    zero_delta = np.zeros(len(rb.family3_modifiable_columns), dtype=float)
    target_z = np.array(
        [family3_score_z_value(idx, target_score_name, baseline_snapshot) for idx in eligible_index],
        dtype=float,
    )
    if direction == "to_green":
        threshold = rb.family3_z_target_green
        positive_mask = target_z > threshold + 1e-12
    else:
        threshold = rb.family3_z_target_red
        positive_mask = target_z < threshold - 1e-12
    selected = pd.Index(eligible_index[positive_mask])
    if sample_limit is not None and sample_limit > 0 and len(selected) > sample_limit and candidate_index is None:
        rng = np.random.default_rng(sample_seed if sample_seed is not None else rb.random_seed)
        sampled = rng.choice(selected.to_numpy(), size=sample_limit, replace=False)
        selected = pd.Index(sampled)
    return selected


def _target_requires_global_panel_context(target_score_name: str) -> bool:
    """Scores tied to a global null geometry must keep the full panel context."""
    return target_score_name in {"z_mahalanobis_sq", "t_iut"}


def epoch_rows(
    *,
    run_id: str,
    mode: str,
    direction: str,
    target_score_name: str,
    target_score_family: str,
    threshold_z: float,
    method_name: str,
    epoch: int,
    row_indices: list[object],
    snapshot: Family3Snapshot,
    baseline_snapshot: Family3Snapshot,
    baseline_x_by_idx: dict[object, np.ndarray],
    candidate_x_by_idx: dict[object, np.ndarray],
    columns: tuple[str, ...],
    settings,
    objective_scope: str,
    objective_loss_total: float,
    objective_loss_score_term: float,
    objective_loss_l1_term: float,
    objective_loss_l2_term: float,
    best_snapshot: Family3Snapshot | None = None,
    best_candidate_x_by_idx: dict[object, np.ndarray] | None = None,
    best_objective_loss_total: float | None = None,
    best_objective_loss_score_term: float | None = None,
    best_objective_loss_l1_term: float | None = None,
    best_objective_loss_l2_term: float | None = None,
    source_fields: dict[str, object] | None = None,
    scale_floor: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_fields = source_fields or {}
    primary_col = settings.robustness_benchmark.primary_composite
    column_positions = {col: pos for pos, col in enumerate(columns)}
    floor_vec = scale_floor if scale_floor is not None else np.ones(len(columns), dtype=float)
    for idx in row_indices:
        firm_id = snapshot.panel.loc[idx, settings.panel_schema.firm_id]
        quarter = snapshot.panel.loc[idx, settings.panel_schema.quarter]
        baseline_target_z = family3_score_z_value(idx, target_score_name, baseline_snapshot)
        current_target_z = family3_score_z_value(idx, target_score_name, snapshot)
        baseline_primary_p = family3_primary_pvalue(idx, baseline_snapshot, settings)
        current_primary_p = family3_primary_pvalue(idx, snapshot, settings)
        baseline_flag_p = family3_flag_pvalue(idx, baseline_snapshot)
        current_flag_p = family3_flag_pvalue(idx, snapshot)
        baseline_status = family3_status_from_p(baseline_flag_p, settings)
        current_status = family3_status_from_p(current_flag_p, settings)
        baseline_target_status = family3_target_status_from_z(baseline_target_z, settings)
        current_target_status = family3_target_status_from_z(current_target_z, settings)
        success = current_target_status == ("GREEN" if direction == "to_green" else "RED")
        baseline_x = baseline_x_by_idx[idx]
        current_x = candidate_x_by_idx[idx]
        scale = np.maximum(np.abs(baseline_x), floor_vec)
        delta_norm = family3_normalized_delta(current_x, baseline_x, scale)
        row_loss_total, row_loss_score, row_loss_l1, row_loss_l2, _ = family3_exact_loss(
            target_score_z=current_target_z,
            direction=direction,
            delta_norm=delta_norm,
            settings=settings,
        )
        best_target_z = np.nan
        best_row_loss_total = np.nan
        best_row_loss_score = np.nan
        best_row_loss_l1 = np.nan
        best_row_loss_l2 = np.nan
        best_primary_p = np.nan
        best_flag_p = np.nan
        best_status = "UNKNOWN"
        best_target_status = "UNKNOWN"
        if best_snapshot is not None and best_candidate_x_by_idx is not None and idx in best_candidate_x_by_idx:
            best_x = best_candidate_x_by_idx[idx]
            best_target_z = family3_score_z_value(idx, target_score_name, best_snapshot)
            best_primary_p = family3_primary_pvalue(idx, best_snapshot, settings)
            best_flag_p = family3_flag_pvalue(idx, best_snapshot)
            best_status = family3_status_from_p(best_flag_p, settings)
            best_target_status = family3_target_status_from_z(best_target_z, settings)
            best_delta_norm = family3_normalized_delta(best_x, baseline_x, scale)
            best_row_loss_total, best_row_loss_score, best_row_loss_l1, best_row_loss_l2, _ = family3_exact_loss(
                target_score_z=best_target_z,
                direction=direction,
                delta_norm=best_delta_norm,
                settings=settings,
            )
        legacy_total = float(objective_loss_total if objective_scope == "cohort" else row_loss_total)
        legacy_score = float(objective_loss_score_term if objective_scope == "cohort" else row_loss_score)
        legacy_l1 = float(objective_loss_l1_term if objective_scope == "cohort" else row_loss_l1)
        legacy_l2 = float(objective_loss_l2_term if objective_scope == "cohort" else row_loss_l2)
        top_driver = ""
        top_driver_abs_delta = 0.0
        if len(columns) > 0:
            abs_delta = np.abs(delta_norm)
            top_pos = int(np.argmax(abs_delta))
            # Guard against argmax's zero-tie default (see build_summary_row /
            # _rank_firm_variables): an all-zero delta must not be reported as
            # if `columns[0]` were a genuine dominant driver.
            if abs_delta[top_pos] > 1e-12:
                top_driver = str(columns[top_pos])
                top_driver_abs_delta = float(abs_delta[top_pos])
        row = {
            "run_id": run_id,
            "method": method_name,
            "mode": mode,
            "direction": direction,
            "target_score_name": target_score_name,
            "target_score_family": target_score_family,
            "target_threshold_z": threshold_z,
            "lambda_l1": settings.robustness_benchmark.family3_lambda_l1,
            "epoch": epoch,
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else idx,
            "firm_id": firm_id,
            "quarter": quarter,
            "primary_composite": primary_col,
            "baseline_primary_p": baseline_primary_p,
            "current_primary_p": current_primary_p,
            "baseline_flag_p": baseline_flag_p,
            "current_flag_p": current_flag_p,
            "baseline_status": baseline_status,
            "current_status": current_status,
            "baseline_target_status": baseline_target_status,
            "current_target_status": current_target_status,
            "success_flag": bool(success),
            "baseline_target_score_z": baseline_target_z,
            "target_score_current": current_target_z,
            "best_target_score_current": best_target_z,
            "objective_scope": objective_scope,
            "objective_loss_total": float(objective_loss_total),
            "objective_loss_score_term": float(objective_loss_score_term),
            "objective_loss_l1_term": float(objective_loss_l1_term),
            "objective_loss_l2_term": float(objective_loss_l2_term),
            "best_objective_loss_total": float(best_objective_loss_total if best_objective_loss_total is not None else objective_loss_total),
            "best_objective_loss_score_term": float(best_objective_loss_score_term if best_objective_loss_score_term is not None else objective_loss_score_term),
            "best_objective_loss_l1_term": float(best_objective_loss_l1_term if best_objective_loss_l1_term is not None else objective_loss_l1_term),
            "best_objective_loss_l2_term": float(best_objective_loss_l2_term if best_objective_loss_l2_term is not None else objective_loss_l2_term),
            "row_loss_total": row_loss_total,
            "row_loss_score_term": row_loss_score,
            "row_loss_l1_term": row_loss_l1,
            "row_loss_l2_term": row_loss_l2,
            "best_row_loss_total": best_row_loss_total,
            "best_row_loss_score_term": best_row_loss_score,
            "best_row_loss_l1_term": best_row_loss_l1,
            "best_row_loss_l2_term": best_row_loss_l2,
            "loss_total": legacy_total,
            "loss_score_term": legacy_score,
            "loss_l1_term": legacy_l1,
            "loss_l2_term": legacy_l2,
            "best_primary_p": best_primary_p,
            "best_flag_p": best_flag_p,
            "best_status": best_status,
            "best_target_status": best_target_status,
            "top_driver": top_driver,
            "top_driver_abs_delta": top_driver_abs_delta,
        }
        for col, base_value, cur_value in zip(columns, baseline_x, current_x):
            row[f"baseline_{col}"] = float(base_value)
            row[f"current_{col}"] = float(cur_value)
            row[f"delta_{col}"] = float(cur_value - base_value)
            if best_candidate_x_by_idx is not None and idx in best_candidate_x_by_idx:
                best_value = float(best_candidate_x_by_idx[idx][column_positions[col]])
                row[f"best_current_{col}"] = best_value
                row[f"best_delta_{col}"] = best_value - float(base_value)
        row.update(source_fields)
        rows.append(row)
    return rows


def build_summary_row(
    *,
    run_id: str,
    method_name: str,
    mode: str,
    direction: str,
    target_score_name: str,
    target_family: str,
    idx: object,
    baseline_snapshot: Family3Snapshot,
    final_snapshot: Family3Snapshot,
    baseline_target_z: float,
    final_target_z: float,
    target_threshold_z: float,
    executed_epochs: int,
    objective_scope: str,
    objective_loss_total: float,
    objective_loss_score_term: float,
    objective_loss_l1_term: float,
    objective_loss_l2_term: float,
    row_loss_total: float,
    row_loss_score_term: float,
    row_loss_l1_term: float,
    row_loss_l2_term: float,
    baseline_x: np.ndarray,
    final_x: np.ndarray,
    mod_cols: tuple[str, ...],
    settings,
    source_fields: dict[str, object] | None = None,
    scale_floor: np.ndarray | None = None,
) -> dict[str, object]:
    baseline_primary_p = family3_primary_pvalue(idx, baseline_snapshot, settings)
    final_primary_p = family3_primary_pvalue(idx, final_snapshot, settings)
    baseline_flag_p = family3_flag_pvalue(idx, baseline_snapshot)
    final_flag_p = family3_flag_pvalue(idx, final_snapshot)
    final_target_status = family3_target_status_from_z(final_target_z, settings)
    row_index = int(idx) if isinstance(idx, (int, np.integer)) else idx
    top_driver = ""
    top_driver_abs_delta = 0.0
    floor_vec = scale_floor if scale_floor is not None else np.ones(len(mod_cols), dtype=float)
    delta_norm = family3_normalized_delta(final_x, baseline_x, np.maximum(np.abs(baseline_x), floor_vec))
    if len(mod_cols) > 0:
        abs_delta = np.abs(delta_norm)
        top_pos = int(np.argmax(abs_delta))
        # Guard against argmax's zero-tie default: on a fully untouched
        # candidate (optimizer left every column at baseline, e.g. no
        # modifiable column could move the exact score for this row) all
        # entries are 0.0 and argmax silently returns position 0 -- without
        # this check `top_driver` would always report the first configured
        # modifiable column (e.g. "atq") as if it were genuinely dominant.
        if abs_delta[top_pos] > 1e-12:
            top_driver = str(mod_cols[top_pos])
            top_driver_abs_delta = float(abs_delta[top_pos])
    summary_row = {
        "run_id": run_id,
        "method": method_name,
        "mode": mode,
        "direction": direction,
        "target_score_name": target_score_name,
        "target_score_family": target_family,
        "row_index": row_index,
        "firm_id": baseline_snapshot.panel.loc[idx, settings.panel_schema.firm_id],
        "quarter": baseline_snapshot.panel.loc[idx, settings.panel_schema.quarter],
        "baseline_primary_p": baseline_primary_p,
        "final_primary_p": final_primary_p,
        "baseline_flag_p": baseline_flag_p,
        "final_flag_p": final_flag_p,
        "baseline_status": family3_status_from_p(baseline_flag_p, settings),
        "final_status": family3_status_from_p(final_flag_p, settings),
        "baseline_target_status": family3_target_status_from_z(baseline_target_z, settings),
        "final_target_status": final_target_status,
        "success": final_target_status == ("GREEN" if direction == "to_green" else "RED"),
        "epochs_run": executed_epochs,
        "objective_scope": objective_scope,
        "objective_loss_total": objective_loss_total,
        "objective_loss_score_term": objective_loss_score_term,
        "objective_loss_l1_term": objective_loss_l1_term,
        "objective_loss_l2_term": objective_loss_l2_term,
        "row_loss_total": row_loss_total,
        "row_loss_score_term": row_loss_score_term,
        "row_loss_l1_term": row_loss_l1_term,
        "row_loss_l2_term": row_loss_l2_term,
        "loss_total": objective_loss_total,
        "loss_score_term": objective_loss_score_term,
        "loss_l1_term": objective_loss_l1_term,
        "loss_l2_term": objective_loss_l2_term,
        "target_threshold_z": target_threshold_z,
        "baseline_target_score_z": baseline_target_z,
        "final_target_score_z": final_target_z,
        "top_driver": top_driver,
        "top_driver_abs_delta": top_driver_abs_delta,
    }
    for col, base_value, final_value in zip(mod_cols, baseline_x, final_x):
        summary_row[f"baseline_{col}"] = float(base_value)
        summary_row[f"final_{col}"] = float(final_value)
        summary_row[f"delta_{col}"] = float(final_value - base_value)
    summary_row.update(source_fields or {})
    return summary_row


def attach_summary_group_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "success" not in summary.columns:
        return summary
    group_cols = ["method", "mode", "target_score_name", "direction"]
    grouped = summary.groupby(group_cols, dropna=False)
    metrics = grouped.agg(
        evasion_rate=("success", "mean"),
        median_cost=("loss_l1_term", "median"),
        p90_cost=("loss_l1_term", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
        median_l2_cost=("loss_l2_term", "median"),
        p90_l2_cost=("loss_l2_term", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
        median_row_cost=("row_loss_l1_term", "median"),
        p90_row_cost=("row_loss_l1_term", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
    )
    return summary.join(metrics, on=group_cols)


def _rank_firm_variables(
    *,
    baseline_matrix: np.ndarray,
    final_matrix: np.ndarray,
    scales: np.ndarray,
    mod_cols: tuple[str, ...],
) -> tuple[str, float, str, int, int]:
    """Return top-driver summaries for one firm-level counterfactual."""

    if baseline_matrix.size == 0 or final_matrix.size == 0 or not mod_cols:
        return "", 0.0, "[]", 0, 0
    delta_norm = np.divide(
        np.asarray(final_matrix, dtype=float) - np.asarray(baseline_matrix, dtype=float),
        scales,
        out=np.zeros_like(scales, dtype=float),
        where=scales > 1e-12,
    )
    abs_norm = np.abs(delta_norm)
    abs_sum_by_var = abs_norm.sum(axis=0)
    touched_by_var = (abs_norm > 1e-12).sum(axis=0)
    n_active_cells = int((abs_norm > 1e-12).sum())
    n_active_variables = int((touched_by_var > 0).sum())
    if not np.isfinite(abs_sum_by_var).any():
        return "", 0.0, "[]", n_active_cells, n_active_variables
    order = np.argsort(-abs_sum_by_var)
    top_pos = int(order[0])
    # Guard against argsort's zero-tie default: if the optimizer left every
    # column at baseline (e.g. no modifiable column could move the exact
    # score for this firm), abs_sum_by_var is all-zero and order[0] would
    # silently report the first configured modifiable column as "dominant".
    if abs_sum_by_var[top_pos] > 1e-12:
        top_driver = str(mod_cols[top_pos])
        top_driver_abs_delta = float(abs_sum_by_var[top_pos])
    else:
        top_driver = ""
        top_driver_abs_delta = 0.0
    top5_payload: list[dict[str, object]] = []
    raw_delta = np.asarray(final_matrix, dtype=float) - np.asarray(baseline_matrix, dtype=float)
    for pos in order[:5]:
        if abs_sum_by_var[pos] <= 1e-12:
            continue
        top5_payload.append(
            {
                "variable": str(mod_cols[int(pos)]),
                "abs_delta_norm_sum": float(abs_sum_by_var[pos]),
                "n_rows_touched": int(touched_by_var[pos]),
                "median_signed_delta": float(np.nanmedian(raw_delta[:, int(pos)])),
            }
        )
    return top_driver, top_driver_abs_delta, json.dumps(top5_payload, ensure_ascii=True), n_active_cells, n_active_variables


def build_firm_summary_row(
    *,
    run_id: str,
    method_name: str,
    mode: str,
    direction: str,
    target_score_name: str,
    target_family: str,
    firm_id: object,
    row_indices: list[object],
    baseline_snapshot: Family3Snapshot,
    final_snapshot: Family3Snapshot,
    executed_epochs: int,
    objective_loss_total: float,
    objective_loss_score_term: float,
    objective_loss_l1_term: float,
    objective_loss_l2_term: float,
    baseline_matrix: np.ndarray,
    final_matrix: np.ndarray,
    scales: np.ndarray,
    mod_cols: tuple[str, ...],
    settings,
    source_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    baseline_target_p = family3_firm_target_pvalue(firm_id, target_score_name, baseline_snapshot, settings)
    final_target_p = family3_firm_target_pvalue(firm_id, target_score_name, final_snapshot, settings)
    baseline_target_z = family3_firm_score_z_value(firm_id, target_score_name, baseline_snapshot, settings)
    final_target_z = family3_firm_score_z_value(firm_id, target_score_name, final_snapshot, settings)
    primary_name = settings.robustness_benchmark.primary_composite
    primary_score_name = str(primary_name).removeprefix("p_")
    baseline_primary_p = family3_firm_target_pvalue(firm_id, primary_score_name, baseline_snapshot, settings)
    final_primary_p = family3_firm_target_pvalue(firm_id, primary_score_name, final_snapshot, settings)
    top_driver, top_driver_abs_delta, top5_json, n_active_cells, n_active_variables = _rank_firm_variables(
        baseline_matrix=baseline_matrix,
        final_matrix=final_matrix,
        scales=scales,
        mod_cols=mod_cols,
    )
    success = family3_target_status_from_z(final_target_z, settings) == ("GREEN" if direction == "to_green" else "RED")
    row: dict[str, object] = {
        "run_id": run_id,
        "method": method_name,
        "mode": mode,
        "direction": direction,
        "target_score_name": target_score_name,
        "target_score_family": target_family,
        "firm_id": firm_id,
        "n_rows_attacked": int(len(row_indices)),
        "row_indices_json": json.dumps([int(v) if isinstance(v, (int, np.integer)) else str(v) for v in row_indices], ensure_ascii=True),
        "baseline_primary_p": baseline_primary_p,
        "final_primary_p": final_primary_p,
        "baseline_target_p": baseline_target_p,
        "final_target_p": final_target_p,
        "baseline_status": family3_target_status_from_z(baseline_target_z, settings),
        "final_status": family3_target_status_from_z(final_target_z, settings),
        "baseline_target_status": family3_target_status_from_z(baseline_target_z, settings),
        "final_target_status": family3_target_status_from_z(final_target_z, settings),
        "success": bool(success),
        "epochs_run": int(executed_epochs),
        "objective_scope": "firm",
        "target_threshold_z": settings.robustness_benchmark.family3_z_target_green if direction == "to_green" else settings.robustness_benchmark.family3_z_target_red,
        "baseline_target_score_z": baseline_target_z,
        "final_target_score_z": final_target_z,
        "objective_loss_total": float(objective_loss_total),
        "objective_loss_score_term": float(objective_loss_score_term),
        "objective_loss_l1_term": float(objective_loss_l1_term),
        "objective_loss_l2_term": float(objective_loss_l2_term),
        "row_loss_total": np.nan,
        "row_loss_score_term": np.nan,
        "row_loss_l1_term": np.nan,
        "row_loss_l2_term": np.nan,
        "loss_total": float(objective_loss_total),
        "loss_score_term": float(objective_loss_score_term),
        "loss_l1_term": float(objective_loss_l1_term),
        "loss_l2_term": float(objective_loss_l2_term),
        "top_driver": top_driver,
        "top_driver_abs_delta": top_driver_abs_delta,
        "top5_drivers_json": top5_json,
        "n_active_cells": n_active_cells,
        "n_active_variables_any_quarter": n_active_variables,
    }
    row.update(source_fields or {})
    return row


def _is_baseline_candidate(candidate: np.ndarray, baseline: np.ndarray) -> bool:
    """Return whether a candidate is numerically identical to the baseline.

    For Family 3 global-context composites (`t_iut`, `z_mahalanobis_sq`), the
    zero-delta path must stay exactly aligned with the baseline snapshot. A
    small tolerance keeps the guard robust to optimizer bookkeeping noise while
    still treating genuine perturbations as changes.
    """

    return bool(
        np.allclose(
            np.asarray(candidate, dtype=float),
            np.asarray(baseline, dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    )


def _baseline_snapshot_from_working_set(working_set: WorkingSet) -> Family3Snapshot:
    """Rebuild the exact baseline snapshot from the working-set cache.

    The zero-delta path must not trigger any recomputation. For global-context
    composites such as `t_iut` and `z_mahalanobis_sq`, even a harmless-looking
    replay can drift away from the baseline if it re-enters detector or Phase 4
    logic. This helper returns a detached snapshot assembled directly from the
    cached baseline state.
    """

    cache = working_set.snapshot_cache
    return Family3Snapshot(
        panel=cache.ratio.panel_with_ratios.reindex(working_set.panel.index).copy(),
        raw_zscores=cache.detector.raw_zscores.reindex(working_set.panel.index).copy(),
        merged_zscores=cache.composite.merged_zscores.reindex(working_set.panel.index).copy(),
        composites=cache.composite.composites.reindex(working_set.panel.index).copy(),
    )


def row_loss_and_snapshot(
    *,
    working_set: WorkingSet,
    row_index: object,
    row_position: int,
    candidate_x: np.ndarray,
    target_score_name: str,
    direction: str,
    base_ctx: ScoreContext,
    settings,
) -> tuple[float, float, float, float, float, Family3Snapshot]:
    baseline_x = working_set.baseline_matrix[row_position]
    frozen_detectors = settings.robustness_benchmark.family3_frozen_detectors_during_optimization
    if _is_baseline_candidate(candidate_x, baseline_x):
        snapshot = _baseline_snapshot_from_working_set(working_set)
    else:
        candidate_panel = family3_apply_candidate_inplace(
            working_set.scratch_panel,
            [row_index],
            np.asarray([candidate_x], dtype=float),
            working_set.columns,
        )
        snapshot = family3_snapshot(
            candidate_panel,
            settings,
            base_ctx=base_ctx,
            changed_columns=working_set.columns,
            frozen_detectors=frozen_detectors,
            changed_index=[row_index],
            snapshot_cache=working_set.snapshot_cache,
            ratio_buffer=working_set.scratch_panel_with_ratios,
            raw_zscores_buffer=working_set.scratch_raw_zscores,
            merged_zscores_buffer=working_set.scratch_merged_zscores,
            composites_buffer=working_set.scratch_composites,
        )
    target_score_z = family3_score_z_value(row_index, target_score_name, snapshot)
    loss_total, loss_score, loss_l1, loss_l2, threshold = family3_exact_loss(
        target_score_z=target_score_z,
        direction=direction,
        delta_norm=family3_normalized_delta(candidate_x, baseline_x, working_set.scales[row_position]),
        settings=settings,
    )
    return loss_total, loss_score, loss_l1, loss_l2, threshold, snapshot


def cohort_loss_and_snapshot(
    *,
    working_set: WorkingSet,
    candidate_matrix: np.ndarray,
    target_score_name: str,
    direction: str,
    base_ctx: ScoreContext,
    settings,
) -> tuple[float, float, float, float, float, Family3Snapshot]:
    """Evaluate one shared cohort candidate with configurable aggregation.

    The default cohort objective is the mean row loss, which matches the main
    XAI specification. A `min` mode is also available for stricter experiments
    where the cohort score is driven by the hardest row to improve.
    """
    frozen_detectors = settings.robustness_benchmark.family3_frozen_detectors_during_optimization
    if _is_baseline_candidate(candidate_matrix, working_set.baseline_matrix):
        snapshot = _baseline_snapshot_from_working_set(working_set)
    else:
        candidate_panel = family3_apply_candidate_inplace(
            working_set.scratch_panel,
            list(working_set.row_indices),
            candidate_matrix,
            working_set.columns,
        )
        snapshot = family3_snapshot(
            candidate_panel,
            settings,
            base_ctx=base_ctx,
            changed_columns=working_set.columns,
            frozen_detectors=frozen_detectors,
            changed_index=list(working_set.row_indices),
            snapshot_cache=working_set.snapshot_cache,
            ratio_buffer=working_set.scratch_panel_with_ratios,
            raw_zscores_buffer=working_set.scratch_raw_zscores,
            merged_zscores_buffer=working_set.scratch_merged_zscores,
            composites_buffer=working_set.scratch_composites,
        )
    target_score_z = np.array(
        [family3_score_z_value(idx, target_score_name, snapshot) for idx in working_set.row_indices],
        dtype=float,
    )
    score_terms = np.array(
        [
            family3_score_loss_term(
                target_score_z=float(z_value),
                direction=direction,
                settings=settings,
            )[0]
            for z_value in target_score_z
        ],
        dtype=float,
    )
    threshold = (
        settings.robustness_benchmark.family3_z_target_green
        if direction == "to_green"
        else settings.robustness_benchmark.family3_z_target_red
    )
    delta_norm = np.divide(
        np.asarray(candidate_matrix, dtype=float) - working_set.baseline_matrix,
        working_set.scales,
        out=np.zeros_like(working_set.scales, dtype=float),
        where=working_set.scales > 1e-12,
    )
    l1_terms = settings.robustness_benchmark.family3_lambda_l1 * np.abs(delta_norm).sum(axis=1)
    l2_terms = settings.robustness_benchmark.family3_lambda_l2 * np.square(delta_norm).sum(axis=1)
    aggregate_mode = getattr(settings.robustness_benchmark, "family3_cohort_loss_mode", "mean")
    if aggregate_mode == "mean":
        score_component = float(np.mean(score_terms)) if score_terms.size else float("nan")
        l1_component = float(np.mean(l1_terms)) if l1_terms.size else float("nan")
        l2_component = float(np.mean(l2_terms)) if l2_terms.size else float("nan")
        total_component = float(score_component + l1_component + l2_component)
    elif aggregate_mode == "min":
        row_totals = score_terms + l1_terms + l2_terms
        if row_totals.size:
            best_pos = int(np.argmax(row_totals))
            score_component = float(score_terms[best_pos])
            l1_component = float(l1_terms[best_pos])
            l2_component = float(l2_terms[best_pos])
            total_component = float(row_totals[best_pos])
        else:
            score_component = float("nan")
            l1_component = float("nan")
            l2_component = float("nan")
            total_component = float("nan")
    else:
        raise ValueError(f"Unsupported Family 3 cohort loss mode {aggregate_mode!r}.")
    return total_component, score_component, l1_component, l2_component, float(threshold), snapshot


def firm_loss_and_snapshot(
    *,
    working_set: WorkingSet,
    row_indices: list[object],
    candidate_matrix: np.ndarray,
    target_score_name: str,
    direction: str,
    base_ctx: ScoreContext,
    settings,
) -> tuple[float, float, float, float, float, Family3Snapshot]:
    """Evaluate one candidate where all rows of one firm move jointly."""

    frozen_detectors = settings.robustness_benchmark.family3_frozen_detectors_during_optimization
    if _is_baseline_candidate(candidate_matrix, working_set.baseline_matrix):
        snapshot = _baseline_snapshot_from_working_set(working_set)
    else:
        candidate_panel = family3_apply_candidate_inplace(
            working_set.scratch_panel,
            list(row_indices),
            candidate_matrix,
            working_set.columns,
        )
        snapshot = family3_snapshot(
            candidate_panel,
            settings,
            base_ctx=base_ctx,
            changed_columns=working_set.columns,
            frozen_detectors=frozen_detectors,
            changed_index=list(row_indices),
            snapshot_cache=working_set.snapshot_cache,
            ratio_buffer=working_set.scratch_panel_with_ratios,
            raw_zscores_buffer=working_set.scratch_raw_zscores,
            merged_zscores_buffer=working_set.scratch_merged_zscores,
            composites_buffer=working_set.scratch_composites,
        )
    firm_id = snapshot.panel.loc[row_indices[0], settings.panel_schema.firm_id]
    target_score_z = family3_firm_score_z_value(firm_id, target_score_name, snapshot, settings)
    delta_norm = np.divide(
        np.asarray(candidate_matrix, dtype=float) - working_set.baseline_matrix,
        working_set.scales,
        out=np.zeros_like(working_set.scales, dtype=float),
        where=working_set.scales > 1e-12,
    )
    loss_total, loss_score, loss_l1, loss_l2, threshold = family3_exact_loss(
        target_score_z=target_score_z,
        direction=direction,
        delta_norm=delta_norm.reshape(-1),
        settings=settings,
    )
    return loss_total, loss_score, loss_l1, loss_l2, threshold, snapshot


def local_firm_single(
    *,
    panel: pd.DataFrame,
    settings,
    base_ctx: ScoreContext,
    method_name: str,
    source_fields: dict[str, object] | None = None,
    epoch_writer: Callable[[list[dict[str, object]]], None] | None = None,
    keep_epoch_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = settings.robustness_benchmark
    mod_cols = tuple(col for col in rb.family3_modifiable_columns if col in panel.columns)
    if not mod_cols:
        return pd.DataFrame([{"method": method_name, "status": "no modifiable columns"}]), pd.DataFrame()
    if not rb.family3_local_firm_case:
        return pd.DataFrame([{"method": method_name, "status": "no local firm requested"}]), pd.DataFrame()
    _warn_if_benford_detectors_are_live(settings=settings, mod_cols=mod_cols)
    baseline_snapshot = family3_snapshot(panel, settings, base_ctx=base_ctx)
    schema = settings.panel_schema
    firm_mask = panel[schema.firm_id].astype(str) == str(rb.family3_local_firm_case)
    row_indices = panel.index[firm_mask].tolist()
    if not row_indices:
        return pd.DataFrame([{"method": method_name, "mode": "local_firm_single", "status": "requested firm_id not found", "firm_id": rb.family3_local_firm_case}]), pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    epoch_rows_buffer: list[dict[str, object]] | None = [] if keep_epoch_rows else None
    # Built once per run (see profiling notes): the C-split reference stats
    # used by the incremental detector patch must come from the FULL panel,
    # not the row-restricted working panel, or they silently corrupt
    # detectors for any C-split firm whose own history isn't entirely in C.
    global_refs = family3_build_global_detector_references(panel, settings, base_ctx=base_ctx)
    scale_floor = family3_column_scale_floor(panel, mod_cols, rb.family3_scale_floor_quantile)
    working_set = family3_build_working_set(
        panel,
        row_indices,
        mod_cols,
        settings,
        base_ctx=base_ctx,
        preserve_global_context=False,
        global_refs=global_refs,
        scale_floor=scale_floor,
    )
    baseline_firm_id = baseline_snapshot.panel.loc[row_indices[0], schema.firm_id]
    for target_score_name in rb.family3_target_scores:
        target_family = family3_target_score_kind(target_score_name)
        if target_family == "composite" and target_score_name not in baseline_snapshot.composites.columns:
            continue
        for direction in rb.family3_directions:
            baseline_target_p = family3_firm_target_pvalue(baseline_firm_id, target_score_name, baseline_snapshot, settings)
            baseline_target_z = family3_firm_score_z_value(baseline_firm_id, target_score_name, baseline_snapshot, settings)
            baseline_target_status = family3_target_status_from_z(baseline_target_z, settings)
            eligible_statuses = {"ORANGE"} if direction == "to_red" else {"ORANGE", "RED"}
            if baseline_target_status not in eligible_statuses:
                summary_rows.append(
                    {
                        "method": method_name,
                        "mode": "local_firm_single",
                        "target_score_name": target_score_name,
                        "direction": direction,
                        "firm_id": baseline_firm_id,
                        "n_rows_attacked": int(len(row_indices)),
                        "baseline_target_p": baseline_target_p,
                        "baseline_target_score_z": baseline_target_z,
                        "baseline_target_status": baseline_target_status,
                        "status": f"firm not eligible for {direction}",
                    }
                )
                continue

            run_id = f"{method_name}_{direction}_{target_score_name}_firm_{baseline_firm_id}_{_utc_run_timestamp()}"
            local_epoch_rows: list[dict[str, object]] = []

            def _evaluate_firm(delta_flat: np.ndarray) -> OptimizationEvaluation:
                delta_matrix = np.asarray(delta_flat, dtype=float).reshape(len(row_indices), len(mod_cols))
                candidate_matrix = np.maximum(working_set.baseline_matrix + delta_matrix * working_set.scales, 0.0)
                loss_total, loss_score, loss_l1, loss_l2, threshold, snapshot = firm_loss_and_snapshot(
                    working_set=working_set,
                    row_indices=row_indices,
                    candidate_matrix=candidate_matrix,
                    target_score_name=target_score_name,
                    direction=direction,
                    base_ctx=base_ctx,
                    settings=settings,
                )
                return OptimizationEvaluation(
                    loss_total=loss_total,
                    loss_score=loss_score,
                    loss_regularizer=loss_l1 + loss_l2,
                    threshold=threshold,
                    artifact=snapshot,
                    payload=candidate_matrix,
                    loss_regularizer_l1=loss_l1,
                    loss_regularizer_l2=loss_l2,
                )

            def _firm_epoch_callback(
                epoch: int,
                evaluation: OptimizationEvaluation,
                best_evaluation: OptimizationEvaluation,
            ) -> None:
                batch_rows = epoch_rows(
                    run_id=run_id,
                    mode="local_firm_single",
                    direction=direction,
                    target_score_name=target_score_name,
                    target_score_family=target_family,
                    threshold_z=evaluation.threshold,
                    method_name=method_name,
                    epoch=epoch,
                    row_indices=row_indices,
                    snapshot=evaluation.artifact,
                    baseline_snapshot=baseline_snapshot,
                    baseline_x_by_idx={idx: working_set.baseline_matrix[pos] for pos, idx in enumerate(row_indices)},
                    candidate_x_by_idx={idx: evaluation.payload[pos] for pos, idx in enumerate(row_indices)},
                    columns=mod_cols,
                    settings=settings,
                    objective_scope="firm",
                    objective_loss_total=evaluation.loss_total,
                    objective_loss_score_term=evaluation.loss_score,
                    objective_loss_l1_term=evaluation.loss_regularizer_l1,
                    objective_loss_l2_term=evaluation.loss_regularizer_l2,
                    best_snapshot=best_evaluation.artifact,
                    best_candidate_x_by_idx={idx: best_evaluation.payload[pos] for pos, idx in enumerate(row_indices)},
                    best_objective_loss_total=best_evaluation.loss_total,
                    best_objective_loss_score_term=best_evaluation.loss_score,
                    best_objective_loss_l1_term=best_evaluation.loss_regularizer_l1,
                    best_objective_loss_l2_term=best_evaluation.loss_regularizer_l2,
                    source_fields={**(source_fields or {}), "local_scope": "firm_single"},
                    scale_floor=scale_floor,
                )
                local_epoch_rows.extend(batch_rows)
                store_epoch_rows(batch_rows, epoch_rows=epoch_rows_buffer, epoch_writer=epoch_writer)

            log_prefix = (
                f"family3_exact: mode=local_firm_single firm={baseline_firm_id} "
                f"n_rows={len(row_indices)} target={target_score_name} direction={direction}"
            )
            if rb.family3_optimizer == "torch_adam":
                _validate_torch_adam_request(
                    direction=direction,
                    target_score_name=target_score_name,
                    loss_name=rb.family3_loss_name,
                )
                descent = run_local_firm_torch_adam(
                    panel=panel,
                    base_ctx=base_ctx,
                    settings=settings,
                    row_indices=row_indices,
                    mod_cols=mod_cols,
                    target_score_name=target_score_name,
                    direction=direction,
                    max_epochs=rb.family3_epochs,
                    max_eps=rb.family3_max_eps,
                    step_size=rb.pgd_step_size,
                    log_prefix=log_prefix,
                    evaluate_fn=_evaluate_firm,
                    epoch_callback=_firm_epoch_callback,
                    early_stop_loss=rb.family3_early_stop_loss,
                    early_stop_patience=rb.family3_early_stop_patience,
                    adam_beta1=rb.family3_adam_beta1,
                    adam_beta2=rb.family3_adam_beta2,
                    adam_eps=rb.family3_adam_eps,
                    plateau_shrink_patience=rb.family3_plateau_shrink_patience,
                    plateau_shrink_factor=rb.family3_plateau_shrink_factor,
                    min_step_size=rb.family3_min_step_size,
                    restore_best_on_shrink=rb.family3_restore_best_on_shrink,
                    reset_moments_on_shrink=rb.family3_reset_moments_on_shrink,
                )
            else:
                descent = run_optimizer(
                    optimizer_name=rb.family3_optimizer,
                    state_dim=len(row_indices) * len(mod_cols),
                    max_epochs=rb.family3_epochs,
                    max_eps=rb.family3_max_eps,
                    step_size=rb.pgd_step_size,
                    fd_step=rb.family3_fd_step,
                    random_seed=rb.random_seed,
                    log_prefix=log_prefix,
                    evaluate_fn=_evaluate_firm,
                    epoch_callback=_firm_epoch_callback,
                    early_stop_loss=rb.family3_early_stop_loss,
                    early_stop_patience=rb.family3_early_stop_patience,
                    adam_beta1=rb.family3_adam_beta1,
                    adam_beta2=rb.family3_adam_beta2,
                    adam_eps=rb.family3_adam_eps,
                    plateau_shrink_patience=rb.family3_plateau_shrink_patience,
                    plateau_shrink_factor=rb.family3_plateau_shrink_factor,
                    min_step_size=rb.family3_min_step_size,
                    restore_best_on_shrink=rb.family3_restore_best_on_shrink,
                    reset_moments_on_shrink=rb.family3_reset_moments_on_shrink,
                )
            best_eval = descent.best_evaluation
            final_snapshot = family3_snapshot(
                family3_candidate_panel(working_set.panel, row_indices, best_eval.payload, mod_cols),
                settings,
                base_ctx=base_ctx,
                changed_columns=mod_cols,
                changed_index=row_indices,
                snapshot_cache=working_set.snapshot_cache,
                ratio_buffer=working_set.scratch_panel_with_ratios,
                raw_zscores_buffer=working_set.scratch_raw_zscores,
                merged_zscores_buffer=working_set.scratch_merged_zscores,
                composites_buffer=working_set.scratch_composites,
            )
            summary_rows.append(
                build_firm_summary_row(
                    run_id=run_id,
                    method_name=method_name,
                    mode="local_firm_single",
                    direction=direction,
                    target_score_name=target_score_name,
                    target_family=target_family,
                    firm_id=baseline_firm_id,
                    row_indices=row_indices,
                    baseline_snapshot=baseline_snapshot,
                    final_snapshot=final_snapshot,
                    executed_epochs=descent.executed_epochs,
                    objective_loss_total=best_eval.loss_total,
                    objective_loss_score_term=best_eval.loss_score,
                    objective_loss_l1_term=best_eval.loss_regularizer_l1,
                    objective_loss_l2_term=best_eval.loss_regularizer_l2,
                    baseline_matrix=working_set.baseline_matrix,
                    final_matrix=best_eval.payload,
                    scales=working_set.scales,
                    mod_cols=mod_cols,
                    settings=settings,
                    source_fields={**(source_fields or {}), "local_scope": "firm_single"},
                )
            )
    summary = attach_summary_group_metrics(pd.DataFrame(summary_rows))
    return summary, pd.DataFrame(epoch_rows_buffer or [])


def global_per_row(
    *,
    panel: pd.DataFrame,
    settings,
    base_ctx: ScoreContext,
    method_name: str,
    candidate_index: pd.Index | None = None,
    source_fields: dict[str, object] | None = None,
    epoch_writer: Callable[[list[dict[str, object]]], None] | None = None,
    keep_epoch_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = settings.robustness_benchmark
    mod_cols = tuple(col for col in rb.family3_modifiable_columns if col in panel.columns)
    if not mod_cols:
        return pd.DataFrame([{"method": method_name, "status": "no modifiable columns"}]), pd.DataFrame()
    _warn_if_benford_detectors_are_live(settings=settings, mod_cols=mod_cols)
    baseline_snapshot = family3_snapshot(panel, settings, base_ctx=base_ctx)
    schema = settings.panel_schema
    summary_rows: list[dict[str, object]] = []
    epoch_rows_buffer: list[dict[str, object]] | None = [] if keep_epoch_rows else None
    # Built once per run (see the normalization-floor audit): a
    # per-column floor derived from each column's own scale across the full
    # panel, instead of one constant shared by every column regardless of its
    # natural units.
    scale_floor = family3_column_scale_floor(panel, mod_cols, rb.family3_scale_floor_quantile)
    # Built once per run and reused across every candidate row below, mirroring
    # `run_cohort_torch_adam`/`run_local_firm_torch_adam` -- these only depend
    # on `base_ctx`, never on which row is being attacked. Without this,
    # `run_rowwise_torch_adam` rebuilds the whole bundle (including the
    # ~11-12s LedoitWolf peer reference fit) from scratch on every single row
    # of a `global_per_row` run (see profiling notes).
    shared_ctx = (
        build_torch_shared_run_context(base_ctx, prefer_gpu=True, scale_floor=scale_floor)
        if rb.family3_optimizer == "torch_adam"
        else None
    )
    firm_panel_cache: dict[object, tuple[pd.DataFrame, pd.DataFrame] | None] = {}
    # Built once per run (see profiling notes): the C-split reference stats
    # used by the incremental detector patch must come from the FULL panel,
    # not the row-restricted working panel, or they silently corrupt
    # detectors for any C-split firm whose own history isn't entirely in C.
    global_refs = family3_build_global_detector_references(panel, settings, base_ctx=base_ctx)
    for target_position, target_score_name in enumerate(rb.family3_target_scores, start=1):
        target_family = family3_target_score_kind(target_score_name)
        if target_family == "composite" and target_score_name not in baseline_snapshot.composites.columns:
            continue
        for direction_position, direction in enumerate(rb.family3_directions, start=1):
            current_candidate_index = target_candidate_index(
                baseline_snapshot,
                settings,
                target_score_name=target_score_name,
                direction=direction,
                candidate_index=candidate_index,
                sample_seed=rb.random_seed + target_position * 100 + direction_position,
                sample_limit=rb.family3_max_rows if candidate_index is None else None,
            )
            if len(current_candidate_index) == 0:
                summary_rows.append({"method": method_name, "mode": "global_per_row", "target_score_name": target_score_name, "direction": direction, "status": "no target-eligible rows"})
                continue
            template_cache: dict[str, object] = {}
            for row_position, idx in enumerate(current_candidate_index.tolist(), start=1):
                firm_key = str(panel.loc[idx, schema.firm_id])
                template = template_cache.get(firm_key)
                if template is None:
                    template = family3_build_working_template(
                        panel,
                        [idx],
                        settings,
                        base_ctx=base_ctx,
                        preserve_global_context=_target_requires_global_panel_context(target_score_name),
                        global_refs=global_refs,
                    )
                    template_cache[firm_key] = template
                working_set = family3_build_working_set(
                    panel,
                    [idx],
                    mod_cols,
                    settings,
                    base_ctx=base_ctx,
                    preserve_global_context=_target_requires_global_panel_context(target_score_name),
                    template=template,
                    scale_floor=scale_floor,
                )
                baseline_x = working_set.baseline_matrix[0]
                scale = working_set.scales[0]
                run_id = f"{method_name}_{direction}_{target_score_name}_{idx}_{_utc_run_timestamp()}"
                local_epoch_rows: list[dict[str, object]] = []

                def _evaluate_row(delta: np.ndarray) -> OptimizationEvaluation:
                    candidate_x = np.maximum(baseline_x + np.asarray(delta, dtype=float) * scale, 0.0)
                    loss_total, loss_score, loss_l1, loss_l2, threshold, snapshot = row_loss_and_snapshot(
                        working_set=working_set,
                        row_index=idx,
                        row_position=0,
                        candidate_x=candidate_x,
                        target_score_name=target_score_name,
                        direction=direction,
                        base_ctx=base_ctx,
                        settings=settings,
                    )
                    return OptimizationEvaluation(
                        loss_total=loss_total,
                        loss_score=loss_score,
                        loss_regularizer=loss_l1 + loss_l2,
                        threshold=threshold,
                        artifact=snapshot,
                        payload=candidate_x,
                        loss_regularizer_l1=loss_l1,
                        loss_regularizer_l2=loss_l2,
                    )

                def _row_epoch_callback(
                    epoch: int,
                    evaluation: OptimizationEvaluation,
                    best_evaluation: OptimizationEvaluation,
                ) -> None:
                    batch_rows = epoch_rows(
                        run_id=run_id,
                        mode="global_per_row",
                        direction=direction,
                        target_score_name=target_score_name,
                        target_score_family=target_family,
                        threshold_z=evaluation.threshold,
                        method_name=method_name,
                        epoch=epoch,
                        row_indices=[idx],
                        snapshot=evaluation.artifact,
                        baseline_snapshot=baseline_snapshot,
                        baseline_x_by_idx={idx: baseline_x},
                        candidate_x_by_idx={idx: evaluation.payload},
                        columns=mod_cols,
                        settings=settings,
                        objective_scope="row",
                        objective_loss_total=evaluation.loss_total,
                        objective_loss_score_term=evaluation.loss_score,
                        objective_loss_l1_term=evaluation.loss_regularizer_l1,
                        objective_loss_l2_term=evaluation.loss_regularizer_l2,
                        best_snapshot=best_evaluation.artifact,
                        best_candidate_x_by_idx={idx: best_evaluation.payload},
                        best_objective_loss_total=best_evaluation.loss_total,
                        best_objective_loss_score_term=best_evaluation.loss_score,
                        best_objective_loss_l1_term=best_evaluation.loss_regularizer_l1,
                        best_objective_loss_l2_term=best_evaluation.loss_regularizer_l2,
                        source_fields=source_fields,
                        scale_floor=scale_floor,
                    )
                    local_epoch_rows.extend(batch_rows)
                    store_epoch_rows(
                        batch_rows,
                        epoch_rows=epoch_rows_buffer,
                        epoch_writer=epoch_writer,
                    )

                log_prefix = (
                    f"family3_exact: mode=global_per_row row={idx} "
                    f"({row_position}/{len(current_candidate_index)}) "
                    f"target={target_score_name} direction={direction}"
                )
                if rb.family3_optimizer == "torch_adam":
                    _validate_torch_adam_request(
                        direction=direction,
                        target_score_name=target_score_name,
                        loss_name=rb.family3_loss_name,
                    )
                    descent = run_rowwise_torch_adam(
                        panel=panel,
                        base_ctx=base_ctx,
                        settings=settings,
                        row_index=idx,
                        mod_cols=mod_cols,
                        target_score_name=target_score_name,
                        direction=direction,
                        max_epochs=rb.family3_epochs,
                        max_eps=rb.family3_max_eps,
                        step_size=rb.pgd_step_size,
                        log_prefix=log_prefix,
                        evaluate_fn=_evaluate_row,
                        epoch_callback=_row_epoch_callback,
                        early_stop_loss=rb.family3_early_stop_loss,
                        early_stop_patience=rb.family3_early_stop_patience,
                        adam_beta1=rb.family3_adam_beta1,
                        adam_beta2=rb.family3_adam_beta2,
                        adam_eps=rb.family3_adam_eps,
                        plateau_shrink_patience=rb.family3_plateau_shrink_patience,
                        plateau_shrink_factor=rb.family3_plateau_shrink_factor,
                        min_step_size=rb.family3_min_step_size,
                        restore_best_on_shrink=rb.family3_restore_best_on_shrink,
                        reset_moments_on_shrink=rb.family3_reset_moments_on_shrink,
                        shared_ctx=shared_ctx,
                        firm_panel_cache=firm_panel_cache,
                    )
                else:
                    descent = run_optimizer(
                        optimizer_name=rb.family3_optimizer,
                        state_dim=len(mod_cols),
                        max_epochs=rb.family3_epochs,
                        max_eps=rb.family3_max_eps,
                        step_size=rb.pgd_step_size,
                        fd_step=rb.family3_fd_step,
                        random_seed=rb.random_seed + row_position * 10_000 + target_position * 100 + direction_position,
                        log_prefix=log_prefix,
                        evaluate_fn=_evaluate_row,
                        epoch_callback=_row_epoch_callback,
                        early_stop_loss=rb.family3_early_stop_loss,
                        early_stop_patience=rb.family3_early_stop_patience,
                        adam_beta1=rb.family3_adam_beta1,
                        adam_beta2=rb.family3_adam_beta2,
                        adam_eps=rb.family3_adam_eps,
                    )
                best_eval = descent.best_evaluation
                selected_epoch_row: dict[str, object] | None = None
                if local_epoch_rows:
                    valid_epoch_rows = [
                        row
                        for row in local_epoch_rows
                        if bool(row.get("success_flag", False))
                    ]
                    if valid_epoch_rows:
                        selected_epoch_row = min(
                            valid_epoch_rows,
                            key=lambda row: (
                                float(row.get("row_loss_l1_term", float("inf")))
                                + float(row.get("row_loss_l2_term", float("inf"))),
                                int(row.get("epoch", 10**9)),
                            ),
                        )
                    elif direction == "to_green":
                        selected_epoch_row = min(
                            local_epoch_rows,
                            key=lambda row: float(row.get("target_score_current", float("inf"))),
                        )
                    else:
                        selected_epoch_row = max(
                            local_epoch_rows,
                            key=lambda row: float(row.get("target_score_current", float("-inf"))),
                        )

                selected_x = best_eval.payload
                selected_target_z = np.nan
                selected_threshold = best_eval.threshold
                selected_row_loss_total = best_eval.loss_total
                selected_row_loss_score = best_eval.loss_score
                selected_row_loss_l1 = best_eval.loss_regularizer_l1
                selected_row_loss_l2 = best_eval.loss_regularizer_l2
                selected_epoch = descent.executed_epochs
                selection_basis = "optimizer_best"
                if selected_epoch_row is not None:
                    selected_x = np.asarray(
                        [float(selected_epoch_row[f"current_{col}"]) for col in mod_cols],
                        dtype=float,
                    )
                    selected_target_z = float(selected_epoch_row.get("target_score_current", np.nan))
                    selected_threshold = float(selected_epoch_row.get("target_threshold_z", best_eval.threshold))
                    selected_row_loss_total = float(selected_epoch_row.get("row_loss_total", best_eval.loss_total))
                    selected_row_loss_score = float(selected_epoch_row.get("row_loss_score_term", best_eval.loss_score))
                    selected_row_loss_l1 = float(selected_epoch_row.get("row_loss_l1_term", best_eval.loss_regularizer_l1))
                    selected_row_loss_l2 = float(selected_epoch_row.get("row_loss_l2_term", best_eval.loss_regularizer_l2))
                    selected_epoch = int(selected_epoch_row.get("epoch", descent.executed_epochs))
                    selection_basis = "min_valid_regularized_cost" if bool(selected_epoch_row.get("success_flag", False)) else "closest_failure"

                final_snapshot = family3_snapshot(
                    family3_candidate_panel(working_set.panel, [idx], np.asarray([selected_x], dtype=float), mod_cols),
                    settings,
                    base_ctx=base_ctx,
                    changed_columns=mod_cols,
                    changed_index=[idx],
                    snapshot_cache=working_set.snapshot_cache,
                    ratio_buffer=working_set.scratch_panel_with_ratios,
                    raw_zscores_buffer=working_set.scratch_raw_zscores,
                    merged_zscores_buffer=working_set.scratch_merged_zscores,
                    composites_buffer=working_set.scratch_composites,
                )
                baseline_target_z = family3_score_z_value(idx, target_score_name, baseline_snapshot)
                final_target_z = (
                    selected_target_z
                    if np.isfinite(selected_target_z)
                    else family3_score_z_value(idx, target_score_name, final_snapshot)
                )
                summary_rows.append(
                    build_summary_row(
                        run_id=run_id,
                        method_name=method_name,
                        mode="global_per_row",
                        direction=direction,
                        target_score_name=target_score_name,
                        target_family=target_family,
                        idx=idx,
                        baseline_snapshot=baseline_snapshot,
                        final_snapshot=final_snapshot,
                        baseline_target_z=baseline_target_z,
                        final_target_z=final_target_z,
                        target_threshold_z=selected_threshold,
                        executed_epochs=selected_epoch,
                        objective_scope="row",
                        objective_loss_total=selected_row_loss_total,
                        objective_loss_score_term=selected_row_loss_score,
                        objective_loss_l1_term=selected_row_loss_l1,
                        objective_loss_l2_term=selected_row_loss_l2,
                        row_loss_total=selected_row_loss_total,
                        row_loss_score_term=selected_row_loss_score,
                        row_loss_l1_term=selected_row_loss_l1,
                        row_loss_l2_term=selected_row_loss_l2,
                        baseline_x=baseline_x,
                        final_x=selected_x,
                        mod_cols=mod_cols,
                        settings=settings,
                        source_fields={
                            **(source_fields or {}),
                            "selected_epoch": selected_epoch,
                            "selection_basis": selection_basis,
                        },
                        scale_floor=scale_floor,
                    )
                )
    summary = attach_summary_group_metrics(pd.DataFrame(summary_rows))
    return summary, pd.DataFrame(epoch_rows_buffer or [])


def global_cohort(
    *,
    panel: pd.DataFrame,
    settings,
    base_ctx: ScoreContext,
    method_name: str,
    candidate_index: pd.Index | None = None,
    source_fields: dict[str, object] | None = None,
    epoch_writer: Callable[[list[dict[str, object]]], None] | None = None,
    keep_epoch_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = settings.robustness_benchmark
    mod_cols = tuple(col for col in rb.family3_modifiable_columns if col in panel.columns)
    if not mod_cols:
        return pd.DataFrame([{"method": method_name, "status": "no modifiable columns"}]), pd.DataFrame()
    _warn_if_benford_detectors_are_live(settings=settings, mod_cols=mod_cols)
    baseline_snapshot = family3_snapshot(panel, settings, base_ctx=base_ctx)
    summary_rows: list[dict[str, object]] = []
    epoch_rows_buffer: list[dict[str, object]] | None = [] if keep_epoch_rows else None
    # Built once per run (see profiling notes): the C-split reference stats
    # used by the incremental detector patch must come from the FULL panel,
    # not the row-restricted working panel, or they silently corrupt
    # detectors for any C-split firm whose own history isn't entirely in C.
    global_refs = family3_build_global_detector_references(panel, settings, base_ctx=base_ctx)
    # Built once per run (see the normalization-floor audit): a
    # per-column floor derived from each column's own scale across the full
    # panel, instead of one constant shared by every column regardless of its
    # natural units.
    scale_floor = family3_column_scale_floor(panel, mod_cols, rb.family3_scale_floor_quantile)
    for target_position, target_score_name in enumerate(rb.family3_target_scores, start=1):
        target_family = family3_target_score_kind(target_score_name)
        if target_family == "composite" and target_score_name not in baseline_snapshot.composites.columns:
            continue
        for direction_position, direction in enumerate(rb.family3_directions, start=1):
            current_candidate_index = target_candidate_index(
                baseline_snapshot,
                settings,
                target_score_name=target_score_name,
                direction=direction,
                candidate_index=candidate_index,
                sample_seed=rb.random_seed + target_position * 100 + direction_position,
                sample_limit=rb.family3_max_rows if candidate_index is None else None,
            )
            if len(current_candidate_index) == 0:
                summary_rows.append({"method": method_name, "mode": "global_cohort", "target_score_name": target_score_name, "direction": direction, "status": "no target-eligible rows"})
                continue
            row_indices = current_candidate_index.tolist()
            working_set = family3_build_working_set(
                panel,
                row_indices,
                mod_cols,
                settings,
                base_ctx=base_ctx,
                preserve_global_context=_target_requires_global_panel_context(target_score_name),
                global_refs=global_refs,
                scale_floor=scale_floor,
            )
            run_id = f"{method_name}_{direction}_{target_score_name}_cohort_{_utc_run_timestamp()}"

            def _evaluate_cohort(delta: np.ndarray) -> OptimizationEvaluation:
                candidate_matrix = np.maximum(working_set.baseline_matrix + np.asarray(delta, dtype=float) * working_set.scales, 0.0)
                loss_total, loss_score, loss_l1, loss_l2, threshold, snapshot = cohort_loss_and_snapshot(
                    working_set=working_set,
                    candidate_matrix=candidate_matrix,
                    target_score_name=target_score_name,
                    direction=direction,
                    base_ctx=base_ctx,
                    settings=settings,
                )
                return OptimizationEvaluation(
                    loss_total=loss_total,
                    loss_score=loss_score,
                    loss_regularizer=loss_l1 + loss_l2,
                    threshold=threshold,
                    artifact=snapshot,
                    payload=candidate_matrix,
                    loss_regularizer_l1=loss_l1,
                    loss_regularizer_l2=loss_l2,
                )

            def _cohort_epoch_callback(
                epoch: int,
                evaluation: OptimizationEvaluation,
                best_evaluation: OptimizationEvaluation,
            ) -> None:
                store_epoch_rows(
                    epoch_rows(
                        run_id=run_id,
                        mode="global_cohort",
                        direction=direction,
                        target_score_name=target_score_name,
                        target_score_family=target_family,
                        threshold_z=evaluation.threshold,
                        method_name=method_name,
                        epoch=epoch,
                        row_indices=row_indices,
                        snapshot=evaluation.artifact,
                        baseline_snapshot=baseline_snapshot,
                        baseline_x_by_idx={idx: working_set.baseline_matrix[pos] for pos, idx in enumerate(row_indices)},
                        candidate_x_by_idx={idx: evaluation.payload[pos] for pos, idx in enumerate(row_indices)},
                        columns=mod_cols,
                        settings=settings,
                        objective_scope="cohort",
                        objective_loss_total=evaluation.loss_total,
                        objective_loss_score_term=evaluation.loss_score,
                        objective_loss_l1_term=evaluation.loss_regularizer_l1,
                        objective_loss_l2_term=evaluation.loss_regularizer_l2,
                        best_snapshot=best_evaluation.artifact,
                        best_candidate_x_by_idx={idx: best_evaluation.payload[pos] for pos, idx in enumerate(row_indices)},
                        best_objective_loss_total=best_evaluation.loss_total,
                        best_objective_loss_score_term=best_evaluation.loss_score,
                        best_objective_loss_l1_term=best_evaluation.loss_regularizer_l1,
                        best_objective_loss_l2_term=best_evaluation.loss_regularizer_l2,
                        source_fields=source_fields,
                        scale_floor=scale_floor,
                    ),
                    epoch_rows=epoch_rows_buffer,
                    epoch_writer=epoch_writer,
                )

            log_prefix = (
                f"family3_exact: mode=global_cohort n_rows={len(row_indices)} "
                f"target={target_score_name} direction={direction}"
            )
            if rb.family3_optimizer == "torch_adam":
                _validate_torch_adam_request(
                    direction=direction,
                    target_score_name=target_score_name,
                    loss_name=rb.family3_loss_name,
                )
                descent = run_cohort_torch_adam(
                    panel=panel,
                    base_ctx=base_ctx,
                    settings=settings,
                    row_indices=row_indices,
                    mod_cols=mod_cols,
                    target_score_name=target_score_name,
                    direction=direction,
                    max_epochs=rb.family3_epochs,
                    max_eps=rb.family3_max_eps,
                    step_size=rb.pgd_step_size,
                    log_prefix=log_prefix,
                    evaluate_fn=_evaluate_cohort,
                    epoch_callback=_cohort_epoch_callback,
                    early_stop_loss=rb.family3_early_stop_loss,
                    early_stop_patience=rb.family3_early_stop_patience,
                    adam_beta1=rb.family3_adam_beta1,
                    adam_beta2=rb.family3_adam_beta2,
                    adam_eps=rb.family3_adam_eps,
                    plateau_shrink_patience=rb.family3_plateau_shrink_patience,
                    plateau_shrink_factor=rb.family3_plateau_shrink_factor,
                    min_step_size=rb.family3_min_step_size,
                    restore_best_on_shrink=rb.family3_restore_best_on_shrink,
                    reset_moments_on_shrink=rb.family3_reset_moments_on_shrink,
                )
            else:
                descent = run_optimizer(
                    optimizer_name=rb.family3_optimizer,
                    state_dim=len(mod_cols),
                    max_epochs=rb.family3_epochs,
                    max_eps=rb.family3_max_eps,
                    step_size=rb.pgd_step_size,
                    fd_step=rb.family3_fd_step,
                    random_seed=rb.random_seed + len(row_indices) * 10_000 + target_position * 100 + direction_position,
                    log_prefix=log_prefix,
                    evaluate_fn=_evaluate_cohort,
                    epoch_callback=_cohort_epoch_callback,
                    early_stop_loss=rb.family3_early_stop_loss,
                    early_stop_patience=rb.family3_early_stop_patience,
                    adam_beta1=rb.family3_adam_beta1,
                    adam_beta2=rb.family3_adam_beta2,
                    adam_eps=rb.family3_adam_eps,
                )
            best_eval = descent.best_evaluation
            final_snapshot = family3_snapshot(
                family3_candidate_panel(working_set.panel, row_indices, best_eval.payload, mod_cols),
                settings,
                base_ctx=base_ctx,
                changed_columns=mod_cols,
                changed_index=row_indices,
                snapshot_cache=working_set.snapshot_cache,
                ratio_buffer=working_set.scratch_panel_with_ratios,
                raw_zscores_buffer=working_set.scratch_raw_zscores,
                merged_zscores_buffer=working_set.scratch_merged_zscores,
                composites_buffer=working_set.scratch_composites,
            )
            for pos, idx in enumerate(row_indices):
                baseline_target_z = family3_score_z_value(idx, target_score_name, baseline_snapshot)
                final_target_z = family3_score_z_value(idx, target_score_name, final_snapshot)
                final_row_loss_total, final_row_loss_score, final_row_loss_l1, final_row_loss_l2, _ = family3_exact_loss(
                    target_score_z=final_target_z,
                    direction=direction,
                    delta_norm=family3_normalized_delta(best_eval.payload[pos], working_set.baseline_matrix[pos], working_set.scales[pos]),
                    settings=settings,
                )
                summary_rows.append(
                    build_summary_row(
                        run_id=run_id,
                        method_name=method_name,
                        mode="global_cohort",
                        direction=direction,
                        target_score_name=target_score_name,
                        target_family=target_family,
                        idx=idx,
                        baseline_snapshot=baseline_snapshot,
                        final_snapshot=final_snapshot,
                        baseline_target_z=baseline_target_z,
                        final_target_z=final_target_z,
                        target_threshold_z=best_eval.threshold,
                        executed_epochs=descent.executed_epochs,
                        objective_scope="cohort",
                        objective_loss_total=best_eval.loss_total,
                        objective_loss_score_term=best_eval.loss_score,
                        objective_loss_l1_term=best_eval.loss_regularizer_l1,
                        objective_loss_l2_term=best_eval.loss_regularizer_l2,
                        row_loss_total=final_row_loss_total,
                        row_loss_score_term=final_row_loss_score,
                        row_loss_l1_term=final_row_loss_l1,
                        row_loss_l2_term=final_row_loss_l2,
                        baseline_x=working_set.baseline_matrix[pos],
                        final_x=best_eval.payload[pos],
                        mod_cols=mod_cols,
                        settings=settings,
                        source_fields=source_fields,
                        scale_floor=scale_floor,
                    )
                )
    summary = attach_summary_group_metrics(pd.DataFrame(summary_rows))
    return summary, pd.DataFrame(epoch_rows_buffer or [])


def run_family3_xai(
    panel: pd.DataFrame,
    settings,
    base_ctx: ScoreContext,
    *,
    method_name: str,
    candidate_index: pd.Index | None = None,
    source_fields: dict[str, object] | None = None,
    epoch_writer: Callable[[list[dict[str, object]]], None] | None = None,
    keep_epoch_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = settings.robustness_benchmark
    summary_parts: list[pd.DataFrame] = []
    epoch_parts: list[pd.DataFrame] = []
    if "global_per_row" in rb.family3_modes:
        s, e = global_per_row(
            panel=panel,
            settings=settings,
            base_ctx=base_ctx,
            method_name=method_name,
            candidate_index=candidate_index,
            source_fields=source_fields,
            epoch_writer=epoch_writer,
            keep_epoch_rows=keep_epoch_rows,
        )
        summary_parts.append(s)
        epoch_parts.append(e)
    if "global_cohort" in rb.family3_modes:
        s, e = global_cohort(
            panel=panel,
            settings=settings,
            base_ctx=base_ctx,
            method_name=method_name,
            candidate_index=candidate_index,
            source_fields=source_fields,
            epoch_writer=epoch_writer,
            keep_epoch_rows=keep_epoch_rows,
        )
        summary_parts.append(s)
        epoch_parts.append(e)
    if "local_firm_single" in rb.family3_modes and rb.family3_local_firm_case:
        s, e = local_firm_single(
            panel=panel,
            settings=settings,
            base_ctx=base_ctx,
            method_name=method_name,
            source_fields=source_fields,
            epoch_writer=epoch_writer,
            keep_epoch_rows=keep_epoch_rows,
        )
        summary_parts.append(s)
        epoch_parts.append(e)
    nonempty_summary_parts = [df for df in summary_parts if not df.empty]
    nonempty_epoch_parts = [df for df in epoch_parts if not df.empty]
    summary = pd.concat(nonempty_summary_parts, ignore_index=True, sort=False) if nonempty_summary_parts else pd.DataFrame()
    epochs = pd.concat(nonempty_epoch_parts, ignore_index=True, sort=False) if nonempty_epoch_parts else pd.DataFrame()
    if not summary.empty and not epochs.empty and "run_id" in summary.columns and "run_id" in epochs.columns:
        epoch_json = (
            epochs.groupby("run_id", dropna=False)
            .apply(lambda grp: grp.replace({np.nan: None}).to_json(orient="records"))
            .rename("epoch_log_json")
            .reset_index()
        )
        summary = summary.merge(epoch_json, on="run_id", how="left")
    return summary, epochs
