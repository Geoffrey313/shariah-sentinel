"""Utility helpers for canonical Family 3 counterfactual robustness.

This module groups the stable typed structures, fast exact row evaluators,
and lightweight reporting helpers used by the counterfactual engine.
"""


from dataclasses import dataclass

import numpy as np
import pandas as pd


COMPOSITE_VALUE_COLUMNS: tuple[str, ...] = (
    "z_plus",
    "z_plus_renorm",
    "breadth",
    "z_mahalanobis_sq",
    "t_iut",
    "z_plus_softmax",
    "z_plus_orth",
)
COMPOSITE_P_COLUMNS: tuple[str, ...] = (
    "p_z_plus",
    "p_z_plus_renorm",
    "p_breadth",
    "p_z_mahalanobis_sq",
    "p_t_iut",
    "p_z_plus_softmax",
    "p_z_plus_orth",
)
# RED verdict namespace (M6): only the four primary composites of eq. (13). The
# softmax/orthogonal variants and breadth are reporting extensions and must not
# decide the family3 RED flag, so the counterfactual erases the same verdict the
# paper presents as canonical (not a wider namespace).
VERDICT_P_COLUMNS: tuple[str, ...] = (
    "p_z_plus",
    "p_z_plus_renorm",
    "p_z_mahalanobis_sq",
    "p_t_iut",
)
@dataclass(frozen=True)
class ScoreContext:
    panel: pd.DataFrame
    raw_zscores: pd.DataFrame
    zscores: pd.DataFrame
    composites: pd.DataFrame
    active: tuple[str, ...]
    weights: np.ndarray
    sigma: np.ndarray
    null: object
    complete_c_rows: np.ndarray
    complete_c_index: np.ndarray


@dataclass(frozen=True)
class Family3Snapshot:
    panel: pd.DataFrame
    raw_zscores: pd.DataFrame
    merged_zscores: pd.DataFrame
    composites: pd.DataFrame


"""Fast single-row Family 3 evaluation helpers.

This module hosts the low-latency evaluation utilities that used to live in
``server.robustness.benchmark``. Keeping them here gives two benefits:

1. The exact Family 3 package exposes a clearer computational boundary.
2. Experimental callers such as the torch surrogate no longer depend on
   private helpers from the benchmark orchestration module.

The helpers below are still intentionally narrow and benchmark-oriented. They
reuse the authoritative score context and null distributions, but only
recompute the small subset of row-sensitive detectors needed for fast
candidate scoring.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.ratio_inputs import get_canonical_sharia_ratios
from src.engine.detector_composite import orthogonal_softmax, softmax_active
from src.engine.proximity import _firm_raw_statistics, _proximity_raw_statistic
from src.engine.peer import _get_global_reference, _resolve_reference_for_row, _score_peer_row
from src.engine.cost_of_debt import _implied_cost_of_debt, _t8_per_firm, _t8_vectorized
from src.engine.pit import pit_empirical
from src.common.methodology import thresholds_for_panel
from src.data.ratios import compute_shariah_ratios
from src.engine.composites import breadth, iut_statistic, mahalanobis_squared, renormalised_truncated_sum, truncated_sum
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED
from src.engine.bootstrap import upper_tail_pvalue

COL_P_Z_MAHALANOBIS = "p_z_mahalanobis_sq"
COL_P_Z_PLUS = "p_z_plus"
COL_P_Z_PLUS_RENORM = "p_z_plus_renorm"
COL_P_BREADTH = "p_breadth"
COL_P_T_IUT = "p_t_iut"
COL_P_Z_PLUS_SOFTMAX = "p_z_plus_softmax"
COL_P_Z_PLUS_ORTH = "p_z_plus_orth"


def _coerce_float(value: object, *, default: float = np.nan) -> float:
    """Return a finite float when possible, otherwise ``default``."""
    numeric = float(pd.to_numeric(value, errors="coerce"))
    return numeric if np.isfinite(numeric) else default


def family3_current_ratios(row: pd.Series) -> np.ndarray:
    """Extract the three canonical Shariah ratios from a row-like object."""
    return np.array(
        [
            _coerce_float(row.get("ratio_debt_adj")),
            _coerce_float(row.get("ratio_cash_adj")),
            _coerce_float(row.get("ratio_income")),
        ],
        dtype=float,
    )


def family3_reference_ratio_stats(base_ctx: ScoreContext) -> dict[str, float]:
    """Compute mean/std calibration statistics for the canonical ratios."""
    work = base_ctx.panel.copy()
    if "_split" in work.columns:
        work = work[work["_split"] == SPLIT_LABEL_INCLUDED].copy()
    ratio_cols = ["ratio_debt_adj", "ratio_cash_adj", "ratio_income"]
    if work.empty:
        work = base_ctx.panel.copy()
    stats: dict[str, float] = {}
    for col, prefix in zip(ratio_cols, ("debt", "cash", "income")):
        vals = pd.to_numeric(work.get(col, pd.Series(np.nan, index=work.index)), errors="coerce")
        center = float(vals.mean()) if vals.notna().any() else 0.0
        scale = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 1.0
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        stats[f"ratio_center_{prefix}"] = center
        stats[f"ratio_scale_{prefix}"] = scale
    return stats


def family3_column_scale_floor(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    quantile: float,
) -> np.ndarray:
    """Per-column normalization floor derived from the panel's own scale.

    Family 3 normalizes each candidate delta by ``max(|baseline|, floor)`` so a
    fixed L1 budget means the same thing for every modifiable column. A single
    constant floor is inconsistent across columns with very different natural
    units (e.g. `atq`, total assets, vs `xintq`, interest expense): a firm
    whose `xintq` happens to sit near zero would get an artificially cheap,
    economically meaningless large relative move, while the same constant is
    negligible for `atq`. Using each column's own typical nonzero magnitude
    across the full panel keeps the floor meaningful in that column's own
    units instead of one arbitrary constant shared by every column.

    Args:
        panel: Full production panel (not the row-restricted working panel).
        columns: Modifiable raw columns, in the order callers will use them.
        quantile: Lower quantile of each column's nonzero absolute values used
            as its floor.

    Returns:
        Array of shape ``(len(columns),)`` with one floor value per column.
    """
    floors = np.ones(len(columns), dtype=float)
    for pos, col in enumerate(columns):
        if col not in panel.columns:
            continue
        values = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=float)
        nonzero_abs = np.abs(values[np.isfinite(values) & (values != 0.0)])
        if nonzero_abs.size == 0:
            continue
        floors[pos] = max(float(np.quantile(nonzero_abs, quantile)), 1e-6)
    return floors


def family3_ratio_scale_vector(base_ctx: ScoreContext) -> np.ndarray:
    """Return ratio standard deviations as a compact vector."""
    stats = family3_reference_ratio_stats(base_ctx)
    return np.array(
        [
            stats["ratio_scale_debt"],
            stats["ratio_scale_cash"],
            stats["ratio_scale_income"],
        ],
        dtype=float,
    )


def finite_or_default(value: object, default: float = 0.0) -> float:
    """Coerce to a finite float, falling back to ``default`` otherwise."""
    return _coerce_float(value, default=default)


@dataclass
class Family3FastRowEvaluator:
    """Evaluate the primary composite p-value for one attacked row.

    The evaluator keeps the authoritative panel-level score context fixed and
    only refreshes the detectors that respond directly to the attacked raw
    variables: ``z4``, ``z57`` and ``z8``.
    """

    base_ctx: ScoreContext
    idx: object
    primary: str
    settings: AnalysisSettings

    _Z4_NAME = "z4"
    _Z57_NAME = "z57"
    _Z8_NAME = "z8"

    def __post_init__(self) -> None:
        self.active = self.base_ctx.active
        self.null = self.base_ctx.null
        self.sigma = self.base_ctx.sigma
        self.weights = self.base_ctx.weights

        active_list = list(self.active)
        self._z_base = self.base_ctx.zscores.loc[self.idx, active_list].to_numpy(dtype=float).copy()

        self._z4_idx = active_list.index(self._Z4_NAME) if self._Z4_NAME in active_list else None
        self._z57_idx = active_list.index(self._Z57_NAME) if self._Z57_NAME in active_list else None
        self._z8_idx = active_list.index(self._Z8_NAME) if self._Z8_NAME in active_list else None

        panel = self.base_ctx.panel
        c_mask = panel["_split"] == SPLIT_LABEL_INCLUDED
        c_panel = panel[c_mask]
        # Computed once and stored so every method below (`_recompute_z4`,
        # `evaluate_details`) reads the panel's actual methodology/country
        # thresholds instead of the hardcoded Malaysia-only `SAC_THRESHOLDS`
        # (see profiling notes) -- required for future non-MY panels.
        self._thresholds = thresholds_for_panel(panel)

        if self._z4_idx is not None and "gvkey" in panel.columns:
            gvkey = panel.at[self.idx, "gvkey"]
            thresholds = self._thresholds
            ratio_cols = [c for c in thresholds if c in panel.columns]
            self._z4_firm_ratios = panel.loc[panel["gvkey"] == gvkey, ratio_cols].copy()
            self._z4_firm_idx = self.idx
            c_ratios_z4 = c_panel[[c for c in thresholds if c in c_panel.columns]]
            raw_c = _firm_raw_statistics(c_panel, c_ratios_z4, thresholds)
            self._z4_pit_ref = np.array([v for v in raw_c.values() if np.isfinite(v)], dtype=float)
        else:
            self._z4_firm_ratios = None
            self._z4_firm_idx = self.idx
            self._z4_pit_ref = np.array([], dtype=float)

        if self._z57_idx is not None:
            c_ratios_df = get_canonical_sharia_ratios(c_panel)
            self._z7_n_dims = len(c_ratios_df.columns) if not c_ratios_df.empty else 0
            if not c_ratios_df.empty:
                row_in_c = self.idx in c_panel.index
                lookup = c_panel if row_in_c else pd.concat([c_panel, panel.loc[[self.idx]]])
                ctx = _get_global_reference(lookup, get_canonical_sharia_ratios(lookup))
                self._z7_ref = _resolve_reference_for_row(self.idx, ctx) if ctx is not None else None
                if self._z7_ref is not None:
                    base_ratio_vec = get_canonical_sharia_ratios(panel.loc[[self.idx]])
                    if not base_ratio_vec.empty:
                        rv = base_ratio_vec.iloc[0].to_numpy(dtype=float)
                        mu, inv, n_ref, _ = self._z7_ref
                        try:
                            z7_base = _score_peer_row(rv, mu, inv, n_ref, self._z7_n_dims)
                            z57_base = float(self._z_base[self._z57_idx])
                            self._z5_baseline = float(z57_base - z7_base) if np.isfinite(z7_base) else z57_base
                        except Exception:
                            self._z5_baseline = float(self._z_base[self._z57_idx])
                    else:
                        self._z5_baseline = float(self._z_base[self._z57_idx])
                else:
                    self._z5_baseline = float(self._z_base[self._z57_idx])
            else:
                self._z7_n_dims = 0
                self._z7_ref = None
                self._z5_baseline = float(self._z_base[self._z57_idx]) if self._z57_idx is not None else 0.0
        else:
            self._z7_n_dims = 0
            self._z7_ref = None
            self._z5_baseline = 0.0

        if self._z8_idx is not None and "gvkey" in panel.columns and "datacqtr" in panel.columns:
            gvkey = panel.at[self.idx, "gvkey"]
            firm_panel = panel.loc[panel["gvkey"] == gvkey].sort_values("datacqtr")
            self._z8_firm_pos = int((firm_panel.index == self.idx).argmax())
            self._z8_cod_history = _implied_cost_of_debt(firm_panel).to_numpy(dtype=float)
            self._z8_xintq = _coerce_float(panel.at[self.idx, "xintq"] if "xintq" in panel.columns else np.nan)
            # Vectorized (see profiling notes): this reference is invariant
            # across rows, but was previously rebuilt via a per-firm Python
            # loop every time a row evaluator is constructed -- with 88
            # evaluators built per local_firm_single run, that meant ~88x
            # the whole C panel's firms re-looped with the old per-firm
            # `_t8_per_firm`. Order across firms doesn't matter here (only
            # the finite T8 *set* feeds the empirical PIT reference).
            c_sorted = c_panel.sort_values(["gvkey", "datacqtr"])
            cod_sorted = _implied_cost_of_debt(c_sorted).to_numpy(dtype=float)
            firm_codes_z8, _ = pd.factorize(c_sorted["gvkey"], sort=False)
            t8_sorted = _t8_vectorized(cod_sorted, firm_codes_z8)
            self._z8_pit_ref = t8_sorted[np.isfinite(t8_sorted)]
        else:
            self._z8_firm_pos = -1
            self._z8_cod_history = np.array([], dtype=float)
            self._z8_xintq = np.nan
            self._z8_pit_ref = np.array([], dtype=float)

        self._null_attr = {
            COL_P_Z_MAHALANOBIS: "z_mahalanobis_sorted",
            COL_P_Z_PLUS: "z_plus_sorted",
            COL_P_Z_PLUS_RENORM: "z_plus_renorm_sorted",
            COL_P_BREADTH: "breadth_sorted",
            COL_P_T_IUT: "t_iut_sorted",
            COL_P_Z_PLUS_SOFTMAX: "z_plus_softmax_sorted",
            COL_P_Z_PLUS_ORTH: "z_plus_orth_sorted",
        }.get(self.primary)

    def _composite_value(self, z_vec: np.ndarray) -> float:
        z_row = np.asarray(z_vec, dtype=float).reshape(1, -1)
        finite_mask = np.isfinite(z_row)
        z_fill = np.where(finite_mask, z_row, 0.0)
        if self.primary == COL_P_Z_MAHALANOBIS:
            return float(mahalanobis_squared(z_row, self.sigma)[0])
        if self.primary == COL_P_Z_PLUS:
            return float(truncated_sum(z_row, self.weights)[0])
        if self.primary == COL_P_Z_PLUS_RENORM:
            return float(
                renormalised_truncated_sum(
                    z_row,
                    self.weights,
                    threshold=self.settings.composites.active_set_threshold,
                    min_active=self.settings.composites.min_active_for_renorm,
                )[0]
            )
        if self.primary == COL_P_BREADTH:
            return float(breadth(z_row, threshold=self.settings.composites.active_set_threshold)[0])
        if self.primary == COL_P_T_IUT:
            return float(iut_statistic(z_row, min_active=self.settings.composites.min_active_for_iut)[0])
        if self.primary == COL_P_Z_PLUS_SOFTMAX:
            return float(
                softmax_active(z_fill, finite_mask, self.weights, gamma=self.settings.composites.softmax_gamma)[0]
            )
        if self.primary == COL_P_Z_PLUS_ORTH:
            return float(
                orthogonal_softmax(
                    z_fill,
                    finite_mask,
                    self.weights,
                    gamma=self.settings.composites.softmax_gamma,
                    Sigma=self.sigma,
                )[0]
            )
        return float("nan")

    def _recompute_z4(self, new_ratios: dict[str, float]) -> float | None:
        if self._z4_firm_ratios is None or len(self._z4_pit_ref) == 0:
            return None
        updated = self._z4_firm_ratios.copy()
        for col, val in new_ratios.items():
            if col in updated.columns:
                updated.at[self._z4_firm_idx, col] = val
        per_ratio = [
            t
            for ratio_name, threshold in self._thresholds.items()
            if ratio_name in updated.columns
            for t in [_proximity_raw_statistic(updated[ratio_name].to_numpy(), threshold)]
            if np.isfinite(t)
        ]
        if not per_ratio:
            return None
        z4_arr = pit_empirical(np.array([float(np.max(per_ratio))]), self._z4_pit_ref)
        return float(z4_arr[0]) if np.isfinite(z4_arr[0]) else None

    def _recompute_z57(self, ratio_vec: np.ndarray) -> float | None:
        if self._z7_ref is None or self._z7_n_dims == 0 or not np.isfinite(ratio_vec).all():
            return None
        mu_ref, inv_ref, n_ref, _ = self._z7_ref
        try:
            z7_new = _score_peer_row(ratio_vec, mu_ref, inv_ref, n_ref, self._z7_n_dims)
        except Exception:
            return None
        if not np.isfinite(z7_new):
            return None
        return float(max(self._z5_baseline, z7_new))

    def _recompute_z8(self, new_dlttq: float, new_dlcq: float) -> float | None:
        if self._z8_firm_pos < 0 or len(self._z8_pit_ref) == 0 or not np.isfinite(self._z8_xintq):
            return None
        new_debt = max(new_dlttq, 0.0) + max(new_dlcq, 0.0)
        if new_debt <= 0:
            return None
        new_cod = self._z8_xintq / new_debt
        cod_series = self._z8_cod_history.copy()
        cod_series[self._z8_firm_pos] = new_cod
        t8_series = _t8_per_firm(cod_series)
        t8_val = t8_series[self._z8_firm_pos]
        if not np.isfinite(t8_val):
            return None
        z8_arr = pit_empirical(np.array([t8_val]), self._z8_pit_ref)
        return float(z8_arr[0]) if np.isfinite(z8_arr[0]) else None

    def _pvalue_from_z_vec(self, z_vec: np.ndarray) -> float:
        if self._null_attr is None:
            return float("nan")
        null_sorted = getattr(self.null, self._null_attr, None)
        if null_sorted is None:
            return float("nan")
        try:
            value = self._composite_value(z_vec)
            if not np.isfinite(value):
                return float("nan")
            return float(upper_tail_pvalue(np.array([value]), null_sorted, self.null.n_replicates)[0])
        except Exception:
            return float("nan")

    def evaluate_details(self, candidate_row: pd.Series) -> dict[str, float]:
        """Return fast-updated detector/composite details for one candidate row."""
        z_vec = self._z_base.copy()
        new_ratios = {col: _coerce_float(candidate_row.get(col)) for col in self._thresholds}
        ratio_vec = np.array(list(new_ratios.values()), dtype=float)
        new_dlttq = finite_or_default(candidate_row.get("dlttq"), 0.0)
        new_dlcq = finite_or_default(candidate_row.get("dlcq"), 0.0)

        if self._z4_idx is not None:
            value = self._recompute_z4(new_ratios)
            if value is not None:
                z_vec[self._z4_idx] = value
        if self._z57_idx is not None:
            value = self._recompute_z57(ratio_vec)
            if value is not None:
                z_vec[self._z57_idx] = value
        if self._z8_idx is not None:
            value = self._recompute_z8(new_dlttq, new_dlcq)
            if value is not None:
                z_vec[self._z8_idx] = value

        primary_value = self._composite_value(z_vec)
        primary_p = self._pvalue_from_z_vec(z_vec)
        return {
            "primary_composite_value": float(primary_value) if np.isfinite(primary_value) else np.nan,
            "primary_composite_p": float(primary_p) if np.isfinite(primary_p) else np.nan,
            "z4": float(z_vec[self._z4_idx]) if self._z4_idx is not None and np.isfinite(z_vec[self._z4_idx]) else np.nan,
            "z57": float(z_vec[self._z57_idx]) if self._z57_idx is not None and np.isfinite(z_vec[self._z57_idx]) else np.nan,
            "z8": float(z_vec[self._z8_idx]) if self._z8_idx is not None and np.isfinite(z_vec[self._z8_idx]) else np.nan,
        }

    def evaluate(self, candidate_row: pd.Series) -> float:
        """Return the fast-evaluated primary composite p-value."""
        details = self.evaluate_details(candidate_row)
        return float(details["primary_composite_p"])


import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd



def family3_summary_table(family3: pd.DataFrame) -> pd.DataFrame:
    if family3.empty:
        return pd.DataFrame()
    work = family3.copy()
    group_cols = [c for c in ("method", "mode", "direction", "target_score_name", "target_score_family", "source_family2_method", "source_rho", "source_delta") if c in work.columns]
    rows: list[dict[str, object]] = []
    for key, group in work.groupby(group_cols, dropna=False) if group_cols else [((), work)]:
        row: dict[str, object] = {"n_rows": int(len(group))}
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            for col, value in zip(group_cols, key):
                row[col] = value
        for col in (
            "success",
            "evasion_rate",
            "median_cost",
            "p90_cost",
            "median_l2_cost",
            "p90_l2_cost",
            "median_row_cost",
            "p90_row_cost",
            "loss_total",
            "loss_l1_term",
            "loss_l2_term",
            "objective_loss_total",
            "objective_loss_score_term",
            "objective_loss_l1_term",
            "objective_loss_l2_term",
            "row_loss_total",
            "row_loss_score_term",
            "row_loss_l1_term",
            "row_loss_l2_term",
        ):
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce")
                row[col] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def family3_source_mechanism_table(family3: pd.DataFrame) -> pd.DataFrame:
    if family3.empty or "source_family2_method" not in family3.columns:
        return pd.DataFrame()
    work = family3.copy()
    group_cols = ["method", "source_family2_method"]
    for maybe in ("source_rho", "source_delta"):
        if maybe in work.columns:
            group_cols.append(maybe)
    rows: list[dict[str, object]] = []
    for _, group in work.groupby(group_cols, dropna=False):
        row: dict[str, object] = {
            "method": group["method"].iloc[0] if "method" in group.columns else "",
            "source_family2_method": group["source_family2_method"].iloc[0],
            "n_rows": int(len(group)),
        }
        for maybe in ("source_rho", "source_delta"):
            if maybe in group.columns:
                row[maybe] = group[maybe].iloc[0]
        for maybe in ("source_n_contaminated", "source_n_red_contaminated", "source_n_orange_contaminated"):
            if maybe in group.columns:
                vals = pd.to_numeric(group[maybe], errors="coerce")
                row[maybe] = float(vals.max()) if vals.notna().any() else np.nan
        for col in (
            "success",
            "evasion_rate",
            "baseline_primary_p",
            "final_primary_p",
            "loss_l1_term",
            "loss_l2_term",
            "loss_total",
            "objective_loss_total",
            "objective_loss_score_term",
            "objective_loss_l1_term",
            "objective_loss_l2_term",
            "row_loss_total",
            "row_loss_score_term",
            "row_loss_l1_term",
            "row_loss_l2_term",
            "median_row_cost",
            "p90_row_cost",
        ):
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce")
                row[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def family3_detailed_table(family3: pd.DataFrame) -> pd.DataFrame:
    if family3.empty:
        return pd.DataFrame()
    preferred_cols = [
        "run_id",
        "method",
        "mode",
        "direction",
        "target_score_name",
        "target_score_family",
        "source_family2_method",
        "source_rho",
        "source_delta",
        "row_index",
        "firm_id",
        "quarter",
        "primary_composite",
        "baseline_primary_p",
        "final_primary_p",
        "baseline_status",
        "final_status",
        "success",
        "epochs_run",
        "objective_scope",
        "target_threshold_z",
        "baseline_target_score_z",
        "final_target_score_z",
        "objective_loss_total",
        "objective_loss_score_term",
        "objective_loss_l1_term",
        "objective_loss_l2_term",
        "row_loss_total",
        "row_loss_score_term",
        "row_loss_l1_term",
        "row_loss_l2_term",
        "loss_total",
        "loss_score_term",
        "loss_l1_term",
        "loss_l2_term",
        "top_driver",
        "top_driver_abs_delta",
    ]
    # The modifiable raw-variable set is configuration-driven (default 19
    # columns), not the legacy 4 SAC-targeted columns. Derive it from the
    # emitted ``delta_*`` columns, which the summary builder writes only for
    # modifiable raw variables (metric columns get explicit keys, never a
    # ``delta_`` prefix), so every attacked variable's before/after/delta values
    # are surfaced rather than silently dropped.
    raw_cols = sorted(c[len("delta_"):] for c in family3.columns if c.startswith("delta_"))
    mod_cols = [f"baseline_{c}" for c in raw_cols if f"baseline_{c}" in family3.columns]
    final_cols = [f"final_{c}" for c in raw_cols if f"final_{c}" in family3.columns]
    delta_cols = [f"delta_{c}" for c in raw_cols]
    keep = [c for c in preferred_cols if c in family3.columns] + mod_cols + final_cols + delta_cols
    return family3[keep].copy()


def family3_case_cards(family3: pd.DataFrame) -> str:
    if family3.empty:
        return ""
    cards: list[str] = []
    note = (
        "Note: `top_driver` is the variable with the largest absolute normalized delta in the selected "
        "counterfactual. It is a useful heuristic, not a causal attribution score."
    )
    for _, row in family3.iterrows():
        row_index = pd.to_numeric(row.get("row_index"), errors="coerce")
        row_label = str(int(row_index)) if np.isfinite(row_index) else "status"
        epochs_run = pd.to_numeric(row.get("epochs_run"), errors="coerce")
        epochs_run_label = int(epochs_run) if np.isfinite(epochs_run) else "NA"
        direction = str(row.get("direction", ""))
        escaped_label = "ESCAPED RED" if direction == "to_green" else "ESCAPED GREEN"
        remained_label = "REMAINED RED" if direction == "to_green" else "REMAINED GREEN"
        status = escaped_label if bool(row.get("success")) else remained_label
        objective_scope = row.get("objective_scope", "row")
        cards.append(
            (
                f"### Row {row_label} — {row.get('firm_id', '')} / {row.get('quarter', '')}\n"
                f"- Status: **{status}**\n"
                f"- Method: `{row.get('method', '')}`\n"
                f"- Mode / direction: `{row.get('mode', '')}` / `{row.get('direction', '')}`\n"
                f"- Target score: `{row.get('target_score_name', '')}`\n"
                f"- Primary p-value: `{row.get('baseline_primary_p', np.nan):.6f}` -> `{row.get('final_primary_p', np.nan):.6f}`\n"
                f"- Epochs: `{epochs_run_label}`\n"
                f"- Target z: `{row.get('baseline_target_score_z', np.nan):.6f}` -> `{row.get('final_target_score_z', np.nan):.6f}`\n"
                f"- Objective scope: `{objective_scope}`\n"
                f"- Objective loss: `{row.get('objective_loss_total', row.get('loss_total', np.nan)):.6f}` "
                f"(score `{row.get('objective_loss_score_term', row.get('loss_score_term', np.nan)):.6f}`, "
                f"L1 `{row.get('objective_loss_l1_term', row.get('loss_l1_term', np.nan)):.6f}`, "
                f"L2 `{row.get('objective_loss_l2_term', row.get('loss_l2_term', np.nan)):.6f}`)\n"
                f"- Row loss: `{row.get('row_loss_total', row.get('loss_total', np.nan)):.6f}` "
                f"(score `{row.get('row_loss_score_term', row.get('loss_score_term', np.nan)):.6f}`, "
                f"L1 `{row.get('row_loss_l1_term', row.get('loss_l1_term', np.nan)):.6f}`, "
                f"L2 `{row.get('row_loss_l2_term', row.get('loss_l2_term', np.nan)):.6f}`)\n"
                f"- Top driver: `{row.get('top_driver', '')}` (|delta| `{row.get('top_driver_abs_delta', np.nan):.6f}`)"
            )
        )
    return note + "\n\n" + "\n\n".join(cards)


def family3_majority_vote_table(family3: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-target top drivers into a simple majority-vote table."""
    if family3.empty or "top_driver" not in family3.columns or "target_score_name" not in family3.columns:
        return pd.DataFrame()
    work = family3.copy()
    work["top_driver"] = work["top_driver"].fillna("").astype(str)
    work = work[work["top_driver"].str.len() > 0].copy()
    if work.empty:
        return pd.DataFrame()
    case_cols = [c for c in ("method", "mode", "direction", "row_index", "firm_id", "quarter") if c in work.columns]
    target_counts = (
        work.groupby(case_cols, dropna=False)["target_score_name"]
        .nunique()
        .rename("n_target_scores")
        .reset_index()
    )
    vote = (
        work.groupby(case_cols + ["top_driver"], dropna=False)
        .agg(
            majority_votes=("target_score_name", "nunique"),
            mean_abs_delta=("top_driver_abs_delta", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            max_abs_delta=("top_driver_abs_delta", lambda s: float(pd.to_numeric(s, errors="coerce").max())),
        )
        .reset_index()
    )
    vote = vote.merge(target_counts, on=case_cols, how="left")
    vote["vote_share"] = vote["majority_votes"] / vote["n_target_scores"]
    vote["driver_vote_basis"] = "max_abs_delta"
    vote["driver_vote_note"] = (
        "Heuristic majority vote over per-target top drivers ranked by |delta|; not a causal attribution."
    )
    vote = vote.sort_values(case_cols + ["majority_votes", "vote_share", "mean_abs_delta"], ascending=[True] * len(case_cols) + [False, False, False])
    return vote


def family3_history_table(family3: pd.DataFrame) -> pd.DataFrame:
    """Expand stored JSON epoch histories into a flat dataframe.

    The reporting layer should be resilient to one malformed payload rather
    than crashing the whole render pass.
    """
    if family3.empty or "epoch_log_json" not in family3.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in family3.iterrows():
        payload = row.get("epoch_log_json")
        if pd.isna(payload):
            continue
        try:
            rows.extend(json.loads(payload))
        except (TypeError, json.JSONDecodeError):
            continue
    return pd.DataFrame(rows)


def write_family3_paper_outputs(
    *,
    family3: pd.DataFrame,
    paper_dir: Path,
    to_latex_table: Callable[..., str],
) -> dict[str, pd.DataFrame]:
    """Render the Family-3 paper-ready artifacts outside the benchmark module."""
    family3_df = family3_summary_table(family3)
    family3_source_df = family3_source_mechanism_table(family3)
    family3_detail_df = family3_detailed_table(family3)
    family3_hist_df = family3_history_table(family3)
    family3_vote_df = family3_majority_vote_table(family3)

    family3_df.to_csv(paper_dir / "family3_summary_table.csv", index=False)
    family3_source_df.to_csv(paper_dir / "family3_source_mechanism_table.csv", index=False)
    family3_detail_df.to_csv(paper_dir / "family3_detailed_cases_table.csv", index=False)
    family3_hist_df.to_csv(paper_dir / "family3_history_table.csv", index=False)
    family3_vote_df.to_csv(paper_dir / "family3_majority_vote_table.csv", index=False)

    if not family3_detail_df.empty:
        (paper_dir / "family3_case_cards.md").write_text(
            family3_case_cards(family3_detail_df),
            encoding="utf-8",
        )

    (paper_dir / "family3_summary_table.tex").write_text(
        to_latex_table(
            family3_df,
            caption="Shariah-targeted local evasion summary.",
            label="tab:family3_summary",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    (paper_dir / "family3_source_mechanism_table.tex").write_text(
        to_latex_table(
            family3_source_df,
            caption="Family-3 evasion summary by source manipulation mechanism.",
            label="tab:family3_source_mechanism",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    (paper_dir / "family3_detailed_cases_table.tex").write_text(
        to_latex_table(
            family3_detail_df,
            caption="Detailed before/after view of Shariah-targeted local evasion cases.",
            label="tab:family3_detailed_cases",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    (paper_dir / "family3_majority_vote_table.tex").write_text(
        to_latex_table(
            family3_vote_df,
            caption="Majority-vote driver ranking across Family-3 target-score variants.",
            label="tab:family3_majority_vote",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )

    return {
        "summary": family3_df,
        "source": family3_source_df,
        "detail": family3_detail_df,
        "history": family3_hist_df,
        "vote": family3_vote_df,
    }


def family3_case_label(row: pd.Series) -> str:
    row_index = row.get("row_index")
    firm_id = row.get("firm_id")
    quarter = row.get("quarter")
    numeric_row = pd.to_numeric(row_index, errors="coerce")
    row_label = str(int(numeric_row)) if np.isfinite(numeric_row) else "status"
    return f"row {row_label} | {firm_id} | {quarter}"
