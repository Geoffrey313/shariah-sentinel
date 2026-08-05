"""Robustness benchmark (families 1 to 4).

Runs correlated-Gaussian contamination (Family 1), realistic manipulation
mechanisms with contamination-AUC tables (Family 2), adversarial attacks
(Family 3), and AnoShift temporal robustness (Family 4). Backs the paper's
contamination-AUC and robustness results; checkpoints each family to CSV so a
run resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.contamination_config import ContaminationSettings
from src.analysis.contamination_mechanisms import apply_m5, apply_m6
from src.analysis.contamination import (
    contaminate_abn_disx_full,
    contaminate_abn_disx_partial,
    contaminate_abn_prod_full,
    contaminate_m1_v2,
    contaminate_m2b,
    contaminate_m3_v2,
    contaminate_m4_v2,
)
try:
    from src.interpretability import (
        build_detector_explanation_table,
        build_red_case_explanations,
    )
except ImportError:
    logging.getLogger(__name__).warning(
        "src.interpretability not available — interpretability tables will be empty."
    )
    def build_detector_explanation_table(*args, **kwargs):
        return pd.DataFrame()
    def build_red_case_explanations(*args, **kwargs):
        return pd.DataFrame()
from src.common.methodology import thresholds_for_panel
from src.data.ratios import compute_shariah_ratios
from src.analysis.counterfactual.core import (
    family3_raw_zscore_settings,
    run_family3_xai,
)
from src.analysis.counterfactual.utils import (
    ScoreContext as Family3ScoreContext,
    family3_case_cards,
    family3_case_label,
    family3_detailed_table,
    family3_history_table,
    family3_source_mechanism_table,
    family3_summary_table,
)
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import REASON_INCLUDED, SPLIT_LABEL_EXCLUDED, SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)

# Mirror the Phase 4 column names locally so lightweight helpers do not need
# to import the entire scoring stack at module import time.
COL_Z_PLUS = "z_plus"
COL_Z_PLUS_RENORM = "z_plus_renorm"
COL_BREADTH = "breadth"
COL_Z_MAHALANOBIS = "z_mahalanobis_sq"
COL_T_IUT = "t_iut"
COL_Z_PLUS_SOFTMAX = "z_plus_softmax"
COL_Z_PLUS_ORTH = "z_plus_orth"

COL_P_Z_PLUS = "p_z_plus"
COL_P_Z_PLUS_RENORM = "p_z_plus_renorm"
COL_P_BREADTH = "p_breadth"
COL_P_Z_MAHALANOBIS = "p_z_mahalanobis_sq"
COL_P_T_IUT = "p_t_iut"
COL_P_Z_PLUS_SOFTMAX = "p_z_plus_softmax"
COL_P_Z_PLUS_ORTH = "p_z_plus_orth"

COMPOSITE_VALUE_COLUMNS: tuple[str, ...] = (
    COL_Z_PLUS,
    COL_Z_PLUS_RENORM,
    COL_BREADTH,
    COL_Z_MAHALANOBIS,
    COL_T_IUT,
    COL_Z_PLUS_SOFTMAX,
    COL_Z_PLUS_ORTH,
)
COMPOSITE_P_COLUMNS: tuple[str, ...] = (
    COL_P_Z_PLUS,
    COL_P_Z_PLUS_RENORM,
    COL_P_BREADTH,
    COL_P_Z_MAHALANOBIS,
    COL_P_T_IUT,
    COL_P_Z_PLUS_SOFTMAX,
    COL_P_Z_PLUS_ORTH,
)
KEY_COVERAGE_COLUMNS: tuple[str, ...] = (
    "cogsq",
    "xrdq",
    "xsgaq",
    "invtq",
    "oibdpq",
    "revtq",
    "dlttq",
    "dlcq",
    "cheq",
    "iditq",
)
FAMILY2_METHOD_TO_LABEL = {
    "threshold_clustering_m1_v2": "_contaminated_m1_v2",
    "temporal_spike_m2b": "_contaminated_m2b",
    "benford_m3_v2": "_contaminated_m3_v2",
    "interstatement_m4_v2": "_contaminated_m4_v2",
    "m5_cod_break": "y",
    "m6_seasonal": "y",
    "abn_disx_full": "_contaminated_abn_disx_full",
    "abn_disx_partial": "_contaminated_abn_disx_partial",
    "abn_prod_full": "_contaminated_abn_prod_full",
}
METHOD_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "correlated_gaussian": ("ratio_debt_adj", "ratio_cash_adj", "ratio_income"),
    "threshold_clustering_m1_v2": ("dlttq", "dlcq", "cheq"),
    "temporal_spike_m2b": ("dlttq", "dlcq", "cheq", "iditq"),
    "benford_m3_v2": ("dlttq", "dlcq", "cheq", "iditq", "revtq"),
    "interstatement_m4_v2": ("niq", "oibdpq", "atq"),
    "m5_cod_break": ("xintq", "dlttq", "dlcq"),
    "m6_seasonal": ("dlttq", "dlcq", "fqtr"),
    "abn_disx_partial": ("xsgaq", "oibdpq"),
    "abn_disx_full": ("xsgaq", "xrdq", "oibdpq"),
    "abn_prod_full": ("revtq", "cogsq", "invtq"),
    "adversarial_evasion": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_general_evasion": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_evasion_no_sac": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_general_evasion_no_sac": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_post_realistic_evasion": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_general_post_realistic_evasion": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_post_realistic_evasion_no_sac": ("dlttq", "dlcq", "cheq", "iditq"),
    "adversarial_general_post_realistic_evasion_no_sac": ("dlttq", "dlcq", "cheq", "iditq"),
    "anoshift": ("quarter",),
}


@dataclass(frozen=True)
class _ScoreContext:
    panel: pd.DataFrame
    zscores: pd.DataFrame
    composites: pd.DataFrame
    active: tuple[str, ...]
    weights: np.ndarray
    sigma: np.ndarray
    null: object
    complete_c_rows: np.ndarray
    complete_c_index: np.ndarray


@dataclass(frozen=True)
class RobustnessBenchmarkOutcome:
    family1: pd.DataFrame
    family2: pd.DataFrame
    family3: pd.DataFrame
    family4: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.astype(object).fillna("").to_numpy().tolist()]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _robustness_output_paths(settings: AnalysisSettings) -> dict[str, Path]:
    """Return canonical output paths for the cross-family benchmark."""
    out_dir = settings.output_layout.robustness_benchmark_dir()
    return {
        "family1": out_dir / "family1_correlated_gaussian.csv",
        "family2": out_dir / "family2_threshold_clustering.csv",
        "family3": out_dir / "family3_adversarial_evasion.csv",
        "family4": out_dir / "family4_anoshift.csv",
        "coverage_csv": out_dir / "coverage_key_variables.csv",
        "coverage_md": out_dir / "coverage_key_variables.md",
        "scoreboard_csv": out_dir / "benchmark_scoreboard.csv",
        "scoreboard_md": out_dir / "benchmark_scoreboard.md",
        "json": out_dir / "robustness_benchmark_summary.json",
        "md": out_dir / "ROBUSTNESS_BENCHMARK.md",
    }


def _write_csv_checkpoint(df: pd.DataFrame, path: Path) -> None:
    """Persist a completed benchmark family without leaving half-written CSVs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _load_family_checkpoint(
    paths: dict[str, Path],
    key: str,
    resume: bool,
) -> pd.DataFrame | None:
    """Load a completed family checkpoint when resume mode is enabled."""
    path = paths[key]
    if not resume or not path.exists():
        return None
    try:
        checkpoint = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        checkpoint = pd.DataFrame()
    log.info("robustness_benchmark: resume loaded %s from %s", key, path)
    return checkpoint


def _coverage_table(panel: pd.DataFrame, columns: tuple[str, ...] = KEY_COVERAGE_COLUMNS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(panel)
    for col in columns:
        if col not in panel.columns:
            rows.append(
                {
                    "column": col,
                    "present_in_panel": False,
                    "n_nonnull": "",
                    "pct_nonnull": "",
                }
            )
            continue
        n_nonnull = int(panel[col].notna().sum())
        rows.append(
            {
                "column": col,
                "present_in_panel": True,
                "n_nonnull": n_nonnull,
                "pct_nonnull": round(100.0 * n_nonnull / max(n, 1), 2),
            }
        )
    return pd.DataFrame(rows)


def _method_availability_table(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    family2: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = list(settings.robustness_benchmark.methods)
    skip_lookup: dict[str, list[str]] = {}
    if not family2.empty and {"method", "entity_type", "status"}.issubset(family2.columns):
        skipped = family2[family2["entity_type"] == "status"].copy()
        for method, group in skipped.groupby("method"):
            skip_lookup[str(method)] = sorted({str(v) for v in group["status"].dropna().tolist() if str(v)})

    for method in methods:
        required = METHOD_REQUIREMENTS.get(method, ())
        missing = [col for col in required if col not in panel.columns]
        coverage_values = []
        for col in required:
            if col in panel.columns:
                coverage_values.append(f"{col}:{100.0 * float(panel[col].notna().mean()):.1f}%")
            else:
                coverage_values.append(f"{col}:missing")
        status = "available"
        if missing:
            status = "missing_inputs"
        elif method in skip_lookup:
            status = "ran_with_skips"
        rows.append(
            {
                "method": method,
                "required_columns": ", ".join(required),
                "column_coverage": "; ".join(coverage_values),
                "status": status,
                "missing_columns": ", ".join(missing),
                "skip_reason": " | ".join(skip_lookup.get(method, [])),
            }
        )
    return pd.DataFrame(rows)


def _nearest_positive_definite(mat: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, min_eig, None)
    repaired = (vecs * vals) @ vecs.T
    return 0.5 * (repaired + repaired.T)


def _family1_ratio_covariance(ctx: _ScoreContext) -> np.ndarray | None:
    cols = ["ratio_debt_adj", "ratio_cash_adj", "ratio_income"]
    ratio_block = (
        ctx.panel.loc[ctx.complete_c_index, cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(ratio_block) < 2:
        return None
    arr = ratio_block.to_numpy(dtype=float)
    try:
        from sklearn.covariance import LedoitWolf

        cov = LedoitWolf().fit(arr).covariance_
    except Exception:
        log.warning(
            "family1_correlated_gaussian: Ledoit-Wolf covariance failed; falling back to sample covariance on %d rows.",
            len(arr),
        )
        cov = np.cov(arr, rowvar=False, ddof=0)
    if cov.shape != (3, 3) or not np.isfinite(cov).all():
        return None
    return _nearest_positive_definite(cov)


def _scoreboard_table(
    family1: pd.DataFrame,
    family2: pd.DataFrame,
    family3: pd.DataFrame,
    family4: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _append_metric_rows(df: pd.DataFrame, family: str) -> None:
        if df.empty or "method" not in df.columns:
            return
        work = df.copy()
        if "entity_type" in work.columns:
            work = work[work["entity_type"].isin(["composite", "detector"])]
        metric_cols = [c for c in ("auc", "detection_rate", "fpr") if c in work.columns]
        if not metric_cols:
            return
        group_cols = ["method"]
        if "entity_type" in work.columns:
            group_cols.append("entity_type")
        summary = (
            work.groupby(group_cols, dropna=False)[metric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        for _, row in summary.iterrows():
            rows.append(
                {
                    "family": family,
                    "method": row.get("method", ""),
                    "entity_type": row.get("entity_type", ""),
                    "mean_auc": round(float(row["auc"]), 4) if pd.notna(row.get("auc")) else "",
                    "mean_detection_rate": round(float(row["detection_rate"]), 4) if pd.notna(row.get("detection_rate")) else "",
                    "mean_fpr": round(float(row["fpr"]), 4) if pd.notna(row.get("fpr")) else "",
                }
            )

    _append_metric_rows(family1, "family1")
    _append_metric_rows(family2, "family2")
    _append_metric_rows(family4, "family4")

    if not family3.empty:
        group_iter = family3.groupby("method", dropna=False) if "method" in family3.columns else [("adversarial_evasion", family3)]
        for method, group in group_iter:
            rows.append(
                {
                    "family": "family3",
                    "method": method,
                    "entity_type": "row_attack",
                    "mean_auc": "",
                    "mean_detection_rate": round(float(pd.to_numeric(group.get("evasion_rate"), errors="coerce").dropna().mean()), 4)
                    if "evasion_rate" in group.columns and pd.to_numeric(group["evasion_rate"], errors="coerce").notna().any()
                    else "",
                    "mean_fpr": "",
                    "median_cost": round(float(pd.to_numeric(group.get("median_cost"), errors="coerce").dropna().mean()), 4)
                    if "median_cost" in group.columns and pd.to_numeric(group["median_cost"], errors="coerce").notna().any()
                    else "",
                    "p90_cost": round(float(pd.to_numeric(group.get("p90_cost"), errors="coerce").dropna().mean()), 4)
                    if "p90_cost" in group.columns and pd.to_numeric(group["p90_cost"], errors="coerce").notna().any()
                    else "",
                }
            )

    return pd.DataFrame(rows)


def _to_latex_table(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "%.4f",
) -> str:
    if df.empty:
        return "% empty table"
    latex = df.to_latex(
        index=False,
        escape=False,
        na_rep="---",
        float_format=(lambda x: float_format % x),
    )
    if caption or label:
        lines = latex.splitlines()
        insert_at = 1 if lines and lines[0].startswith("\\begin{tabular}") else 0
        wrapped = ["\\begin{table}[!htbp]", "\\centering"]
        if caption:
            wrapped.append(f"\\caption{{{caption}}}")
        if label:
            wrapped.append(f"\\label{{{label}}}")
        wrapped.extend(lines)
        wrapped.append("\\end{table}")
        latex = "\n".join(wrapped)
    return latex


def _family2_auc_table(family2: pd.DataFrame) -> pd.DataFrame:
    if family2.empty:
        return pd.DataFrame()
    work = family2.copy()
    work = work[
        (work.get("entity_type") == "composite")
        & (work.get("entity_name") == "z_mahalanobis_sq")
        & work["auc"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame()
    pivot = work.pivot_table(
        index="method",
        columns="rho",
        values="auc",
        aggfunc="mean",
    ).reset_index()
    rho_cols = [c for c in pivot.columns if c != "method"]
    rename = {"method": "Mechanism"}
    for col in rho_cols:
        rename[col] = rf"$\rho = {100*float(col):.0f}\%$"
    pivot = pivot.rename(columns=rename)
    return pivot


def _detector_contribution_table(
    panel: pd.DataFrame,
    zscores: pd.DataFrame,
    composites: pd.DataFrame,
    family2: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    p_cols = [c for c in ("p_z_plus", "p_z_plus_renorm", "p_z_mahalanobis_sq", "p_t_iut") if c in composites.columns]
    if not p_cols:
        return pd.DataFrame()
    det_cols = [c for c in zscores.columns if c.startswith("z")]
    if not det_cols:
        return pd.DataFrame()
    work = zscores.copy()
    work.index = panel.index
    comp = composites.copy()
    comp.index = panel.index
    min_p = comp[p_cols].min(axis=1)
    red_threshold = settings.robustness.red_threshold
    red_mask = min_p < red_threshold
    if not red_mask.any():
        return pd.DataFrame()
    red_z = work.loc[red_mask, det_cols].apply(pd.to_numeric, errors="coerce")
    dominant = red_z.idxmax(axis=1, skipna=True)
    rows: list[dict[str, object]] = []
    auc_lookup = {}
    if not family2.empty:
        auc_rows = family2[
            (family2.get("entity_type") == "detector")
            & family2["auc"].notna()
        ]
        for detector, val in auc_rows.groupby("entity_name")["auc"].mean().items():
            auc_lookup[str(detector)] = float(val)
    for det in det_cols:
        vals = pd.to_numeric(red_z[det], errors="coerce")
        rows.append(
            {
                "Detector": det,
                "Dominant %": round(100.0 * (dominant == det).mean(), 1),
                "z > 2 %": round(100.0 * (vals > 2.0).fillna(False).mean(), 1),
                "Contam. AUC": round(auc_lookup[det], 2) if det in auc_lookup else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _detector_red_profile_table(
    panel: pd.DataFrame,
    zscores: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    p_cols = [c for c in ("p_z_plus", "p_z_plus_renorm", "p_z_mahalanobis_sq", "p_t_iut") if c in composites.columns]
    det_cols = [c for c in zscores.columns if c.startswith("z")]
    if not p_cols or not det_cols:
        return pd.DataFrame()
    work = zscores.copy()
    work.index = panel.index
    comp = composites.copy()
    comp.index = panel.index
    min_p = comp[p_cols].min(axis=1)
    red_mask = min_p < settings.robustness.red_threshold
    c_mask = pd.Series(panel["_split"] == SPLIT_LABEL_INCLUDED, index=panel.index) if "_split" in panel.columns else pd.Series(False, index=panel.index)
    if not red_mask.any():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for det in det_cols:
        red_vals = pd.to_numeric(work.loc[red_mask, det], errors="coerce")
        c_vals = pd.to_numeric(work.loc[c_mask, det], errors="coerce")
        red_mean = float(red_vals.mean()) if red_vals.notna().any() else np.nan
        c_mean = float(c_vals.mean()) if c_vals.notna().any() else np.nan
        rows.append(
            {
                "Detector": det,
                "RED mean z": red_mean,
                "C mean z": c_mean,
                "Lift": red_mean - c_mean if np.isfinite(red_mean) and np.isfinite(c_mean) else np.nan,
                "RED q90 z": float(red_vals.quantile(0.9)) if red_vals.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Lift", ascending=False, na_position="last").reset_index(drop=True)
    return out


def _top_red_cases_table(
    panel: pd.DataFrame,
    composites: pd.DataFrame,
    settings: AnalysisSettings,
    top_n: int = 15,
) -> pd.DataFrame:
    p_cols = [c for c in ("p_z_plus", "p_z_plus_renorm", "p_z_mahalanobis_sq", "p_t_iut") if c in composites.columns]
    if not p_cols:
        return pd.DataFrame()
    comp = composites.copy()
    comp.index = panel.index
    comp["_min_p"] = comp[p_cols].min(axis=1)
    top = comp[comp["_min_p"] < settings.robustness.red_threshold].copy()
    top = top.sort_values("_min_p", ascending=True).head(top_n)
    if top.empty:
        return pd.DataFrame()
    firm_col = settings.panel_schema.firm_id
    quarter_col = settings.panel_schema.quarter
    rows: list[dict[str, object]] = []
    for idx, row in top.iterrows():
        rows.append(
            {
                "row_index": int(idx),
                "firm_id": panel.loc[idx, firm_col] if firm_col in panel.columns else "",
                "quarter": panel.loc[idx, quarter_col] if quarter_col in panel.columns else "",
                "min_p": float(row["_min_p"]),
                "p_z_mahalanobis_sq": float(pd.to_numeric(pd.Series([row.get("p_z_mahalanobis_sq")]), errors="coerce").iat[0]) if "p_z_mahalanobis_sq" in row.index else np.nan,
                "z_mahalanobis_sq": float(pd.to_numeric(pd.Series([row.get("z_mahalanobis_sq")]), errors="coerce").iat[0]) if "z_mahalanobis_sq" in row.index else np.nan,
                "p_z_plus": float(pd.to_numeric(pd.Series([row.get("p_z_plus")]), errors="coerce").iat[0]) if "p_z_plus" in row.index else np.nan,
                "p_t_iut": float(pd.to_numeric(pd.Series([row.get("p_t_iut")]), errors="coerce").iat[0]) if "p_t_iut" in row.index else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _anoshift_summary_table(family4: pd.DataFrame) -> pd.DataFrame:
    if family4.empty:
        return pd.DataFrame()
    work = family4.copy()
    metric_cols = [c for c in ("fpr", "detection_rate", "auc") if c in work.columns]
    if not metric_cols or "method" not in work.columns:
        return pd.DataFrame()
    group_cols = ["method"]
    if "kind" in work.columns:
        group_cols.append("kind")
    if "protocol" in work.columns:
        group_cols.append("protocol")
    if "calibration_split" in work.columns:
        group_cols.append("calibration_split")
    for candidate in ("split", "target_split"):
        if candidate in work.columns:
            group_cols.append(candidate)
            break
    if "year" in work.columns:
        group_cols.append("year")
    summary = (
        work.groupby(group_cols, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    return summary


def _anoshift_calibration_table(family4: pd.DataFrame) -> pd.DataFrame:
    if family4.empty:
        return pd.DataFrame()
    work = family4.copy()
    keep_cols = [
        c for c in [
            "protocol",
            "calibration_split",
            "split",
            "target_split",
            "year",
            "kind",
            "entity_name",
            "n_rows",
            "fpr",
            "detection_rate",
            "auc",
            "status",
        ] if c in work.columns
    ]
    if not keep_cols:
        return pd.DataFrame()
    out = work[keep_cols].copy()
    out = out.sort_values(
        [c for c in ["protocol", "calibration_split", "kind", "split", "year", "entity_name"] if c in out.columns],
        kind="stable",
    ).reset_index(drop=True)
    return out


def _family2_power_curve_table(family2: pd.DataFrame) -> pd.DataFrame:
    if family2.empty:
        return pd.DataFrame()
    work = family2.copy()
    required = {"method", "entity_type", "entity_name", "rho", "delta", "detection_rate"}
    if not required.issubset(work.columns):
        return pd.DataFrame()
    work = work[
        (work["entity_type"] == "composite")
        & (work["entity_name"] == "z_mahalanobis_sq")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    out = (
        work.groupby(["method", "rho", "delta"], dropna=False)[["detection_rate", "auc", "fpr"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    return out.sort_values(["method", "rho", "delta"]).reset_index(drop=True)


def _write_paper_outputs(
    *,
    out_dir: Path,
    panel: pd.DataFrame,
    zscores: pd.DataFrame,
    composites: pd.DataFrame,
    family1: pd.DataFrame,
    family2: pd.DataFrame,
    family3: pd.DataFrame,
    family4: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paper_dir = out_dir / "paper_ready"
    paper_dir.mkdir(parents=True, exist_ok=True)
    threshold_debt, threshold_cash, threshold_income = _threshold_values(panel)

    coverage_df = _coverage_table(panel)
    scoreboard_df = _scoreboard_table(family1, family2, family3, family4)
    contam_auc_df = _family2_auc_table(family2)
    detector_contrib_df = _detector_contribution_table(panel, zscores, composites, family2, settings)
    availability_df = _method_availability_table(panel, settings, family2)
    family3_df = family3_summary_table(family3)
    family3_source_df = family3_source_mechanism_table(family3)
    family3_detail_df = family3_detailed_table(family3)
    family3_hist_df = family3_history_table(family3)
    anoshift_df = _anoshift_summary_table(family4)
    anoshift_calibration_df = _anoshift_calibration_table(family4)
    detector_profile_df = _detector_red_profile_table(panel, zscores, composites, settings)
    top_red_df = _top_red_cases_table(panel, composites, settings)
    family2_power_df = _family2_power_curve_table(family2)
    detector_expl_df = build_detector_explanation_table(family2)
    red_case_expl_df = build_red_case_explanations(
        panel,
        zscores,
        composites,
        family2,
        firm_col=settings.panel_schema.firm_id,
        quarter_col=settings.panel_schema.quarter,
        red_threshold=settings.robustness.red_threshold,
        top_n=25,
    )

    coverage_df.to_csv(paper_dir / "coverage_table.csv", index=False)
    scoreboard_df.to_csv(paper_dir / "scoreboard_table.csv", index=False)
    contam_auc_df.to_csv(paper_dir / "contamination_auc_table.csv", index=False)
    detector_contrib_df.to_csv(paper_dir / "detector_contribution_table.csv", index=False)
    availability_df.to_csv(paper_dir / "method_availability_table.csv", index=False)
    family3_df.to_csv(paper_dir / "family3_summary_table.csv", index=False)
    family3_source_df.to_csv(paper_dir / "family3_source_mechanism_table.csv", index=False)
    family3_detail_df.to_csv(paper_dir / "family3_detailed_cases_table.csv", index=False)
    family3_hist_df.to_csv(paper_dir / "family3_history_table.csv", index=False)
    anoshift_df.to_csv(paper_dir / "anoshift_summary_table.csv", index=False)
    anoshift_calibration_df.to_csv(paper_dir / "anoshift_calibration_table.csv", index=False)
    detector_profile_df.to_csv(paper_dir / "interpretability_detector_profile_table.csv", index=False)
    top_red_df.to_csv(paper_dir / "interpretability_top_red_cases_table.csv", index=False)
    family2_power_df.to_csv(paper_dir / "family2_power_curve_table.csv", index=False)
    detector_expl_df.to_csv(paper_dir / "interpretability_detector_explanations_table.csv", index=False)
    red_case_expl_df.to_csv(paper_dir / "interpretability_red_case_explanations.csv", index=False)
    if not family3_detail_df.empty:
        (paper_dir / "family3_case_cards.md").write_text(
            family3_case_cards(family3_detail_df),
            encoding="utf-8",
        )
    if not red_case_expl_df.empty:
        (paper_dir / "interpretability_red_case_cards.md").write_text(
            "\n\n".join(
                [
                    (
                        f"### {row.get('firm_id', '')} — {row.get('quarter', '')}\n"
                        f"- Trigger: `{row.get('trigger_composite', '')}` with p={row.get('trigger_p', '')}\n"
                        f"- Primary detector: `{row.get('dominant_detector', '')}` ({row.get('short_name', '')})\n"
                        f"- Hypothesis: {row.get('hypothesis', '')}\n"
                        f"- Business readout: {row.get('business_readout', '')}\n"
                        f"- Benchmark support: {row.get('benchmark_support', '')}\n"
                        f"- Snapshot: {row.get('supporting_snapshot', '')}\n"
                        f"- Caution: {row.get('caution', '')}"
                    )
                    for _, row in red_case_expl_df.iterrows()
                ]
            ),
            encoding="utf-8",
        )

    (paper_dir / "coverage_table.tex").write_text(
        _to_latex_table(coverage_df, caption="Coverage of key robustness variables.", label="tab:coverage_vars", float_format="%.2f"),
        encoding="utf-8",
    )
    (paper_dir / "scoreboard_table.tex").write_text(
        _to_latex_table(scoreboard_df, caption="Main robustness metrics by family and method.", label="tab:robust_scoreboard", float_format="%.4f"),
        encoding="utf-8",
    )
    (paper_dir / "contamination_auc_table.tex").write_text(
        _to_latex_table(contam_auc_df, caption=r"Contamination AUC ($Z^2_{Mah}$) by method and contamination rate.", label="tab:contam_modern", float_format="%.2f"),
        encoding="utf-8",
    )
    (paper_dir / "detector_contribution_table.tex").write_text(
        _to_latex_table(detector_contrib_df, caption="Per-detector contribution to RED verdicts.", label="tab:detector_contrib_modern", float_format="%.2f"),
        encoding="utf-8",
    )
    (paper_dir / "method_availability_table.tex").write_text(
        _to_latex_table(availability_df, caption="Availability of robustness methods on the current panel.", label="tab:method_availability", float_format="%.2f"),
        encoding="utf-8",
    )
    (paper_dir / "family3_summary_table.tex").write_text(
        _to_latex_table(family3_df, caption="Shariah-targeted local evasion summary.", label="tab:family3_summary", float_format="%.4f"),
        encoding="utf-8",
    )
    (paper_dir / "family3_source_mechanism_table.tex").write_text(
        _to_latex_table(
            family3_source_df,
            caption="Family-3 evasion summary by source manipulation mechanism.",
            label="tab:family3_source_mechanism",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    (paper_dir / "family3_detailed_cases_table.tex").write_text(
        _to_latex_table(
            family3_detail_df,
            caption="Detailed before/after view of Shariah-targeted local evasion cases.",
            label="tab:family3_detailed_cases",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    (paper_dir / "anoshift_summary_table.tex").write_text(
        _to_latex_table(anoshift_df, caption="AnoShift summary under fixed global calibration.", label="tab:anoshift_summary", float_format="%.4f"),
        encoding="utf-8",
    )
    (paper_dir / "anoshift_calibration_table.tex").write_text(
        _to_latex_table(
            anoshift_calibration_df,
            caption="AnoShift detailed calibration and evaluation rows by protocol.",
            label="tab:anoshift_calibration_detail",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    (paper_dir / "interpretability_detector_profile_table.tex").write_text(
        _to_latex_table(detector_profile_df, caption="Detector profile on RED rows versus included reference rows.", label="tab:interpretability_detector_profile", float_format="%.3f"),
        encoding="utf-8",
    )
    (paper_dir / "interpretability_top_red_cases_table.tex").write_text(
        _to_latex_table(top_red_df, caption="Top RED cases by minimum composite p-value.", label="tab:top_red_cases", float_format="%.4f"),
        encoding="utf-8",
    )
    (paper_dir / "family2_power_curve_table.tex").write_text(
        _to_latex_table(family2_power_df, caption="Family-2 power curve summary on the Mahalanobis composite.", label="tab:family2_power_curves", float_format="%.4f"),
        encoding="utf-8",
    )
    (paper_dir / "interpretability_detector_explanations_table.tex").write_text(
        _to_latex_table(detector_expl_df, caption="Detector-level business interpretation guide.", label="tab:detector_explanations", float_format="%.3f"),
        encoding="utf-8",
    )
    (paper_dir / "interpretability_red_case_explanations.tex").write_text(
        _to_latex_table(
            red_case_expl_df[[
                c for c in [
                    "firm_id",
                    "quarter",
                    "trigger_composite",
                    "trigger_p",
                    "dominant_detector",
                    "short_name",
                    "hypothesis",
                    "benchmark_support",
                ] if c in red_case_expl_df.columns
            ]],
            caption="Row-level explanations for the strongest RED cases.",
            label="tab:red_case_explanations",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not contam_auc_df.empty:
            plot_df = contam_auc_df.set_index("Mechanism")
            ax = plot_df.plot(kind="bar", figsize=(10, 5))
            ax.set_ylabel("AUC")
            ax.set_title("Contamination AUC by mechanism")
            ax.legend(title="Rate")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(paper_dir / "contamination_auc_bar.png", dpi=180)
            plt.close(fig)

        if not coverage_df.empty:
            plot_cov = coverage_df[coverage_df["present_in_panel"] == True].copy()
            plot_cov["pct_nonnull"] = pd.to_numeric(plot_cov["pct_nonnull"], errors="coerce")
            ax = plot_cov.plot(x="column", y="pct_nonnull", kind="bar", legend=False, figsize=(10, 4))
            ax.set_ylabel("% non-null")
            ax.set_title("Coverage of robustness input variables")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(paper_dir / "coverage_bar.png", dpi=180)
            plt.close(fig)

        if not detector_contrib_df.empty:
            det_plot = detector_contrib_df.sort_values("Dominant %", ascending=False).head(10)
            ax = det_plot.plot(x="Detector", y="Dominant %", kind="bar", legend=False, figsize=(10, 4))
            ax.set_ylabel("Dominant share among RED rows (%)")
            ax.set_title("Most dominant detectors on RED rows")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(paper_dir / "detector_dominance_bar.png", dpi=180)
            plt.close(fig)

        if not detector_profile_df.empty:
            prof_plot = detector_profile_df.head(10).copy()
            ax = prof_plot.plot(x="Detector", y="Lift", kind="bar", legend=False, figsize=(10, 4))
            ax.set_ylabel("RED minus C mean z")
            ax.set_title("Detector lift on RED rows")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(paper_dir / "interpretability_detector_lift_bar.png", dpi=180)
            plt.close(fig)

        if not family2_power_df.empty:
            power_plot = family2_power_df.copy()
            rho_min = sorted(power_plot["rho"].dropna().unique().tolist())[0]
            power_plot = power_plot[power_plot["rho"] == rho_min]
            if not power_plot.empty:
                pivot = power_plot.pivot_table(index="delta", columns="method", values="detection_rate", aggfunc="mean")
                ax = pivot.plot(marker="o", figsize=(10, 5))
                ax.set_ylabel("Detection rate")
                ax.set_title(f"Family-2 power curves at rho={rho_min:.2f}")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(paper_dir / "family2_power_curves.png", dpi=180)
                plt.close(fig)

                scatter_df = power_plot.dropna(subset=["auc", "fpr"]).copy()
                if not scatter_df.empty:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    for method, grp in scatter_df.groupby("method"):
                        ax.scatter(grp["fpr"], grp["auc"], label=method, s=50)
                    ax.set_xlabel("False-positive rate")
                    ax.set_ylabel("AUC")
                    ax.set_title(f"Family-2 AUC vs FPR at rho={rho_min:.2f}")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    fig.savefig(paper_dir / "family2_auc_vs_fpr_scatter.png", dpi=180)
                    plt.close(fig)

        if not anoshift_df.empty:
            split_col = "split" if "split" in anoshift_df.columns else "target_split"
            pivot = anoshift_df.pivot_table(index="method", columns=split_col, values="fpr", aggfunc="mean")
            if not pivot.empty:
                ax = pivot.plot(kind="bar", figsize=(10, 5))
                ax.set_ylabel("FPR")
                ax.set_title("AnoShift false-positive rate by split (fixed global calibration)")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(paper_dir / "anoshift_fpr_bar.png", dpi=180)
                plt.close(fig)

        # TODO(family3): these plots consume the retired PGD epsilon-grid schema
        # (epsilon_rel / median_cost / pvalue_by_eps_json). The counterfactual
        # package emits an epoch-based schema instead; the guards below skip
        # cleanly until epoch-based equivalents are rebuilt against
        # epoch_log_json.
        if not family3_df.empty and {"epsilon_rel", "evasion_rate"}.issubset(family3_df.columns):
            family3_plot = family3_df.dropna(subset=["epsilon_rel", "evasion_rate"]).copy()
            if not family3_plot.empty:
                ax = family3_plot.plot(x="epsilon_rel", y="evasion_rate", marker="o", legend=False, figsize=(7, 4))
                ax.set_ylabel("Evasion rate")
                ax.set_title("Adversarial evasion rate by relative epsilon")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(paper_dir / "family3_evasion_rate_curve.png", dpi=180)
                plt.close(fig)

                if "median_cost" in family3_plot.columns:
                    ax = family3_plot.plot(x="epsilon_rel", y="median_cost", marker="o", legend=False, figsize=(7, 4))
                    ax.set_ylabel("Median attack cost")
                    ax.set_title("Adversarial median cost by relative epsilon")
                    fig = ax.get_figure()
                    fig.tight_layout()
                    fig.savefig(paper_dir / "family3_cost_curve.png", dpi=180)
                    plt.close(fig)

        if not family3_detail_df.empty and "pvalue_by_eps_json" in family3_detail_df.columns:
            alpha = settings.robustness_benchmark.alpha
            pvalue_rows = []
            for _, row in family3_detail_df.iterrows():
                raw = row.get("pvalue_by_eps_json")
                if not pd.notna(raw):
                    continue
                try:
                    eps_map: dict = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                label = family3_case_label(row)
                for eps_val, pval in sorted(eps_map.items(), key=lambda t: float(t[0])):
                    pvalue_rows.append({"case_label": label, "eps_rel": float(eps_val), "pvalue": float(pval)})
            if pvalue_rows:
                pv_df = pd.DataFrame(pvalue_rows)
                fig, ax = plt.subplots(figsize=(8, 5))
                for case_label, grp in pv_df.groupby("case_label", dropna=False):
                    grp = grp.sort_values("eps_rel")
                    ax.plot(grp["eps_rel"], grp["pvalue"], marker="o", label=case_label)
                ax.axhline(
                    alpha,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"α = {alpha}",
                )
                ax.set_xlabel("Relative epsilon (attack budget)")
                ax.set_ylabel("Composite p-value")
                ax.set_title("Family 3 — p-value across epsilon grid (per attacked row)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(paper_dir / "family3_pvalue_by_eps.png", dpi=180)
                plt.close(fig)

        # TODO(family3): the per-row loss/ratio/raw trajectory diagnostics below
        # read the retired epoch-history columns (loss, ratio_debt_adj, dlttq,
        # ...). The counterfactual package's epoch_log_json expands to a
        # different column set, so this block is guarded on the legacy `loss`
        # column and skips until rebuilt against the epoch schema.
        if not family3_hist_df.empty and "loss" in family3_hist_df.columns:
            diag_dir = paper_dir / "family3_diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)

            multi_hist = family3_hist_df.copy()
            multi_hist["case_label"] = multi_hist.apply(family3_case_label, axis=1)

            loss_multi = multi_hist.dropna(subset=["epoch", "loss"]).copy()
            if not loss_multi.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                for case_label, grp in loss_multi.groupby("case_label", dropna=False):
                    grp = grp.sort_values("epoch")
                    ax.plot(grp["epoch"], grp["loss"], marker="o", label=case_label)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Surrogate loss")
                ax.set_title("Family 3 loss trajectory by attacked row")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(paper_dir / "family3_loss_by_row.png", dpi=180)
                plt.close(fig)

            ratio_specs = [
                ("ratio_debt_adj", "Debt ratio", threshold_debt, "family3_ratio_debt_by_row.png"),
                ("ratio_cash_adj", "Cash ratio", threshold_cash, "family3_ratio_cash_by_row.png"),
                ("ratio_income", "Income ratio", threshold_income, "family3_ratio_income_by_row.png"),
            ]
            for ratio_col, ratio_label, threshold, filename in ratio_specs:
                ratio_multi = multi_hist.dropna(subset=["epoch", ratio_col]).copy()
                if ratio_multi.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                for case_label, grp in ratio_multi.groupby("case_label", dropna=False):
                    grp = grp.sort_values("epoch")
                    ax.plot(grp["epoch"], grp[ratio_col], marker="o", label=case_label)
                ax.axhline(threshold, color="black", linestyle="--", alpha=0.5, label="Threshold")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(ratio_label)
                ax.set_title(f"Family 3 {ratio_label.lower()} trajectory by attacked row")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(paper_dir / filename, dpi=180)
                plt.close(fig)

            raw_specs = [
                ("dlttq", "Long-term debt", "family3_raw_dlttq_by_row.png"),
                ("dlcq", "Short-term debt", "family3_raw_dlcq_by_row.png"),
                ("cheq", "Cash", "family3_raw_cheq_by_row.png"),
                ("iditq", "Interest income", "family3_raw_iditq_by_row.png"),
            ]
            for raw_col, raw_label, filename in raw_specs:
                raw_multi = multi_hist.dropna(subset=["epoch", raw_col]).copy()
                if raw_multi.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                for case_label, grp in raw_multi.groupby("case_label", dropna=False):
                    grp = grp.sort_values("epoch")
                    ax.plot(grp["epoch"], grp[raw_col], marker="o", label=case_label)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(raw_label)
                ax.set_title(f"Family 3 {raw_label.lower()} trajectory by attacked row")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(paper_dir / filename, dpi=180)
                plt.close(fig)

            for (row_index, firm_id, quarter), grp in family3_hist_df.groupby(["row_index", "firm_id", "quarter"], dropna=False):
                safe_stem = f"row_{int(row_index)}_{str(firm_id)}_{str(quarter)}".replace("/", "_").replace(" ", "_")

                loss_df = grp.dropna(subset=["loss"]).copy()
                if not loss_df.empty:
                    ax = loss_df.plot(x="epoch", y="loss", marker="o", legend=False, figsize=(7, 4))
                    ax.set_ylabel("Surrogate loss")
                    ax.set_title(f"Family 3 loss trajectory — row {int(row_index)}")
                    fig = ax.get_figure()
                    fig.tight_layout()
                    fig.savefig(diag_dir / f"{safe_stem}_loss.png", dpi=180)
                    plt.close(fig)

                ratio_df = grp.dropna(subset=["epoch"]).copy()
                ratio_cols = [c for c in ["ratio_debt_adj", "ratio_cash_adj", "ratio_income"] if c in ratio_df.columns]
                if ratio_cols:
                    ax = ratio_df.plot(x="epoch", y=ratio_cols, marker="o", figsize=(8, 4))
                    ax.axhline(threshold_debt, color="C0", linestyle="--", alpha=0.4)
                    ax.axhline(threshold_cash, color="C1", linestyle="--", alpha=0.4)
                    ax.axhline(threshold_income, color="C2", linestyle="--", alpha=0.4)
                    ax.set_ylabel("Ratio value")
                    ax.set_title(f"Family 3 ratio trajectory — row {int(row_index)}")
                    fig = ax.get_figure()
                    fig.tight_layout()
                    fig.savefig(diag_dir / f"{safe_stem}_ratios.png", dpi=180)
                    plt.close(fig)

                raw_cols = [c for c in ["dlttq", "dlcq", "cheq", "iditq"] if c in ratio_df.columns]
                if raw_cols:
                    ax = ratio_df.plot(x="epoch", y=raw_cols, marker="o", figsize=(8, 4))
                    ax.set_ylabel("Raw value")
                    ax.set_title(f"Family 3 raw-variable trajectory — row {int(row_index)}")
                    fig = ax.get_figure()
                    fig.tight_layout()
                    fig.savefig(diag_dir / f"{safe_stem}_raw.png", dpi=180)
                    plt.close(fig)
    except Exception as exc:
        log.warning("paper_ready plots skipped: %s", exc)

    paths["paper_dir"] = paper_dir
    return paths


def _benchmark_settings(base: AnalysisSettings) -> AnalysisSettings:
    rb = base.robustness_benchmark
    inj = base.injection.model_copy(
        update={
            "delta_grid": tuple(rb.delta_grid),
            "alpha_grid": (rb.alpha,),
            "primary_alpha_for_mde": rb.alpha,
        }
    )
    return base.model_copy(
        update={
            "bootstrap": base.bootstrap.model_copy(
                update={
                    "n_replicates": rb.bootstrap_replicates,
                    "random_seed": rb.random_seed,
                }
            ),
            "injection": inj,
        }
    )


def _year_from_quarter(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.slice(0, 4), errors="coerce")


def _threshold_values(panel: pd.DataFrame) -> tuple[float, float, float]:
    thresholds = thresholds_for_panel(panel)
    return (
        thresholds["ratio_debt_adj"],
        thresholds["ratio_cash_adj"],
        thresholds["ratio_income"],
    )


def _threshold_context(panel: pd.DataFrame) -> dict[str, float]:
    debt, cash, income = _threshold_values(panel)
    return {
        "threshold_debt": debt,
        "threshold_cash": cash,
        "threshold_income": income,
    }


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(score)
    if mask.sum() == 0:
        return float("nan")
    y = y_true[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score[mask]))


def _detector_flag_metrics(
    zscores: pd.DataFrame,
    labels: np.ndarray,
    alpha: float,
    *,
    rho: float,
    delta: float,
    method: str,
) -> list[dict[str, object]]:
    from scipy.stats import norm

    crit = float(norm.ppf(1.0 - alpha))
    rows: list[dict[str, object]] = []
    for col in [c for c in zscores.columns if c.startswith("z")]:
        vals = pd.to_numeric(zscores[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            continue
        flags = vals[finite] >= crit
        y = labels[finite]
        rows.append(
            {
                "method": method,
                "rho": rho,
                "delta": delta,
                "entity_type": "detector",
                "entity_name": col,
                "n_finite": int(finite.sum()),
                "auc": _safe_auc(y, vals[finite]),
                "detection_rate": float(flags[y == 1].mean()) if (y == 1).any() else float("nan"),
                "fpr": float(flags[y == 0].mean()) if (y == 0).any() else float("nan"),
            }
        )
    return rows


def _composite_metrics(
    composites: pd.DataFrame,
    labels: np.ndarray,
    alpha: float,
    *,
    rho: float,
    delta: float,
    method: str,
    extra: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    extra = extra or {}
    for value_col, p_col in zip(COMPOSITE_VALUE_COLUMNS, COMPOSITE_P_COLUMNS):
        vals = pd.to_numeric(composites[value_col], errors="coerce").to_numpy(dtype=float)
        pvals = pd.to_numeric(composites[p_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals) & np.isfinite(pvals)
        if finite.sum() == 0:
            continue
        flags = pvals[finite] < alpha
        y = labels[finite]
        row = {
            "method": method,
            "rho": rho,
            "delta": delta,
            "entity_type": "composite",
            "entity_name": value_col,
            "pvalue_col": p_col,
            "n_finite": int(finite.sum()),
            "auc": _safe_auc(y, vals[finite]),
            "detection_rate": float(flags[y == 1].mean()) if (y == 1).any() else float("nan"),
            "fpr": float(flags[y == 0].mean()) if (y == 0).any() else float("nan"),
        }
        row.update(extra)
        rows.append(row)
    return rows


def _build_primary_null(
    complete_c: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    settings: AnalysisSettings,
) -> object:
    from src.engine.bootstrap import run_non_parametric_bootstrap, run_parametric_bootstrap

    mode = settings.bootstrap.primary_mode
    if mode == "non_parametric":
        return run_non_parametric_bootstrap(
            reference_rows=complete_c,
            sigma=sigma,
            weights=weights,
            settings=settings,
        )
    if mode == "parametric":
        return run_parametric_bootstrap(
            sigma=sigma,
            weights=weights,
            settings=settings,
        )
    raise ValueError(f"Unsupported bootstrap primary_mode: {mode!r}")


_CACHED_NULL = None
_CACHED_SIGMA = None


def _cache_baseline(ctx: "_ScoreContext") -> None:
    """Cache the baseline null and sigma for reuse across _score_panel calls."""
    global _CACHED_NULL, _CACHED_SIGMA
    _CACHED_NULL = ctx.null
    _CACHED_SIGMA = ctx.sigma


def _score_panel(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    *,
    null_override: object = None,
    sigma_override: np.ndarray | None = None,
) -> _ScoreContext:
    from src.engine.zscores import compute_zscores
    from src.analysis.reference_sample import run_phase0
    from src.analysis.dependence import run_phase2
    from src.analysis.composite_scoring import _build_weights, run_phase4

    # Use cached baseline null/sigma when available (avoids redundant bootstrap)
    if null_override is None and _CACHED_NULL is not None:
        null_override = _CACHED_NULL
    if sigma_override is None and _CACHED_SIGMA is not None:
        sigma_override = _CACHED_SIGMA

    # Skip Phase 0 when _split is already present (pre-populated by the caller).
    if "_split" in panel.columns:
        p0_panel = panel
    else:
        p0 = run_phase0(panel=panel, settings=settings, write_outputs=False)
        p0_panel = p0.panel

    zc = compute_zscores(panel=p0_panel, settings=settings, write_outputs=False)

    # Phase 2 (Ledoit-Wolf) is skipped when sigma_override is provided.
    if sigma_override is not None:
        sigma_for_p4: object = sigma_override
        p2_ledoit = None
    else:
        p2 = run_phase2(panel=p0_panel, settings=settings, zscores=zc.zscores, write_outputs=False)
        sigma_for_p4 = p2.ledoit_wolf
        p2_ledoit = p2.ledoit_wolf

    try:
        p4 = run_phase4(
            panel=p0_panel,
            settings=settings,
            zscores=zc.zscores,
            sigma_override=sigma_for_p4,
            null_override=null_override,
            write_outputs=False,
        )
    except (ValueError, KeyError):
        # sigma_override shape mismatches active set (active detectors changed) —
        # fall back to a fresh Phase 2. Rare in practice.
        p2 = run_phase2(panel=p0_panel, settings=settings, zscores=zc.zscores, write_outputs=False)
        p2_ledoit = p2.ledoit_wolf
        p4 = run_phase4(
            panel=p0_panel,
            settings=settings,
            zscores=zc.zscores,
            sigma_override=p2_ledoit,
            null_override=null_override,
            write_outputs=False,
        )

    active = tuple(p4.summary["active_detectors"])
    weights = _build_weights(active, settings)

    if sigma_override is not None and p2_ledoit is None:
        sigma = np.asarray(sigma_override, dtype=float)
    else:
        sigma = p2_ledoit.loc[list(active), list(active)].to_numpy(dtype=float)

    z_matrix = zc.zscores.loc[:, list(active)].to_numpy(dtype=float)
    mask_c = (p0_panel["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()
    complete_mask = mask_c & np.isfinite(z_matrix).all(axis=1)
    complete_c = z_matrix[complete_mask]
    complete_index = p0_panel.index.to_numpy()[complete_mask]

    if null_override is not None:
        null = null_override
    else:
        null = _build_primary_null(complete_c=complete_c, sigma=sigma, weights=weights, settings=settings)

    return _ScoreContext(
        panel=p0_panel,
        zscores=zc.zscores,
        composites=p4.composites,
        active=active,
        weights=weights,
        sigma=sigma,
        null=null,
        complete_c_rows=complete_c,
        complete_c_index=complete_index,
    )


def _score_panel_family3(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    *,
    null_override: object = None,
    sigma_override: np.ndarray | None = None,
) -> Family3ScoreContext:
    """Score a panel and adapt it to the Family-3 package ``ScoreContext``.

    The package needs the pre-merge (raw) z-scores in addition to the merged
    frame produced by ``_score_panel``: ``family3_snapshot`` patches raw
    detector outputs incrementally during the counterfactual search. Raw
    z-scores are obtained by rescoring with the detector merge rules disabled.
    """
    from src.engine.zscores import compute_zscores

    ctx = _score_panel(
        panel,
        settings,
        null_override=null_override,
        sigma_override=sigma_override,
    )
    raw_zc = compute_zscores(
        panel=ctx.panel,
        settings=family3_raw_zscore_settings(settings),
        write_outputs=False,
    )
    return Family3ScoreContext(
        panel=ctx.panel,
        raw_zscores=raw_zc.zscores,
        zscores=ctx.zscores,
        composites=ctx.composites,
        active=ctx.active,
        weights=ctx.weights,
        sigma=ctx.sigma,
        null=ctx.null,
        complete_c_rows=ctx.complete_c_rows,
        complete_c_index=ctx.complete_c_index,
    )


def _systematic_baseline(ctx: _ScoreContext, settings: AnalysisSettings) -> pd.DataFrame:
    from src.analysis.injection import run_phase5

    systematic_only = tuple(a for a in settings.injection.archetypes if a.name == "systematic")
    baseline_settings = settings.model_copy(
        update={
            "injection": settings.injection.model_copy(
                update={"archetypes": systematic_only}
            )
        }
    )
    phase5 = run_phase5(
        panel=ctx.panel,
        settings=baseline_settings,
        zscores=ctx.zscores,
        sigma_override=ctx.sigma,
        write_outputs=False,
    )
    df = phase5.power_curves.copy()
    return df[df["archetype"] == "systematic"].copy()


# ── Parallel scoring of contamination cells (Families 1 & 2) ────────────────
# Each contamination cell is scored independently by ``_score_panel``. The only
# input shared across cells is the baseline null/sigma — calibrated once on the
# uncontaminated reference sample and reused for every cell. We pass that
# baseline in as an EXPLICIT override rather than relying on the module-level
# ``_CACHED_*`` globals, because a pooled worker process starts with those
# globals empty and would otherwise rebuild a divergent null from the
# contaminated panel. With the override the pooled result is bit-identical to
# the serial path. Job construction stays sequential (it advances a shared RNG);
# only this scoring step fans out, and results are reassembled in job order.
#
# The workers are picklable module-level functions (closures cannot cross a
# process boundary), and a PROCESS pool — not threads — is required because
# ``_score_panel`` mutates the ``_CACHED_*`` module globals and we want each
# worker's BLAS thread-pool isolated under the OMP/MKL caps.


def _score_family1_job(
    job: tuple,
    settings: AnalysisSettings,
    null_override: object,
    sigma_override: object,
) -> list[dict[str, object]]:
    contaminated_panel, labels, rho, delta, baseline_rates = job
    scored = _score_panel(
        contaminated_panel,
        settings,
        null_override=null_override,
        sigma_override=sigma_override,
    )
    return _composite_metrics(
        scored.composites,
        labels,
        settings.robustness_benchmark.alpha,
        rho=rho,
        delta=delta,
        method="correlated_gaussian",
        extra={"baseline_phase5_systematic_detection_rate": baseline_rates},
    )


def _score_family2_job(
    job: tuple,
    settings: AnalysisSettings,
    null_override: object,
    sigma_override: object,
) -> list[dict[str, object]]:
    df, label_col, method, rho, delta = job
    ctx = _score_panel(
        df,
        settings,
        null_override=null_override,
        sigma_override=sigma_override,
    )
    labels = df[label_col].fillna(0).astype(int).to_numpy()
    rb = settings.robustness_benchmark
    result_rows = list(
        _detector_flag_metrics(ctx.zscores, labels, rb.alpha, rho=rho, delta=delta, method=method)
    )
    result_rows.extend(
        _composite_metrics(ctx.composites, labels, rb.alpha, rho=rho, delta=delta, method=method)
    )
    return result_rows


def _map_scoring_jobs(
    fn,
    jobs: list,
    settings: AnalysisSettings,
    null_override: object,
    sigma_override: object,
    *,
    label: str,
) -> list[list[dict[str, object]]]:
    """Score contamination cells serially or across a process pool.

    Order-preserving: ``result[i]`` corresponds to ``jobs[i]`` regardless of
    completion order, so the emitted rows are identical to the serial path.
    ``workers <= 1`` (or a single job) runs inline — the default, byte-for-byte
    the previous behaviour. A larger pool uses the ``spawn`` start method to
    avoid fork-with-live-threadpool deadlocks; the per-worker re-import cost is
    paid once and amortised over the cells.
    """
    workers = int(getattr(settings.robustness_benchmark, "workers", 1) or 1)
    n = len(jobs)
    if workers <= 1 or n <= 1:
        results: list = []
        for i, job in enumerate(jobs):
            log.info("%s: cell %d/%d", label, i + 1, n)
            results.append(fn(job, settings, null_override, sigma_override))
        return results

    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    results = [None] * n
    max_in_flight = min(n, max(workers, workers * 2))
    log.info(
        "%s: scoring %d cells across %d workers (spawn; max_in_flight=%d)",
        label,
        n,
        workers,
        max_in_flight,
    )
    ctx_mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx_mp) as ex:
        fut_to_i = {}
        next_i = 0
        while next_i < max_in_flight:
            fut_to_i[ex.submit(fn, jobs[next_i], settings, null_override, sigma_override)] = next_i
            next_i += 1

        done = 0
        while fut_to_i:
            completed, _ = wait(fut_to_i, return_when=FIRST_COMPLETED)
            for fut in completed:
                i = fut_to_i.pop(fut)
                results[i] = fut.result()
                done += 1
                log.info("%s: cell %d/%d done", label, done, n)
                if next_i < n:
                    fut_to_i[
                        ex.submit(fn, jobs[next_i], settings, null_override, sigma_override)
                    ] = next_i
                    next_i += 1
    return results


def _simulate_correlated_gaussian(
    ctx: _ScoreContext,
    settings: AnalysisSettings,
    baseline_phase5: pd.DataFrame,
) -> pd.DataFrame:
    # Suppress noisy logging during benchmark inner loops
    scoring_logger = logging.getLogger("src.scoring")
    detector_logger = logging.getLogger("src.engine")
    prev_scoring = scoring_logger.level
    prev_detector = detector_logger.level
    scoring_logger.setLevel(logging.WARNING)
    detector_logger.setLevel(logging.WARNING)

    rb = settings.robustness_benchmark
    rng = np.random.default_rng(rb.random_seed)
    reference_index = ctx.complete_c_index
    if reference_index.size == 0:
        return pd.DataFrame()
    ratio_cov = _family1_ratio_covariance(ctx)
    if ratio_cov is None:
        log.warning("family1_correlated_gaussian: insufficient finite ratio rows for covariance estimation; skipping.")
        return pd.DataFrame()

    # Phase 1: build all contaminated panels sequentially (preserves rng state).
    jobs: list[tuple[pd.DataFrame, np.ndarray, float, float, dict]] = []
    n = reference_index.shape[0]
    for rho in rb.rho_grid:
        for delta in rb.delta_grid:
            n_contam = min(max(1, int(round(rho * n))), n)
            selected = rng.choice(n, size=n_contam, replace=False)
            contaminated_panel = ctx.panel.copy()
            contaminated_panel["_family1_corr_label"] = 0
            modified_indices: list[object] = []
            ratio_noise = rng.multivariate_normal(
                mean=np.zeros(3, dtype=float),
                cov=(delta ** 2) * ratio_cov,
                size=n_contam,
                method="cholesky",
            )

            for pos, noise in zip(selected, ratio_noise):
                idx = reference_index[int(pos)]
                if idx not in contaminated_panel.index:
                    continue
                row = contaminated_panel.loc[idx]
                atq = pd.to_numeric(pd.Series([row.get("atq")]), errors="coerce").iat[0]
                revtq = pd.to_numeric(pd.Series([row.get("revtq")]), errors="coerce").iat[0]
                if not np.isfinite(atq) or atq <= 0 or not np.isfinite(revtq) or revtq <= 0:
                    continue
                base_debt = pd.to_numeric(pd.Series([row.get("ratio_debt_adj")]), errors="coerce").iat[0]
                base_cash = pd.to_numeric(pd.Series([row.get("ratio_cash_adj")]), errors="coerce").iat[0]
                base_income = pd.to_numeric(pd.Series([row.get("ratio_income")]), errors="coerce").iat[0]
                if not (np.isfinite(base_debt) and np.isfinite(base_cash) and np.isfinite(base_income)):
                    continue

                target_debt = max(base_debt + float(noise[0]), 0.0)
                target_cash = max(base_cash + float(noise[1]), 0.0)
                target_income = max(base_income + float(noise[2]), 0.0)

                sukuk_ratio = float(row.get("sukuk_ratio_t", 0.0) or 0.0)
                islamic_cash_ratio = float(row.get("islamic_cash_ratio_t", 0.0) or 0.0)
                dlcq = pd.to_numeric(pd.Series([row.get("dlcq")]), errors="coerce").iat[0]
                if not np.isfinite(dlcq):
                    dlcq = 0.0

                debt_factor = max(1.0 - sukuk_ratio, 1e-9)
                cash_factor = max(1.0 - islamic_cash_ratio, 1e-9)
                contaminated_panel.at[idx, "dlttq"] = max((target_debt * atq - dlcq) / debt_factor, 0.0)
                contaminated_panel.at[idx, "cheq"] = max((target_cash * atq) / cash_factor, 0.0)
                contaminated_panel.at[idx, "iditq"] = max(target_income * revtq, 0.0)
                contaminated_panel.at[idx, "_family1_corr_label"] = 1
                modified_indices.append(idx)

            contaminated_panel = _recompute_ratios_for_rows(contaminated_panel, modified_indices)
            labels = contaminated_panel["_family1_corr_label"].fillna(0).astype(int).to_numpy()
            baseline_match = baseline_phase5[
                (baseline_phase5["delta"] == delta)
                & (baseline_phase5["alpha"] == rb.alpha)
            ]
            baseline_rates = {
                r["composite"]: float(r["detection_rate"])
                for _, r in baseline_match.iterrows()
            }
            jobs.append((contaminated_panel, labels, rho, delta, baseline_rates))

    if not jobs:
        return pd.DataFrame()

    # Phase 2: score contaminated panels (serial or across a process pool).
    # ``ctx`` is the uncontaminated base context, so ``ctx.null``/``ctx.sigma``
    # are exactly the baseline cached by the orchestrator and reused for every
    # cell in the serial path — pass them as explicit overrides so a pooled
    # worker reproduces the serial numbers.
    rows: list[dict[str, object]] = []
    scored_cells = _map_scoring_jobs(
        _score_family1_job,
        jobs,
        settings,
        ctx.null,
        ctx.sigma,
        label="Family 1 (Gaussian)",
    )
    for result_rows in scored_cells:
        rows.extend(result_rows)
    log.info("Family 1 (Gaussian): done")

    # Restore logging
    scoring_logger.setLevel(prev_scoring)
    detector_logger.setLevel(prev_detector)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if "baseline_phase5_systematic_detection_rate" in out.columns:
        out["baseline_phase5_systematic_detection_rate"] = out.apply(
            lambda r: r["baseline_phase5_systematic_detection_rate"].get(r["entity_name"], np.nan)
            if isinstance(r["baseline_phase5_systematic_detection_rate"], dict)
            else np.nan,
            axis=1,
        )
    return out


def _build_realistic_contamination_jobs(panel: pd.DataFrame, settings: AnalysisSettings) -> list[tuple | dict]:
    rb = settings.robustness_benchmark
    legacy_settings = ContaminationSettings()
    threshold_debt, _, _ = _threshold_values(panel)

    jobs: list[tuple | dict] = []
    for method in rb.realistic_methods:
        for rho in rb.rho_grid:
            for delta in rb.delta_grid:
                try:
                    if method == "threshold_clustering_m1_v2":
                        result = contaminate_m1_v2(
                            panel, rate=rho, random_state=rb.random_seed, slack_max=threshold_debt * delta
                        )
                    elif method == "temporal_spike_m2b":
                        result = contaminate_m2b(
                            panel, rate=rho, random_state=rb.random_seed, jump_scale=delta
                        )
                    elif method == "benford_m3_v2":
                        result = contaminate_m3_v2(
                            panel, rate=rho, random_state=rb.random_seed
                        )
                    elif method == "interstatement_m4_v2":
                        result = contaminate_m4_v2(
                            panel, rate=rho, random_state=rb.random_seed, delta_scale=max(delta, 1e-6)
                        )
                    elif method == "m5_cod_break":
                        base = panel.copy()
                        base["y"] = 0
                        contaminated = apply_m5(
                            base,
                            rho=rho,
                            settings=legacy_settings,
                            rng=np.random.default_rng(rb.random_seed),
                        )
                        modified_idx = contaminated.index[contaminated["y"].fillna(0).astype(int) == 1].tolist()
                        contaminated = _recompute_ratios_for_rows(contaminated, modified_idx)
                        result = SimpleNamespace(df=contaminated, label_col="y")
                    elif method == "m6_seasonal":
                        base = panel.copy()
                        base["y"] = 0
                        contaminated = apply_m6(
                            base,
                            rho=rho,
                            settings=legacy_settings,
                            rng=np.random.default_rng(rb.random_seed),
                        )
                        modified_idx = contaminated.index[
                            (contaminated["y"].fillna(0).astype(int) == 1)
                            & (pd.to_numeric(contaminated.get("fqtr"), errors="coerce") == 4)
                        ].tolist()
                        contaminated = _recompute_ratios_for_rows(contaminated, modified_idx)
                        result = SimpleNamespace(df=contaminated, label_col="y")
                    elif method == "abn_disx_full":
                        result = contaminate_abn_disx_full(
                            panel, rate=rho, random_state=rb.random_seed, delta=delta
                        )
                    elif method == "abn_disx_partial":
                        result = contaminate_abn_disx_partial(
                            panel, rate=rho, random_state=rb.random_seed, delta=delta
                        )
                    elif method == "abn_prod_full":
                        result = contaminate_abn_prod_full(
                            panel, rate=rho, random_state=rb.random_seed, delta=delta
                        )
                    else:
                        continue
                    jobs.append((result.df, result.label_col, method, rho, delta))
                except ValueError as exc:
                    jobs.append({
                        "method": method,
                        "rho": rho,
                        "delta": delta,
                        "entity_type": "status",
                        "entity_name": "skipped",
                        "status": str(exc),
                    })
    return jobs


def _evaluate_realistic_manipulations(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    null_override: object = None,
    sigma_override: object = None,
) -> pd.DataFrame:

    jobs = _build_realistic_contamination_jobs(panel, settings)

    # ``dict`` jobs are pre-computed error rows (construction failures) that pass
    # straight through; ``tuple`` jobs are the contaminated panels to score.
    score_job_positions = [(i, j) for i, j in enumerate(jobs) if isinstance(j, tuple)]
    score_jobs = [j for _, j in score_job_positions]
    rows_by_job: list[list[dict[str, object]]] = [
        [j] if isinstance(j, dict) else [] for j in jobs
    ]

    # Suppress noisy per-detector logging during benchmark inner loop
    scoring_logger = logging.getLogger("src.scoring")
    detector_logger = logging.getLogger("src.engine")
    prev_scoring = scoring_logger.level
    prev_detector = detector_logger.level
    scoring_logger.setLevel(logging.WARNING)
    detector_logger.setLevel(logging.WARNING)

    if score_jobs:
        # ``null_override``/``sigma_override`` are the orchestrator's cached
        # baseline (see _map_scoring_jobs); passing them explicitly lets pooled
        # workers reproduce the serial numbers. When None (serial callers /
        # tests), _score_panel falls back to the _CACHED_* module globals.
        scored_cells = _map_scoring_jobs(
            _score_family2_job,
            score_jobs,
            settings,
            null_override,
            sigma_override,
            label="Family 2 (realistic)",
        )
        for (job_i, _), result_rows in zip(score_job_positions, scored_cells):
            rows_by_job[job_i] = result_rows
        log.info("Family 2 (realistic): done")

    scoring_logger.setLevel(prev_scoring)
    detector_logger.setLevel(prev_detector)

    rows: list[dict[str, object]] = []
    for result_rows in rows_by_job:
        rows.extend(result_rows)
    return pd.DataFrame(rows)


def _recompute_ratios_for_rows(
    panel: pd.DataFrame,
    indices: list[object] | np.ndarray,
) -> pd.DataFrame:
    if len(indices) == 0:
        return panel
    updated = panel.copy()
    target_index = updated.index.intersection(pd.Index(indices))
    if target_index.empty:
        return updated
    subset = updated.loc[target_index].copy()
    recomputed = compute_shariah_ratios(
        subset,
        log_coverage=False,
        warn_on_missing_connectors=False,
    )
    assign_index = updated.index.intersection(recomputed.index)
    if assign_index.empty:
        return updated
    assign_cols = [c for c in recomputed.columns if c in updated.columns]
    for col in assign_cols:
        updated.loc[assign_index, col] = recomputed.loc[assign_index, col].to_numpy()
    return updated


def _override_split_to_iid(panel_with_split: pd.DataFrame, iid_years: tuple[int, int]) -> pd.DataFrame:
    out = panel_with_split.copy()
    quarter_col = "datacqtr" if "datacqtr" in out.columns else "quarter"
    years = _year_from_quarter(out[quarter_col])
    orig_c = out["_split"] == SPLIT_LABEL_INCLUDED
    iid_mask = orig_c & years.between(iid_years[0], iid_years[1], inclusive="both")
    out.loc[~iid_mask, "_split"] = SPLIT_LABEL_EXCLUDED
    out.loc[~iid_mask, "_split_reason"] = "anoshift_non_iid"
    out.loc[iid_mask, "_split_reason"] = REASON_INCLUDED
    return out


def _restrict_c_to_year_window(panel_with_split: pd.DataFrame, year_window: tuple[int, int]) -> pd.DataFrame:
    out = panel_with_split.copy()
    quarter_col = "datacqtr" if "datacqtr" in out.columns else "quarter"
    years = _year_from_quarter(out[quarter_col])
    orig_c = out["_split"] == SPLIT_LABEL_INCLUDED
    window_mask = orig_c & years.between(year_window[0], year_window[1], inclusive="both")
    out.loc[~window_mask, "_split"] = SPLIT_LABEL_EXCLUDED
    out.loc[~window_mask, "_split_reason"] = "anoshift_outside_calibration_window"
    out.loc[window_mask, "_split_reason"] = REASON_INCLUDED
    return out


def _restricted_m1_v2(
    panel: pd.DataFrame,
    eligible_mask: pd.Series,
    *,
    rate: float,
    slack_max: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_state)
    out = panel.copy()
    out["_anoshift_m1_label"] = 0
    eligible_index = out.index[eligible_mask.fillna(False)].to_numpy()
    if eligible_index.size == 0:
        return out, out["_anoshift_m1_label"].to_numpy(dtype=int)

    sample_n = min(max(1, int(round(rate * len(eligible_index)))), len(eligible_index))
    selected = pd.Index(rng.choice(eligible_index, size=sample_n, replace=False))
    threshold_debt, threshold_cash, threshold_income = _threshold_values(out)
    for idx in selected:
        row = out.loc[idx]
        touched = False
        if pd.notna(row.get("ratio_debt_adj")) and row["ratio_debt_adj"] > threshold_debt:
            atq = float(row["atq"])
            dlcq = float(row["dlcq"])
            sukuk_ratio = float(row.get("sukuk_ratio_t", 0.0) or 0.0)
            debt_factor = max(1.0 - sukuk_ratio, 1e-9)
            raw_target = max(((threshold_debt - slack_max) * atq - dlcq) / debt_factor, 0.0)
            out.at[idx, "dlttq"] = raw_target
            touched = True
        if pd.notna(row.get("ratio_cash_adj")) and row["ratio_cash_adj"] > threshold_cash:
            atq = float(row["atq"])
            islamic_cash_ratio = float(row.get("islamic_cash_ratio_t", 0.0) or 0.0)
            cash_factor = max(1.0 - islamic_cash_ratio, 1e-9)
            out.at[idx, "cheq"] = max(((threshold_cash - slack_max) * atq) / cash_factor, 0.0)
            touched = True
        if pd.notna(row.get("ratio_income")) and row["ratio_income"] > threshold_income:
            out.at[idx, "iditq"] = max((threshold_income - slack_max) * float(row["revtq"]), 0.0)
            touched = True
        if touched:
            out.at[idx, "_anoshift_m1_label"] = 1
    out = compute_shariah_ratios(out)
    return out, out["_anoshift_m1_label"].to_numpy(dtype=int)


def _run_anoshift(panel: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    from src.analysis.composite_scoring import _attach_pvalues, _composite_columns

    # Suppress noisy logging during AnoShift inner loops
    scoring_logger = logging.getLogger("src.scoring")
    detector_logger = logging.getLogger("src.engine")
    prev_scoring = scoring_logger.level
    prev_detector = detector_logger.level
    scoring_logger.setLevel(logging.WARNING)
    detector_logger.setLevel(logging.WARNING)

    rb = settings.robustness_benchmark
    base_p0 = panel if "_split" in panel.columns else panel
    years = _year_from_quarter(base_p0[settings.panel_schema.quarter])
    original_c = base_p0["_split"] == SPLIT_LABEL_INCLUDED
    rows: list[dict[str, object]] = []
    c_years = sorted(int(y) for y in pd.Series(years[original_c]).dropna().unique().tolist())

    def _append_fpr_rows(ctx: _ScoreContext, *, protocol: str, calibration_split: str) -> None:
        for split_name, year_range in rb.anoshift_splits.items():
            split_mask = original_c & years.between(year_range[0], year_range[1], inclusive="both")
            if split_mask.sum() == 0:
                continue
            for value_col, p_col in zip(COMPOSITE_VALUE_COLUMNS, COMPOSITE_P_COLUMNS):
                if p_col not in ctx.composites.columns:
                    continue
                pvals = pd.to_numeric(ctx.composites.loc[split_mask, p_col], errors="coerce")
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "fpr",
                        "protocol": protocol,
                        "calibration_split": calibration_split,
                        "split": split_name,
                        "entity_name": value_col,
                        "pvalue_col": p_col,
                        "n_rows": int(pvals.notna().sum()),
                        "fpr": float((pvals < rb.alpha).mean()) if pvals.notna().any() else float("nan"),
                    }
                )

    def _append_annual_rows(ctx: _ScoreContext, *, protocol: str, calibration_split: str) -> None:
        year_values = sorted(int(y) for y in pd.Series(years[original_c]).dropna().unique().tolist())
        for year in year_values:
            year_mask = original_c & (years == year)
            if year_mask.sum() == 0:
                continue
            p_col = rb.primary_composite
            if p_col not in ctx.composites.columns:
                continue
            pvals = pd.to_numeric(ctx.composites.loc[year_mask, p_col], errors="coerce")
            rows.append(
                {
                    "method": "anoshift_yearly",
                    "kind": "annual_fpr",
                    "protocol": protocol,
                    "calibration_split": calibration_split,
                    "year": int(year),
                    "entity_name": p_col.replace("p_", ""),
                    "pvalue_col": p_col,
                    "n_rows": int(pvals.notna().sum()),
                    "fpr": float((pvals < rb.alpha).mean()) if pvals.notna().any() else float("nan"),
                }
            )

    def _append_power_rows(
        calibration_ctx: _ScoreContext,
        *,
        protocol: str,
        calibration_split: str,
    ) -> None:
        eval_split_names = [name for name in ("near", "far") if name in rb.anoshift_splits]
        for i_split, split_name in enumerate(eval_split_names):
            year_range = rb.anoshift_splits[split_name]
            eval_mask = original_c & years.between(year_range[0], year_range[1], inclusive="both")
            eligible_index = calibration_ctx.panel.index[eval_mask]
            if len(eligible_index) == 0:
                continue

            z_eval = calibration_ctx.zscores.loc[eligible_index, list(calibration_ctx.active)].to_numpy(dtype=float)
            split_complete = z_eval[np.isfinite(z_eval).all(axis=1)]
            if split_complete.size:
                for i_rho, rho in enumerate(rb.rho_grid):
                    for i_delta, delta in enumerate(rb.delta_grid):
                        _seed = rb.random_seed + i_split * len(rb.rho_grid) * len(rb.delta_grid) + i_rho * len(rb.delta_grid) + i_delta
                        rng = np.random.default_rng(_seed)
                        n_rows = split_complete.shape[0]
                        n_contam = min(max(1, int(round(rho * n_rows))), n_rows)
                        selected = rng.choice(n_rows, size=n_contam, replace=False)
                        labels = np.zeros(n_rows, dtype=int)
                        labels[selected] = 1
                        z_rows = split_complete.copy()
                        noise = rng.multivariate_normal(
                            mean=np.zeros(len(calibration_ctx.active), dtype=float),
                            cov=(delta ** 2) * calibration_ctx.sigma,
                            size=n_contam,
                            method="cholesky",
                        )
                        z_rows[selected] = z_rows[selected] + noise
                        comp = _composite_columns(z_rows, calibration_ctx.sigma, calibration_ctx.weights, settings)
                        pvals = _attach_pvalues(comp, calibration_ctx.null)
                        comp_df = pd.DataFrame(comp)
                        for name, arr in pvals.items():
                            comp_df[name] = arr
                        for row in _composite_metrics(
                            comp_df,
                            labels,
                            rb.alpha,
                            rho=rho,
                            delta=delta,
                            method="anoshift_correlated_gaussian",
                        ):
                            row["kind"] = "power"
                            row["protocol"] = protocol
                            row["calibration_split"] = calibration_split
                            row["split"] = split_name
                            rows.append(row)

            for i_rho, rho in enumerate(rb.rho_grid):
                for i_delta, delta in enumerate(rb.delta_grid):
                    _seed = rb.random_seed + i_split * len(rb.rho_grid) * len(rb.delta_grid) + i_rho * len(rb.delta_grid) + i_delta
                    contaminated_panel, labels = _restricted_m1_v2(
                        calibration_ctx.panel,
                        eval_mask,
                        rate=rho,
                        slack_max=_threshold_values(calibration_ctx.panel)[0] * delta,
                        random_state=_seed,
                    )
                    try:
                        contaminated_ctx = _score_panel(
                            contaminated_panel,
                            settings,
                            sigma_override=calibration_ctx.sigma,
                            null_override=calibration_ctx.null,
                        )
                    except ValueError as exc:
                        rows.append(
                            {
                                "method": "anoshift_threshold_clustering_m1_v2",
                                "kind": "status",
                                "protocol": protocol,
                                "calibration_split": calibration_split,
                                "split": split_name,
                                "rho": rho,
                                "delta": delta,
                                "status": f"skipped: {exc}",
                            }
                        )
                        continue
                    for row in _composite_metrics(
                        contaminated_ctx.composites,
                        labels,
                        rb.alpha,
                        rho=rho,
                        delta=delta,
                        method="anoshift_threshold_clustering_m1_v2",
                    ):
                        row["kind"] = "power"
                        row["protocol"] = protocol
                        row["calibration_split"] = calibration_split
                        row["split"] = split_name
                        rows.append(row)

    try:
        global_ctx = _score_panel(base_p0, settings)
        _append_fpr_rows(global_ctx, protocol="global_c_near_far", calibration_split="global_c")
        _append_annual_rows(global_ctx, protocol="global_c_near_far", calibration_split="global_c")
        _append_power_rows(global_ctx, protocol="global_c_near_far", calibration_split="global_c")
    except Exception as exc:
        log.warning("anoshift: global calibration skipped because scoring failed: %s", exc)
        rows.append(
            {
                "method": "anoshift",
                "kind": "status",
                "protocol": "global_c_near_far",
                "calibration_split": "global_c",
                "status": f"global_calibration_skipped: {exc}",
            }
        )

    iid_years = rb.anoshift_splits.get("iid")
    if iid_years is not None:
        iid_start, iid_end = iid_years
        candidate_ends = [y for y in c_years if y >= iid_end]
        fitted_iid_ctx: _ScoreContext | None = None
        fitted_iid_end: int | None = None
        last_iid_exc: Exception | None = None
        try:
            from tqdm import tqdm
            end_iter = tqdm(candidate_ends, desc="Family 4 AnoShift (iid)", unit="year", leave=False)
        except ImportError:
            end_iter = candidate_ends
        for end_year in end_iter:
            iid_window = (iid_start, end_year)
            iid_only_panel = _restrict_c_to_year_window(base_p0, iid_window)
            try:
                candidate_ctx = _score_panel(iid_only_panel, settings)
                complete_rows = int(len(candidate_ctx.complete_c_index))
                log.info(
                    "anoshift iid_transfer: start=%d end=%d complete_c_rows=%d min_required=%d",
                    iid_start,
                    end_year,
                    complete_rows,
                    int(rb.anoshift_min_complete_rows),
                )
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "calibration_diagnostic",
                        "protocol": "iid_transfer",
                        "calibration_split": f"{iid_start}-{end_year}",
                        "complete_c_rows": complete_rows,
                        "min_complete_rows": int(rb.anoshift_min_complete_rows),
                        "status": "candidate_scored",
                    }
                )
                if complete_rows < rb.anoshift_min_complete_rows:
                    last_iid_exc = ValueError(
                        f"only {complete_rows} complete z-vectors in C (< {rb.anoshift_min_complete_rows})"
                    )
                    rows.append(
                        {
                            "method": "anoshift",
                            "kind": "calibration_diagnostic",
                            "protocol": "iid_transfer",
                            "calibration_split": f"{iid_start}-{end_year}",
                            "complete_c_rows": complete_rows,
                            "min_complete_rows": int(rb.anoshift_min_complete_rows),
                            "status": str(last_iid_exc),
                        }
                    )
                    continue
                fitted_iid_ctx = candidate_ctx
                fitted_iid_end = end_year
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "calibration_diagnostic",
                        "protocol": "iid_transfer",
                        "calibration_split": f"{iid_start}-{end_year}",
                        "complete_c_rows": complete_rows,
                        "min_complete_rows": int(rb.anoshift_min_complete_rows),
                        "status": "selected",
                    }
                )
                break
            except Exception as exc:
                last_iid_exc = exc
                log.warning("anoshift iid_transfer: start=%d end=%d failed: %s", iid_start, end_year, exc)
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "calibration_diagnostic",
                        "protocol": "iid_transfer",
                        "calibration_split": f"{iid_start}-{end_year}",
                        "complete_c_rows": np.nan,
                        "min_complete_rows": int(rb.anoshift_min_complete_rows),
                        "status": f"failed: {exc}",
                    }
                )
                continue
        try:
            if fitted_iid_ctx is None or fitted_iid_end is None:
                raise last_iid_exc or ValueError("no valid iid calibration window found")
            transfer_ctx = _score_panel(
                base_p0,
                settings,
                sigma_override=fitted_iid_ctx.sigma,
                null_override=fitted_iid_ctx.null,
            )
            calib_label = f"{iid_start}-{fitted_iid_end}"
            log.info(
                "anoshift iid_transfer: selected calibration window %s with complete_c_rows=%d",
                calib_label,
                int(len(fitted_iid_ctx.complete_c_index)),
            )
            _append_fpr_rows(transfer_ctx, protocol="iid_transfer", calibration_split=calib_label)
            _append_annual_rows(transfer_ctx, protocol="iid_transfer", calibration_split=calib_label)
            _append_power_rows(transfer_ctx, protocol="iid_transfer", calibration_split=calib_label)
        except Exception as exc:
            log.warning("anoshift: iid transfer calibration skipped because scoring failed: %s", exc)
            rows.append(
                {
                    "method": "anoshift",
                    "kind": "status",
                    "protocol": "iid_transfer",
                    "calibration_split": "iid",
                    "status": f"iid_transfer_skipped: {exc}",
                }
            )

    adaptive_start_years = tuple(rb.anoshift_adaptive_start_years)
    if not adaptive_start_years:
        if c_years:
            derived: list[int] = []
            first = c_years[0]
            max_year = c_years[-1]
            for start in range(first, max_year, 2):
                if start in c_years and start < max_year:
                    derived.append(start)
            adaptive_start_years = tuple(derived[:4])

    try:
        from tqdm import tqdm
        adapt_iter = tqdm(list(adaptive_start_years), desc="Family 4 AnoShift (adaptive)", unit="start", leave=False)
    except ImportError:
        adapt_iter = adaptive_start_years
    for start_year in adapt_iter:
        candidate_ends = [y for y in c_years if y >= start_year]
        if not candidate_ends:
            rows.append(
                {
                    "method": "anoshift",
                    "kind": "status",
                    "protocol": "adaptive_transfer",
                    "calibration_split": f"{start_year}-?",
                    "status": "adaptive_transfer_skipped: no calibration years available",
                }
            )
            continue

        fitted_ctx: _ScoreContext | None = None
        fitted_end_year: int | None = None
        last_exc: Exception | None = None
        for end_year in candidate_ends:
            calibration_panel = _restrict_c_to_year_window(base_p0, (start_year, end_year))
            try:
                candidate_ctx = _score_panel(calibration_panel, settings)
                log.info(
                    "anoshift adaptive_transfer: start=%d end=%d complete_c_rows=%d min_required=%d",
                    start_year,
                    end_year,
                    int(len(candidate_ctx.complete_c_index)),
                    int(rb.anoshift_min_complete_rows),
                )
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "calibration_diagnostic",
                        "protocol": "adaptive_transfer",
                        "calibration_split": f"{start_year}-{end_year}",
                        "complete_c_rows": int(len(candidate_ctx.complete_c_index)),
                        "min_complete_rows": int(rb.anoshift_min_complete_rows),
                        "status": "candidate_scored",
                    }
                )
                if int(len(candidate_ctx.complete_c_index)) < rb.anoshift_min_complete_rows:
                    last_exc = ValueError(
                        f"only {len(candidate_ctx.complete_c_index)} complete z-vectors in C (< {rb.anoshift_min_complete_rows})"
                    )
                    rows.append(
                        {
                            "method": "anoshift",
                            "kind": "calibration_diagnostic",
                            "protocol": "adaptive_transfer",
                            "calibration_split": f"{start_year}-{end_year}",
                            "complete_c_rows": int(len(candidate_ctx.complete_c_index)),
                            "min_complete_rows": int(rb.anoshift_min_complete_rows),
                            "status": str(last_exc),
                        }
                    )
                    continue
                fitted_ctx = candidate_ctx
                fitted_end_year = end_year
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "calibration_diagnostic",
                        "protocol": "adaptive_transfer",
                        "calibration_split": f"{start_year}-{end_year}",
                        "complete_c_rows": int(len(candidate_ctx.complete_c_index)),
                        "min_complete_rows": int(rb.anoshift_min_complete_rows),
                        "status": "selected",
                    }
                )
                break
            except Exception as exc:
                last_exc = exc
                log.warning("anoshift adaptive_transfer: start=%d end=%d failed: %s", start_year, end_year, exc)
                rows.append(
                    {
                        "method": "anoshift",
                        "kind": "calibration_diagnostic",
                        "protocol": "adaptive_transfer",
                        "calibration_split": f"{start_year}-{end_year}",
                        "complete_c_rows": np.nan,
                        "min_complete_rows": int(rb.anoshift_min_complete_rows),
                        "status": f"failed: {exc}",
                    }
                )
                continue

        if fitted_ctx is None or fitted_end_year is None:
            rows.append(
                {
                    "method": "anoshift",
                    "kind": "status",
                    "protocol": "adaptive_transfer",
                    "calibration_split": f"{start_year}-?",
                    "status": f"adaptive_transfer_skipped: {last_exc}",
                }
            )
            continue

        future_mask = original_c & (years > fitted_end_year)
        if int(future_mask.sum()) == 0:
            rows.append(
                {
                    "method": "anoshift",
                    "kind": "status",
                    "protocol": "adaptive_transfer",
                    "calibration_split": f"{start_year}-{fitted_end_year}",
                    "status": "adaptive_transfer_skipped: no future evaluation rows after calibration window",
                }
            )
            continue

        try:
            transfer_ctx = _score_panel(
                base_p0,
                settings,
                sigma_override=fitted_ctx.sigma,
                null_override=fitted_ctx.null,
            )
            calib_label = f"{start_year}-{fitted_end_year}"
            log.info(
                "anoshift adaptive_transfer: selected calibration window %s with complete_c_rows=%d; future_eval_rows=%d",
                calib_label,
                int(len(fitted_ctx.complete_c_index)),
                int(future_mask.sum()),
            )
            _append_fpr_rows(transfer_ctx, protocol="adaptive_transfer", calibration_split=calib_label)
            _append_annual_rows(transfer_ctx, protocol="adaptive_transfer", calibration_split=calib_label)
            _append_power_rows(transfer_ctx, protocol="adaptive_transfer", calibration_split=calib_label)
        except Exception as exc:
            rows.append(
                {
                    "method": "anoshift",
                    "kind": "status",
                    "protocol": "adaptive_transfer",
                    "calibration_split": f"{start_year}-{fitted_end_year}",
                    "status": f"adaptive_transfer_scoring_failed: {exc}",
                }
            )

    # Restore logging
    scoring_logger.setLevel(prev_scoring)
    detector_logger.setLevel(prev_detector)
    return pd.DataFrame(rows)


def _run_family3_post_realistic(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
    *,
    method_name: str,
) -> pd.DataFrame:
    """Run the Family-3 counterfactual over realistic-contamination cells.

    ``run_family3_xai`` attacks a single already-contaminated panel; the outer
    contamination loop (build cells -> score each -> attack) stays here, issuing
    one ``run_family3_xai`` call per cell. Row eligibility within each cell is
    decided per target-score inside the package from the full contaminated set,
    so we pass the whole contaminated index as candidates rather than the old
    primary-composite RED pre-filter (the RED count is kept only for logging).
    """
    rows: list[pd.DataFrame] = []
    jobs = _build_realistic_contamination_jobs(panel, settings)
    rb = settings.robustness_benchmark

    for job in jobs:
        if isinstance(job, dict):
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "method": method_name,
                            "source_family2_method": job.get("method"),
                            "source_rho": job.get("rho"),
                            "source_delta": job.get("delta"),
                            "status": job.get("status", "contamination build failed"),
                        }
                    ]
                )
            )
            continue

        contaminated_panel, label_col, source_method, rho, delta = job
        labels = contaminated_panel[label_col].fillna(0).astype(int)
        contaminated_index = contaminated_panel.index[labels == 1]
        if len(contaminated_index) == 0:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "method": method_name,
                            "source_family2_method": source_method,
                            "source_rho": rho,
                            "source_delta": delta,
                            "status": "no contaminated rows",
                        }
                    ]
                )
            )
            continue

        try:
            contaminated_ctx = _score_panel_family3(contaminated_panel, settings)
        except Exception as exc:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "method": method_name,
                            "source_family2_method": source_method,
                            "source_rho": rho,
                            "source_delta": delta,
                            "status": f"scoring_failed: {exc}",
                        }
                    ]
                )
            )
            continue

        if rb.primary_composite not in contaminated_ctx.composites.columns:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "method": method_name,
                            "source_family2_method": source_method,
                            "source_rho": rho,
                            "source_delta": delta,
                            "status": f"missing primary composite {rb.primary_composite}",
                        }
                    ]
                )
            )
            continue

        contaminated_primary = pd.to_numeric(
            contaminated_ctx.composites.loc[
                contaminated_ctx.composites.index.intersection(contaminated_index),
                rb.primary_composite,
            ],
            errors="coerce",
        )
        n_red_contaminated = int((contaminated_primary < rb.alpha).sum())
        log.info(
            "family3_post_realistic: method=%s source=%s rho=%.3f delta=%.3f contaminated=%d red_contaminated=%d",
            method_name,
            source_method,
            rho,
            delta,
            int(len(contaminated_index)),
            n_red_contaminated,
        )

        summary, _epochs = run_family3_xai(
            contaminated_panel,
            settings,
            contaminated_ctx,
            method_name=method_name,
            candidate_index=contaminated_index,
            source_fields={
                "source_family2_method": source_method,
                "source_rho": rho,
                "source_delta": delta,
                "source_n_contaminated": int(len(contaminated_index)),
                "source_n_red_contaminated": n_red_contaminated,
            },
        )
        if not summary.empty:
            rows.append(summary)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def run_robustness_benchmark(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    write_outputs: bool = True,
    resume: bool = True,
) -> RobustnessBenchmarkOutcome:
    settings = _benchmark_settings(settings or AnalysisSettings())
    paths = _robustness_output_paths(settings) if write_outputs else {}
    base_ctx = _score_panel(panel, settings)
    _cache_baseline(base_ctx)
    log.info("robustness_benchmark: baseline cached (null + sigma reused for all cells)")
    baseline_phase5 = _systematic_baseline(base_ctx, settings)

    method_set = set(settings.robustness_benchmark.methods)
    family1_enabled = "correlated_gaussian" in method_set
    family2_enabled = any(m in method_set for m in settings.robustness_benchmark.realistic_methods)
    family3_enabled = any(
        m in method_set
        for m in (
            "adversarial_evasion",
            "adversarial_general_evasion",
            "adversarial_evasion_no_sac",
            "adversarial_general_evasion_no_sac",
            "adversarial_post_realistic_evasion",
            "adversarial_general_post_realistic_evasion",
            "adversarial_post_realistic_evasion_no_sac",
            "adversarial_general_post_realistic_evasion_no_sac",
        )
    )
    family4_enabled = "anoshift" in method_set

    family1 = (
        _load_family_checkpoint(paths, "family1", resume)
        if write_outputs and family1_enabled
        else None
    )
    if family1 is None:
        family1 = (
            _simulate_correlated_gaussian(base_ctx, settings, baseline_phase5)
            if family1_enabled
            else pd.DataFrame()
        )
        if write_outputs:
            _write_csv_checkpoint(family1, paths["family1"])
            log.info("robustness_benchmark: checkpointed family1 to %s", paths["family1"])

    family2 = (
        _load_family_checkpoint(paths, "family2", resume)
        if write_outputs and family2_enabled
        else None
    )
    if family2 is None:
        family2 = (
            _evaluate_realistic_manipulations(
                base_ctx.panel,
                settings,
                null_override=base_ctx.null,
                sigma_override=base_ctx.sigma,
            )
            if family2_enabled
            else pd.DataFrame()
        )
        if write_outputs:
            _write_csv_checkpoint(family2, paths["family2"])
            log.info("robustness_benchmark: checkpointed family2 to %s", paths["family2"])

    family3 = (
        _load_family_checkpoint(paths, "family3", resume)
        if write_outputs and family3_enabled
        else None
    )
    if family3 is None:
        family3_parts: list[pd.DataFrame] = []
        # The canonical counterfactual package retired the targeted-vs-general
        # (`surrogate_mode`) and SAC-projector (`enforce_sac`) attack axes: only
        # two behaviourally distinct methods survive — a global evasion pass over
        # the base panel and a post-realistic pass over contaminated cells. The
        # legacy `_general`/`_no_sac` method names therefore fold into these two
        # rather than re-running as relabelled duplicates.
        direct_requested = any(
            m in method_set
            for m in (
                "adversarial_evasion",
                "adversarial_general_evasion",
                "adversarial_evasion_no_sac",
                "adversarial_general_evasion_no_sac",
            )
        )
        post_realistic_requested = any(
            m in method_set
            for m in (
                "adversarial_post_realistic_evasion",
                "adversarial_general_post_realistic_evasion",
                "adversarial_post_realistic_evasion_no_sac",
                "adversarial_general_post_realistic_evasion_no_sac",
            )
        )
        if direct_requested:
            family3_base_ctx = _score_panel_family3(panel, settings)
            summary, _epochs = run_family3_xai(
                panel,
                settings,
                family3_base_ctx,
                method_name="adversarial_evasion",
            )
            if not summary.empty:
                family3_parts.append(summary)
        if post_realistic_requested:
            post_summary = _run_family3_post_realistic(
                panel,
                settings,
                method_name="adversarial_post_realistic_evasion",
            )
            if not post_summary.empty:
                family3_parts.append(post_summary)
        family3 = pd.concat(family3_parts, ignore_index=True) if family3_parts else pd.DataFrame()
        if write_outputs:
            _write_csv_checkpoint(family3, paths["family3"])
            log.info("robustness_benchmark: checkpointed family3 to %s", paths["family3"])

    family4 = (
        _load_family_checkpoint(paths, "family4", resume)
        if write_outputs and family4_enabled
        else None
    )
    if family4 is None:
        family4 = (
            _run_anoshift(base_ctx.panel, settings)
            if family4_enabled
            else pd.DataFrame()
        )
        if write_outputs:
            _write_csv_checkpoint(family4, paths["family4"])
            log.info("robustness_benchmark: checkpointed family4 to %s", paths["family4"])

    summary = {
        "methods": list(settings.robustness_benchmark.methods),
        "alpha": settings.robustness_benchmark.alpha,
        "rho_grid": list(settings.robustness_benchmark.rho_grid),
        "delta_grid": list(settings.robustness_benchmark.delta_grid),
        "bootstrap_replicates": settings.robustness_benchmark.bootstrap_replicates,
        "key_variable_coverage": _coverage_table(panel).to_dict(orient="records"),
        "family_rows": {
            "family1_correlated_gaussian": int(len(family1)),
            "family2_threshold_clustering": int(len(family2)),
            "family3_adversarial_evasion": int(len(family3)),
            "family4_anoshift": int(len(family4)),
        },
    }

    if write_outputs:
        out_dir = settings.output_layout.robustness_benchmark_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        coverage_df = _coverage_table(panel)
        scoreboard_df = _scoreboard_table(family1, family2, family3, family4)

        coverage_df.to_csv(paths["coverage_csv"], index=False)
        scoreboard_df.to_csv(paths["scoreboard_csv"], index=False)
        paths["coverage_md"].write_text(_markdown_table(coverage_df), encoding="utf-8")
        paths["scoreboard_md"].write_text(_markdown_table(scoreboard_df), encoding="utf-8")
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        md_lines = [
            "# Robustness Benchmark",
            "",
            "This benchmark complements `phase5_injection`.",
            "",
            "- `family1_correlated_gaussian.csv`: covariance-aware mean-shift benchmark.",
            "- `family2_threshold_clustering.csv`: raw-data manipulation benchmark (`M1_v2`, `M2b` temporal spike, `M3_v2`, `M4_v2`, REM-style variants when available).",
            "- `family3_adversarial_evasion.csv`: Shariah-targeted local evasion benchmark on the configured raw variables "
            f"({', '.join(f'`{c}`' for c in settings.robustness_benchmark.family3_modifiable_columns)}).",
            "- `family4_anoshift.csv`: temporal transfer benchmark using a fixed global calibration on the honest reference sample, then near/far evaluation over time.",
            "- `coverage_key_variables.csv`: current panel coverage of raw variables needed by the realistic manipulations.",
            "- `benchmark_scoreboard.csv`: compact mean metrics by family and method.",
            "",
            "The benchmark keeps `phase5_injection` as a faster z-score-level baseline.",
            "",
            "Interpretation notes:",
            "- `M2b` is a temporal spike / gap mechanism, not a literal ABN_CFO implementation.",
            "- `ABN_PROD` and `ABN_DISX` probe REM-style manipulations that may bypass the Shariah-ratio layer and therefore stress non-SAC detector dimensions.",
            "- `Family 3` does not test full general evasion over all accounting variables; it tests whether RED cases can be weakened through Shariah-targeted local raw-variable adjustments.",
            "- `Family 4` is currently reported as a fixed-calibration temporal drift stress test rather than the stricter IID-only transfer protocol from the tex.",
            "",
            "## Key Variable Coverage",
            "",
            _markdown_table(coverage_df),
            "",
            "## Scoreboard",
            "",
            _markdown_table(scoreboard_df),
        ]
        paths["md"].write_text("\n".join(md_lines), encoding="utf-8")
        paths.update(
            _write_paper_outputs(
                out_dir=out_dir,
                panel=base_ctx.panel,
                zscores=base_ctx.zscores,
                composites=base_ctx.composites,
                family1=family1,
                family2=family2,
                family3=family3,
                family4=family4,
                settings=settings,
            )
        )
        log.info("robustness_benchmark: wrote %d files to %s", len(paths), out_dir)

    return RobustnessBenchmarkOutcome(
        family1=family1,
        family2=family2,
        family3=family3,
        family4=family4,
        summary=summary,
        paths=paths,
    )


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the cross-family robustness benchmark standalone."
    )
    parser.add_argument(
        "--country",
        default=AnalysisSettings().output_layout.country_code_lower,
        help="Lower-case country code subfolder.",
    )
    parser.add_argument(
        "--strict-compliance",
        action="store_true",
        default=False,
        help="Use the strict-compliance reference sample.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Recompute all enabled benchmark families instead of loading completed family CSV checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    defaults = AnalysisSettings()

    ref_update = {}
    if args.strict_compliance:
        ref_update["require_ratio_compliance"] = True

    top_update = {
        "output_layout": defaults.output_layout.model_copy(
            update={"country_code_lower": args.country}
        ),
    }
    if ref_update:
        top_update["reference_sample"] = defaults.reference_sample.model_copy(
            update=ref_update
        )

    settings = defaults.model_copy(update=top_update)

    # Load panel
    panel_path = settings.output_layout.phase0_dir() / "panel_with_split.parquet"
    if not panel_path.exists():
        base = args.country.split("_ex_")[0].split("_compliant_only")[0]
        panel_path = settings.output_layout.root / Path(
            str(settings.output_layout.panel_relative).format(country=base)
        )
    log.info("Loading panel from %s", panel_path)
    panel = pd.read_parquet(panel_path)

    run_robustness_benchmark(panel=panel, settings=settings, resume=not args.no_resume)


if __name__ == "__main__":
    main()
