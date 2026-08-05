"""Canonical public package for Family 3 counterfactual robustness."""

from __future__ import annotations

from importlib import import_module

from src.analysis.counterfactual.utils import (
    COMPOSITE_P_COLUMNS,
    COMPOSITE_VALUE_COLUMNS,
    Family3Snapshot,
    ScoreContext,
)

METHOD_NAME = "counterfactual_xai"
DEFAULT_OPTIMIZER = "torch_adam"
PUBLIC_MODES: tuple[str, ...] = ("global_per_row", "global_cohort", "local_firm_single")


def run_family3_counterfactual(*args, **kwargs):
    from src.analysis.counterfactual.core import run_family3_xai
    return run_family3_xai(*args, **kwargs)


__all__ = [
    "COMPOSITE_P_COLUMNS",
    "COMPOSITE_VALUE_COLUMNS",
    "DEFAULT_OPTIMIZER",
    "Family3FastRowEvaluator",
    "Family3Snapshot",
    "METHOD_NAME",
    "PUBLIC_MODES",
    "ScoreContext",
    "append_epoch_rows_csv",
    "cohort_loss_and_snapshot",
    "epoch_rows",
    "family3_candidate_panel",
    "family3_case_cards",
    "family3_case_label",
    "family3_column_scale_floor",
    "family3_composites_from_zscores",
    "family3_current_ratios",
    "family3_detector_dependencies",
    "family3_detailed_table",
    "family3_exact_loss",
    "family3_firm_score_z_value",
    "family3_firm_target_pvalue",
    "family3_flag_pvalue",
    "family3_flag_pvalues",
    "family3_history_table",
    "family3_impacted_detectors",
    "family3_majority_vote_table",
    "family3_merge_zscores",
    "family3_normalized_delta",
    "family3_primary_pvalue",
    "family3_ratio_scale_vector",
    "family3_raw_score_value",
    "family3_raw_zscore_settings",
    "family3_reference_ratio_stats",
    "family3_score_z_value",
    "family3_snapshot",
    "family3_source_mechanism_table",
    "family3_status_from_p",
    "family3_summary_table",
    "family3_target_score_kind",
    "family3_target_status_from_z",
    "family3_working_panel",
    "family3_z_equivalent_from_p",
    "finite_or_default",
    "global_cohort",
    "global_per_row",
    "local_firm_single",
    "row_loss_and_snapshot",
    "run_family3_counterfactual",
    "run_family3_xai",
    "sample_orange_index",
    "store_epoch_rows",
    "target_candidate_index",
    "write_family3_paper_outputs",
]

_EXPORT_TO_MODULE = {
    "append_epoch_rows_csv": "src.analysis.counterfactual.core",
    "cohort_loss_and_snapshot": "src.analysis.counterfactual.core",
    "epoch_rows": "src.analysis.counterfactual.core",
    "global_cohort": "src.analysis.counterfactual.core",
    "global_per_row": "src.analysis.counterfactual.core",
    "local_firm_single": "src.analysis.counterfactual.core",
    "row_loss_and_snapshot": "src.analysis.counterfactual.core",
    "run_family3_xai": "src.analysis.counterfactual.core",
    "sample_orange_index": "src.analysis.counterfactual.core",
    "store_epoch_rows": "src.analysis.counterfactual.core",
    "target_candidate_index": "src.analysis.counterfactual.core",
    "family3_exact_loss": "src.analysis.counterfactual.core",
    "family3_firm_score_z_value": "src.analysis.counterfactual.core",
    "family3_firm_target_pvalue": "src.analysis.counterfactual.core",
    "family3_flag_pvalue": "src.analysis.counterfactual.core",
    "family3_flag_pvalues": "src.analysis.counterfactual.core",
    "family3_normalized_delta": "src.analysis.counterfactual.core",
    "family3_primary_pvalue": "src.analysis.counterfactual.core",
    "family3_raw_score_value": "src.analysis.counterfactual.core",
    "family3_score_z_value": "src.analysis.counterfactual.core",
    "family3_status_from_p": "src.analysis.counterfactual.core",
    "family3_target_score_kind": "src.analysis.counterfactual.core",
    "family3_target_status_from_z": "src.analysis.counterfactual.core",
    "family3_z_equivalent_from_p": "src.analysis.counterfactual.core",
    "family3_candidate_panel": "src.analysis.counterfactual.core",
    "family3_composites_from_zscores": "src.analysis.counterfactual.core",
    "family3_detector_dependencies": "src.analysis.counterfactual.core",
    "family3_impacted_detectors": "src.analysis.counterfactual.core",
    "family3_merge_zscores": "src.analysis.counterfactual.core",
    "family3_raw_zscore_settings": "src.analysis.counterfactual.core",
    "family3_snapshot": "src.analysis.counterfactual.core",
    "family3_working_panel": "src.analysis.counterfactual.core",
    "Family3FastRowEvaluator": "src.analysis.counterfactual.utils",
    "family3_current_ratios": "src.analysis.counterfactual.utils",
    "family3_ratio_scale_vector": "src.analysis.counterfactual.utils",
    "family3_reference_ratio_stats": "src.analysis.counterfactual.utils",
    "finite_or_default": "src.analysis.counterfactual.utils",
    "family3_case_cards": "src.analysis.counterfactual.utils",
    "family3_case_label": "src.analysis.counterfactual.utils",
    "family3_column_scale_floor": "src.analysis.counterfactual.utils",
    "family3_detailed_table": "src.analysis.counterfactual.utils",
    "family3_history_table": "src.analysis.counterfactual.utils",
    "family3_majority_vote_table": "src.analysis.counterfactual.utils",
    "family3_source_mechanism_table": "src.analysis.counterfactual.utils",
    "family3_summary_table": "src.analysis.counterfactual.utils",
    "write_family3_paper_outputs": "src.analysis.counterfactual.utils",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
