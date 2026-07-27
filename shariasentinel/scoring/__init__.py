"""Scoring analysis — phases 0..7 with lazy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AnalysisSettings",
    "apply_sector_filter",
    "compute_zscores",
    "load_zscores",
    "run_full_analysis",
    "run_phase0",
    "run_phase1",
    "run_phase2",
    "run_phase3",
    "run_phase4",
    "run_phase4b",
    "run_phase5",
    "run_phase6",
    "run_phase7",
    "run_robustness_benchmark",
]

_EXPORT_MAP = {
    "AnalysisSettings": ("shariasentinel.scoring.config", "AnalysisSettings"),
    "apply_sector_filter": ("shariasentinel.scoring.sector_filter", "apply_sector_filter"),
    "compute_zscores": ("shariasentinel.scoring.compute_zscores", "compute_zscores"),
    "load_zscores": ("shariasentinel.scoring.compute_zscores", "load_zscores"),
    "run_full_analysis": ("shariasentinel.scoring.run_analysis", "run_full_analysis"),
    "run_phase0": ("shariasentinel.scoring.phase0_reference_sample", "run_phase0"),
    "run_phase1": ("shariasentinel.scoring.phase1_calibration", "run_phase1"),
    "run_phase2": ("shariasentinel.scoring.phase2_dependence", "run_phase2"),
    "run_phase3": ("shariasentinel.scoring.phase3_detector_qc", "run_phase3"),
    "run_phase4": ("shariasentinel.scoring.phase4_composites", "run_phase4"),
    "run_phase4b": ("shariasentinel.scoring.phase4b_confidence", "run_phase4b"),
    "run_phase5": ("shariasentinel.scoring.phase5_injection", "run_phase5"),
    "run_phase6": ("shariasentinel.scoring.phase6_robustness", "run_phase6"),
    "run_phase7": ("shariasentinel.scoring.phase7_fdr", "run_phase7"),
    "run_robustness_benchmark": ("shariasentinel.robustness", "run_robustness_benchmark"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
