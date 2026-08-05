"""Cross-family robustness benchmark exports with lazy loading."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "RobustnessBenchmarkOutcome",
    "run_robustness_benchmark",
    "apply_contamination",
    "apply_contamination_bundle",
]

_EXPORT_MAP = {
    "RobustnessBenchmarkOutcome": ("src.analysis.benchmark", "RobustnessBenchmarkOutcome"),
    "run_robustness_benchmark": ("src.analysis.benchmark", "run_robustness_benchmark"),
    "apply_contamination": ("src.analysis.contamination", "apply_contamination"),
    "apply_contamination_bundle": ("src.analysis.contamination", "apply_contamination_bundle"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
