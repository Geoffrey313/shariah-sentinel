"""Detector package exports with lazy loading."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "compute_composite",
    "compute_cross_statement_relations",
    "describe_cross_statement_relation_status",
    "detect_benford",
    "detect_cross_statement_coherence",
    "detect_joint_zscore",
    "detect_mscore",
    "detect_peer",
    "detect_temporal",
    "detect_threshold_proximity",
    "detect_zipf",
]

_EXPORT_MAP = {
    "detect_benford": ("src.engine.benford", "detect_benford"),
    "detect_zipf": ("src.engine.zipf", "detect_zipf"),
    "detect_mscore": ("src.engine.mscore", "detect_mscore"),
    "detect_threshold_proximity": ("src.engine.proximity", "detect_threshold_proximity"),
    "compute_cross_statement_relations": ("src.engine.coherence", "compute_cross_statement_relations"),
    "describe_cross_statement_relation_status": ("src.engine.coherence", "describe_cross_statement_relation_status"),
    "detect_cross_statement_coherence": ("src.engine.coherence", "detect_cross_statement_coherence"),
    "detect_joint_zscore": ("src.engine.coherence", "detect_joint_zscore"),
    "detect_temporal": ("src.engine.temporal", "detect_temporal"),
    "detect_peer": ("src.engine.peer", "detect_peer"),
    "compute_composite": ("src.engine.detector_composite", "compute_composite"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
