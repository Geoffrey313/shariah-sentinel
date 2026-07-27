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
    "detect_benford": ("shariasentinel.detectors.d1_benford", "detect_benford"),
    "detect_zipf": ("shariasentinel.detectors.d2_zipf", "detect_zipf"),
    "detect_mscore": ("shariasentinel.detectors.d3_mscore", "detect_mscore"),
    "detect_threshold_proximity": ("shariasentinel.detectors.d4_proximity", "detect_threshold_proximity"),
    "compute_cross_statement_relations": ("shariasentinel.detectors.d5_coherence", "compute_cross_statement_relations"),
    "describe_cross_statement_relation_status": ("shariasentinel.detectors.d5_coherence", "describe_cross_statement_relation_status"),
    "detect_cross_statement_coherence": ("shariasentinel.detectors.d5_coherence", "detect_cross_statement_coherence"),
    "detect_joint_zscore": ("shariasentinel.detectors.d5_coherence", "detect_joint_zscore"),
    "detect_temporal": ("shariasentinel.detectors.d6_temporal", "detect_temporal"),
    "detect_peer": ("shariasentinel.detectors.d7_peer", "detect_peer"),
    "compute_composite": ("shariasentinel.detectors.composite", "compute_composite"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
