"""Backward-compatible shim for the canonical reduced-WP ``z5`` detector."""

from __future__ import annotations

import pandas as pd

from .d5_coherence import (
    compute_cross_statement_relations,
    describe_cross_statement_relation_status,
    detect_cross_statement_coherence,
)


def detect_joint_zscore(df: pd.DataFrame) -> pd.Series:
    """Backward-compatible alias for the canonical reduced-WP ``z5`` detector."""
    return detect_cross_statement_coherence(df)


__all__ = [
    "compute_cross_statement_relations",
    "describe_cross_statement_relation_status",
    "detect_cross_statement_coherence",
    "detect_joint_zscore",
]
