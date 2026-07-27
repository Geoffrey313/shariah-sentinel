"""Contamination framework — controlled H₀/H₁ benchmark.

Validates the full detection pipeline end-to-end by injecting
known manipulations into raw financial variables and measuring
AUC against ground-truth labels. Companion to the z-score-level
injection in Phase 5.
"""

from .config import ContaminationSettings
from .run_contamination import run_contamination_study
from .run_frontier import run_detection_frontier
from .run_firm_level import run_firm_level_study

__all__ = [
    "ContaminationSettings",
    "run_contamination_study",
    "run_detection_frontier",
    "run_firm_level_study",
]
