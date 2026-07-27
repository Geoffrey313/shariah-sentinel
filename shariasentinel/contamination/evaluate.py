"""Evaluation metrics for the contamination benchmark.

Computes AUC, precision, recall against ground-truth labels.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

log = logging.getLogger(__name__)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic.

    Args:
        scores: Higher = more anomalous.
        labels: 1 = contaminated, 0 = clean.

    Returns:
        AUC in [0, 1]. Returns NaN if either class is empty.
    """
    finite = np.isfinite(scores)
    s, y = scores[finite], labels[finite]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)
    u1 = ranks[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n0))


def compute_precision_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    null_sorted: np.ndarray,
) -> dict:
    """Precision and recall at significance level α.

    Uses the bootstrap null to determine the critical value c_α,
    then flags observations with score > c_α.
    """
    if null_sorted.size == 0:
        return {"alpha": alpha, "precision": float("nan"),
                "recall": float("nan"), "n_flagged": 0}

    c_alpha = np.percentile(null_sorted, 100 * (1 - alpha))
    finite = np.isfinite(scores)
    s, y = scores[finite], labels[finite]

    flagged = s > c_alpha
    n_flagged = int(flagged.sum())
    tp = int((flagged & (y == 1)).sum())
    n_contam = int((y == 1).sum())

    precision = tp / n_flagged if n_flagged > 0 else float("nan")
    recall = tp / n_contam if n_contam > 0 else float("nan")

    return {
        "alpha": alpha,
        "precision": float(precision),
        "recall": float(recall),
        "n_flagged": n_flagged,
        "tp": tp,
    }


def compute_effective_contamination_rate(
    z_clean: np.ndarray,
    z_contam: np.ndarray,
    epsilon: float = 0.5,
) -> float:
    """Fraction of contaminated rows with a detectable z-score shift.

    A contaminated row is "effectively contaminated" if the L2 norm
    of its z-score change exceeds ε.
    """
    if z_clean.shape[0] == 0:
        return 0.0
    diff = np.linalg.norm(z_contam - z_clean, axis=1)
    return float((diff > epsilon).mean())


def evaluate_cell(
    composites: pd.DataFrame,
    labels: np.ndarray,
    null_distributions: dict[str, np.ndarray],
    alpha_grid: tuple[float, ...],
) -> dict:
    """Evaluate one (mechanism, ρ) cell across all composites.

    Returns a dict with AUC per composite and precision/recall at
    each α.
    """
    composite_cols = [
        "z_plus", "z_plus_renorm", "z_mahalanobis_sq", "t_iut",
    ]
    results: dict = {}

    for comp in composite_cols:
        if comp not in composites.columns:
            continue
        scores = composites[comp].to_numpy(dtype=float)
        # Higher score = more anomalous for all composites
        # (t_iut: higher min = more unanimously anomalous)
        auc = compute_auc(scores, labels)
        results[f"auc_{comp}"] = auc

        null_key = comp
        if null_key in null_distributions:
            for alpha in alpha_grid:
                pr = compute_precision_recall(
                    scores, labels, alpha, null_distributions[null_key],
                )
                results[f"precision_{comp}_a{alpha}"] = pr["precision"]
                results[f"recall_{comp}_a{alpha}"] = pr["recall"]

    return results
