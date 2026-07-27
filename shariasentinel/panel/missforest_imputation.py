"""MissForest-style imputation via sklearn IterativeImputer.

Uses ``IterativeImputer`` with ``RandomForestRegressor`` as the
per-variable estimator — functionally equivalent to the MissForest
algorithm (Stekhoven & Buhlmann, 2012) without a third-party dependency.

Design rules:

- Only imputes the numeric financial columns listed in the call. Firm
  identifiers, dates, flags, and categorical columns are untouched.
- Every imputed cell receives ``<col>_source = "missforest"`` so
  downstream detectors can filter or sensitivity-test.
- The imputer is fitted on the entire panel (or the subset specified by
  ``fit_mask``) and transforms all rows — this is the standard
  MissForest protocol where the model iterates over all variables.
- Preprocessing: signed-log transform + winsorization to tame extreme
  outliers before the RF fits, then inverse-transform after.

Caveats (documented in ``docs/preprocessing/imputation_comparison.md``):

- MissForest reconstructs the joint distribution well (KS ~0.11 on raw
  variables) but can wash out anomaly signals by smoothing digits (z1),
  enforcing cross-statement consistency (z5), and dampening temporal
  variance (z6). Use the ``HYBRID`` strategy (annual disagg first, then
  MissForest for residual NaN) to minimise this risk.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

log = logging.getLogger(__name__)

DEFAULT_MAX_ITER: int = 10
DEFAULT_N_ESTIMATORS: int = 100
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_WINSORIZE_PERCENTILE: float = 0.01


def _signed_log(x: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)`` — stabilises extreme monetary values."""
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_exp(x: np.ndarray) -> np.ndarray:
    """Inverse of ``_signed_log``."""
    return np.sign(x) * np.expm1(np.abs(x))


def _winsorize(arr: np.ndarray, pct: float) -> np.ndarray:
    """Clip to ``[pct, 1 - pct]`` quantiles per column."""
    out = arr.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size < 10:
            continue
        lo, hi = np.nanquantile(finite, [pct, 1.0 - pct])
        out[:, j] = np.clip(col, lo, hi)
    return out


def run_missforest_imputation(
    panel: pd.DataFrame,
    impute_columns: list[str] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    winsorize_pct: float = DEFAULT_WINSORIZE_PERCENTILE,
) -> pd.DataFrame:
    """Apply MissForest-style iterative RF imputation to the panel.

    Args:
        panel: Quarterly panel — modified in place (a copy is made
            internally for the imputation block).
        impute_columns: Numeric columns to impute. Defaults to the
            standard financial column set.
        max_iter: Maximum IterativeImputer rounds.
        n_estimators: Trees per RandomForestRegressor.
        random_state: Seed for reproducibility.
        winsorize_pct: Percentile for pre-imputation winsorization.

    Returns:
        The panel with NaN cells filled and ``<col>_source`` tags
        updated to ``"missforest"`` for every imputed cell.
    """
    if impute_columns is None:
        impute_columns = [
            "atq", "dlttq", "dlcq", "ltq", "cheq", "chq", "chsq",
            "revtq", "iditq", "nopiq", "xintq", "oibdpq", "niq",
            "actq", "lctq", "rectq", "invtq", "xsgaq", "cogsq", "xrdq",
        ]

    present = [c for c in impute_columns if c in panel.columns]
    if not present:
        log.warning("missforest: no imputable columns found; returning panel unchanged.")
        return panel

    panel = panel.copy()

    # Record which cells are NaN before imputation for source-tagging.
    was_nan: dict[str, pd.Series] = {}
    for col in present:
        was_nan[col] = panel[col].isna()

    # Extract numeric block, apply signed-log + winsorization.
    block = panel[present].to_numpy(dtype=float)
    n_missing_before = np.isnan(block).sum()
    if n_missing_before == 0:
        log.info("missforest: no missing values; nothing to impute.")
        return panel

    block_log = _signed_log(block)
    block_log = _winsorize(block_log, winsorize_pct)

    log.info(
        "missforest: imputing %d columns × %d rows (%d NaN cells), "
        "max_iter=%d, n_estimators=%d",
        len(present), len(panel), int(n_missing_before),
        max_iter, n_estimators,
    )

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        ),
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    imputed_log = imputer.fit_transform(block_log)
    imputed = _signed_exp(imputed_log)

    n_missing_after = np.isnan(imputed).sum()
    log.info(
        "missforest: done. NaN before=%d, after=%d, filled=%d",
        int(n_missing_before), int(n_missing_after),
        int(n_missing_before - n_missing_after),
    )

    # Write imputed values back and tag sources.
    for i, col in enumerate(present):
        panel[col] = imputed[:, i]

        source_col = f"{col}_source"
        if source_col not in panel.columns:
            panel[source_col] = "quarterly_observed"
        # Only tag cells that WERE NaN and are now filled.
        newly_filled = was_nan[col] & panel[col].notna()
        panel.loc[newly_filled, source_col] = "missforest"

    return panel
