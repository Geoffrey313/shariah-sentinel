"""H₀ generator — clean reference sample.

Two modes:

- **bootstrap** (original): Resample individual rows from C with
  replacement. Fast, but destroys temporal and firm structure.
  Suitable for calibrating the null distribution.

- **panel** (new default): Use the actual C rows from the panel,
  preserving firm identity, temporal ordering, and cross-quarter
  coherence. Required for realistic contamination evaluation
  because detectors like z6 (temporal) and z57 (peer) rely on
  temporal and firm structure.
"""
from __future__ import annotations

import logging
from math import floor

import numpy as np
import pandas as pd

from .config import ContaminationSettings

log = logging.getLogger(__name__)


def generate_h0(
    panel: pd.DataFrame,
    settings: ContaminationSettings,
    mode: str = "panel",
) -> pd.DataFrame:
    """Generate a clean benchmark sample from C.

    Args:
        panel: Panel with ``_split`` column (output of Phase 0).
        settings: Contamination config.
        mode: ``"panel"`` preserves firm/temporal structure (default);
              ``"bootstrap"`` resamples individual rows (legacy).

    Returns:
        DataFrame with a ``y`` column set to 0 (clean).
    """
    if "_split" not in panel.columns:
        raise KeyError("h0_generator: panel lacks _split — run Phase 0 first.")

    c_rows = panel[panel["_split"] == "C"]

    if mode == "bootstrap":
        return _generate_h0_bootstrap(c_rows, settings)
    elif mode == "panel":
        return _generate_h0_panel(c_rows, settings)
    else:
        raise ValueError(f"h0_generator: unknown mode {mode!r}")


def _generate_h0_panel(
    c_rows: pd.DataFrame,
    settings: ContaminationSettings,
) -> pd.DataFrame:
    """Use actual C rows, preserving firm and temporal structure.

    If h0_sample_size < |C|, sample whole firms (not individual rows)
    to maintain within-firm time series integrity.
    """
    n_available = len(c_rows)
    n_target = settings.h0_sample_size

    if n_target >= n_available:
        h0 = c_rows.copy()
    else:
        # Sample whole firms to hit the target size
        firm_col = "gvkey"
        if firm_col not in c_rows.columns:
            # Fallback: random row sample (no firm structure available)
            rng = np.random.default_rng(settings.random_seed)
            idx = rng.choice(c_rows.index, size=n_target, replace=False)
            h0 = c_rows.loc[idx].copy()
        else:
            rng = np.random.default_rng(settings.random_seed)
            firms = c_rows[firm_col].unique()
            rng.shuffle(firms)
            # Accumulate firms until we reach the target size
            selected_firms = []
            count = 0
            for firm in firms:
                n_firm = (c_rows[firm_col] == firm).sum()
                selected_firms.append(firm)
                count += n_firm
                if count >= n_target:
                    break
            h0 = c_rows[c_rows[firm_col].isin(selected_firms)].copy()

    h0["y"] = 0
    log.info(
        "h0_generator (panel mode): %d rows from %d firms "
        "(%d available in C).",
        len(h0),
        h0["gvkey"].nunique() if "gvkey" in h0.columns else -1,
        len(c_rows),
    )
    return h0


def _generate_h0_bootstrap(
    c_rows: pd.DataFrame,
    settings: ContaminationSettings,
) -> pd.DataFrame:
    """Legacy: resample individual rows with replacement from C.

    WARNING: Destroys temporal ordering and firm structure. Detectors
    that rely on within-firm time series (z6) or peer context (z57)
    will not see realistic signals.
    """
    rng = np.random.default_rng(settings.random_seed)
    n = min(settings.h0_sample_size, len(c_rows))
    idx = rng.choice(c_rows.index, size=n, replace=True)

    h0 = c_rows.loc[idx].copy().reset_index(drop=True)
    h0["_original_idx"] = idx
    h0["y"] = 0

    log.info(
        "h0_generator (bootstrap mode): sampled %d rows from C "
        "(%d available).",
        n, len(c_rows),
    )
    return h0
