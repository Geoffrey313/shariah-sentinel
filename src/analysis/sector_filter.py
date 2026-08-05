"""Sector exclusion filter.

Applied once at the start of the scoring pipeline before Phase 0.
Every downstream phase (C, detectors, composites, bootstrap, FDR)
automatically adjusts because they all consume the filtered panel.

Usage::

    from src.analysis.sector_filter import apply_sector_filter
    panel = apply_sector_filter(panel, settings)
"""
from __future__ import annotations

import logging

import pandas as pd

from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)


def apply_sector_filter(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Exclude sectors listed in ``settings.exclude_sectors``.

    Args:
        panel: Full panel dataframe with a sector column.
        settings: Configuration — reads ``exclude_sectors`` and
            ``panel_schema.sector``.

    Returns:
        Filtered panel (copy). If ``exclude_sectors`` is empty,
        returns the original panel unchanged (no copy).
    """
    if not settings.exclude_sectors:
        return panel

    sector_col = settings.panel_schema.sector
    if sector_col not in panel.columns:
        log.warning(
            "sector_filter: column %r not found; skipping exclusion.",
            sector_col,
        )
        return panel

    before = len(panel)
    before_firms = panel[settings.panel_schema.firm_id].nunique()

    mask = ~panel[sector_col].isin(settings.exclude_sectors)
    filtered = panel.loc[mask].copy().reset_index(drop=True)

    after = len(filtered)
    after_firms = filtered[settings.panel_schema.firm_id].nunique()

    log.info(
        "sector_filter: excluded %s → %d rows removed (%d → %d), "
        "%d firms removed (%d → %d).",
        list(settings.exclude_sectors),
        before - after, before, after,
        before_firms - after_firms, before_firms, after_firms,
    )
    return filtered


def exclusion_suffix(settings: AnalysisSettings) -> str:
    """Return a directory suffix for excluded-sector runs.

    Empty string if no exclusion (default output path). Otherwise
    ``_ex_<sector1>_<sector2>`` with spaces replaced by underscores
    and lowercased.

    Used by the orchestrator to route outputs to a distinct
    directory so full-panel and excluded-sector results coexist.
    """
    if not settings.exclude_sectors:
        return ""
    names = "_".join(
        s.lower().replace(" ", "_") for s in sorted(settings.exclude_sectors)
    )
    return f"_ex_{names}"


def compliant_suffix(settings: AnalysisSettings) -> str:
    """Return ``_compliant_only`` when the flag is active, else ``""``."""
    if not settings.compliant_only:
        return ""
    return "_compliant_only"
