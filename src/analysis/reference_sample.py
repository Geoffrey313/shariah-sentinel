"""Phase 0 — reference sample curation.

Implements the Sprint 1 prerequisite of ``docs/scores/analysis_plan_v2.md`` and
the reference populations defined in Rémi's math framework
(``docs/math-framework/framework.md``).

The framework requires three populations:

- ``C`` — global presumed-honest sample. Used for the empirical PIT of z₃
  (framework §9.2 D3) and for the non-parametric bootstrap of Phase 4
  composites (§3.3, §6.3).
- ``C_s`` — sector-stratified slices of ``C``. Used for z₅ cross-statement
  residual standardisation (§9.2 D5) and z₇ peer moments (§9.2 D7).
- Peer groups — subsets of ``C`` sharing a firm's peer key. Used for
  Hotelling's T² on each row (§9.2 D7).

The module is deliberately data-only: it emits labels and diagnostic
artefacts, but does not mutate the panel parquet. Downstream phases read the
diagnostic JSON / parquet and apply the labels on their own copy of the panel.

Deliverables (under ``outputs/scores/<country>/phase0_reference_sample/``):

- ``reference_sample.json`` — summary statistics and the per-sector
  / per-peer-group coverage tables the plan calls for.
- ``panel_with_split.parquet`` — the input panel plus two added columns
  ``_split`` (``"C"`` or ``"NOT_C"``) and ``_split_reason`` (why a row was
  excluded from ``C``). The detector modules already consume ``_split``.
- ``C_sector_sizes.csv`` — rows per sector of ``C``; the plan calls it this.
- ``peer_group_sizes.csv`` — size per peer group (z₇).
- ``C_definition.md`` — human-readable summary of the inclusion rule and
  contamination notes; written once per run so a reviewer can audit the
  sample without re-reading code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.methodology import screening_thresholds_for_panel

from src.common.config import AnalysisSettings

log = logging.getLogger(__name__)


# Exclusion reasons reported in ``_split_reason``. Values kept stable so that
# downstream phases can filter on them without guessing wording. The
# authority-label reason embeds the resolved column name (``sac_shariah``
# on legacy MYS panels, ``compliance_shariah`` on generic panels) — see
# :func:`resolve_label_column`.
REASON_INCLUDED: str = "included"
REASON_NOT_SAC_COMPLIANT: str = "sac_shariah != 1"
REASON_NOT_CLEAN_SAMPLE: str = "clean_sample != 1"
REASON_MISSING_REQUIRED_COLUMN: str = "missing_required_column"
REASON_RATIO_VIOLATION: str = "ratio exceeds active authority threshold"

SPLIT_LABEL_INCLUDED: str = "C"
SPLIT_LABEL_EXCLUDED: str = "NOT_C"


def resolve_label_column(panel: pd.DataFrame, settings: AnalysisSettings) -> str:
    """Column carrying the authority compliance label for this panel.

    Prefers the generic tri-state ``compliance_shariah`` (stamped by
    country-aware panel builds); falls back to the legacy MYS
    ``sac_shariah`` so every historical artefact keeps working. The
    reference sample C always selects strictly ``== 1`` on the resolved
    column — with a positive-only authority (DFM) the label is 1/NA and
    ``fillna(0)`` keeps unknown firms out of C without ever branding
    them non-compliant.
    """
    schema = settings.panel_schema
    if schema.compliance_shariah in panel.columns:
        return schema.compliance_shariah
    return schema.sac_shariah


@dataclass(frozen=True)
class ReferenceSampleOutcome:
    """Return value of :func:`run_phase0`.

    Attributes:
        panel: Input panel augmented with ``_split`` / ``_split_reason``.
        summary: Dictionary serialised to ``reference_sample.json``.
        sector_sizes: Size of ``C_s`` per sector.
        peer_group_sizes: Size of each peer group within ``C``.
        paths: Resolved output paths, keyed by deliverable name.
    """

    panel: pd.DataFrame
    summary: dict
    sector_sizes: pd.DataFrame
    peer_group_sizes: pd.DataFrame
    paths: dict[str, Path]


def _require_columns(panel: pd.DataFrame, required: list[str]) -> list[str]:
    """Return the subset of ``required`` columns missing from ``panel``.

    Missing required columns abort Phase 0 — downstream phases cannot make
    inference calls without the sac label or the firm/quarter keys.
    """
    return [col for col in required if col not in panel.columns]


def _build_inclusion_mask(
    panel: pd.DataFrame,
    settings: AnalysisSettings,
) -> tuple[pd.Series, pd.Series]:
    """Compute the inclusion mask for ``C`` and the exclusion reason per row.

    The mask is the logical AND of the configured inclusion predicates; the
    reason string records the first predicate that failed so the diagnostic
    CSV explains every exclusion. Rows that satisfy all predicates receive
    :data:`REASON_INCLUDED`.
    """
    schema = settings.panel_schema
    rules = settings.reference_sample
    label_col = resolve_label_column(panel, settings)

    reasons = pd.Series(REASON_INCLUDED, index=panel.index, dtype="object")
    mask = pd.Series(True, index=panel.index, dtype=bool)

    if rules.require_sac_compliant:
        # Strict == 1: NA (unknown under a positive-only authority) is
        # excluded from C but the reason string names the column so the
        # diagnostic CSV stays honest about which label ruled.
        sac_ok = panel[label_col].fillna(0).astype(int) == 1
        reasons = reasons.mask(mask & ~sac_ok, f"{label_col} != 1")
        mask &= sac_ok

    if rules.require_clean_sample:
        if schema.clean_sample not in panel.columns:
            log.warning(
                "phase0: clean_sample column %r absent; treating rule as satisfied.",
                schema.clean_sample,
            )
        else:
            clean_ok = panel[schema.clean_sample].fillna(0).astype(int) == 1
            reasons = reasons.mask(mask & ~clean_ok, REASON_NOT_CLEAN_SAMPLE)
            mask &= clean_ok

    if rules.require_ratio_compliance:
        # Exclude rows where any recomputed authority ratio exceeds its
        # threshold. A row passes if ALL available ratios are within
        # bounds (NaN = pass). Thresholds resolve per the panel's
        # methodology (SAC 33/33/5, DFM 30/30/10) — never hardcoded here.
        ratio_ok = pd.Series(True, index=panel.index, dtype=bool)
        for col, cap in screening_thresholds_for_panel(panel).items():
            if col in panel.columns:
                exceeds = panel[col] > cap
                ratio_ok &= ~exceeds.fillna(False)
        n_excluded = (mask & ~ratio_ok).sum()
        reasons = reasons.mask(mask & ~ratio_ok, REASON_RATIO_VIOLATION)
        mask &= ratio_ok
        log.info(
            "phase0: --strict-compliance excluded %d rows "
            "with ratio violations from C.",
            n_excluded,
        )

    return mask, reasons


def _tabulate_sector_sizes(
    panel_c: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Return per-sector counts with the D5 / sample-level thresholds attached.

    ``meets_min_sector_size``  → precondition from Phase 0 rules.
    ``meets_d5_floor``         → precondition from framework §9.2 D5.
    """
    schema = settings.panel_schema
    rules = settings.reference_sample
    d5_floor = settings.detector_preconditions.cross_statement.min_sector_reference

    counts = (
        panel_c.groupby(schema.sector, dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    counts["meets_min_sector_size"] = counts["n_rows"] >= rules.min_sector_size
    counts["meets_d5_floor"] = counts["n_rows"] >= d5_floor
    return counts.sort_values("n_rows", ascending=False).reset_index(drop=True)


def _tabulate_peer_group_sizes(
    panel_c: pd.DataFrame,
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Per-peer-group row count with the z7 precondition attached."""
    rules = settings.reference_sample
    d7_floor = settings.detector_preconditions.peer.min_peer_size

    counts = (
        panel_c.groupby(rules.peer_group_column, dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    counts["meets_peer_group_min_size"] = counts["n_rows"] >= rules.peer_group_min_size
    counts["meets_d7_floor"] = counts["n_rows"] >= d7_floor
    return counts.sort_values("n_rows", ascending=False).reset_index(drop=True)


def _write_c_definition_markdown(
    path: Path,
    settings: AnalysisSettings,
    summary: dict,
) -> None:
    """Write ``C_definition.md`` alongside the JSON summary.

    A single file documents: the inclusion rule, size checks, contamination
    notes, and which detectors are usable per sector. Keeps audit friction
    low — no need to diff code to re-read the definition.
    """
    rules = settings.reference_sample
    lines: list[str] = []
    lines.append("# Reference Sample C — Definition")
    lines.append("")
    lines.append("Generated by ``scores.analysis.phase0_reference_sample``.")
    lines.append("")
    lines.append("## Inclusion rule")
    lines.append("")
    lines.append(f"- require_sac_compliant      = {rules.require_sac_compliant}")
    lines.append(f"- require_clean_sample       = {rules.require_clean_sample}")
    lines.append(f"- require_ratio_compliance   = {rules.require_ratio_compliance}")
    lines.append("")
    lines.append("A row is kept in ``C`` iff all active predicates are satisfied.")
    lines.append("")
    lines.append("## Size thresholds")
    lines.append("")
    lines.append(f"- min |C|      = {rules.min_global_size}  (framework §9.2 D3)")
    lines.append(f"- min |C_s|    = {rules.min_sector_size}  (framework §9.2 D5)")
    lines.append(f"- min N_peer   = {rules.peer_group_min_size}  (framework §9.2 D7)")
    lines.append("")
    lines.append("## Observed sizes")
    lines.append("")
    lines.append(f"- |C|           = {summary['global']['size']}")
    lines.append(f"- sectors in C  = {summary['sector']['n_sectors']} "
                 f"({summary['sector']['n_meets_min_sector_size']} meet the min size)")
    lines.append(f"- peer groups   = {summary['peer_groups']['n_groups']} "
                 f"({summary['peer_groups']['n_meets_peer_group_min_size']} meet the min size)")
    lines.append("")
    lines.append("## Contamination notes")
    lines.append("")
    lines.append(
        "The current panel does not expose a delisting flag, so the "
        "`not delisted` clause of analysis_plan_v2.md is replaced by "
        "`clean_sample == 1` (flag written by `add_clean_sample_flag` in "
        "`panel_creation/build_panel.py`). Known restated firms are not "
        "excluded — if an audit-report list becomes available, extend "
        "`ReferenceSampleSettings` with a `restated_firms` deny-list rather "
        "than filtering inline."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase0(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    write_outputs: bool = True,
) -> ReferenceSampleOutcome:
    """Curate ``C``, ``C_s`` and peer groups on ``panel``.

    The panel is **not** mutated in place; a copy is returned with two added
    columns (``_split`` and ``_split_reason``). This keeps the function pure
    and lets callers diff the returned frame against ``panel`` trivially.

    Args:
        panel: Quarterly panel produced by ``panel_creation.main`` — must
            contain the columns named in ``settings.panel_schema``.
        settings: Configuration. Defaults to :class:`AnalysisSettings` with
            the plan-of-record values.
        write_outputs: If ``True``, persist the JSON / parquet / CSV / MD
            deliverables under ``settings.output_layout.phase0_dir()``.

    Returns:
        A :class:`ReferenceSampleOutcome` with the split-labelled panel,
        summary dictionary, per-sector table, per-peer-group table and the
        resolved paths.

    Raises:
        KeyError: If required schema columns are missing from ``panel``.
        ValueError: If ``|C| < min_global_size`` (framework §9.2 D3 floor).
    """
    settings = settings or AnalysisSettings()
    schema = settings.panel_schema
    label_col = resolve_label_column(panel, settings)

    required = [
        schema.firm_id, schema.quarter, label_col, schema.sector,
    ]
    missing = _require_columns(panel, required)
    if missing:
        raise KeyError(
            f"phase0: panel missing required columns {missing}; "
            f"update `PanelSchemaSettings` or rebuild the panel."
        )

    # Tri-state visibility: log the raw 1 / 0 / NA counts up front so a
    # positive-only authority list (DFM — no 0s, many NAs) is impossible
    # to mistake for "everything non-compliant".
    label_num = pd.to_numeric(panel[label_col], errors="coerce")
    label_counts = {
        "compliant_1": int((label_num == 1).sum()),
        "noncompliant_0": int((label_num == 0).sum()),
        "unknown_na": int(label_num.isna().sum()),
    }
    log.info(
        "phase0: authority label %r — 1: %d | 0: %d | NA: %d",
        label_col, label_counts["compliant_1"],
        label_counts["noncompliant_0"], label_counts["unknown_na"],
    )

    mask, reasons = _build_inclusion_mask(panel, settings)

    out = panel.copy()
    out["_split"] = pd.Series(
        [SPLIT_LABEL_INCLUDED if keep else SPLIT_LABEL_EXCLUDED for keep in mask],
        index=out.index,
        dtype="object",
    )
    out["_split_reason"] = reasons

    panel_c = out.loc[mask]

    if len(panel_c) < settings.reference_sample.min_global_size:
        raise ValueError(
            f"phase0: |C| = {len(panel_c)} < min_global_size = "
            f"{settings.reference_sample.min_global_size}. Loosen the "
            f"inclusion rule or rebuild the panel; no detector can calibrate "
            f"below the §9.2 D3 floor."
        )

    sector_sizes = _tabulate_sector_sizes(panel_c, settings)
    peer_group_sizes = _tabulate_peer_group_sizes(panel_c, settings)

    n_sectors_ok = int(sector_sizes["meets_min_sector_size"].sum())
    n_peers_ok = int(peer_group_sizes["meets_peer_group_min_size"].sum())

    exclusion_breakdown = (
        reasons[~mask].value_counts(dropna=False).to_dict() if (~mask).any() else {}
    )

    summary: dict = {
        "inclusion_rule": {
            "require_sac_compliant": settings.reference_sample.require_sac_compliant,
            "require_clean_sample": settings.reference_sample.require_clean_sample,
            "require_ratio_compliance": settings.reference_sample.require_ratio_compliance,
        },
        "authority_label": {"column": label_col, **label_counts},
        "panel": {"n_rows": int(len(panel)), "n_firms": int(panel[schema.firm_id].nunique())},
        "global": {
            "size": int(len(panel_c)),
            "n_firms": int(panel_c[schema.firm_id].nunique()),
            "min_required": settings.reference_sample.min_global_size,
        },
        "sector": {
            "n_sectors": int(sector_sizes.shape[0]),
            "min_sector_size": settings.reference_sample.min_sector_size,
            "n_meets_min_sector_size": n_sectors_ok,
        },
        "peer_groups": {
            "peer_group_column": settings.reference_sample.peer_group_column,
            "n_groups": int(peer_group_sizes.shape[0]),
            "peer_group_min_size": settings.reference_sample.peer_group_min_size,
            "n_meets_peer_group_min_size": n_peers_ok,
        },
        "exclusion_breakdown": {str(k): int(v) for k, v in exclusion_breakdown.items()},
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase0_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["json"] = out_dir / "reference_sample.json"
        paths["panel_with_split"] = out_dir / "panel_with_split.parquet"
        paths["sector_sizes_csv"] = out_dir / "C_sector_sizes.csv"
        paths["peer_group_sizes_csv"] = out_dir / "peer_group_sizes.csv"
        paths["c_definition_md"] = out_dir / "C_definition.md"

        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        out.to_parquet(paths["panel_with_split"], index=False)
        sector_sizes.to_csv(paths["sector_sizes_csv"], index=False)
        peer_group_sizes.to_csv(paths["peer_group_sizes_csv"], index=False)
        _write_c_definition_markdown(paths["c_definition_md"], settings, summary)

        log.info("phase0: wrote %d deliverables to %s", len(paths), out_dir)

    return ReferenceSampleOutcome(
        panel=out,
        summary=summary,
        sector_sizes=sector_sizes,
        peer_group_sizes=peer_group_sizes,
        paths=paths,
    )
