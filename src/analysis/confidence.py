"""Phase 4b — confidence qualifier κ (framework §A.3).

The qualifier turns the *shape* of the active-set z-score vector into a
categorical tag:

- ``targeted``   — one detector dominates the signal (low entropy).
- ``mixed``      — intermediate concentration.
- ``systematic`` — signal spread across many detectors (high entropy).
- ``single``     — only one active detector; entropy is trivially zero and
  the tag is uninformative — reported as a distinct label.
- ``inactive``   — no active detector; κ undefined.

Following ``analysis_plan_v2.md`` the working definition of ``S_F*`` is

    S_F*(z) = - Σ_{j ∈ A} p_j log p_j,  p_j = z_j / Σ_{j' ∈ A} z_{j'}

normalised by ``log|A|`` so ``S_F* ∈ [0, 1]``. Rémi has flagged the exact
formulation as TBD — it is isolated in :func:`_normalized_entropy_row` so
it can be swapped in one place once he commits.

Thresholds for the ``targeted``/``mixed``/``systematic`` split are taken
from quantiles of ``S_F*`` evaluated on the reference sample ``C`` — the
class boundaries therefore adapt to the actual detector battery.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.composites import active_set_indicator
from src.engine.zscores import load_zscores
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)


COL_S_F_STAR: str = "s_f_star"
COL_KAPPA: str = "kappa"


@dataclass(frozen=True)
class Phase4bOutcome:
    """Return value of :func:`run_phase4b`."""

    kappa_frame: pd.DataFrame
    summary: dict
    paths: dict[str, Path]


def _normalized_entropy_row(z_row: np.ndarray, threshold: float) -> float:
    """Normalised Shannon entropy of the active-set z-scores for one row.

    Returns ``NaN`` when fewer than two detectors are active (entropy is
    0 by convention for ``|A| = 1``, which callers handle with a separate
    ``'single'`` label). The normalisation by ``log|A|`` maps the result
    to ``[0, 1]``.
    """
    finite = np.isfinite(z_row)
    active_mask = finite & (z_row > threshold)
    active_vals = z_row[active_mask]
    n_active = int(active_vals.size)
    if n_active < 2:
        return np.nan
    total = active_vals.sum()
    if total <= 0.0:
        return np.nan
    probs = active_vals / total
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy / np.log(n_active))


def _normalized_entropy_matrix(
    z_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Vectorised wrapper applying :func:`_normalized_entropy_row` per row."""
    out = np.empty(z_matrix.shape[0], dtype=float)
    for i in range(z_matrix.shape[0]):
        out[i] = _normalized_entropy_row(z_matrix[i], threshold=threshold)
    return out


def _assign_kappa(
    s_f_star: np.ndarray,
    active_counts: np.ndarray,
    cutoffs: tuple[float, float],
    settings: AnalysisSettings,
) -> np.ndarray:
    """Discretise ``S_F*`` into ``κ`` labels using ``C`` quantile cutoffs."""
    conf = settings.confidence
    low, high = cutoffs
    labels = np.full(s_f_star.shape, conf.label_inactive, dtype=object)
    single = active_counts == 1
    labels[single] = conf.label_single
    multi = active_counts >= 2
    # Only rows with a defined S_F* (multi-active) get the quantile split.
    targeted = multi & (s_f_star <= low)
    systematic = multi & (s_f_star >= high)
    mixed = multi & ~targeted & ~systematic
    labels[targeted] = conf.label_targeted
    labels[systematic] = conf.label_systematic
    labels[mixed] = conf.label_mixed
    return labels


def run_phase4b(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    zscores: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> Phase4bOutcome:
    """Compute ``S_F*`` and κ per row.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0`.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        zscores: Optional pre-computed z-scores. When ``None`` the Phase 2
            cache is loaded from disk.
        write_outputs: If ``True``, persist the per-row parquet and JSON
            summary under ``settings.output_layout.phase4b_dir()``.

    Returns:
        A :class:`Phase4bOutcome` with the κ frame, summary dict and paths.

    Raises:
        KeyError: If ``_split`` is missing.
    """
    settings = settings or AnalysisSettings()
    if "_split" not in panel.columns:
        raise KeyError("phase4b: panel lacks `_split` — run Phase 0 first.")
    conf = settings.confidence

    zscores = zscores if zscores is not None else load_zscores(settings)
    if len(zscores) != len(panel):
        raise ValueError(
            f"phase4b: zscores length {len(zscores)} != panel length {len(panel)}."
        )
    zscores = zscores.copy()
    zscores.index = panel.index
    configured = tuple(settings.zscores.active_detectors())
    detectors_present = [d for d in configured if d in zscores.columns]
    z_matrix = zscores.loc[:, detectors_present].to_numpy(dtype=float)

    threshold = settings.composites.active_set_threshold
    active = active_set_indicator(z_matrix, threshold=threshold)
    active_counts = active.sum(axis=1).astype(int)

    s_f_star = _normalized_entropy_matrix(z_matrix, threshold=threshold)

    mask_c = (panel["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()
    s_f_star_c = s_f_star[mask_c]
    s_f_star_c = s_f_star_c[np.isfinite(s_f_star_c)]
    if s_f_star_c.size < conf.min_active_for_kappa:
        log.warning(
            "phase4b: only %d finite S_F* values on C — κ cutoffs fall back "
            "to the unit interval.",
            int(s_f_star_c.size),
        )
        low, high = conf.targeted_quantile_cutoff, conf.systematic_quantile_cutoff
    else:
        low = float(np.quantile(s_f_star_c, conf.targeted_quantile_cutoff))
        high = float(np.quantile(s_f_star_c, conf.systematic_quantile_cutoff))

    kappa = _assign_kappa(
        s_f_star=s_f_star,
        active_counts=active_counts,
        cutoffs=(low, high),
        settings=settings,
    )

    schema = settings.panel_schema
    kappa_frame = pd.DataFrame(
        {
            schema.firm_id: panel[schema.firm_id].values,
            schema.quarter: panel[schema.quarter].values,
            COL_S_F_STAR: s_f_star,
            "n_active": active_counts,
            COL_KAPPA: kappa,
        },
        index=panel.index,
    )

    summary: dict = {
        "cutoffs": {
            "targeted_quantile": conf.targeted_quantile_cutoff,
            "systematic_quantile": conf.systematic_quantile_cutoff,
            "targeted_threshold_value": low,
            "systematic_threshold_value": high,
        },
        "kappa_counts": {
            label: int((kappa == label).sum())
            for label in [conf.label_targeted, conf.label_mixed,
                          conf.label_systematic, conf.label_single,
                          conf.label_inactive]
        },
        "n_rows": int(len(panel)),
        "n_rows_with_defined_kappa": int((active_counts >= conf.min_active_for_kappa).sum()),
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase4b_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["kappa_parquet"] = out_dir / "kappa.parquet"
        paths["json"] = out_dir / "phase4b_confidence.json"
        kappa_frame.to_parquet(paths["kappa_parquet"], index=False)
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("phase4b: wrote %d deliverables to %s", len(paths), out_dir)

    return Phase4bOutcome(kappa_frame=kappa_frame, summary=summary, paths=paths)
