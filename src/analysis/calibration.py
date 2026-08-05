"""Phase 1 — per-detector null calibration on reference sample ``C``.

Framework §9.1 claims each ``z_j`` is marginally ``N(0, 1)`` under ``H_0`` once
the PIT is correctly applied. That claim depends on ``F_j`` being well-
specified; several detectors (z₂ with plug-in ``σ̂_0²``, z₃ empirical on
``C``, z₅ / z₆ with approximate ``χ²_q`` calibrations, z₇ Hotelling ``T²``)
only achieve this asymptotically or approximately. Phase 1 checks the claim
empirically on ``C`` and flags any detector whose parametric null is
rejected — that detector is forced to non-parametric bootstrap in Phase 4.

Deliverables (under ``outputs/scores/<country>/phase1_null_calibration/``):

- ``null_calibration.json`` — per-detector moments, KS/AD stats, bias
  verdict, variance-ratio verdict, list of detectors flagged for bootstrap.
- ``qq_quantiles.parquet`` — per-detector empirical quantiles plus the
  corresponding ``N(0, 1)`` theoretical quantiles (for QQ-plot decks).
- ``subsample_moments.csv`` — per-year and per-sector mean/std/n for each
  detector, subject to ``min_subsample_size``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.engine.zscores import load_zscores
from src.common.config import AnalysisSettings
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

log = logging.getLogger(__name__)


# Verdict strings stored in the JSON summary. Kept as module constants so
# downstream phases can filter on them without guessing spelling.
VERDICT_OK: str = "ok"
VERDICT_BIASED: str = "biased"
VERDICT_MIS_SCALED: str = "mis_scaled"
VERDICT_NON_NORMAL: str = "non_normal"
VERDICT_NO_DATA: str = "no_data"


@dataclass(frozen=True)
class CalibrationOutcome:
    """Return value of :func:`run_phase1`."""

    summary: dict
    qq_quantiles: pd.DataFrame
    subsample_moments: pd.DataFrame
    paths: dict[str, Path]


def _per_detector_moments(values: np.ndarray) -> dict:
    """Mean / std / skew / excess-kurtosis / n with robust NaN guard."""
    clean = values[np.isfinite(values)]
    n = int(clean.size)
    if n < 2:
        return {"n": n, "mean": float("nan"), "std": float("nan"),
                "skew": float("nan"), "excess_kurtosis": float("nan")}
    return {
        "n": n,
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)),
        "skew": float(stats.skew(clean, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(clean, fisher=True, bias=False)),
    }


def _normality_tests(values: np.ndarray, settings: AnalysisSettings) -> dict:
    """KS + Anderson-Darling against ``N(0, 1)`` with explicit p-values."""
    clean = values[np.isfinite(values)]
    if clean.size < settings.calibration.min_subsample_size:
        return {"ks_stat": float("nan"), "ks_pvalue": float("nan"),
                "anderson_stat": float("nan"), "anderson_critical_1pct": float("nan")}
    ks_res = stats.kstest(clean, "norm")
    ad_res = stats.anderson(clean, dist="norm")
    # Anderson's critical_values align with significance_levels; pick 1%
    # (index of 1.0 in the returned significance_levels array).
    sig_levels = np.asarray(ad_res.significance_level, dtype=float)
    one_pct_idx = int(np.argmin(np.abs(sig_levels - 1.0)))
    return {
        "ks_stat": float(ks_res.statistic),
        "ks_pvalue": float(ks_res.pvalue),
        "anderson_stat": float(ad_res.statistic),
        "anderson_critical_1pct": float(ad_res.critical_values[one_pct_idx]),
    }


def _verdict(moments: dict, tests: dict, settings: AnalysisSettings) -> list[str]:
    """Collect verdict tags for one detector. Empty list ≡ ``ok``."""
    cal = settings.calibration
    if moments["n"] < cal.min_subsample_size:
        return [VERDICT_NO_DATA]
    tags: list[str] = []
    if np.isfinite(moments["mean"]) and abs(moments["mean"] - cal.target_mean) > cal.bias_tolerance:
        tags.append(VERDICT_BIASED)
    if np.isfinite(moments["std"]):
        var = moments["std"] ** 2
        ratio_diff = abs(var - cal.target_std ** 2) / (cal.target_std ** 2)
        if ratio_diff > cal.variance_ratio_tolerance:
            tags.append(VERDICT_MIS_SCALED)
    if np.isfinite(tests["ks_pvalue"]) and tests["ks_pvalue"] < cal.ks_alpha:
        tags.append(VERDICT_NON_NORMAL)
    return tags


def _qq_quantiles_table(
    zscores_c: pd.DataFrame,
    detectors: tuple[str, ...],
    n_quantiles: int,
) -> pd.DataFrame:
    """Emit empirical + theoretical quantiles for each detector.

    Evenly-spaced probabilities in ``(0, 1)`` avoid the undefined endpoints
    of ``Φ^{-1}(0)`` / ``Φ^{-1}(1)``.
    """
    probs = (np.arange(1, n_quantiles + 1, dtype=float)) / (n_quantiles + 1)
    frames: list[pd.DataFrame] = []
    for name in detectors:
        series = zscores_c[name].dropna().to_numpy(dtype=float)
        if series.size == 0:
            continue
        empirical = np.quantile(series, probs)
        theoretical = stats.norm.ppf(probs)
        frames.append(
            pd.DataFrame(
                {
                    "detector": name,
                    "prob": probs,
                    "empirical_quantile": empirical,
                    "theoretical_quantile": theoretical,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["detector", "prob", "empirical_quantile", "theoretical_quantile"]
    )


def _subsample_moments(
    zscores_c: pd.DataFrame,
    panel_c: pd.DataFrame,
    detectors: tuple[str, ...],
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Per-year and per-sector moments for each detector.

    The framework recommends faceted distributions (§9.1 bullet "Sub-sample
    calibration"). Sub-samples smaller than
    :attr:`CalibrationSettings.min_subsample_size` are dropped so the report
    doesn't amplify noise.
    """
    schema = settings.panel_schema
    year_col = settings.calibration.year_column
    if year_col not in panel_c.columns:
        log.warning("phase1: year column %r missing; sub-sample year stats skipped.", year_col)
    facets: list[tuple[str, pd.Series]] = []
    if year_col in panel_c.columns:
        facets.append(("year", panel_c[year_col]))
    if schema.sector in panel_c.columns:
        facets.append(("sector", panel_c[schema.sector]))

    rows: list[dict] = []
    min_n = settings.calibration.min_subsample_size
    for facet_name, keys in facets:
        for value, sub_idx in keys.groupby(keys.fillna("__NA__")).groups.items():
            sub = zscores_c.loc[sub_idx]
            for det in detectors:
                arr = sub[det].dropna().to_numpy(dtype=float)
                if arr.size < min_n:
                    continue
                rows.append(
                    {
                        "facet": facet_name,
                        "value": value,
                        "detector": det,
                        "n": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def run_phase1(
    panel: pd.DataFrame,
    settings: AnalysisSettings | None = None,
    zscores: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> CalibrationOutcome:
    """Run per-detector null calibration on ``C``.

    Args:
        panel: Output of :func:`phase0_reference_sample.run_phase0`. The
            function reads ``_split`` to identify rows in ``C`` and
            optionally the ``year``/``sector`` columns for sub-sample stats.
        settings: Configuration. Defaults to :class:`AnalysisSettings`.
        zscores: Optional pre-computed z-score frame. When ``None`` the
            cache produced by :func:`compute_zscores` is loaded from disk.
        write_outputs: If ``True``, persist the JSON / parquet / CSV
            deliverables under ``settings.output_layout.phase1_dir()``.

    Returns:
        A :class:`CalibrationOutcome` wrapping the JSON summary, QQ-plot
        quantile frame, sub-sample moments and resolved paths.
    """
    settings = settings or AnalysisSettings()
    if "_split" not in panel.columns:
        raise KeyError("phase1: panel lacks `_split` — run Phase 0 first.")

    zscores = zscores if zscores is not None else load_zscores(settings)
    # Align on panel index — zscores was written in panel row order.
    if len(zscores) != len(panel):
        raise ValueError(
            f"phase1: zscores length {len(zscores)} != panel length "
            f"{len(panel)}; re-run `compute_zscores` after any panel rebuild."
        )
    zscores = zscores.copy()
    zscores.index = panel.index

    detectors = tuple(settings.zscores.active_detectors())
    mask_c = panel["_split"] == SPLIT_LABEL_INCLUDED
    zscores_c = zscores.loc[mask_c, list(detectors)]
    panel_c = panel.loc[mask_c]

    per_detector: dict[str, dict] = {}
    for name in detectors:
        arr = zscores_c[name].to_numpy(dtype=float)
        moments = _per_detector_moments(arr)
        tests = _normality_tests(arr, settings)
        verdict_tags = _verdict(moments, tests, settings)
        per_detector[name] = {
            "moments": moments,
            "tests": tests,
            "verdict_tags": verdict_tags,
            "verdict": VERDICT_OK if not verdict_tags else ",".join(verdict_tags),
            "forced_bootstrap": bool(verdict_tags) and VERDICT_NO_DATA not in verdict_tags,
        }

    qq_df = _qq_quantiles_table(
        zscores_c=zscores_c,
        detectors=detectors,
        n_quantiles=settings.calibration.qq_quantiles,
    )
    subsample = _subsample_moments(
        zscores_c=zscores_c,
        panel_c=panel_c,
        detectors=detectors,
        settings=settings,
    )

    summary: dict = {
        "c_rows": int(mask_c.sum()),
        "targets": {
            "mean": settings.calibration.target_mean,
            "std": settings.calibration.target_std,
        },
        "thresholds": {
            "bias_tolerance": settings.calibration.bias_tolerance,
            "variance_ratio_tolerance": settings.calibration.variance_ratio_tolerance,
            "ks_alpha": settings.calibration.ks_alpha,
            "anderson_alpha": settings.calibration.anderson_alpha,
        },
        "per_detector": per_detector,
        "forced_bootstrap": [d for d, v in per_detector.items() if v["forced_bootstrap"]],
    }

    paths: dict[str, Path] = {}
    if write_outputs:
        out_dir = settings.output_layout.phase1_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["json"] = out_dir / "null_calibration.json"
        paths["qq_parquet"] = out_dir / "qq_quantiles.parquet"
        paths["subsample_csv"] = out_dir / "subsample_moments.csv"
        paths["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        qq_df.to_parquet(paths["qq_parquet"], index=False)
        subsample.to_csv(paths["subsample_csv"], index=False)
        log.info("phase1: wrote %d deliverables to %s", len(paths), out_dir)

    return CalibrationOutcome(
        summary=summary,
        qq_quantiles=qq_df,
        subsample_moments=subsample,
        paths=paths,
    )
