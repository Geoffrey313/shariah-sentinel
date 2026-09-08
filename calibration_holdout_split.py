"""Chronological calibration/test split.

Answers the concern that apparent uniformity on the reference sample C is in-sample:
the same C estimates the empirical PIT, the covariance Sigma-hat and the bootstrap
null, and the KS uniformity check is then read on that same C. A single chronological
hold-out on the MYS anchor calibrates the whole row-level machinery (PIT / Ledoit-Wolf
Sigma-hat / bootstrap null) on the clean rows of the EARLY years only, then evaluates
calibration on the clean rows of the HELD-OUT later years, which never touched the
calibration.

Mechanism (no pipeline modification -- the canonical path is untouched):
  * run Phase 0 once to get the canonical C / NOT_C split;
  * derive the calendar year from the quarter column and fix a train/test boundary;
  * rebuild the panel with _split = "C" kept ONLY on clean rows in the training years
    (clean rows in the test years become NOT_C). Because _score_panel skips Phase 0
    when _split is pre-populated, and because the z-score empirical PIT, the covariance
    and the bootstrap null all calibrate on the _split==C rows, this calibrates
    EVERYTHING on the training-year C alone;
  * score every row with that frozen training calibration and read, on the clean rows,
    the per-detector uniformity (KS of z vs N(0,1)) and the row-level RED rate --
    IN-sample on the training-year clean rows, OUT-of-sample on the held-out test-year
    clean rows.

Invariants: opt-in, MYS-only, reported ALONGSIDE the canonical run. The reported
headline (324/421) and full-C calibration are NOT changed; the bootstrap-cache globals
are reset so the split genuinely recalibrates. A degraded out-of-sample result is a
finding to report honestly next to the temporal-drift analysis.

Requires the licensed panels under outputs/ (not self-contained).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.common.config import AnalysisSettings
from src.analysis.reference_sample import run_phase0, SPLIT_LABEL_INCLUDED, SPLIT_LABEL_EXCLUDED
from src.analysis import benchmark
from src.analysis.benchmark import _score_panel
from src.analysis.composite_scoring import (
    VERDICT_P_COLS, compute_verdict,
    COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_Z_MAHALANOBIS, COL_T_IUT,
    COL_P_Z_PLUS, COL_P_Z_PLUS_RENORM, COL_P_Z_MAHALANOBIS, COL_P_T_IUT,
)
from src.analysis.injection import _composites_matrix, _pvalues_for_composites


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _year_of(quarter: pd.Series) -> np.ndarray:
    """Calendar year from the quarter label (handles '2000-Q3', '2000Q4', etc.)."""
    return quarter.astype(str).str.extract(r"(\d{4})")[0].astype("float").to_numpy()


def _evaluate(z_active: pd.DataFrame, mask: np.ndarray, active, sigma, weights,
              null, settings, red_thr, amber_thr) -> dict:
    """Per-detector KS(z vs N(0,1)) + composite-p uniformity + RED rate on a row subset."""
    Z = z_active.to_numpy(dtype=float)[mask]
    ks_det = {}
    for j, det in enumerate(active):
        col = Z[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 20:
            st = stats.kstest(col, "norm")
            ks_det[det] = {"ks": float(st.statistic), "p": float(st.pvalue), "n": int(col.size)}
    comps = _composites_matrix(Z, sigma, weights, settings)
    pv = _pvalues_for_composites(comps, null)
    pdf = pd.DataFrame({
        COL_P_Z_PLUS: pv[COL_Z_PLUS], COL_P_Z_PLUS_RENORM: pv[COL_Z_PLUS_RENORM],
        COL_P_Z_MAHALANOBIS: pv[COL_Z_MAHALANOBIS], COL_P_T_IUT: pv[COL_T_IUT],
    })
    verdict = compute_verdict(pdf, VERDICT_P_COLS, red_thr, amber_thr)
    red_rate = float((verdict == "RED").mean())
    amber_rate = float((verdict == "AMBER").mean())
    pt = np.asarray(pv[COL_T_IUT], dtype=float)
    pt = pt[np.isfinite(pt)]
    ks_tiut = stats.kstest(pt, "uniform") if pt.size >= 20 else None
    return {
        "n_rows": int(mask.sum()),
        "red_rate": red_rate, "amber_rate": amber_rate,
        "t_iut_p_uniformity_ks": (None if ks_tiut is None else
                                  {"ks": float(ks_tiut.statistic), "p": float(ks_tiut.pvalue)}),
        "per_detector_ks_normal": ks_det,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="mys")
    ap.add_argument("--boundary-year", type=int, default=2019,
                    help="first TEST year; train = years < boundary, test = years >= boundary. "
                         "Default 2019 = 70/30 by clean-year count (16 train / 7 test years), "
                         "pre-registered from the C distribution, independent of any OOS metric.")
    ap.add_argument("--boot-B", type=int, default=None, help="bootstrap B (default: config)")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--smoke", action="store_true", help="strict wiring/logic check (small B)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    boot_B = (args.boot_B or 2000) if args.smoke else args.boot_B
    base = AnalysisSettings()
    upd = {"output_layout": base.output_layout.model_copy(
        update={"country_code_lower": args.country.lower()})}
    if boot_B is not None:
        upd["bootstrap"] = base.bootstrap.model_copy(update={"n_replicates": boot_B})
    settings = base.model_copy(update=upd)
    qcol = settings.panel_schema.quarter
    red_thr = settings.figures.red_threshold
    amber_thr = settings.figures.amber_threshold

    panel = pd.read_parquet(settings.output_layout.panel_path())
    panel_sha = _sha256_df(panel)

    p0 = run_phase0(panel=panel, settings=settings, write_outputs=False)
    p0panel = p0.panel.reset_index(drop=True)
    year = _year_of(p0panel[qcol])
    C_canon = (p0panel["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()

    b = args.boundary_year
    train = year < b
    test = year >= b
    trainC = C_canon & train & np.isfinite(year)
    testC = C_canon & test & np.isfinite(year)

    panel_w6 = p0panel.copy()
    split_w6 = np.where(trainC, SPLIT_LABEL_INCLUDED, SPLIT_LABEL_EXCLUDED)
    panel_w6["_split"] = split_w6
    # Traceability (does not affect the calculation): mark why each row's split
    # changed -- train_C stays C, the held-out clean rows carry holdout_not_C so the
    # artifact distinguishes them from genuinely-excluded (non-clean) rows.
    if "_split_reason" in panel_w6.columns:
        reason = panel_w6["_split_reason"].astype(object).to_numpy().copy()
    else:
        reason = np.array([""] * len(panel_w6), dtype=object)
    reason[trainC] = "train_C"
    reason[testC] = "holdout_not_C"
    panel_w6["_split_reason"] = reason

    assert not (split_w6 == SPLIT_LABEL_INCLUDED)[test].any(), "test-year row leaked into training C"
    tr_years = sorted(int(y) for y in set(year[trainC].astype(int)))
    te_years = sorted(int(y) for y in set(year[testC].astype(int)))
    print(f"[split] boundary={b} | train C rows={int(trainC.sum())} ({tr_years[0]}-{tr_years[-1]}, "
          f"{len(tr_years)}y) | test C rows={int(testC.sum())} ({te_years[0]}-{te_years[-1]}, "
          f"{len(te_years)}y) | panel_sha={panel_sha[:12]}", flush=True)

    benchmark._CACHED_NULL = None
    benchmark._CACHED_SIGMA = None
    t0 = time.time()
    ctx = _score_panel(panel_w6, settings)
    active = tuple(ctx.active)
    sigma = np.asarray(ctx.sigma, dtype=float)
    weights = np.asarray(ctx.weights, dtype=float)
    null = ctx.null
    z_active = ctx.zscores.reindex(ctx.panel.index)[list(active)].reset_index(drop=True)
    print(f"[calib] trained on C_train in {time.time()-t0:.0f}s | active={active} | "
          f"bootstrap B={settings.bootstrap.n_replicates}", flush=True)

    in_sample = _evaluate(z_active, trainC, active, sigma, weights, null, settings, red_thr, amber_thr)
    out_sample = _evaluate(z_active, testC, active, sigma, weights, null, settings, red_thr, amber_thr)

    result = {
        "scope": ("chronological calibration/test split (MYS anchor): PIT + Sigma-hat + bootstrap "
                  "null calibrated on the training-year clean rows only, evaluated on the held-out "
                  "later-year clean rows. Reported alongside the canonical run; the headline and the "
                  "full-C calibration are NOT changed."),
        "guardrails": {
            "canonical_untouched": True, "mys_only": True,
            "bootstrap_cache_reset": True, "rho_cal_unchanged": settings.fdr.exceedance_rho_cal,
        },
        "config": {
            "country": args.country, "boundary_year": b, "seed": args.seed,
            "bootstrap_B": settings.bootstrap.n_replicates, "smoke": args.smoke,
            "panel_sha256": panel_sha, "active": list(active),
            "n_train_C": int(trainC.sum()), "n_test_C": int(testC.sum()),
            "train_years": [tr_years[0], tr_years[-1]], "test_years": [te_years[0], te_years[-1]],
            "red_threshold": red_thr, "amber_threshold": amber_thr,
        },
        "in_sample_train_C": in_sample,
        "out_of_sample_test_C": out_sample,
    }
    print(f"[in-sample  train C] RED={in_sample['red_rate']:.4f} AMBER={in_sample['amber_rate']:.4f}", flush=True)
    print(f"[out-sample test  C] RED={out_sample['red_rate']:.4f} AMBER={out_sample['amber_rate']:.4f}", flush=True)

    out_path = Path(args.out) if args.out else Path(
        f"outputs/scores/{args.country}/calibration_holdout/calibration_holdout"
        f"{'_smoke' if args.smoke else ''}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
