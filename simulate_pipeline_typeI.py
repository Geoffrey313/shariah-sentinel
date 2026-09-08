"""End-to-end type-I simulation of the full scoring pipeline under a global null.

Scope (read this before quoting a number). This does NOT simulate a raw-Compustat
data-generating process from scratch. It measures the **type-I error of the full
scoring pipeline conditional on the empirical clean-firm detector / null
distribution**: the calibration (reference sample C, PIT, Ledoit-Wolf covariance
Sigma-hat, the row-level bootstrap null, and the AR(1) exceedance null cache at
rho_cal) is fit ONCE on the real clean data and then held FIXED, exactly as a
deployed screen would be, and applied to panels that are null by construction.
This is materially stronger than the stage-wise (aggregation-only) result (which conditioned on
the calibrated row-level p-values); it exercises composites -> row-level bootstrap
p-values -> Holm RED verdict -> firm-level episodic exceedance -> Benjamini-Hochberg.

Two null branches (both with the calibration frozen):

  * primary   -- firm-block bootstrap of the REAL clean-firm z-trajectories
                 (rows with _split==C), resampled whole-firm with replacement.
                 Cross-detector AND within-firm dependence are empirical, no
                 parametric assumption. This is the credible empirical evidence.
  * parametric -- z_t ~ N(0, Sigma-hat) with an AR(1) serial copula of
                 autocorrelation rho_true swept over {0.14, 0.20, 0.30, 0.40}.
                 rho_true=0.14 ~ the C-estimated dependence; the sweep quantifies
                 the safety margin of the conservative calibration rho_cal=0.30.

Reported per branch, with Monte-Carlo error:
  * row-level RED false-positive rate vs the nominal Holm cut (red_threshold/m);
  * firm-level P(>=1 false discovery) at q<=0.01 and q<=0.05;
  * mean BH FDR (degenerates to P(>=1) under the global null -- reported as such);
  * mean and p95 number of false discoveries at each q.

Acceptance: realized end-to-end type-I <= nominal, or the
gap quantified and explained.

This is a research/validation script; it is NOT part of the production pipeline
and imports internal analysis helpers to avoid recomputing the calibration per
replicate. Requires the licensed panels under outputs/ (not self-contained).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import AnalysisSettings
from src.analysis.composite_scoring import (
    VERDICT_P_COLS, compute_verdict,
    COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_BREADTH, COL_Z_MAHALANOBIS, COL_T_IUT,
    COL_P_Z_PLUS, COL_P_Z_PLUS_RENORM, COL_P_Z_MAHALANOBIS, COL_P_T_IUT,
)
from src.analysis.fdr import (
    _exceedance_S, _exceedance_null_S, _benjamini_hochberg,
)
from src.analysis.injection import _composites_matrix, _pvalues_for_composites
from src.analysis.benchmark import _score_panel
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED


# ── calibration (frozen once from the real clean data) ───────────────────────

class FrozenCalibration:
    """Everything fit on the real panel and held fixed across null replicates."""

    def __init__(self, settings: AnalysisSettings):
        self.settings = settings
        panel = pd.read_parquet(settings.output_layout.panel_path())
        self.panel_sha256 = _sha256_df(panel)
        ctx = _score_panel(panel, settings)               # Phase0 -> z -> LW -> Phase4
        self.active = tuple(ctx.active)
        self.sigma = np.asarray(ctx.sigma, dtype=float)
        self.weights = np.asarray(ctx.weights, dtype=float)
        self.null = ctx.null                              # frozen row-level bootstrap null

        p = ctx.panel.reset_index(drop=True)
        z = ctx.zscores.reindex(ctx.panel.index)[list(self.active)].reset_index(drop=True)
        comp = ctx.composites.reset_index(drop=True)
        firm_col = settings.panel_schema.firm_id
        quarter_col = settings.panel_schema.quarter

        # clean-firm donor blocks: rows with _split==C, ordered by (firm, quarter)
        c_mask = (p["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()
        cdf = pd.DataFrame({
            "firm": p[firm_col].to_numpy()[c_mask],
            "q": p[quarter_col].to_numpy()[c_mask],
            "_row": np.where(c_mask)[0],
        }).sort_values(["firm", "q"], kind="stable")
        Z = z.to_numpy(dtype=float)
        self.donor_blocks: list[np.ndarray] = [
            Z[g["_row"].to_numpy()] for _, g in cdf.groupby("firm", sort=False)
        ]
        self.donor_K = np.array([b.shape[0] for b in self.donor_blocks], dtype=int)

        # real firm K-distribution (finite p_t_iut per firm) -> parametric branch + n_firms
        realK = (
            comp.assign(_f=p[firm_col].to_numpy())
                .groupby("_f", sort=False)[COL_P_T_IUT]
                .apply(lambda s: int(np.isfinite(s.to_numpy(dtype=float)).sum()))
        )
        self.real_K_values = realK[realK > 0].to_numpy(dtype=int)
        self.n_firms = int(self.real_K_values.size)

        # cholesky of Sigma-hat for the parametric branch
        self.chol = np.linalg.cholesky(_pd_project(self.sigma))


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _pd_project(sigma: np.ndarray) -> np.ndarray:
    """Nearest-ish SPD: symmetrise and floor eigenvalues so cholesky succeeds."""
    s = 0.5 * (sigma + sigma.T)
    w, V = np.linalg.eigh(s)
    w = np.clip(w, 1e-10, None)
    return (V * w) @ V.T


# ── exceedance null cache (frozen; depends only on K, rho_cal, taus, B) ───────

def build_exceedance_cache(K_max: int, rho_cal: float, taus: np.ndarray, B: int,
                           seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: _exceedance_null_S(k, rho_cal, taus, B, rng) for k in range(1, K_max + 1)}


# ── scoring a null panel through the frozen calibration ──────────────────────

def score_null_panel(Z: np.ndarray, firm_ids: np.ndarray, cal: FrozenCalibration,
                     cache: dict[int, np.ndarray], taus: np.ndarray, B_exc: int,
                     red_threshold: float, amber_threshold: float,
                     q_levels: tuple[float, ...]) -> dict:
    comps = _composites_matrix(Z, cal.sigma, cal.weights, cal.settings)
    pv = _pvalues_for_composites(comps, cal.null)
    pdf = pd.DataFrame({
        COL_P_Z_PLUS: pv[COL_Z_PLUS], COL_P_Z_PLUS_RENORM: pv[COL_Z_PLUS_RENORM],
        COL_P_Z_MAHALANOBIS: pv[COL_Z_MAHALANOBIS], COL_P_T_IUT: pv[COL_T_IUT],
    })
    verdict = compute_verdict(pdf, VERDICT_P_COLS, red_threshold, amber_threshold)
    red_rate = float((verdict == "RED").mean())
    amber_rate = float((verdict == "AMBER").mean())

    # firm-level episodic exceedance on p_t_iut vs the frozen cache
    pt = pv[COL_T_IUT]
    n_firms = int(firm_ids.max()) + 1
    firm_p = np.full(n_firms, np.nan, dtype=float)
    order = np.argsort(firm_ids, kind="stable")
    fids = firm_ids[order]
    pts = pt[order]
    bounds = np.searchsorted(fids, np.arange(n_firms + 1))
    for f in range(n_firms):
        seg = pts[bounds[f]:bounds[f + 1]]
        S, K = _exceedance_S(seg, taus)
        if K == 0:
            continue
        a = cache[K]
        ge = len(a) - np.searchsorted(a, S, side="left")
        firm_p[f] = (1.0 + ge) / (B_exc + 1.0)

    q = _benjamini_hochberg(firm_p)
    disc = {f"{lv}": int(np.nansum(q <= lv)) for lv in q_levels}
    return {"red_rate": red_rate, "amber_rate": amber_rate,
            "n_rows": int(Z.shape[0]), "n_firms": n_firms, "disc": disc}


# ── null-panel generators ────────────────────────────────────────────────────

def draw_primary(cal: FrozenCalibration, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Firm-block bootstrap: resample whole clean-firm z-blocks with replacement."""
    pick = rng.integers(0, len(cal.donor_blocks), size=cal.n_firms)
    blocks = [cal.donor_blocks[i] for i in pick]
    Z = np.vstack(blocks)
    firm_ids = np.repeat(np.arange(cal.n_firms), [cal.donor_blocks[i].shape[0] for i in pick])
    return Z, firm_ids


def draw_parametric(cal: FrozenCalibration, rho_true: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """z_t ~ N(0, Sigma-hat) with an AR(1) serial copula of autocorrelation rho_true;
    firm lengths K drawn from the real finite-p_t_iut K-distribution."""
    Ks = rng.choice(cal.real_K_values, size=cal.n_firms, replace=True)
    d = cal.sigma.shape[0]
    s = np.sqrt(1.0 - rho_true * rho_true)
    blocks = []
    firm_ids = []
    for f, K in enumerate(Ks):
        eps = rng.standard_normal((K, d)) @ cal.chol.T   # N(0, Sigma), iid over t
        z = np.empty((K, d), dtype=float)
        z[0] = eps[0]
        for t in range(1, K):
            z[t] = rho_true * z[t - 1] + s * eps[t]      # marginal N(0,Sigma), AR(1) serial
        blocks.append(z)
        firm_ids.append(np.full(K, f, dtype=int))
    return np.vstack(blocks), np.concatenate(firm_ids)


# ── replicate loop + aggregation ─────────────────────────────────────────────

def run_branch(name: str, generator, cal: FrozenCalibration, cache, taus, B_exc,
               red_threshold, amber_threshold, q_levels, n_outer, base_seed) -> dict:
    red_rates, amber_rates = [], []
    disc = {f"{lv}": [] for lv in q_levels}
    t0 = time.time()
    for r in range(n_outer):
        rng = np.random.default_rng(base_seed + r)
        Z, firm_ids = generator(cal, rng)
        m = score_null_panel(Z, firm_ids, cal, cache, taus, B_exc,
                             red_threshold, amber_threshold, q_levels)
        red_rates.append(m["red_rate"])
        amber_rates.append(m["amber_rate"])
        for lv in q_levels:
            disc[f"{lv}"].append(m["disc"][f"{lv}"])
    red = np.array(red_rates)
    out = {
        "n_outer": n_outer,
        "n_firms": cal.n_firms,
        "row_red_fpr": {"mean": float(red.mean()),
                        "mc_se": float(red.std(ddof=1) / np.sqrt(n_outer)) if n_outer > 1 else float("nan")},
        "row_amber_rate": {"mean": float(np.mean(amber_rates))},
        "firm_level": {},
        "seconds": round(time.time() - t0, 1),
    }
    for lv in q_levels:
        d = np.array(disc[f"{lv}"], dtype=float)
        p_any = float((d >= 1).mean())
        out["firm_level"][f"q<={lv}"] = {
            "P_ge1_false_discovery": p_any,
            "P_ge1_mc_se": float(np.sqrt(p_any * (1 - p_any) / n_outer)),
            # under the GLOBAL null every discovery is false, so BH-FDR == P(>=1)
            "mean_BH_FDR_is_P_ge1": p_any,
            "mean_false_discoveries": float(d.mean()),
            "false_discoveries_mc_se": float(d.std(ddof=1) / np.sqrt(n_outer)) if n_outer > 1 else float("nan"),
            "p95_false_discoveries": float(np.percentile(d, 95)),
            "max_false_discoveries": int(d.max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="mys")
    ap.add_argument("--mode", choices=["primary", "parametric", "both"], default="both")
    ap.add_argument("--n-outer", type=int, default=500)
    ap.add_argument("--rho-true", type=float, nargs="+", default=[0.14, 0.20, 0.30, 0.40])
    ap.add_argument("--setup-B", type=int, default=None, help="row-level bootstrap B for the frozen calibration (default: config)")
    ap.add_argument("--exceedance-B", type=int, default=None, help="exceedance null cache B (default: config)")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate wiring (n_outer=20, small B)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_outer = args.n_outer if args.n_outer != 500 else 20
        setup_B = args.setup_B or 1000
        exc_B = args.exceedance_B or 3000
    else:
        setup_B = args.setup_B
        exc_B = args.exceedance_B

    base = AnalysisSettings()
    # --country drives the panel/outputs paths, not just the JSON name.
    upd = {"output_layout": base.output_layout.model_copy(
        update={"country_code_lower": args.country.lower()})}
    if setup_B is not None:
        upd["bootstrap"] = base.bootstrap.model_copy(update={"n_replicates": setup_B})
    settings = base.model_copy(update=upd)
    fdr = settings.fdr
    taus = np.asarray(fdr.exceedance_taus, dtype=float)
    rho_cal = fdr.exceedance_rho_cal
    B_exc = exc_B if exc_B is not None else fdr.exceedance_B
    # verdict thresholds from settings (compute_verdict applies Holm via /m internally)
    red_threshold = settings.figures.red_threshold
    amber_threshold = settings.figures.amber_threshold
    q_levels = tuple(sorted(fdr.q_levels))

    print(f"[setup] freezing calibration (bootstrap B={settings.bootstrap.n_replicates}) ...", flush=True)
    t0 = time.time()
    cal = FrozenCalibration(settings)
    print(f"[setup] done in {time.time()-t0:.0f}s | active={cal.active} | "
          f"n_firms={cal.n_firms} | donors={len(cal.donor_blocks)} | "
          f"K in [{cal.donor_K.min()},{cal.donor_K.max()}] | panel_sha256={cal.panel_sha256[:12]}", flush=True)

    K_max = int(max(cal.donor_K.max(), cal.real_K_values.max()))
    print(f"[setup] building exceedance null cache K=1..{K_max} at rho_cal={rho_cal}, B={B_exc} ...", flush=True)
    cache = build_exceedance_cache(K_max, rho_cal, taus, B_exc, seed=fdr.exceedance_seed)

    result = {
        "scope": ("full scoring-pipeline type-I conditional on the empirical clean-firm "
                  "detector/null distribution (calibration frozen from real C; NOT a raw "
                  "Compustat data-generating process)"),
        "config": {
            "country": args.country, "n_outer": args.n_outer, "seed": args.seed,
            "setup_bootstrap_B": settings.bootstrap.n_replicates, "exceedance_B": B_exc,
            "rho_cal": rho_cal, "taus": list(map(float, taus)),
            "red_threshold": red_threshold, "amber_threshold": amber_threshold,
            "q_levels": list(q_levels), "headline_composite": COL_P_T_IUT,
            "nominal_row_red_fwer": red_threshold, "smoke": args.smoke,
            "panel_sha256": cal.panel_sha256, "n_firms": cal.n_firms,
        },
        "branches": {},
    }

    if args.mode in ("primary", "both"):
        print(f"[primary] firm-block bootstrap of real clean-firm z-blocks, n_outer={args.n_outer} ...", flush=True)
        result["branches"]["primary_firm_block_bootstrap"] = run_branch(
            "primary", draw_primary, cal, cache, taus, B_exc,
            red_threshold, amber_threshold, q_levels, args.n_outer, base_seed=args.seed)
        print(f"[primary] {json.dumps(result['branches']['primary_firm_block_bootstrap'], indent=2)}", flush=True)

    if args.mode in ("parametric", "both"):
        result["branches"]["parametric_ar1"] = {}
        for rho_true in args.rho_true:
            print(f"[parametric] N(0,Sigma)+AR(1) rho_true={rho_true}, n_outer={args.n_outer} ...", flush=True)
            gen = lambda cal, rng, _r=rho_true: draw_parametric(cal, _r, rng)
            br = run_branch(f"param_{rho_true}", gen, cal, cache, taus, B_exc,
                            red_threshold, amber_threshold, q_levels, args.n_outer,
                            base_seed=args.seed + int(round(rho_true * 1000)))
            br["rho_true"] = rho_true
            result["branches"]["parametric_ar1"][f"rho_true={rho_true}"] = br

    out_path = Path(args.out) if args.out else Path(
        f"outputs/scores/{args.country}/typeI_pipeline/simulate_pipeline_typeI"
        f"{'_smoke' if args.smoke else ''}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
