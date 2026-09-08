"""Firm-cluster bootstrap robustness of the row-level null.

The production row-level bootstrap null resamples single reference rows i.i.d.
(src/engine/bootstrap.py::_simulate_non_parametric), treating a firm's quarters as
exchangeable. This script rebuilds the SAME null with a firm-level CLUSTER resample
(whole firms drawn with replacement, _simulate_non_parametric_cluster) and reports
how much the null critical values and the resulting discovery counts move. It is a
robustness check reported ALONGSIDE the canonical run, never a replacement.

Invariants enforced here:
  * opt-in only -- the production pipeline default resampler stays i.i.d. row; this
    script is the only caller that passes resample_mode="cluster".
  * the firm-level episodic-exceedance calibration is UNCHANGED: rho_cal stays at
    fdr.exceedance_rho_cal and the exceedance null cache is frozen. ONLY the
    row-level bootstrap null differs (row vs cluster), isolating the row-bootstrap
    exchangeability effect from the firm-aggregation calibration.
  * the firm-level path REUSES the shipped exceedance functions
    (_exceedance_firm_pvalues / _discovery_counts) so the canonical frozen null
    reproduces the reported headline (324 @ q<=0.01 / 421 @ q<=0.05) EXACTLY; a hard
    self-check aborts at full resolution if it does not. "row" is the matched
    Monte-Carlo baseline and "cluster" is the stress test.

Procedure. Freeze the calibration once (reference sample C, PIT, Ledoit-Wolf
Sigma-hat, active set, weights). Hold the observed composites and the exceedance
null cache fixed (firm K is null-invariant, so the cache is built once and reused).
For each scheme in {row, cluster} and each of n_outer null draws (matched on B):
build the composite null -> bootstrap p-values of the observed composites ->
row-level Holm RED count -> firm-level episodic-exceedance headline (q<=0.01,
q<=0.05). Report mean / p5 / p95 / MC-SE of the critical values and the counts, and
the cluster-vs-row delta.

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

from src.common.config import AnalysisSettings
from src.engine.bootstrap import (
    _simulate_non_parametric,
    _simulate_non_parametric_cluster,
    _composite_null_from_draws,
)
from src.analysis.composite_scoring import (
    VERDICT_P_COLS, compute_verdict,
    COL_Z_PLUS, COL_Z_PLUS_RENORM, COL_Z_MAHALANOBIS, COL_T_IUT,
    COL_P_Z_PLUS, COL_P_Z_PLUS_RENORM, COL_P_Z_MAHALANOBIS, COL_P_T_IUT,
)
from src.analysis.injection import _composites_matrix, _pvalues_for_composites
from src.analysis.fdr import (
    _exceedance_S, _exceedance_null_S, _exceedance_firm_pvalues,
    _benjamini_hochberg, _discovery_counts,
)
from src.analysis.benchmark import _score_panel
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED

_PT_COL = "p_t_iut"  # working column name for the firm-level exceedance frames


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


class Frozen:
    """Calibration + observed composites held fixed across every null draw."""

    def __init__(self, settings: AnalysisSettings):
        panel = pd.read_parquet(settings.output_layout.panel_path())
        self.panel_sha256 = _sha256_df(panel)
        ctx = _score_panel(panel, settings)                 # reference sample -> z -> LW -> composites
        self.settings = settings
        self.firm_col = settings.panel_schema.firm_id
        self.active = tuple(ctx.active)
        self.sigma = np.asarray(ctx.sigma, dtype=float)
        self.weights = np.asarray(ctx.weights, dtype=float)
        self.null_canonical = ctx.null                      # the shipped row-level null

        p = ctx.panel.reset_index(drop=True)
        z = ctx.zscores.reindex(ctx.panel.index)[list(self.active)].reset_index(drop=True)
        Zall = z.to_numpy(dtype=float)

        # observed composites on the FULL panel (fixed; only the null varies)
        self.obs_comps = _composites_matrix(Zall, self.sigma, self.weights, settings)

        # observed firm ids 0..n_firms-1 (all firms in the panel)
        firm_codes, _ = pd.factorize(p[self.firm_col].to_numpy())
        self.obs_firm_ids = firm_codes.astype(int)
        self.n_firms_obs = int(firm_codes.max()) + 1

        # clean-firm reference rows + aligned firm ids (what C's bootstrap resamples)
        c_mask = (p["_split"] == SPLIT_LABEL_INCLUDED).to_numpy()
        finite = np.isfinite(Zall).all(axis=1)
        cm = c_mask & finite
        self.complete_c = Zall[cm]
        self.firm_ids_c = firm_codes[cm].astype(int)
        self.n_ref = int(self.complete_c.shape[0])
        self.n_firms_c = int(np.unique(self.firm_ids_c).size)

    def pt_for_null(self, null) -> np.ndarray:
        """Observed T_IUT bootstrap p-values under a given row-level null."""
        pv = _pvalues_for_composites(self.obs_comps, null)
        return pv, np.asarray(pv[COL_T_IUT], dtype=float)


def _draw_null(scheme: str, fr: Frozen, B: int, rng: np.random.Generator):
    if scheme == "row":
        draws = _simulate_non_parametric(fr.complete_c, B, rng)
    elif scheme == "cluster":
        draws = _simulate_non_parametric_cluster(fr.complete_c, fr.firm_ids_c, B, rng)
    else:
        raise ValueError(scheme)
    return _composite_null_from_draws(draws, fr.sigma, fr.weights, fr.settings, scheme)


def _firm_pvalues_cached(pt: np.ndarray, firm_ids: np.ndarray, firm_col: str,
                         taus: np.ndarray, cache: dict[int, np.ndarray],
                         B_exc: int) -> pd.Series:
    """Firm-level exceedance p-values, mirroring the shipped _exceedance_firm_pvalues
    but with a PRE-BUILT cache (firm K is null-invariant, so the cache — hence the
    RNG draws — is identical to the shipped path; validated once against the real
    function in main)."""
    df = pd.DataFrame({firm_col: firm_ids, _PT_COL: pt})
    obs = df.groupby(firm_col, observed=True)[_PT_COL].apply(
        lambda s: _exceedance_S(s.to_numpy(dtype=float), taus)
    )
    S_obs = obs.apply(lambda t: t[0])
    K = obs.apply(lambda t: t[1]).astype(int)
    p = pd.Series(np.nan, index=obs.index, dtype=float)
    for k, a in cache.items():
        idx = K.index[(K == k).to_numpy()]
        if len(idx) == 0:
            continue
        S = S_obs.loc[idx].to_numpy(dtype=float)
        ge = len(a) - np.searchsorted(a, S, side="left")
        p.loc[idx] = (1.0 + ge) / (B_exc + 1.0)
    return p


def _headline(null, fr: Frozen, cache, taus, B_exc, red_thr, amber_thr,
              q_levels, m) -> dict:
    """Row-level RED count + firm-level exceedance headline for one null."""
    pv, pt = fr.pt_for_null(null)
    pdf = pd.DataFrame({
        COL_P_Z_PLUS: pv[COL_Z_PLUS], COL_P_Z_PLUS_RENORM: pv[COL_Z_PLUS_RENORM],
        COL_P_Z_MAHALANOBIS: pv[COL_Z_MAHALANOBIS], COL_P_T_IUT: pv[COL_T_IUT],
    })
    verdict = compute_verdict(pdf, VERDICT_P_COLS, red_thr, amber_thr)
    red_count = int((verdict == "RED").sum())

    fp = _firm_pvalues_cached(pt, fr.obs_firm_ids, fr.firm_col, taus, cache, B_exc)
    q = _benjamini_hochberg(fp.to_numpy(dtype=float))
    disc = _discovery_counts(pd.Series(q), q_levels)   # keys "q<=0.01", "q<=0.05"

    alpha = red_thr / m
    crit = {
        "t_iut": float(np.quantile(null.t_iut_sorted, 1.0 - alpha)),
        "z_mahalanobis": float(np.quantile(null.z_mahalanobis_sorted, 1.0 - alpha)),
    }
    return {"red_count": red_count, "disc": disc, "crit": crit}


def _agg(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    n = a.size
    return {
        "mean": float(a.mean()),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "mc_se": float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "min": float(a.min()), "max": float(a.max()),
    }


def run_scheme(scheme, fr, cache, taus, B, B_exc, red_thr, amber_thr,
               qkeys, m, n_outer, base_seed) -> dict:
    red, crit_t, crit_m = [], [], []
    disc = {k: [] for k in qkeys}
    t0 = time.time()
    for r in range(n_outer):
        rng = np.random.default_rng(base_seed + r)
        null = _draw_null(scheme, fr, B, rng)
        h = _headline(null, fr, cache, taus, B_exc, red_thr, amber_thr, QLEVELS, m)
        red.append(h["red_count"])
        crit_t.append(h["crit"]["t_iut"])
        crit_m.append(h["crit"]["z_mahalanobis"])
        for k in qkeys:
            disc[k].append(h["disc"][k])
    return {
        "scheme": scheme, "n_outer": n_outer, "B": B, "seconds": round(time.time() - t0, 1),
        "row_red_count": _agg(red),
        "critical_value_t_iut": _agg(crit_t),
        "critical_value_z_mahalanobis": _agg(crit_m),
        "firm_discoveries": {k: _agg(disc[k]) for k in qkeys},
    }


# q-levels are fixed from config in main(); kept module-level for the closures above
QLEVELS: tuple[float, ...] = ()


def main() -> None:
    global QLEVELS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="mys")
    ap.add_argument("--n-outer", type=int, default=300)
    ap.add_argument("--boot-B", type=int, default=None, help="row-level bootstrap B per null draw (default: config)")
    ap.add_argument("--exceedance-B", type=int, default=None, help="exceedance null cache B (default: config)")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate wiring (n_outer=15, small B)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_outer = args.n_outer if args.n_outer != 300 else 15
        boot_B = args.boot_B or 2000
        exc_B = args.exceedance_B or 5000
    else:
        boot_B = args.boot_B
        exc_B = args.exceedance_B

    base = AnalysisSettings()
    upd = {"output_layout": base.output_layout.model_copy(
        update={"country_code_lower": args.country.lower()})}
    if boot_B is not None:
        upd["bootstrap"] = base.bootstrap.model_copy(update={"n_replicates": boot_B})
    settings = base.model_copy(update=upd)
    fdr = settings.fdr
    taus = np.asarray(fdr.exceedance_taus, dtype=float)
    rho_cal = fdr.exceedance_rho_cal
    B_exc = exc_B if exc_B is not None else fdr.exceedance_B
    exc_seed = fdr.exceedance_seed
    red_thr = settings.figures.red_threshold
    amber_thr = settings.figures.amber_threshold
    QLEVELS = tuple(sorted(fdr.q_levels))
    qkeys = [f"q<={q:.3g}" for q in QLEVELS]
    m = len(VERDICT_P_COLS)
    B = settings.bootstrap.n_replicates

    print(f"[setup] freezing calibration (bootstrap B={B}) ...", flush=True)
    t0 = time.time()
    fr = Frozen(settings)
    print(f"[setup] done in {time.time()-t0:.0f}s | active={fr.active} | n_ref(C)={fr.n_ref} | "
          f"firms(C)={fr.n_firms_c} | firms(obs)={fr.n_firms_obs} | "
          f"panel_sha256={fr.panel_sha256[:12]}", flush=True)

    # ── authoritative canonical headline via the shipped exceedance function ─
    pv_canon, pt_canon = fr.pt_for_null(fr.null_canonical)
    df_canon = pd.DataFrame({fr.firm_col: fr.obs_firm_ids, _PT_COL: pt_canon})
    print(f"[setup] canonical firm exceedance p-values via the shipped path "
          f"(rho_cal={rho_cal}, B_exc={B_exc}, seed={exc_seed}) ...", flush=True)
    fp_canon = _exceedance_firm_pvalues(df_canon, _PT_COL, fr.firm_col, taus, rho_cal, B_exc, exc_seed)
    q_canon = _benjamini_hochberg(fp_canon.to_numpy(dtype=float))
    disc_canon = _discovery_counts(pd.Series(q_canon), QLEVELS)
    red_canon = int((compute_verdict(pd.DataFrame({
        COL_P_Z_PLUS: pv_canon[COL_Z_PLUS], COL_P_Z_PLUS_RENORM: pv_canon[COL_Z_PLUS_RENORM],
        COL_P_Z_MAHALANOBIS: pv_canon[COL_Z_MAHALANOBIS], COL_P_T_IUT: pv_canon[COL_T_IUT],
    }), VERDICT_P_COLS, red_thr, amber_thr) == "RED").sum())
    print(f"[selfcheck] canonical frozen null -> RED={red_canon} | firm disc={disc_canon}", flush=True)

    # hard guardrail at full resolution: must reproduce the reported headline exactly
    EXPECT = {"q<=0.01": 324, "q<=0.05": 421}
    full_resolution = (
        (not args.smoke) and (B >= base.bootstrap.n_replicates) and (B_exc >= base.fdr.exceedance_B)
    )
    if full_resolution:
        got = {k: disc_canon.get(k) for k in EXPECT}
        if got != EXPECT:
            raise SystemExit(
                f"[selfcheck FAILED] canonical frozen null gave firm disc {got}, expected "
                f"{EXPECT}. The calibration is misaligned (panel / bootstrap B / exceedance "
                f"cache) — aborting before the {args.n_outer}-draw run.")
        print(f"[selfcheck OK] canonical frozen null reproduces the reported headline {EXPECT}.",
              flush=True)

    # ── shared exceedance cache for the fast n_outer path (built ONCE, identical
    #    construction to the shipped path; firm K is null-invariant) ───────────
    K_canon = df_canon.groupby(fr.firm_col, observed=True)[_PT_COL].apply(
        lambda s: _exceedance_S(s.to_numpy(dtype=float), taus)[1]
    ).astype(int)
    rng = np.random.default_rng(exc_seed)
    cache = {k: _exceedance_null_S(k, rho_cal, taus, B_exc, rng)
             for k in sorted(int(x) for x in K_canon.unique() if x > 0)}
    # validate the shared-cache fast path == the authoritative canonical headline
    fp_fast = _firm_pvalues_cached(pt_canon, fr.obs_firm_ids, fr.firm_col, taus, cache, B_exc)
    disc_fast = _discovery_counts(pd.Series(_benjamini_hochberg(fp_fast.to_numpy(dtype=float))), QLEVELS)
    if disc_fast != disc_canon:
        raise SystemExit(f"[internal] shared-cache path {disc_fast} != authoritative {disc_canon}")
    print("[selfcheck OK] shared-cache fast path matches the shipped path exactly.", flush=True)

    result = {
        "scope": ("firm-cluster vs i.i.d.-row resample of the row-level bootstrap null, "
                  "reported alongside the canonical run; the exceedance calibration "
                  "(rho_cal) and the headline are NOT changed."),
        "guardrails": {
            "rho_cal_frozen": rho_cal, "exceedance_cache_frozen": True,
            "production_default_resample": "row (i.i.d.)", "canonical_untouched": True,
        },
        "config": {
            "country": args.country, "n_outer": args.n_outer, "seed": args.seed,
            "bootstrap_B": B, "exceedance_B": B_exc, "taus": list(map(float, taus)),
            "red_threshold": red_thr, "amber_threshold": amber_thr, "holm_m": m,
            "holm_cut_per_composite": red_thr / m, "q_levels": list(QLEVELS),
            "headline_composite": COL_P_T_IUT, "smoke": args.smoke,
            "panel_sha256": fr.panel_sha256, "n_ref_C": fr.n_ref,
            "n_firms_C": fr.n_firms_c, "n_firms_obs": fr.n_firms_obs,
        },
        "canonical_selfcheck": {"red_count": red_canon, "disc": disc_canon},
        "schemes": {},
    }

    for scheme in ("row", "cluster"):
        print(f"[{scheme}] {args.n_outer} null draws ...", flush=True)
        result["schemes"][scheme] = run_scheme(
            scheme, fr, cache, taus, B, B_exc, red_thr, amber_thr, qkeys, m,
            args.n_outer, base_seed=args.seed + (0 if scheme == "row" else 100000))
        print(f"[{scheme}] {json.dumps(result['schemes'][scheme], indent=2)}", flush=True)

    rmean = lambda s, k: result["schemes"][s][k]["mean"]
    delta = {
        "critical_value_t_iut_rel": (rmean("cluster", "critical_value_t_iut")
                                     / rmean("row", "critical_value_t_iut") - 1.0),
        "critical_value_z_mahalanobis_rel": (rmean("cluster", "critical_value_z_mahalanobis")
                                             / rmean("row", "critical_value_z_mahalanobis") - 1.0),
        "row_red_count_abs": rmean("cluster", "row_red_count") - rmean("row", "row_red_count"),
    }
    for k in qkeys:
        delta[f"firm_disc_{k}_abs"] = (result["schemes"]["cluster"]["firm_discoveries"][k]["mean"]
                                       - result["schemes"]["row"]["firm_discoveries"][k]["mean"])
    result["cluster_vs_row_delta"] = delta

    out_path = Path(args.out) if args.out else Path(
        f"outputs/scores/{args.country}/cluster_bootstrap/cluster_bootstrap"
        f"{'_smoke' if args.smoke else ''}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
