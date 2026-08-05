"""ShariaSentinel — paper-reproduction entry point.

Single argparse CLI to reproduce the anomaly-screening results from the
reconstructed per-country panels shipped under ``data/``. Run one analysis or
the whole pipeline. This repository is reproduction-only: no unit tests, no raw
data, no application/serving code.

Examples:
  python reproduce.py pipeline    --country mys
  python reproduce.py score       --country mys
  python reproduce.py benchmark   --country mys
  python reproduce.py family3-cf  --country mys --rows 500 --near-boundary
  python reproduce.py family3-pgd --country mys --rows 5
  python reproduce.py all         --country mys
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.common.config import AnalysisSettings  # noqa: E402

COUNTRIES = ("mys", "idn", "uae", "sau", "pak")
SAC4 = ("dlttq", "dlcq", "cheq", "iditq")  # SAC-ratio raw columns (PGD attack surface)


def settings_for(country: str) -> AnalysisSettings:
    """Settings pinned to this repo's reconstructed data root + country."""
    s = AnalysisSettings()
    ol = s.output_layout.model_copy(update={"country_code_lower": country, "root": str(REPO / "data")})
    return s.model_copy(update={"output_layout": ol})


def load_panel(settings: AnalysisSettings) -> pd.DataFrame:
    path = settings.output_layout.phase0_dir() / "panel_with_split.parquet"
    if not path.exists():
        sys.exit(f"reconstructed panel not found: {path}")
    return pd.read_parquet(path)


# ── subcommands ─────────────────────────────────────────────────────────────
def cmd_ablation(args) -> None:
    """Detector-ablation table: rerun the pipeline under the full model and the
    four ablations (no annual cash-flow recovery, no z5/z7 merge, no
    Monte-Carlo Benford, no empirical PIT) and write ablation_results.csv."""
    from src.analysis.ablation import run_ablation_study
    print(f"[ablation] {args.country}: running full model + 4 ablations", flush=True)
    run_ablation_study(args.country)
    print("[ablation] done", flush=True)


def cmd_pipeline(args) -> None:
    """Full phase 0-7 pipeline from the reconstructed panel: reference sample
    (with the ratio-compliance screen), z-scores, composites + bootstrap, and
    the firm-level FDR. This reproduces the flag rate, calibration, and
    false-discovery tables of the paper."""
    from src.analysis.pipeline import run_full_analysis
    s = settings_for(args.country)
    print(f"[pipeline] {args.country}: running phases 0-7", flush=True)
    run_full_analysis(settings=s)
    print("[pipeline] done", flush=True)



def cmd_score(args) -> None:
    from src.engine.zscores import compute_zscores
    s = settings_for(args.country)
    panel = load_panel(s)
    print(f"[score] {args.country}: panel {panel.shape}", flush=True)
    zc = compute_zscores(panel=panel, settings=s, write_outputs=True)
    out = s.output_layout.scores_dir() / "z_scores.parquet" if hasattr(s.output_layout, "scores_dir") else None
    print(f"[score] done — z-scores columns: {list(zc.zscores.columns)}", flush=True)


def cmd_benchmark(args) -> None:
    from src.analysis.benchmark import run_robustness_benchmark
    s = settings_for(args.country)
    panel = load_panel(s)
    print(f"[benchmark] {args.country}: running families 1-4 (this is slow)", flush=True)
    run_robustness_benchmark(panel=panel, settings=s, write_outputs=True, resume=not args.no_resume)
    print("[benchmark] done", flush=True)


def cmd_family3_cf(args) -> None:
    from src.analysis.benchmark import _score_panel_family3
    from src.analysis.counterfactual.core import run_family3_xai
    s = settings_for(args.country)
    rb = s.robustness_benchmark.model_copy(update={
        "family3_max_rows": args.rows,
        "family3_target_scores": (args.target,),
        "family3_directions": (args.direction,),
        "family3_modes": ("global_per_row",),
    })
    s = s.model_copy(update={"robustness_benchmark": rb})
    panel = load_panel(s)
    base_ctx = _score_panel_family3(panel, s)
    candidate_index = None
    if args.near_boundary:
        comp = base_ctx.composites.copy()
        comp["_p"] = pd.to_numeric(comp[rb.primary_composite], errors="coerce")
        candidate_index = comp[comp["_p"] < rb.alpha].sort_values("_p", ascending=False).index[: args.rows]
        print(f"[family3-cf] near-boundary RED cohort: {len(candidate_index)} rows", flush=True)
    summary, _ = run_family3_xai(panel, s, base_ctx, method_name="reproduction", candidate_index=candidate_index)
    n = int(summary["success"].sum()) if "success" in summary else 0
    print(f"[family3-cf] {summary.shape[0]} attacked, {n} flips "
          f"({100 * n / max(summary.shape[0], 1):.1f}%)", flush=True)
    out = s.output_layout.robustness_benchmark_dir() / f"family3_cf_{args.direction}_{args.target}_top{args.rows}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"[family3-cf] wrote {out}", flush=True)


def cmd_family3_pgd(args) -> None:
    from src.analysis.benchmark import _score_panel_family3
    from src.analysis.counterfactual.torch import run_family3_torch_surrogate
    s = settings_for(args.country)
    rb = s.robustness_benchmark
    panel = load_panel(s)
    ctx = _score_panel_family3(panel, s)
    comp = ctx.composites.copy()
    comp["_p"] = pd.to_numeric(comp[rb.primary_composite], errors="coerce")
    targets = list(comp[comp["_p"] < rb.alpha].sort_values("_p").index[: args.rows])
    print(f"[family3-pgd] SAC-projected PGD on {len(targets)} highest-risk RED rows", flush=True)
    eps_levels = [0.0, 0.05]
    evaded = {e: 0 for e in eps_levels}
    for idx in targets:
        for e in eps_levels:
            res, _ = run_family3_torch_surrogate(
                panel=panel, base_ctx=ctx, settings=s, row_index=idx,
                surrogate_mode="threshold", eps_grid=(e,), iterations=25,
                enforce_sac=True, modifiable_columns=SAC4)
            if res.final_score >= rb.alpha:
                evaded[e] += 1
    for e in eps_levels:
        print(f"[family3-pgd] eps={e}: {evaded[e]}/{len(targets)} evaded "
              f"({100 * evaded[e] / max(len(targets), 1):.1f}%)", flush=True)


def cmd_all(args) -> None:
    cmd_score(args)
    cmd_benchmark(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--country", default="mys", choices=COUNTRIES)

    sp = sub.add_parser("pipeline", help="full phase 0-7 (ref sample, z-scores, composites, FDR)"); add_common(sp); sp.set_defaults(func=cmd_pipeline)
    sp = sub.add_parser("ablation", help="detector-ablation table (full model + 4 ablations)"); add_common(sp); sp.set_defaults(func=cmd_ablation)
    sp = sub.add_parser("score", help="compute detector z-scores"); add_common(sp); sp.set_defaults(func=cmd_score)
    sp = sub.add_parser("benchmark", help="robustness benchmark (families 1-4)"); add_common(sp)
    sp.add_argument("--no-resume", action="store_true"); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("family3-cf", help="Family-3 counterfactual (RED->GREEN)"); add_common(sp)
    sp.add_argument("--rows", type=int, default=500); sp.add_argument("--target", default="z_plus_renorm")
    sp.add_argument("--direction", default="to_green"); sp.add_argument("--near-boundary", action="store_true")
    sp.set_defaults(func=cmd_family3_cf)
    sp = sub.add_parser("family3-pgd", help="Family-3 SAC-projected PGD evasion"); add_common(sp)
    sp.add_argument("--rows", type=int, default=5); sp.set_defaults(func=cmd_family3_pgd)
    sp = sub.add_parser("all", help="score + benchmark"); add_common(sp)
    sp.add_argument("--no-resume", action="store_true"); sp.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
