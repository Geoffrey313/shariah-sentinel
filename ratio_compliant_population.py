"""Flag rate on the ratio-compliant population: the counts behind the reported shares.

Scope. The verdict section reports the RED share on the rows whose Compustat-recomputed
ratios satisfy the three authority caps (ratio_debt_adj/ratio_cash_adj <= 0.33,
ratio_income <= 0.05), split by the authority label. The shares were readable but their
two terms were not: the reference bundle ships the debt ratio only, so the three-cap
restriction is not recomputable from it, and the counts appeared in no output. This
script emits them, so every share in that paragraph carries a published numerator and
denominator, and the cross-authority flag-rate figure becomes regenerable.

Caps. The screened population is defined by the caps an authority actually applies,
via ``screening_thresholds_for_panel``: Indonesia screens no cash cap, and the
canonical registry carries a non-binding placeholder there that would drop one row.
The two readings coincide for the four other authorities.

It changes no verdict and no headline: the tri-state verdict is the canonical
Holm/Bonferroni-corrected rule of ``compute_verdict`` applied to the scored composites
as shipped, and the restriction only selects which rows are counted.

Requires the licensed panels (not self-contained); the composites come from the
reference bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.config import AnalysisSettings
from src.common.methodology import screening_thresholds_for_panel
from src.analysis.composite_scoring import VERDICT_P_COLS, compute_verdict
from src.analysis.reference_sample import resolve_label_column


def _ratio_compliant_mask(panel: pd.DataFrame, caps: dict[str, float]) -> pd.Series:
    """Rows that breach none of the caps; a missing ratio never counts as a breach."""
    ok = pd.Series(True, index=panel.index)
    for col, cap in caps.items():
        if col in panel.columns:
            ok &= ~(pd.to_numeric(panel[col], errors="coerce") > cap).fillna(False)
    return ok


def _share(n_red: int, n_rows: int) -> dict:
    return {"n_rows": int(n_rows), "n_red": int(n_red),
            "red_share": round(n_red / n_rows, 6) if n_rows else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="mys")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = AnalysisSettings()
    settings = base.model_copy(update={"output_layout": base.output_layout.model_copy(
        update={"country_code_lower": args.country.lower(),
                "root": str(Path(__file__).resolve().parent / "data")})})

    panel = pd.read_parquet(settings.output_layout.panel_path()).reset_index(drop=True)
    composites = pd.read_parquet(
        settings.output_layout.phase4_dir() / "composites_panel.parquet").reset_index(drop=True)
    quarter = settings.panel_schema.quarter
    if not (panel[quarter].astype(str).values == composites[quarter].astype(str).values).all():
        raise ValueError("panel and composites are not row-aligned on the quarter column.")

    caps = screening_thresholds_for_panel(panel)
    ok = _ratio_compliant_mask(panel, caps)
    red = compute_verdict(
        composites, p_cols=VERDICT_P_COLS,
        red_threshold=settings.figures.red_threshold,
        amber_threshold=settings.figures.amber_threshold,
    ).reset_index(drop=True) == "RED"

    label = resolve_label_column(panel, settings)
    labelled = pd.to_numeric(panel[label], errors="coerce").fillna(0) == 1

    result = {
        "scope": ("RED share on the ratio-compliant population: rows breaching none of the "
                  "authority caps, under the canonical tri-state verdict. Selects rows, "
                  "changes no verdict and no headline."),
        "caps": {k: float(v) for k, v in caps.items()},
        "label_column": label,
        "all": _share(int((red & ok).sum()), int(ok.sum())),
        "authority_labelled": _share(int((red & ok & labelled).sum()), int((ok & labelled).sum())),
        "unlabelled": _share(int((red & ok & ~labelled).sum()), int((ok & ~labelled).sum())),
        "panel_rows": int(len(panel)),
    }

    out = Path(args.out) if args.out else (
        settings.output_layout.phase4_dir().parent / "qualitative"
        / "ratio_compliant_population.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
