"""Contamination mechanisms for the robustness benchmark.

Row-level manipulations injected into raw financial variables to measure
detection AUC against ground-truth labels:

- M1: threshold projection with bounded debt adjustment
- M2: temporal spike and gap mechanism
- M3: Benford distortion by distribution-aware first-digit forcing
- M4: inter-statement inconsistency through raw relation shocks
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.common.methodology import thresholds_for_panel
from src.data.ratios import compute_shariah_ratios

log = logging.getLogger(__name__)

EPS = 1e-9
BENFORD_PROBS = np.log10(1.0 + 1.0 / np.arange(1, 10))
ROW_LEVEL_MECHANISMS = frozenset({"m1", "m3", "m4"})


@dataclass(frozen=True)
class ContaminationResult:
    df: pd.DataFrame
    mechanism: str
    label_col: str
    rate_target: float
    rate_effective: float
    n_total: int
    n_contaminated: int
    details: dict[str, float | int | str]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = num.notna() & den.notna() & (den.abs() > EPS)
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def _clip_rate(rate: float) -> float:
    return float(np.clip(rate, 0.0, 1.0))


def _first_digit(value: float) -> int | None:
    if not np.isfinite(value) or value == 0:
        return None
    x = abs(float(value))
    while x >= 10:
        x /= 10.0
    while x < 1:
        x *= 10.0
    digit = int(x)
    return digit if 1 <= digit <= 9 else None


def _force_first_digit(value: float, target_digit: int, rng: np.random.Generator) -> float:
    """Preserve sign and order of magnitude while changing the leading digit."""
    sign = -1.0 if value < 0 else 1.0
    x = abs(float(value))
    if not np.isfinite(x) or x <= 0:
        return value
    exponent = np.floor(np.log10(x))
    scale = 10.0 ** exponent
    # Keep the value inside the target-digit bucket rather than collapsing
    # every modification to the same mantissa.
    mantissa = target_digit + rng.uniform(0.05, 0.95)
    return sign * mantissa * scale


def _recompute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    return compute_shariah_ratios(df)


def _threshold_values(df: pd.DataFrame) -> tuple[float, float, float]:
    thresholds = thresholds_for_panel(df)
    return (
        thresholds["ratio_debt_adj"],
        thresholds["ratio_cash_adj"],
        thresholds["ratio_income"],
    )


def _candidate_rows_m1(df: pd.DataFrame) -> pd.Index:
    threshold_debt, threshold_cash, threshold_income = _threshold_values(df)
    ratios = df[["ratio_debt_adj", "ratio_cash_adj", "ratio_income"]].apply(pd.to_numeric, errors="coerce")
    candidate_mask = (
        ratios["ratio_debt_adj"].gt(threshold_debt).fillna(False)
        | ratios["ratio_cash_adj"].gt(threshold_cash).fillna(False)
        | ratios["ratio_income"].gt(threshold_income).fillna(False)
    )
    return pd.Index(df.index[candidate_mask])


def _candidate_rows_m3(df: pd.DataFrame, *, candidate_columns: tuple[str, ...], restrict_to_positive_income_flag: bool) -> pd.Index:
    present_columns = [col for col in candidate_columns if col in df.columns]
    if not present_columns:
        return pd.Index([])
    row_pool = pd.Index(df.index)
    if restrict_to_positive_income_flag and "ratio_income" in df.columns:
        _, _, threshold_income = _threshold_values(df)
        row_pool = pd.Index(df.index[df["ratio_income"].gt(threshold_income).fillna(False)])
        if len(row_pool) == 0:
            row_pool = pd.Index(df.index)
    return row_pool


def _candidate_rows_m4(df: pd.DataFrame, *, relation: str) -> pd.Index:
    relation_specs = {
        "interest_to_debt": ("xintq", "dlttq", "dlcq"),
        "earnings_gap": ("niq", "oibdpq"),
        "working_capital_proxy": ("actq", "cheq", "lctq", "dlcq"),
    }
    required = relation_specs[relation]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        return pd.Index([])
    numeric = df[list(required)].apply(pd.to_numeric, errors="coerce")
    return pd.Index(df.index[numeric.notna().all(axis=1)])


def _finalize_result(
    df: pd.DataFrame,
    *,
    mechanism: str,
    label_col: str,
    rate_target: float,
    details: dict[str, float | int | str],
) -> ContaminationResult:
    df = _recompute_ratios(df)
    n_total = len(df)
    n_contaminated = int(df[label_col].fillna(0).astype(int).sum())
    rate_effective = float(n_contaminated / n_total) if n_total else 0.0
    return ContaminationResult(
        df=df,
        mechanism=mechanism,
        label_col=label_col,
        rate_target=rate_target,
        rate_effective=rate_effective,
        n_total=n_total,
        n_contaminated=n_contaminated,
        details=details,
    )


def contaminate_m1_v2(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    slack_max: float = 0.015,
) -> ContaminationResult:
    """Project non-compliant rows toward the nearest feasible threshold.

    Debt adjustment uses the strongest feasible modification. When the exact
    threshold-implied debt would require a negative ``dlttq``, we project to
    the feasible boundary (``dlttq = 0``) instead of skipping the row.
    """
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_m1_v2"
    out = df.copy()
    out[label_col] = 0

    required = {"atq", "dlttq", "dlcq", "cheq", "revtq", "iditq"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"M1 v2 missing required columns: {missing}")

    candidates = _candidate_rows_m1(out).to_numpy()
    n_target = min(int(round(rate * len(out))), len(candidates))
    if n_target <= 0:
        return _finalize_result(
            out,
            mechanism="m1_v2",
            label_col=label_col,
            rate_target=rate,
            details={"status": "no candidates"},
        )

    selected = pd.Index(rng.choice(candidates, size=n_target, replace=False))
    threshold_debt, threshold_cash, threshold_income = _threshold_values(out)
    debt_projected = 0
    threshold_hits = 0

    for idx in selected:
        row = out.loc[idx]
        touched = False
        slack = float(rng.uniform(0.0, slack_max))

        if pd.notna(row.get("ratio_debt_adj")) and row["ratio_debt_adj"] > threshold_debt:
            atq = float(row["atq"])
            dlcq = float(row["dlcq"])
            sukuk_ratio = float(row.get("sukuk_ratio_t", 0.0) or 0.0)
            debt_factor = max(1.0 - sukuk_ratio, EPS)
            target_adjusted = max((threshold_debt - slack) * atq - dlcq, 0.0)
            raw_target = target_adjusted / debt_factor
            if raw_target < 0.0:
                raw_target = 0.0
                debt_projected += 1
            if raw_target == 0.0:
                debt_projected += 1
            out.at[idx, "dlttq"] = raw_target
            touched = True

        if pd.notna(row.get("ratio_cash_adj")) and row["ratio_cash_adj"] > threshold_cash:
            atq = float(row["atq"])
            islamic_cash_ratio = float(row.get("islamic_cash_ratio_t", 0.0) or 0.0)
            cash_factor = max(1.0 - islamic_cash_ratio, EPS)
            raw_target = max(((threshold_cash - slack) * atq) / cash_factor, 0.0)
            out.at[idx, "cheq"] = raw_target
            touched = True

        if pd.notna(row.get("ratio_income")) and row["ratio_income"] > threshold_income:
            revtq = float(row["revtq"])
            raw_target = max((threshold_income - slack) * revtq, 0.0)
            out.at[idx, "iditq"] = raw_target
            touched = True

        if touched:
            out.at[idx, label_col] = 1
            threshold_hits += 1

    return _finalize_result(
        out,
        mechanism="m1_v2",
        label_col=label_col,
        rate_target=rate,
        details={
            "n_selected": n_target,
            "n_touched": threshold_hits,
            "debt_projected_to_boundary": debt_projected,
        },
    )


def contaminate_m2b(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    jump_scale: float = 0.20,
    id_col: str = "gvkey",
    time_col: str = "datacqtr",
) -> ContaminationResult:
    """Create a local temporal spike / gap rather than smoothing.

    This is intentionally a new mechanism distinct from M2. It targets temporal
    discontinuities by shocking one quarter inside the recent history of a firm.
    """
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_m2b"
    out = df.copy()
    out[label_col] = 0

    required = {id_col, time_col, "iditq", "revtq"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"M2b missing required columns: {missing}")

    ordered = out.sort_values([id_col, time_col]).copy()
    firms = []
    for gid, grp in ordered.groupby(id_col, sort=False):
        if len(grp) < 4:
            continue
        firms.append((gid, grp.index.to_list()))

    if not firms:
        return _finalize_result(
            ordered,
            mechanism="m2b",
            label_col=label_col,
            rate_target=rate,
            details={"status": "no eligible firms"},
        )

    target_rows = max(1, int(round(rate * len(ordered))))
    shuffled = rng.permutation(len(firms))
    touched_rows = 0
    touched_firms = 0

    for pos in shuffled:
        gid, idxs = firms[int(pos)]
        tail = idxs[-4:]
        spike_idx = tail[int(rng.integers(1, len(tail) - 1))]
        row = ordered.loc[spike_idx]

        if not np.isfinite(row["revtq"]) or abs(float(row["revtq"])) <= EPS:
            continue

        direction = 1.0 if rng.random() < 0.5 else -1.0
        base_income = float(row["iditq"])
        amplitude = max(abs(base_income), abs(float(row["revtq"])) * 0.01) * jump_scale
        ordered.at[spike_idx, "iditq"] = base_income + direction * amplitude
        ordered.at[spike_idx, label_col] = 1
        touched_rows += 1
        touched_firms += 1
        if touched_rows >= target_rows:
            break

    return _finalize_result(
        ordered,
        mechanism="m2b",
        label_col=label_col,
        rate_target=rate,
        details={
            "n_target_rows": target_rows,
            "n_touched_firms": touched_firms,
            "jump_scale": jump_scale,
        },
    )


def contaminate_m3_v2(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    candidate_columns: tuple[str, ...] = ("revtq", "atq", "iditq", "dlttq"),
    restrict_to_positive_income_flag: bool = True,
) -> ContaminationResult:
    """Distort first-digit frequencies without collapsing to a single target.

    The mechanism increases the global deviation from Benford by forcing values
    toward digits that are already overrepresented relative to Benford. When no
    clear excess exists, it falls back to a high-digit bias on {7, 8, 9}.
    """
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_m3_v2"
    out = df.copy()
    out[label_col] = 0

    present_columns = [col for col in candidate_columns if col in out.columns]
    if not present_columns:
        raise ValueError("M3 v2 has no candidate columns available in the dataframe.")

    row_pool = _candidate_rows_m3(
        out,
        candidate_columns=candidate_columns,
        restrict_to_positive_income_flag=restrict_to_positive_income_flag,
    )

    target_rows = min(max(1, int(round(rate * len(out)))), len(row_pool))
    selected = pd.Index(rng.choice(np.asarray(row_pool), size=target_rows, replace=False))

    # Estimate the current first-digit distribution on all candidate values.
    digits: list[int] = []
    for col in present_columns:
        values = pd.to_numeric(out[col], errors="coerce")
        digits.extend(d for d in values.map(_first_digit).dropna().astype(int).tolist())

    empirical = np.zeros(9, dtype=float)
    if digits:
        counts = np.bincount(np.asarray(digits), minlength=10)[1:10].astype(float)
        empirical = counts / counts.sum()
    excess = empirical - BENFORD_PROBS
    if np.all(~np.isfinite(excess)) or np.nanmax(excess) <= 0:
        target_digits = np.array([7, 8, 9], dtype=int)
        weights = np.array([0.2, 0.3, 0.5], dtype=float)
    else:
        target_digits = np.arange(1, 10, dtype=int)
        weights = np.clip(excess, 0.0, None)
        if weights.sum() <= 0:
            weights = np.array([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.20, 0.30], dtype=float)
        weights = weights / weights.sum()

    touched = 0
    for idx in selected:
        eligible_cols = []
        for col in present_columns:
            value = pd.to_numeric(pd.Series([out.at[idx, col]]), errors="coerce").iat[0]
            if np.isfinite(value) and value != 0:
                eligible_cols.append((col, float(value)))
        if not eligible_cols:
            continue
        col, value = eligible_cols[int(rng.integers(0, len(eligible_cols)))]
        target_digit = int(rng.choice(target_digits, p=weights))
        out.at[idx, col] = _force_first_digit(value, target_digit, rng)
        out.at[idx, label_col] = 1
        touched += 1

    return _finalize_result(
        out,
        mechanism="m3_v2",
        label_col=label_col,
        rate_target=rate,
        details={
            "n_target_rows": target_rows,
            "n_touched": touched,
            "candidate_columns": ",".join(present_columns),
        },
    )


def contaminate_m4_v2(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    delta_scale: float = 1.0,
    relation: Literal["interest_to_debt", "earnings_gap", "working_capital_proxy"] = "interest_to_debt",
) -> ContaminationResult:
    """Create a true inter-statement inconsistency by shocking one side only."""
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_m4_v2"
    out = df.copy()
    out[label_col] = 0

    relation_specs = {
        "interest_to_debt": {"required": ("xintq", "dlttq", "dlcq"), "shock_col": "xintq"},
        "earnings_gap": {"required": ("niq", "oibdpq"), "shock_col": "niq"},
        "working_capital_proxy": {"required": ("actq", "cheq", "lctq", "dlcq"), "shock_col": "actq"},
    }
    spec = relation_specs[relation]
    missing = sorted(set(spec["required"]) - set(out.columns))
    if missing:
        raise ValueError(f"M4 v2 missing required columns for {relation}: {missing}")

    candidates = _candidate_rows_m4(out, relation=relation).to_numpy()
    n_target = min(max(1, int(round(rate * len(out)))), len(candidates))
    if n_target <= 0:
        return _finalize_result(
            out,
            mechanism="m4_v2",
            label_col=label_col,
            rate_target=rate,
            details={"status": "no candidates", "relation": relation},
        )

    selected = pd.Index(rng.choice(candidates, size=n_target, replace=False))
    shock_col = spec["shock_col"]
    scale = float(pd.to_numeric(out[shock_col], errors="coerce").std(ddof=0))
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    shock = max(scale * delta_scale, EPS)

    for idx in selected:
        out.at[idx, shock_col] = float(out.at[idx, shock_col]) + shock
        out.at[idx, label_col] = 1

    return _finalize_result(
        out,
        mechanism="m4_v2",
        label_col=label_col,
        rate_target=rate,
        details={
            "relation": relation,
            "shock_col": shock_col,
            "shock_size": shock,
            "n_selected": n_target,
        },
    )


def contaminate_abn_disx_partial(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    delta: float = 0.15,
    expense_col: str = "xsgaq",
    earnings_col: str = "oibdpq",
) -> ContaminationResult:
    """Partial REM discretionary-expense cut using SG&A only.

    The panel does not currently expose R&D (`xrdq`) consistently, so this
    v1 mechanism targets the available discretionary-expense channel:

        xsgaq' = xsgaq * (1 - delta)
        oibdpq' = oibdpq + xsgaq * delta

    This preserves the operating-earnings interpretation better than editing
    the final ratio directly.
    """
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_abn_disx_partial"
    out = df.copy()
    out[label_col] = 0

    required = {"atq", expense_col, earnings_col}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"ABN_DISX partial missing required columns: {missing}")

    numeric = out[list(required)].apply(pd.to_numeric, errors="coerce")
    candidates = pd.Index(out.index[numeric.notna().all(axis=1)])
    n_target = min(max(1, int(round(rate * len(out)))), len(candidates))
    if n_target <= 0:
        return _finalize_result(
            out,
            mechanism="abn_disx_partial",
            label_col=label_col,
            rate_target=rate,
            details={"status": "no candidates"},
        )

    selected = pd.Index(rng.choice(candidates.to_numpy(), size=n_target, replace=False))
    touched = 0
    for idx in selected:
        xsga = float(out.at[idx, expense_col])
        if not np.isfinite(xsga) or xsga <= 0:
            continue
        cut = xsga * delta
        out.at[idx, expense_col] = max(xsga - cut, 0.0)
        out.at[idx, earnings_col] = float(out.at[idx, earnings_col]) + cut
        out.at[idx, label_col] = 1
        touched += 1

    return _finalize_result(
        out,
        mechanism="abn_disx_partial",
        label_col=label_col,
        rate_target=rate,
        details={
            "n_selected": n_target,
            "n_touched": touched,
            "delta": delta,
            "expense_col": expense_col,
            "earnings_col": earnings_col,
        },
    )


def contaminate_abn_disx_full(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    delta: float = 0.15,
    sga_col: str = "xsgaq",
    rnd_col: str = "xrdq",
    earnings_col: str = "oibdpq",
) -> ContaminationResult:
    """Full REM discretionary-expense cut using SG&A plus R&D when available.

    The cut is applied proportionally across the two discretionary-expense
    channels and added back to operating earnings.
    """
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_abn_disx_full"
    out = df.copy()
    out[label_col] = 0

    required = {"atq", sga_col, rnd_col, earnings_col}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"ABN_DISX full missing required columns: {missing}")

    numeric = out[list(required)].apply(pd.to_numeric, errors="coerce")
    candidates = pd.Index(out.index[numeric.notna().all(axis=1)])
    n_target = min(max(1, int(round(rate * len(out)))), len(candidates))
    if n_target <= 0:
        return _finalize_result(
            out,
            mechanism="abn_disx_full",
            label_col=label_col,
            rate_target=rate,
            details={"status": "no candidates"},
        )

    selected = pd.Index(rng.choice(candidates.to_numpy(), size=n_target, replace=False))
    touched = 0
    for idx in selected:
        xsga = float(out.at[idx, sga_col])
        xrd = float(out.at[idx, rnd_col])
        total_disx = xsga + xrd
        if not np.isfinite(total_disx) or total_disx <= 0:
            continue
        cut_total = total_disx * delta
        xsga_share = xsga / max(total_disx, EPS)
        xrd_share = xrd / max(total_disx, EPS)
        out.at[idx, sga_col] = max(xsga - cut_total * xsga_share, 0.0)
        out.at[idx, rnd_col] = max(xrd - cut_total * xrd_share, 0.0)
        out.at[idx, earnings_col] = float(out.at[idx, earnings_col]) + cut_total
        out.at[idx, label_col] = 1
        touched += 1

    return _finalize_result(
        out,
        mechanism="abn_disx_full",
        label_col=label_col,
        rate_target=rate,
        details={
            "n_selected": n_target,
            "n_touched": touched,
            "delta": delta,
            "sga_col": sga_col,
            "rnd_col": rnd_col,
            "earnings_col": earnings_col,
        },
    )


def contaminate_abn_prod_full(
    df: pd.DataFrame,
    *,
    rate: float,
    random_state: int = 42,
    delta: float = 0.15,
    cogs_col: str = "cogsq",
    inventory_col: str = "invtq",
    revenue_col: str = "revtq",
) -> ContaminationResult:
    """Approximate an overproduction-style REM shock on COGS and inventory.

    A fraction of contemporaneous revenue is shifted from current-period COGS
    into inventory, which mimics temporary margin inflation through inventory
    build-up.
    """
    rng = np.random.default_rng(random_state)
    rate = _clip_rate(rate)
    label_col = "_contaminated_abn_prod_full"
    out = df.copy()
    out[label_col] = 0

    required = {cogs_col, inventory_col, revenue_col}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"ABN_PROD full missing required columns: {missing}")

    numeric = out[list(required)].apply(pd.to_numeric, errors="coerce")
    candidates = pd.Index(out.index[numeric.notna().all(axis=1)])
    n_target = min(max(1, int(round(rate * len(out)))), len(candidates))
    if n_target <= 0:
        return _finalize_result(
            out,
            mechanism="abn_prod_full",
            label_col=label_col,
            rate_target=rate,
            details={"status": "no candidates"},
        )

    selected = pd.Index(rng.choice(candidates.to_numpy(), size=n_target, replace=False))
    touched = 0
    for idx in selected:
        cogs = float(out.at[idx, cogs_col])
        invt = float(out.at[idx, inventory_col])
        revt = float(out.at[idx, revenue_col])
        if not np.isfinite(cogs) or not np.isfinite(invt) or not np.isfinite(revt):
            continue
        if cogs <= 0 or revt <= 0:
            continue
        inventory_build = min(cogs * 0.5, revt * delta)
        if inventory_build <= 0:
            continue
        out.at[idx, cogs_col] = max(cogs - inventory_build, 0.0)
        out.at[idx, inventory_col] = invt + inventory_build
        out.at[idx, label_col] = 1
        touched += 1

    return _finalize_result(
        out,
        mechanism="abn_prod_full",
        label_col=label_col,
        rate_target=rate,
        details={
            "n_selected": n_target,
            "n_touched": touched,
            "delta": delta,
            "cogs_col": cogs_col,
            "inventory_col": inventory_col,
            "revenue_col": revenue_col,
        },
    )


def apply_contamination(
    df: pd.DataFrame,
    *,
    mechanism: str,
    rate: float,
    random_state: int = 42,
    **kwargs,
) -> ContaminationResult:
    mechanism = mechanism.lower()
    if mechanism == "m1":
        return contaminate_m1_v2(df, rate=rate, random_state=random_state, **kwargs)
    if mechanism == "m2b":
        return contaminate_m2b(df, rate=rate, random_state=random_state, **kwargs)
    if mechanism == "m3":
        return contaminate_m3_v2(df, rate=rate, random_state=random_state, **kwargs)
    if mechanism == "m4":
        return contaminate_m4_v2(df, rate=rate, random_state=random_state, **kwargs)
    if mechanism == "abn_disx_partial":
        return contaminate_abn_disx_partial(df, rate=rate, random_state=random_state, **kwargs)
    if mechanism == "abn_disx_full":
        return contaminate_abn_disx_full(df, rate=rate, random_state=random_state, **kwargs)
    if mechanism == "abn_prod_full":
        return contaminate_abn_prod_full(df, rate=rate, random_state=random_state, **kwargs)
    raise ValueError(f"Unknown contamination mechanism: {mechanism}")


def apply_contamination_bundle(
    df: pd.DataFrame,
    *,
    mechanisms: list[str],
    rate: float,
    random_state: int = 42,
    shared_lines: bool = True,
) -> ContaminationResult:
    """Apply multiple row-level mechanisms on the same selected lines.

    For now, bundles are restricted to row-level mechanisms (`m1`, `m3`, `m4`).
    `m2b` remains separate because it operates on temporal firm sequences.
    """
    cleaned = [m.lower() for m in mechanisms]
    unknown = [m for m in cleaned if m not in {"m1", "m2b", "m3", "m4"}]
    if unknown:
        raise ValueError(f"Unknown bundle mechanisms: {unknown}")
    if any(m not in ROW_LEVEL_MECHANISMS for m in cleaned):
        raise ValueError("Bundle mode currently supports only row-level mechanisms: m1, m3, m4.")
    if not cleaned:
        raise ValueError("At least one mechanism is required.")

    rng = np.random.default_rng(random_state)
    out = df.copy()
    bundle_label = "_contaminated_bundle_v2"
    out[bundle_label] = 0

    candidate_sets: dict[str, pd.Index] = {
        "m1": _candidate_rows_m1(out),
        "m3": _candidate_rows_m3(out, candidate_columns=("revtq", "atq", "iditq", "dlttq"), restrict_to_positive_income_flag=True),
        "m4": _candidate_rows_m4(out, relation="interest_to_debt"),
    }
    working = [set(candidate_sets[m].tolist()) for m in cleaned]
    if shared_lines:
        common = set.intersection(*working) if working else set()
        pool = pd.Index(sorted(common))
    else:
        union = set.union(*working) if working else set()
        pool = pd.Index(sorted(union))

    target_rows = min(max(1, int(round(_clip_rate(rate) * len(out)))), len(pool))
    if target_rows <= 0:
        return _finalize_result(
            out,
            mechanism="+".join(cleaned) + "_bundle_v2",
            label_col=bundle_label,
            rate_target=rate,
            details={"status": "no shared candidates", "shared_lines": str(shared_lines)},
        )

    selected = pd.Index(rng.choice(pool.to_numpy(), size=target_rows, replace=False))

    # Apply selected mechanisms on the same selected rows. These are deliberately
    # simple row-level transforms that mirror the single-mechanism v2 logic.
    per_mechanism_counts: dict[str, int] = {}
    threshold_debt, threshold_cash, threshold_income = _threshold_values(out)
    for mech in cleaned:
        label_col = f"_contaminated_{mech}_bundle_v2"
        out[label_col] = 0
        touched = 0

        if mech == "m1":
            for idx in selected:
                row = out.loc[idx]
                changed = False
                slack = float(rng.uniform(0.0, 0.015))
                if pd.notna(row.get("ratio_debt_adj")) and row["ratio_debt_adj"] > threshold_debt:
                    atq = float(row["atq"])
                    dlcq = float(row["dlcq"])
                    sukuk_ratio = float(row.get("sukuk_ratio_t", 0.0) or 0.0)
                    debt_factor = max(1.0 - sukuk_ratio, EPS)
                    target_adjusted = max((threshold_debt - slack) * atq - dlcq, 0.0)
                    out.at[idx, "dlttq"] = max(target_adjusted / debt_factor, 0.0)
                    changed = True
                if pd.notna(row.get("ratio_cash_adj")) and row["ratio_cash_adj"] > threshold_cash:
                    atq = float(row["atq"])
                    islamic_cash_ratio = float(row.get("islamic_cash_ratio_t", 0.0) or 0.0)
                    cash_factor = max(1.0 - islamic_cash_ratio, EPS)
                    out.at[idx, "cheq"] = max(((threshold_cash - slack) * atq) / cash_factor, 0.0)
                    changed = True
                if pd.notna(row.get("ratio_income")) and row["ratio_income"] > threshold_income:
                    revtq = float(row["revtq"])
                    out.at[idx, "iditq"] = max((threshold_income - slack) * revtq, 0.0)
                    changed = True
                if changed:
                    out.at[idx, label_col] = 1
                    touched += 1

        elif mech == "m3":
            digits: list[int] = []
            for col in ("revtq", "atq", "iditq", "dlttq"):
                if col in out.columns:
                    digits.extend(d for d in pd.to_numeric(out[col], errors="coerce").map(_first_digit).dropna().astype(int).tolist())
            empirical = np.zeros(9, dtype=float)
            if digits:
                counts = np.bincount(np.asarray(digits), minlength=10)[1:10].astype(float)
                empirical = counts / counts.sum()
            excess = empirical - BENFORD_PROBS
            weights = np.clip(excess, 0.0, None)
            if weights.sum() <= 0:
                target_digits = np.array([7, 8, 9], dtype=int)
                weights = np.array([0.2, 0.3, 0.5], dtype=float)
            else:
                target_digits = np.arange(1, 10, dtype=int)
                weights = weights / weights.sum()
            for idx in selected:
                candidates = []
                for col in ("revtq", "atq", "iditq", "dlttq"):
                    if col not in out.columns:
                        continue
                    value = pd.to_numeric(pd.Series([out.at[idx, col]]), errors="coerce").iat[0]
                    if np.isfinite(value) and value != 0:
                        candidates.append((col, float(value)))
                if not candidates:
                    continue
                col, value = candidates[int(rng.integers(0, len(candidates)))]
                digit = int(rng.choice(target_digits, p=weights))
                out.at[idx, col] = _force_first_digit(value, digit, rng)
                out.at[idx, label_col] = 1
                touched += 1

        elif mech == "m4":
            shock_scale = float(pd.to_numeric(out["xintq"], errors="coerce").std(ddof=0)) if "xintq" in out.columns else 1.0
            if not np.isfinite(shock_scale) or shock_scale <= EPS:
                shock_scale = 1.0
            shock = shock_scale
            for idx in selected:
                if "xintq" not in out.columns or pd.isna(out.at[idx, "xintq"]):
                    continue
                out.at[idx, "xintq"] = float(out.at[idx, "xintq"]) + shock
                out.at[idx, label_col] = 1
                touched += 1

        per_mechanism_counts[f"{mech}_touched"] = touched
        out[bundle_label] = out[bundle_label] | out[label_col].astype(int)
        out = _recompute_ratios(out)

    out[bundle_label] = out[bundle_label].astype(int)
    return _finalize_result(
        out,
        mechanism="+".join(cleaned) + "_bundle",
        label_col=bundle_label,
        rate_target=rate,
        details={
            "shared_lines": str(shared_lines),
            "n_selected_shared_rows": target_rows,
            "mechanisms": ",".join(cleaned),
            **per_mechanism_counts,
        },
    )
