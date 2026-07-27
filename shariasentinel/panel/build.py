
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .mapping import build_mapping_compustat, diagnose_missing_sc_codes

log = logging.getLogger(__name__)

_OANCFQ_REQUIRED: set[str] = {"gvkey", "fyearq", "fqtr", "oancfy"}


def derive_oancfq(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``oancfq`` derived from the YTD cumulative ``oancfy``.

    For each ``(gvkey, fyearq)`` group:
    - Q1 with observed ``oancfy``: ``oancfq = oancfy``
    - later quarters with the immediately preceding quarter observed:
      ``oancfq_t = oancfy_t - oancfy_{t-1}``
    - otherwise: ``oancfq = NaN`` because the exact quarterly flow is not
      identifiable from the cumulative series alone.
    """
    missing = _OANCFQ_REQUIRED - set(df.columns)
    if missing:
        log.warning("derive_oancfq: missing columns %s — oancfq set to NaN.", sorted(missing))
        out = df.copy()
        out["oancfq"] = np.nan
        return out

    if df["oancfy"].isna().all():
        log.warning("derive_oancfq: oancfy all-NaN — oancfq set to NaN.")
        out = df.copy()
        out["oancfq"] = np.nan
        return out

    out = df.copy().sort_values(["gvkey", "fyearq", "fqtr"])
    grouped = out.groupby(["gvkey", "fyearq"], sort=False)

    prev_oancfy = grouped["oancfy"].shift(1)
    prev_fqtr = grouped["fqtr"].shift(1)
    curr_fqtr = pd.to_numeric(out["fqtr"], errors="coerce")

    is_q1 = curr_fqtr.eq(1)
    has_immediate_prev_quarter = prev_fqtr.eq(curr_fqtr - 1)
    curr_oancfy = pd.to_numeric(out["oancfy"], errors="coerce")

    out["oancfq"] = np.nan
    out.loc[is_q1, "oancfq"] = curr_oancfy.loc[is_q1]

    exact_diff_mask = (
        ~is_q1
        & has_immediate_prev_quarter
        & curr_oancfy.notna()
        & prev_oancfy.notna()
    )
    out.loc[exact_diff_mask, "oancfq"] = (
        curr_oancfy.loc[exact_diff_mask] - prev_oancfy.loc[exact_diff_mask]
    )
    out = out.reindex(df.index)

    n_src = int(df["oancfy"].notna().sum())
    n_out = int(out["oancfq"].notna().sum())
    n_q1_exact = int((is_q1 & curr_oancfy.notna()).sum())
    n_diff_exact = int(exact_diff_mask.sum())
    n_unresolved = int(curr_oancfy.notna().sum() - n_q1_exact - n_diff_exact)
    log.info(
        "derive_oancfq: oancfy non-null=%d → oancfq non-null=%d (%.1f%%); "
        "exact_q1=%d exact_diff=%d unresolved=%d.",
        n_src,
        n_out,
        100 * n_out / max(n_src, 1),
        n_q1_exact,
        n_diff_exact,
        n_unresolved,
    )
    return out


def build_panel_sac_anchored(
    df_comp: pd.DataFrame,
    df_sc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a quarterly panel anchored on SAC Malaysia evaluation dates.

    Business rules:
    - Keep all quarterly rows for firms that were Shariah-compliant at least once.
    - For each SAC list, attach the status to the last available fiscal close
      prior to or on the list date (priority: Q4 > Q3 > Q2 > Q1).
    - sac_shariah = 1 if Shariah-Compliant, 0 otherwise.

    Output: 1 row = 1 gvkey x datacqtr
    """
    mapping = build_mapping_compustat(df_comp)

    # ------------------------------------------------------------------
    # 1) SC universe + mapping
    # ------------------------------------------------------------------
    sc = df_sc.copy()
    sc["date_effective"] = pd.to_datetime(sc["date_effective"], errors="coerce")
    sc["stock_code_z4"] = sc["stock_code"].astype(int).astype(str).str.zfill(4)

    sc_map = sc.merge(
        mapping[["gvkey", "bursa_code_flex"]],
        left_on="stock_code_z4",
        right_on="bursa_code_flex",
        how="left",
    )
    sc_map["gvkey"] = pd.to_numeric(sc_map["gvkey"], errors="coerce")

    # Firms ever compliant -> final universe
    first_compliant = (
        sc_map.loc[sc_map["shariah_status"] == "Shariah-Compliant"]
        .dropna(subset=["gvkey", "date_effective"])
        .groupby("gvkey")["date_effective"]
        .min()
    )

    # ------------------------------------------------------------------
    # 2) Compustat quarterly panel
    # ------------------------------------------------------------------
    comp = df_comp.copy()
    comp["gvkey"] = pd.to_numeric(comp["gvkey"], errors="coerce")
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")

    # Deduplicate on (gvkey, datacqtr)
    comp = (
        comp.sort_values("datadate")
            .drop_duplicates(subset=["gvkey", "datacqtr"], keep="last")
            .copy()
    )

    comp["panel_key"] = comp["gvkey"].astype(str) + "__" + comp["datacqtr"].astype(str)

    # Keep all quarterly rows for firms that were ever compliant
    comp = comp[comp["gvkey"].isin(first_compliant.index)].copy()

    # Fiscal columns required for the anchor logic
    if "fyearq" in comp.columns:
        comp["fiscal_year"] = pd.to_numeric(comp["fyearq"], errors="coerce")
    else:
        comp["fiscal_year"] = comp["datadate"].dt.year

    if "fqtr" not in comp.columns:
        raise ValueError("Column 'fqtr' is required for Geoffrey logic.")

    comp["fqtr_num"] = pd.to_numeric(comp["fqtr"], errors="coerce")

    # ------------------------------------------------------------------
    # 3) Build the annual anchor row per firm-year
    #    Priority: Q4 > Q3 > Q2 > Q1
    # ------------------------------------------------------------------
    anchor_rows = (
        comp.dropna(subset=["gvkey", "fiscal_year", "fqtr_num", "datadate"])
            .sort_values(
                ["gvkey", "fiscal_year", "fqtr_num", "datadate"],
                ascending=[True, True, False, False],
                kind="mergesort",
            )
            .drop_duplicates(subset=["gvkey", "fiscal_year"], keep="first")
            .copy()
    )

    # Keep only valid anchor rows; updates will be written back via stable panel_key
    anchor_rows = anchor_rows.dropna(subset=["gvkey", "datadate"]).copy()

    # ------------------------------------------------------------------
    # 4) For each SAC event, find the last fiscal close <= list date
    # ------------------------------------------------------------------
    sc_events = (
        sc_map.dropna(subset=["gvkey", "date_effective"])
        .copy()
        .assign(gvkey=lambda x: x["gvkey"].astype("int64"))
        .sort_values(["date_effective", "gvkey"])
        .reset_index(drop=True)
    )

    anchor_rows["gvkey"] = anchor_rows["gvkey"].astype("int64")
    anchor_rows = (
        anchor_rows.sort_values(["datadate", "gvkey"])
        .reset_index(drop=True)
    )

    sac_anchor = pd.merge_asof(
        sc_events[
            ["gvkey", "date_effective", "quarter", "company_name", "shariah_status"]
        ],
        anchor_rows[["panel_key", "gvkey", "datadate"]],
        by="gvkey",
        left_on="date_effective",
        right_on="datadate",
        direction="backward",
        allow_exact_matches=True,
    )

    # If multiple SAC lists fall on the same fiscal row, keep the most recent one
    sac_anchor = (
        sac_anchor.dropna(subset=["panel_key"])
        .sort_values(["panel_key", "date_effective"])
        .drop_duplicates(subset=["panel_key"], keep="last")
        .copy()
    )

    # ------------------------------------------------------------------
    # 5) Build final output
    # ------------------------------------------------------------------
    panel = comp.copy()

    panel["sac_shariah"] = 0
    panel["sac_date_effective"] = pd.NaT
    panel["sac_source_quarter"] = pd.NA

    if not sac_anchor.empty:
        sac_updates = sac_anchor[["panel_key", "date_effective", "quarter", "shariah_status"]].copy()
        sac_updates["sac_shariah_new"] = (
            sac_updates["shariah_status"] == "Shariah-Compliant"
        ).astype(int)
        sac_updates = sac_updates.rename(columns={
            "date_effective": "sac_date_effective_new",
            "quarter": "sac_source_quarter_new",
        })

        panel = panel.merge(
            sac_updates[["panel_key", "sac_shariah_new", "sac_date_effective_new", "sac_source_quarter_new"]],
            on="panel_key",
            how="left",
        )

        panel["sac_shariah"] = panel["sac_shariah_new"].fillna(0).astype(int)
        panel["sac_date_effective"] = panel["sac_date_effective_new"]
        panel["sac_source_quarter"] = panel["sac_source_quarter_new"]

        panel = panel.drop(columns=[
            "sac_shariah_new",
            "sac_date_effective_new",
            "sac_source_quarter_new",
        ], errors="ignore")

    # ------------------------------------------------------------------
    # 6) Derive oancfq from YTD oancfy BEFORE dropping fiscal columns
    # ------------------------------------------------------------------
    panel = derive_oancfq(panel)

    # ------------------------------------------------------------------
    # 7) Drop dispensable columns
    # ------------------------------------------------------------------
    drop_cols = [
        "panel_key", "sedol",

        # Accounting columns not needed for ratios
        "dd1q", "dlsq", "notesq", "rectrq", "xintdy",

        # Intermediate technical columns
        "fiscal_year", "fqtr_num", "bursa_code_flex",
    ]

    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns], errors="ignore")

    # Reorder: key columns first
    preferred_order = [
        "gvkey", "conm", "isin",
        "datacqtr", "datadate", "fqtr",
        "sac_shariah", "sac_date_effective", "sac_source_quarter",
    ]
    existing_first = [c for c in preferred_order if c in panel.columns]
    remaining = [c for c in panel.columns if c not in existing_first]
    panel = panel[existing_first + remaining].copy()

    return panel


def build_panel_compliance_anchored(
    df_comp: pd.DataFrame,
    df_events: pd.DataFrame,
    policy,
) -> pd.DataFrame:
    """Generic quarterly panel anchored on authority review dates, by gvkey.

    The UAE (and any future country whose list can be resolved to gvkeys
    upfront) bypasses the Bursa-specific machinery of
    :func:`build_panel_sac_anchored` — int-cast stock codes, ISIN-derived
    4-digit joins, the literal ``"Shariah-Compliant"`` status — and anchors
    directly on ``(gvkey, date_effective)`` events produced by the
    compliance loader + crosswalk.

    Business rules:
    - **Every** firm in the Compustat file stays in the panel (per the
      slice-2 decision the ~120 non-DFM UAE firms keep their anomaly
      scores; they simply carry no screening verdict).
    - For each event, the status attaches to the last available fiscal
      close on or before the list date (priority Q4 > Q3 > Q2 > Q1) —
      same anchor rule as the SAC path.
    - ``compliance_shariah`` is tri-state: 1 = compliant per the event,
      0 = explicitly non-compliant (only if the authority ever emits
      negative events — DFM does not), **NaN = unknown** (no event, or a
      firm absent from the authority list). With ``policy.positive_only``
      no 0 is ever written.
    - ``methodology_key`` is stamped on every row so downstream threshold
      resolution (detectors, Phase 0) follows the right rulebook.

    Args:
        df_comp: Cleaned Compustat quarterly frame (``load_compustat_csv``).
        df_events: One row per (gvkey, review date):
            ``gvkey, date_effective, quarter, compliant`` (int 0/1).
        policy: :class:`server.panel.methodology.CountryCompliancePolicy`.

    Output: 1 row = 1 gvkey × datacqtr.
    """
    # ------------------------------------------------------------------
    # 1) Compustat quarterly panel — full universe
    # ------------------------------------------------------------------
    comp = df_comp.copy()
    comp["gvkey"] = pd.to_numeric(comp["gvkey"], errors="coerce")
    # ns resolution on both asof keys — the Pydantic-validated frame can
    # carry second-resolution datetimes, and merge_asof refuses mixed units.
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce").astype("datetime64[ns]")

    comp = (
        comp.sort_values("datadate")
            .drop_duplicates(subset=["gvkey", "datacqtr"], keep="last")
            .copy()
    )
    comp["panel_key"] = comp["gvkey"].astype(str) + "__" + comp["datacqtr"].astype(str)

    if "fyearq" in comp.columns:
        comp["fiscal_year"] = pd.to_numeric(comp["fyearq"], errors="coerce")
    else:
        comp["fiscal_year"] = comp["datadate"].dt.year

    if "fqtr" not in comp.columns:
        raise ValueError("Column 'fqtr' is required for the anchor logic.")
    comp["fqtr_num"] = pd.to_numeric(comp["fqtr"], errors="coerce")

    # ------------------------------------------------------------------
    # 2) Annual anchor row per firm-year (priority Q4 > Q3 > Q2 > Q1)
    # ------------------------------------------------------------------
    anchor_rows = (
        comp.dropna(subset=["gvkey", "fiscal_year", "fqtr_num", "datadate"])
            .sort_values(
                ["gvkey", "fiscal_year", "fqtr_num", "datadate"],
                ascending=[True, True, False, False],
                kind="mergesort",
            )
            .drop_duplicates(subset=["gvkey", "fiscal_year"], keep="first")
            .dropna(subset=["gvkey", "datadate"])
            .copy()
    )
    anchor_rows["gvkey"] = anchor_rows["gvkey"].astype("int64")
    anchor_rows = anchor_rows.sort_values(["datadate", "gvkey"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3) Attach each event to the last fiscal close <= list date
    # ------------------------------------------------------------------
    events = df_events.copy()
    events["gvkey"] = pd.to_numeric(events["gvkey"], errors="coerce")
    events = events.dropna(subset=["gvkey", "date_effective"]).copy()
    events["gvkey"] = events["gvkey"].astype("int64")
    events["date_effective"] = pd.to_datetime(
        events["date_effective"], errors="coerce"
    ).astype("datetime64[ns]")
    events = events.sort_values(["date_effective", "gvkey"]).reset_index(drop=True)

    n_event_firms = events["gvkey"].nunique()

    anchored = pd.merge_asof(
        events[["gvkey", "date_effective", "quarter", "compliant"]],
        anchor_rows[["panel_key", "gvkey", "datadate"]],
        by="gvkey",
        left_on="date_effective",
        right_on="datadate",
        direction="backward",
        allow_exact_matches=True,
    )
    anchored = (
        anchored.dropna(subset=["panel_key"])
        .sort_values(["panel_key", "date_effective"])
        .drop_duplicates(subset=["panel_key"], keep="last")
        .copy()
    )

    # Coverage cap (positive-only lists): the backward as-of join attaches a
    # firm's FIRST list event to the last fiscal close BEFORE it — which, for a
    # list whose history starts mid-panel (Boubyan/SAU begins 2021), backdates
    # a compliant label onto pre-coverage fiscal closes (e.g. a 2020-Q4 row
    # labelled from a 2021 event). Those pre-coverage rows then contaminate the
    # Phase-0 reference sample. Drop any anchor whose fiscal close predates the
    # list's first observation so labels never precede the authority's coverage.
    # No-op for lists whose coverage spans the panel (MYS/UAE) — their earliest
    # anchor closes are already >= the first list date.
    if policy.positive_only and not events.empty and not anchored.empty:
        coverage_start = events["date_effective"].min()
        n_before = len(anchored)
        anchored = anchored[anchored["datadate"] >= coverage_start].copy()
        n_capped = n_before - len(anchored)
        if n_capped:
            log.info(
                "compliance coverage cap [%s]: dropped %d anchor(s) with a "
                "fiscal close before the list start %s.",
                policy.authority_label, n_capped,
                pd.Timestamp(coverage_start).date(),
            )

    # ------------------------------------------------------------------
    # 4) Final output — tri-state label, methodology stamp
    # ------------------------------------------------------------------
    panel = comp.copy()
    panel["compliance_shariah"] = np.nan          # NaN = unknown
    panel["compliance_date_effective"] = pd.NaT
    panel["compliance_source_quarter"] = pd.NA
    panel["methodology_key"] = policy.methodology_key

    if not anchored.empty:
        updates = anchored.rename(columns={
            "compliant": "_compliance_new",
            "date_effective": "_date_effective_new",
            "quarter": "_source_quarter_new",
        })
        panel = panel.merge(
            updates[["panel_key", "_compliance_new", "_date_effective_new", "_source_quarter_new"]],
            on="panel_key",
            how="left",
        )
        hit = panel["_compliance_new"].notna()
        panel.loc[hit, "compliance_shariah"] = panel.loc[hit, "_compliance_new"].astype(float)
        panel.loc[hit, "compliance_date_effective"] = panel.loc[hit, "_date_effective_new"]
        panel.loc[hit, "compliance_source_quarter"] = panel.loc[hit, "_source_quarter_new"]
        panel = panel.drop(
            columns=["_compliance_new", "_date_effective_new", "_source_quarter_new"],
            errors="ignore",
        )

    if policy.positive_only and (panel["compliance_shariah"] == 0).any():
        raise ValueError(
            f"{policy.authority_label} is a positive-only list but the panel "
            f"carries compliance_shariah == 0 rows — event construction bug."
        )

    # ------------------------------------------------------------------
    # 5) Derive oancfq from YTD oancfy BEFORE dropping fiscal columns
    # ------------------------------------------------------------------
    panel = derive_oancfq(panel)

    # ------------------------------------------------------------------
    # 6) Drop dispensable columns, order keys first
    # ------------------------------------------------------------------
    drop_cols = [
        "panel_key", "sedol",
        "dd1q", "dlsq", "notesq", "rectrq", "xintdy",
        "fiscal_year", "fqtr_num",
    ]
    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns], errors="ignore")

    preferred_order = [
        "gvkey", "conm", "isin",
        "datacqtr", "datadate", "fqtr",
        "compliance_shariah", "compliance_date_effective",
        "compliance_source_quarter", "methodology_key",
    ]
    existing_first = [c for c in preferred_order if c in panel.columns]
    remaining = [c for c in panel.columns if c not in existing_first]
    panel = panel[existing_first + remaining].copy()

    n1 = int((panel["compliance_shariah"] == 1).sum())
    n0 = int((panel["compliance_shariah"] == 0).sum())
    nna = int(panel["compliance_shariah"].isna().sum())
    log.info(
        "build_panel_compliance_anchored[%s]: %d rows, %d firms | "
        "events: %d rows / %d firms, %d anchored | "
        "compliance_shariah 1: %d, 0: %d, NA: %d",
        policy.methodology_key, len(panel), panel["gvkey"].nunique(),
        len(events), n_event_firms, len(anchored),
        n1, n0, nna,
    )
    return panel


def expand_to_full_quarter_grid(
    panel: pd.DataFrame,
    group_col: str = "gvkey",
    quarter_col: str = "datacqtr",
    date_col: str = "datadate",
    carry_id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each firm, insert missing quarterly rows between its first and last observation.

    New rows have NaN for all financial columns; only `group_col`, `quarter_col`,
    `date_col`, `fqtr`, and identity columns (`carry_id_cols`) are populated.

    `carry_id_cols` are forward- then backward-filled so every new row
    is identifiable (firm name, ISIN, etc.).

    Parameters
    ----------
    carry_id_cols : non-financial columns to propagate to new rows
        (default: ["conm", "isin"])

    Returns
    -------
    Expanded DataFrame sorted by (group_col, quarter_col), index reset.
    Adds boolean column `_synthetic_row` (True for inserted rows).
    """
    if carry_id_cols is None:
        carry_id_cols = ["conm", "isin"]

    df = panel.copy()
    df["_synthetic_row"] = False

    # Parse datacqtr -> pd.Period to enumerate quarters
    try:
        df["_period"] = pd.PeriodIndex(df[quarter_col].astype(str), freq="Q")
    except Exception:
        log.warning(
            "expand_to_full_quarter_grid: could not parse '%s' as periods. "
            "Quarter expansion disabled.", quarter_col
        )
        return df

    new_rows: list[dict] = []
    existing_periods_by_firm: dict = (
        df.groupby(group_col, sort=False)["_period"]
        .apply(set)
        .to_dict()
    )

    for gvkey, grp in df.groupby(group_col, sort=False):
        p_min = grp["_period"].min()
        p_max = grp["_period"].max()
        full_range = pd.period_range(p_min, p_max, freq="Q")
        existing = existing_periods_by_firm[gvkey]
        missing = [p for p in full_range if p not in existing]

        for p in missing:
            new_row = {
                group_col:        gvkey,
                quarter_col:      str(p),
                date_col:         p.end_time.normalize(),
                "fqtr":           p.quarter,
                "_period":        p,
                "_synthetic_row": True,
            }

            if "sac_shariah" in df.columns:
                new_row["sac_shariah"] = 0

            new_rows.append(new_row)

    if not new_rows:
        df = df.drop(columns=["_period"])
        return df.sort_values([group_col, quarter_col]).reset_index(drop=True)

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sort_values([group_col, "_period"]).reset_index(drop=True)
    df = df.drop(columns=["_period"])

    # Propagate firm identifiers to new rows
    for col in carry_id_cols:
        if col in df.columns:
            df[col] = df.groupby(group_col, sort=False)[col].ffill()
            df[col] = df.groupby(group_col, sort=False)[col].bfill()

    n_added = len(new_rows)
    n_firms_touched = len({r[group_col] for r in new_rows})
    log.info(
        "expand_to_full_quarter_grid: %d synthetic rows added for %d firms",
        n_added, n_firms_touched,
    )
    return df


def apply_forward_fill(
    panel: pd.DataFrame,
    cols: list[str],
    group_col: str = "gvkey",
    date_col: str = "datadate",
    fill_limit: int = 4,
) -> pd.DataFrame:
    """
    Forward-fill `cols` within each firm (`group_col`), sorted by date (`date_col`).

    For each column XXX in `cols`:
      - XXX_fillna (Int8)  : 1 if the value was missing and filled by ffill, 
                            0 otherwise, limit = fill_limit
      - XXX_lag    (Int16) : number of quarters since the last observed source value
                             (0 = directly observed, k = carried k quarters forward,
                              NA = no prior observation in the group)

    Example: if Q4 2019 fills Q1/Q2/Q3 2020, the lags are 1, 2, 3.

    NaN with no prior value in the group remain NaN (no backward imputation).

    Parameters
    ----------
    panel    : quarterly DataFrame (1 row = 1 gvkey x quarter)
    cols     : columns to forward-fill
    group_col: firm identifier column (default: 'gvkey')
    date_col : date column for temporal ordering (default: 'datadate')

    Returns
    -------
    DataFrame with XXX_fillna and XXX_lag columns added, NaN filled.
    """
    df = panel.copy()

    # Sort to guarantee temporal order within each firm
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    # Sequential intra-firm index (0, 1, 2, ...) — reused across all columns.
    # Since each row = 1 quarter (after expand), this counter equals distance in quarters.
    gidx = df.groupby(group_col, sort=False).cumcount()

    cols_present = [c for c in cols if c in df.columns]
    cols_absent  = [c for c in cols if c not in df.columns]
    if cols_absent:
        log.warning("apply_forward_fill — absent columns ignored: %s", cols_absent)

    for col in cols_present:
        was_null_orig = df[col].isna().copy()

        # Forward-fill within each firm group
        df[col] = df.groupby(group_col, sort=False)[col].ffill(limit=fill_limit)

        # Flag: 1 if the value was missing and has been filled
        df[f"{col}_fillna"] = (was_null_orig & df[col].notna()).astype("Int8")

        # Confidence: distance in quarters from the source observation.
        # last_obs_pos[i] = gidx of the last non-NaN observation before position i
        #                   (NaN if no prior observation exists in the group)
        last_obs_pos = (
            gidx.where(~was_null_orig)           # NaN at originally missing positions
                .groupby(df[group_col], sort=False)
                .transform(lambda x: x.ffill())  # carry forward within group
        )
        # lag = distance from current position to source position
        # -> 0 for direct observations, k for values carried k steps forward
        # -> NaN (cast to pd.NA Int16) if no prior observation exists
        df[f"{col}_lag"] = (gidx - last_obs_pos).astype("Int16")
        df.loc[df[col].isna(), f"{col}_lag"] = pd.NA

        n_filled     = int(df[f"{col}_fillna"].sum())
        n_still_null = int(df[col].isna().sum())
        conf_filled  = df.loc[df[f"{col}_fillna"] == 1, f"{col}_lag"]
        med_conf     = float(conf_filled.median()) if not conf_filled.empty else float("nan")
        max_conf     = int(conf_filled.max())      if not conf_filled.empty else 0
        log.info(
            "ffill %-12s: %d filled, %d still NaN | median lag=%.1f max=%d",
            col, n_filled, n_still_null, med_conf, max_conf,
        )

    return df


def add_clean_sample_flag(
    panel: pd.DataFrame,
    sanity_bounds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add `clean_sample` (Int8): 1 if the row is a good default analysis sample.

    Conditions (all must hold):
      - not a synthetic row (`_synthetic_row == False`)
      - no forward-filled balance-sheet input (`bs_any_ffilled == 0`)
      - no ratio exceeds its sanity bound (NaN ratios do not count as violations)

    Intended as a convenience shortcut for casual analysis:
        clean = panel[panel["clean_sample"] == 1]

    For specific research questions, apply tailored filters directly on the
    flag columns instead (`_synthetic_row`, `bs_any_ffilled`, `sac_shariah`,
    `sector`, etc.) rather than relying on this default.

    Must be called AFTER `compute_shariah_ratios` and `add_balance_sheet_ffill_flag`.
    """
    if sanity_bounds is None:
        sanity_bounds = {
            "ratio_debt": 5.0,
            "ratio_debt_adj": 5.0,
            "ratio_cash": 1.0,
            "ratio_cash_adj": 1.0,
            "ratio_income": 2.0,
        }

    df = panel.copy()

    keep = pd.Series(True, index=df.index)

    if "_synthetic_row" in df.columns:
        keep &= ~df["_synthetic_row"].fillna(False).astype(bool)

    if "bs_any_ffilled" in df.columns:
        keep &= df["bs_any_ffilled"].fillna(0).astype(int) == 0

    for col, bound in sanity_bounds.items():
        if col in df.columns:
            keep &= ~(df[col] > bound).fillna(False)

    df["clean_sample"] = keep.astype("Int8")

    n_clean = int((df["clean_sample"] == 1).sum())
    log.info(
        "clean_sample: %d / %d rows (%.2f%%) pass default integrity filter",
        n_clean, len(df), 100 * n_clean / max(len(df), 1),
    )
    return df


def add_balance_sheet_ffill_flag(
    panel: pd.DataFrame,
    bs_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add `bs_any_ffilled` (Int8): 1 if ANY balance-sheet input on that row was
    forward-filled, 0 if all were directly observed.

    Ffill-induced accounting inconsistencies (e.g. `dlttq + dlcq > ltq`,
    `cheq > atq`) concentrate in rows where items were ffilled from different
    source quarters. Downstream analyses can filter on `bs_any_ffilled == 0`
    for strict integrity, or keep all rows for maximum coverage.

    Requires `apply_forward_fill` to have been called first (reads `{col}_fillna`).
    """
    if bs_cols is None:
        bs_cols = ["atq", "dlttq", "dlcq", "ltq", "cheq"]

    fillna_cols = [f"{c}_fillna" for c in bs_cols if f"{c}_fillna" in panel.columns]
    if not fillna_cols:
        log.warning("add_balance_sheet_ffill_flag: no *_fillna columns found — skipping")
        return panel

    df = panel.copy()
    df["bs_any_ffilled"] = (
        df[fillna_cols].fillna(0).astype("int8").max(axis=1).astype("Int8")
    )
    n_flagged = int((df["bs_any_ffilled"] == 1).sum())
    log.info(
        "bs_any_ffilled: %d rows flagged (%.2f%%) — at least one of %s was ffilled",
        n_flagged, 100 * n_flagged / max(len(df), 1), bs_cols,
    )
    return df


def apply_firm_level_zero_fill(
    panel: pd.DataFrame,
    cols: list[str],
    group_col: str = "gvkey",
) -> pd.DataFrame:
    """
    Set NaN to 0 only for firms that never report the variable across their
    entire history. Firms with at least one observation are left untouched —
    those are handled by `apply_forward_fill`.

    Rationale: NaN->0 is only economically safe when the firm structurally
    lacks the item (e.g. a debt-free firm has no `dlttq`). For firms that
    sometimes report and sometimes don't, NaN is a reporting gap, not a
    genuine zero, and zeroing creates artificial spikes in derived ratios.

    For each column XXX in `cols`, adds:
      - XXX_zerofilled (Int8) : 1 if the value was zeroed by this rule

    Parameters
    ----------
    panel    : quarterly DataFrame (post-ffill)
    cols     : columns eligible for firm-level zero-fill
               (typically: dlttq, dlcq, nopiq)
    group_col: firm identifier column (default: 'gvkey')

    Returns
    -------
    DataFrame with NaN->0 applied at the firm level for never-reporting firms,
    and a XXX_zerofilled flag column per processed variable.
    """
    df = panel.copy()

    cols_present = [c for c in cols if c in df.columns]
    cols_absent  = [c for c in cols if c not in df.columns]
    if cols_absent:
        log.warning("apply_firm_level_zero_fill — absent columns ignored: %s", cols_absent)

    for col in cols_present:
        # Per-firm: is the variable ever observed?
        firm_has_obs = df.groupby(group_col, sort=False)[col].transform(
            lambda s: s.notna().any()
        )

        # Row is NaN AND firm never reports this variable -> safe to zero
        zero_mask = df[col].isna() & ~firm_has_obs

        df[f"{col}_zerofilled"] = zero_mask.astype("Int8")
        df.loc[zero_mask, col] = 0.0

        n_rows_zeroed  = int(zero_mask.sum())
        n_firms_zeroed = int(df.loc[zero_mask, group_col].nunique())
        log.info(
            "firm-level zero-fill %-12s: %d rows zeroed across %d firms",
            col, n_rows_zeroed, n_firms_zeroed,
        )

    return df


def build_ffill_report(
    panel: pd.DataFrame,
    cols: list[str],
    group_col: str = "gvkey",
    date_col: str = "datadate",
    name_col: str = "conm",
    quarter_col: str = "datacqtr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate two diagnostic tables after `expand_to_full_quarter_grid`
    + `apply_forward_fill`.

    summary (one row per variable)
    --------------------------------
    variable | n_filled | n_still_nan | pct_still_nan | n_firms_still_nan
    | max_quarter_still_nan | conf_median | conf_max | conf_gt4_pct

    firms (one row per (variable, firm) where NaN remain)
    -------------------------------------------------------
    variable | gvkey | firm_name | n_nan_remaining
    | first_quarter_nan | last_quarter_nan
    """
    cols_present = [c for c in cols if c in panel.columns]

    summary_rows: list[dict] = []
    firm_rows:    list[pd.DataFrame] = []

    for col in cols_present:
        fillna_col = f"{col}_fillna"
        lag_col    = f"{col}_lag"

        # Filled values
        n_filled = int(panel[fillna_col].sum()) if fillna_col in panel.columns else 0

        # Residual NaN
        mask_null     = panel[col].isna()
        n_still_nan   = int(mask_null.sum())
        pct_still_nan = round(100 * n_still_nan / max(len(panel), 1), 2)

        # Firms with residual NaN
        sub = panel.loc[mask_null, [group_col, name_col, quarter_col]].copy()
        n_firms_still_nan   = sub[group_col].nunique()
        max_quarter_overall = sub[quarter_col].max() if not sub.empty else pd.NA

        # Confidence stats (filled values only)
        if lag_col in panel.columns and fillna_col in panel.columns:
            conf_filled  = panel.loc[panel[fillna_col] == 1, lag_col].dropna()
            conf_median  = round(float(conf_filled.median()), 1) if len(conf_filled) else pd.NA
            conf_max     = int(conf_filled.max())                if len(conf_filled) else pd.NA
            conf_gt4_pct = round(
                100 * (conf_filled > 4).sum() / max(len(conf_filled), 1), 1
            ) if len(conf_filled) else pd.NA
        else:
            conf_median = conf_max = conf_gt4_pct = pd.NA

        summary_rows.append({
            "variable":              col,
            "n_filled":              n_filled,
            "n_still_nan":           n_still_nan,
            "pct_still_nan":         pct_still_nan,
            "n_firms_still_nan":     n_firms_still_nan,
            "max_quarter_still_nan": max_quarter_overall,
            "conf_median_quarters":  conf_median,
            "conf_max_quarters":     conf_max,
            "conf_gt4q_pct":         conf_gt4_pct,   # % of fills carried > 4 quarters (> 1 year)
        })

        # Firm-level detail (residual NaN)
        if not sub.empty:
            grp = (
                sub.groupby([group_col, name_col])[quarter_col]
                .agg(
                    n_nan_remaining="count",
                    first_quarter_nan="min",
                    last_quarter_nan="max",
                )
                .reset_index()
                .rename(columns={group_col: "gvkey", name_col: "firm_name"})
                .sort_values("last_quarter_nan", ascending=False)
            )
            grp.insert(0, "variable", col)
            firm_rows.append(grp)

    summary = pd.DataFrame(summary_rows)

    firms = (
        pd.concat(firm_rows, ignore_index=True)
        if firm_rows
        else pd.DataFrame(
            columns=["variable", "gvkey", "firm_name",
                     "n_nan_remaining", "first_quarter_nan", "last_quarter_nan"]
        )
    )

    return summary, firms


def build_merge_diagnostics(
    df_sc: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_comp_pre_pydantic: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build three diagnostic tables for SC Malaysia ↔ Compustat coverage.

      1) firm_diag   : one row per SC stock code
      2) event_diag  : one row per SC list event
      3) summary     : counts and percentages
    """
    mapping = build_mapping_compustat(df_comp)

    sc = df_sc.copy()
    sc["date_effective"] = pd.to_datetime(sc["date_effective"], errors="coerce")
    sc["stock_code_z4"] = sc["stock_code"].astype(int).astype(str).str.zfill(4)

    # ------------------------------------------------------------------
    # 1) Firm-level table
    # ------------------------------------------------------------------
    sc_firms = (
        sc.sort_values("date_effective")
          .drop_duplicates("stock_code_z4", keep="last")
          [["stock_code_z4", "company_name", "market", "sector"]]
          .copy()
    )

    map_codes = mapping[["gvkey", "conm", "isin", "bursa_code_flex", "status_flex"]].copy()

    firm_diag = sc_firms.merge(
        map_codes,
        left_on="stock_code_z4",
        right_on="bursa_code_flex",
        how="left",
    )

    firm_diag["matched_in_compustat"] = firm_diag["gvkey"].notna().astype(int)

    # Diagnostic reasons for missing firms
    if df_comp_pre_pydantic is not None:
        missing_codes = firm_diag.loc[firm_diag["matched_in_compustat"] == 0, "stock_code_z4"].tolist()
        diagnosed = diagnose_missing_sc_codes(missing_codes, df_comp_pre_pydantic, df_sc)
        if diagnosed:
            ddf = pd.DataFrame(diagnosed).rename(columns={"bursa_code": "stock_code_z4"})
            firm_diag = firm_diag.merge(ddf, on="stock_code_z4", how="left")

    # Harmonise columns
    if "gvkey_compustat" in firm_diag.columns:
        firm_diag["gvkey_reference"] = firm_diag["gvkey_compustat"]
    else:
        firm_diag["gvkey_reference"] = pd.NA

    # ------------------------------------------------------------------
    # 2) Event/list-level table
    # ------------------------------------------------------------------
    event_diag = sc[
        ["source", "date_effective", "quarter", "stock_code_z4", "company_name", "market", "sector", "shariah_status"]
    ].copy()

    for col in ["diagnostic", "detail"]:
        if col not in firm_diag.columns:
            firm_diag[col] = pd.NA

    event_diag = event_diag.merge(
        firm_diag[
            [
                "stock_code_z4", "matched_in_compustat", "diagnostic", "detail",
                "gvkey", "conm", "isin", "gvkey_reference"
            ]
        ],
        on="stock_code_z4",
        how="left",
    )

    event_diag["list_name"] = (
        event_diag["source"].fillna("SC Malaysia").astype(str)
        + " | "
        + event_diag["quarter"].fillna("").astype(str)
        + " | "
        + event_diag["date_effective"].dt.strftime("%Y-%m-%d")
    )

    # ------------------------------------------------------------------
    # 3) Summary table
    # ------------------------------------------------------------------
    sc_unique_firms = int(sc_firms["stock_code_z4"].nunique())
    matched_firms   = int(firm_diag["matched_in_compustat"].sum())
    lost_firms      = int((firm_diag["matched_in_compustat"] == 0).sum())

    summary = pd.DataFrame([{
        "sc_unique_firms": sc_unique_firms,
        "matched_firms":   matched_firms,
        "lost_firms":      lost_firms,
        "lost_pct":        round(100 * lost_firms / max(sc_unique_firms, 1), 2),
    }])

    return firm_diag, event_diag, summary
