"""Benford detector (z1).

Pools reported currency figures over a rolling firm-quarter window and maps
the Benford chi-square statistic to the common anomaly z-scale.

Two calibration paths:

- ``chi2_asymptotic`` — original PIT via ``χ²₈`` CDF. Requires ``N_fig ≥
  min_obs`` (109 under the ``E_d ≥ 5`` rule).
- ``monte_carlo`` — build an empirical CDF of ``T_1`` by drawing
  multinomial samples from the Benford probabilities at the same
  ``N_fig``, then PIT through that CDF. Drops the ``E_d ≥ 5`` constraint
  and can score rows with as few as ``min_obs_mc`` figures.
- ``auto`` — per-row dispatch: MC when ``N_fig < chi2_minimum_figures``
  and χ² otherwise.

The MC null for a given ``N_fig`` is cached across rows so the cost is
amortised — typical firm panels have only a few distinct ``N_fig``
values once figures are pooled.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.engine.pit import pit_chi2, pit_empirical
from src.common.columns import resolve_present_columns

log = logging.getLogger(__name__)

BENFORD_DF: int = 8
MIN_OBS_BENFORD: int = 109
MIN_OBS_BENFORD_MC: int = 30
POOLING_WINDOW_QUARTERS: int = 8
DEFAULT_CALIBRATION_MODE: str = "auto"
DEFAULT_MC_REPLICATES: int = 20_000
DEFAULT_MC_SEED: int = 2

# Minimum number of C reference z1 values required for the second-stage
# empirical PIT. Reuses the same §9.2 D3 floor as the other detectors that
# calibrate against C.
_MIN_EMPIRICAL_PIT_REF: int = 30

DEFAULT_BENFORD_VALUE_COLUMNS: tuple[str, ...] = (
    "revtq", "iditq", "nopiq", "xintq", "xintdy", "niq", "oibdpq",
    "atq", "dlttq", "dlcq", "ltq", "cheq", "chq", "chsq",
    "actq", "lctq", "rectq", "invtq", "rectrq", "dd1q", "dlsq", "notesq",
)

BENFORD_PROBS: np.ndarray = np.array(
    [np.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=float
)


def _row_digit_counts(
    df: pd.DataFrame,
    value_columns: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised first-digit counts per row across the available value columns."""
    available = resolve_present_columns(df, value_columns, DEFAULT_BENFORD_VALUE_COLUMNS)
    n_rows = len(df)
    if not available:
        return np.zeros((n_rows, 9), dtype=float), np.zeros(n_rows, dtype=int)

    matrix = np.column_stack(
        [pd.to_numeric(df[column], errors="coerce").abs().to_numpy(dtype=float) for column in available]
    )
    valid = np.isfinite(matrix) & (matrix > 0)
    safe = np.where(valid, matrix, 1.0)
    exponents = np.floor(np.log10(safe))
    scaled = safe / np.power(10.0, exponents)
    digits = np.clip(scaled, 1.0, 9.999999999).astype(int)
    digits = np.where(valid, digits, 0)

    counts = np.zeros((n_rows, 9), dtype=float)
    for digit in range(1, 10):
        counts[:, digit - 1] = (digits == digit).sum(axis=1)
    n_fig = valid.sum(axis=1).astype(int)
    return counts, n_fig


def _benford_chi2_from_counts(
    observed: np.ndarray,
    min_obs: int,
) -> tuple[float, int]:
    """Chi-square statistic from pre-aggregated first-digit counts."""
    observed = np.asarray(observed, dtype=float)
    n = int(observed.sum())
    if n < min_obs:
        return np.nan, n
    expected = n * BENFORD_PROBS
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    return chi2_stat, n


def _rolling_benford_chi2_fast(
    df: pd.DataFrame,
    row_counts: np.ndarray,
    row_nfig: np.ndarray,
    min_obs: int,
    pooling_window_quarters: int,
) -> tuple[pd.Series, pd.Series]:
    """Rolling Benford chi-square using pre-aggregated digit counts."""
    chi2_values = np.full(len(df), np.nan, dtype=float)
    nfig_values = np.zeros(len(df), dtype=np.int64)
    if "gvkey" not in df.columns:
        return (
            pd.Series(chi2_values, index=df.index, dtype=float),
            pd.Series(nfig_values, index=df.index, dtype="int64"),
        )

    if "datacqtr" not in df.columns:
        for _, idx in df.groupby("gvkey", sort=False).groups.items():
            pos = df.index.get_indexer(pd.Index(idx))
            stat, n = _benford_chi2_from_counts(row_counts[pos].sum(axis=0), min_obs=min_obs)
            chi2_values[pos] = stat
            nfig_values[pos] = n
        return (
            pd.Series(chi2_values, index=df.index, dtype=float),
            pd.Series(nfig_values, index=df.index, dtype="int64"),
        )

    ordered = df.sort_values(["gvkey", "datacqtr"], kind="stable")
    ordered_pos = df.index.get_indexer(ordered.index)
    ordered_gvkeys = ordered["gvkey"].to_numpy()
    if len(ordered_gvkeys) == 0:
        return (
            pd.Series(chi2_values, index=df.index, dtype=float),
            pd.Series(nfig_values, index=df.index, dtype="int64"),
        )
    group_starts = np.concatenate(
        [[0], np.flatnonzero(ordered_gvkeys[1:] != ordered_gvkeys[:-1]) + 1]
    )
    group_ends = np.concatenate([group_starts[1:], [len(ordered_pos)]])

    for start, end in zip(group_starts, group_ends, strict=False):
        firm_pos = ordered_pos[start:end]
        counts_firm = row_counts[firm_pos]
        nfig_firm = row_nfig[firm_pos]
        if counts_firm.size == 0:
            continue
        csum_counts = np.vstack([np.zeros((1, 9), dtype=float), counts_firm.cumsum(axis=0)])
        csum_nfig = np.concatenate([[0], nfig_firm.cumsum()])
        horizon = end - start
        end_offsets = np.arange(horizon, dtype=int)
        start_offsets = np.maximum(end_offsets - pooling_window_quarters + 1, 0)

        observed = csum_counts[end_offsets + 1] - csum_counts[start_offsets]
        n_vec = (csum_nfig[end_offsets + 1] - csum_nfig[start_offsets]).astype(np.int64)
        nfig_values[firm_pos] = n_vec

        valid = n_vec >= min_obs
        if not valid.any():
            continue
        observed_valid = observed[valid]
        n_valid = n_vec[valid].astype(float)
        expected = n_valid[:, None] * BENFORD_PROBS[None, :]
        chi2_valid = np.sum((observed_valid - expected) ** 2 / expected, axis=1)
        chi2_values[firm_pos[valid]] = chi2_valid

    return (
        pd.Series(chi2_values, index=df.index, dtype=float),
        pd.Series(nfig_values, index=df.index, dtype="int64"),
    )


def _monte_carlo_benford_null(
    n_figures: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Empirical null distribution of ``T_1`` at the given ``N_fig``.

    Draws ``n_replicates`` multinomial samples from Benford probabilities,
    computes ``T_1`` on each, and returns the sorted array. The sorted
    array is consumed by :func:`pit_empirical` which already handles the
    ``Φ⁻¹`` conversion and the numerical clipping at the tails.
    """
    counts = rng.multinomial(n_figures, BENFORD_PROBS, size=n_replicates).astype(float)
    expected = n_figures * BENFORD_PROBS
    stats = np.sum((counts - expected) ** 2 / expected, axis=1)
    return stats


def _apply_monte_carlo_pit(
    chi2: pd.Series,
    nfig: pd.Series,
    mask: np.ndarray,
    n_replicates: int,
    seed: int,
    null_cache: dict[int, np.ndarray] | None = None,
) -> pd.Series:
    """PIT the chi-square values via per-``N_fig`` Monte-Carlo nulls.

    Args:
        null_cache: Optional externally-owned ``{N_fig: sorted_null_array}``
            cache. When provided, an ``N_fig`` already present is reused
            verbatim (bit-identical, no new draw) instead of being redrawn,
            and any newly drawn null is written back into it. Passing the
            same dict across the initial full-panel call and later
            firm-restricted Family-3 patch calls lets the patch reuse
            exactly the nulls the full call already produced, as long as it
            doesn't need an ``N_fig`` that never appeared in that first call
            (see profiling notes). Omit to get the original self-contained
            behavior (a fresh, call-local cache every time).
    """
    z = pd.Series(np.nan, index=chi2.index, dtype=float)
    if not mask.any():
        return z
    # Cache MC nulls keyed on the integer figure count.
    rng_master = np.random.default_rng(seed)
    sub = chi2[mask].dropna()
    unique_n = np.unique(nfig.loc[sub.index].to_numpy(dtype=int))
    null_by_n: dict[int, np.ndarray] = null_cache if null_cache is not None else {}
    for n in unique_n:
        if n <= 0 or int(n) in null_by_n:
            continue
        rng_child = np.random.default_rng(rng_master.integers(0, 2 ** 62))
        null_by_n[int(n)] = _monte_carlo_benford_null(
            n_figures=int(n), n_replicates=n_replicates, rng=rng_child,
        )
    # Apply row-by-n-class for vectorised empirical PIT.
    for n in unique_n:
        null_arr = null_by_n.get(int(n))
        if null_arr is None:
            continue
        class_mask = (nfig == n) & mask & chi2.notna()
        if not class_mask.any():
            continue
        values = chi2.loc[class_mask].to_numpy(dtype=float)
        z.loc[class_mask] = pit_empirical(values, null_arr)
    return z


def detect_benford(
    df: pd.DataFrame,
    value_columns: Sequence[str] | None = None,
    min_obs: int = MIN_OBS_BENFORD,
    min_obs_mc: int = MIN_OBS_BENFORD_MC,
    pooling_window_quarters: int = POOLING_WINDOW_QUARTERS,
    calibration_mode: str = DEFAULT_CALIBRATION_MODE,
    chi2_minimum_figures: int = MIN_OBS_BENFORD,
    monte_carlo_replicates: int = DEFAULT_MC_REPLICATES,
    monte_carlo_seed: int = DEFAULT_MC_SEED,
    mc_null_cache: dict[int, np.ndarray] | None = None,
    capture: dict[str, object] | None = None,
) -> pd.Series:
    """Compute the ``z1`` Benford detector.

    Args:
        df: Input dataframe with ``gvkey`` and monetary figure columns.
        value_columns: Optional explicit list of numeric figure columns.
        min_obs: Pooled-figure floor under ``chi2_asymptotic``.
        min_obs_mc: Pooled-figure floor under ``monte_carlo`` (smaller
            than ``min_obs`` — MC drops the ``E_d ≥ 5`` constraint).
        pooling_window_quarters: Rolling window used when ``datacqtr`` is
            available. Falls back to one pooled firm-level statistic when
            quarter labels are absent.
        calibration_mode: ``'chi2_asymptotic'`` / ``'monte_carlo'`` /
            ``'auto'``. ``auto`` picks MC when ``N_fig <
            chi2_minimum_figures`` and χ² otherwise.
        chi2_minimum_figures: Threshold that splits MC and χ² paths under
            ``auto``.
        monte_carlo_replicates: Number of multinomial draws used to build
            the MC null for each unique ``N_fig``.
        monte_carlo_seed: Seed for MC draws.
        mc_null_cache: Optional externally-owned ``{N_fig: null_array}``
            cache forwarded to :func:`_apply_monte_carlo_pit`. Lets a
            caller reuse the exact MC nulls a prior call already drew (see
            profiling notes, the Family-3 incremental patch).
        capture: Optional dict the caller can pass to receive
            ``capture["ref_values"]`` -- the pre-second-stage, ``C``-split
            calibrated z1 values used for the empirical re-PIT below.
            Lets a caller replay that exact second-stage calibration later
            without recomputing the whole detector.

    Returns:
        A series aligned on ``df.index``. Rows below the effective floor
        remain NaN.
    """
    if "gvkey" not in df.columns:
        log.warning("detect_benford: missing `gvkey`; returning NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    mode = calibration_mode
    if mode not in {"chi2_asymptotic", "monte_carlo", "auto"}:
        raise ValueError(f"detect_benford: unknown calibration_mode {mode!r}.")

    # The floor used when pooling rows; under ``auto`` we collect rows
    # that MC can score (≥ min_obs_mc) as well as those χ² can — so the
    # pool floor is the smaller of the two.
    effective_floor = min_obs_mc if mode in {"monte_carlo", "auto"} else min_obs

    row_counts, row_nfig = _row_digit_counts(df, value_columns)
    if int(row_nfig.sum()) == 0:
        log.warning("detect_benford: no monetary figure columns available; returning NaN.")
        return pd.Series(np.nan, index=df.index, dtype=float)

    chi2, nfig = _rolling_benford_chi2_fast(
        df=df,
        row_counts=row_counts,
        row_nfig=row_nfig,
        min_obs=effective_floor,
        pooling_window_quarters=pooling_window_quarters,
    )
    finite = chi2.notna()
    if not finite.any():
        log.warning(
            "detect_benford: no firm reached the effective figure floor (%d).",
            effective_floor,
        )
        return pd.Series(np.nan, index=df.index, dtype=float)

    z = pd.Series(np.nan, index=df.index, dtype=float)

    if mode == "chi2_asymptotic":
        mask = finite & (nfig >= min_obs)
        if mask.any():
            z.loc[mask] = pit_chi2(
                chi2.loc[mask].to_numpy(dtype=float), degrees_of_freedom=BENFORD_DF,
            )

    elif mode == "monte_carlo":
        mask = (finite & (nfig >= min_obs_mc)).to_numpy(dtype=bool)
        z = _apply_monte_carlo_pit(
            chi2=chi2, nfig=nfig, mask=mask,
            n_replicates=monte_carlo_replicates, seed=monte_carlo_seed,
            null_cache=mc_null_cache,
        )

    else:  # auto
        chi2_mask = finite & (nfig >= chi2_minimum_figures)
        mc_mask = finite & (nfig >= min_obs_mc) & ~chi2_mask
        if chi2_mask.any():
            z.loc[chi2_mask] = pit_chi2(
                chi2.loc[chi2_mask].to_numpy(dtype=float),
                degrees_of_freedom=BENFORD_DF,
            )
        if mc_mask.any():
            mc_z = _apply_monte_carlo_pit(
                chi2=chi2, nfig=nfig, mask=mc_mask.to_numpy(dtype=bool),
                n_replicates=monte_carlo_replicates, seed=monte_carlo_seed,
                null_cache=mc_null_cache,
            )
            z = z.where(~mc_mask, mc_z)

    # Second-stage empirical PIT on ``C`` — same treatment as ``d4``. The
    # parametric / MC calibration maps ``T_1`` to an N(0,1) under a theoretical
    # null (Benford multinomial), but the panel itself deviates from Benford
    # systematically (ffill artefacts, limited digit diversity), so honest
    # firms come out with a large positive bias. Applying an empirical PIT
    # against the z1 distribution on ``C`` re-centres the scale — an honest
    # firm now scores ≈ 0 by construction, and injection shifts are read
    # against the actual honest panel rather than a clean theoretical null.
    if "_split" in df.columns:
        ref_values = z.loc[df["_split"] == "C"].to_numpy(dtype=float)
        ref_values = ref_values[np.isfinite(ref_values)]
        if capture is not None:
            capture["ref_values"] = ref_values
        if ref_values.size >= _MIN_EMPIRICAL_PIT_REF:
            finite_mask = np.isfinite(z)
            z.loc[finite_mask] = pit_empirical(
                z.loc[finite_mask].to_numpy(dtype=float), ref_values,
            )
        else:
            log.warning(
                "detect_benford: only %d C reference values for empirical PIT "
                "(< %d); skipping the second-stage calibration.",
                int(ref_values.size), _MIN_EMPIRICAL_PIT_REF,
            )

    scored = int(np.isfinite(z).sum())
    log.info(
        "detect_benford: mode=%s, pooled window=%d, floor χ²=%d / MC=%d, scored=%d rows.",
        mode, pooling_window_quarters, min_obs, min_obs_mc, scored,
    )
    return z
