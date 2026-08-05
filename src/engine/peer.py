"""Peer or cross-sectional distance detector.

This detector implements the active ``j=7`` Working Paper detector:

- compute a Mahalanobis distance between a firm's canonical Sharia ratio vector
  and a calibration reference from split ``C``
- prefer a peer-group-conditioned reference when a grouping column exists
- otherwise fall back to the global cross-section on ``C``
- apply the Hotelling/F or chi-square PIT depending on the reference regime

Current peer hierarchy:
- try the configured grouping columns in order
- use the first sufficiently populated peer bucket
- otherwise fall back to the global cross-sectional ``C``

Future extension explicitly planned:
- super-sector mappings or grouped sectors for thin peer buckets
- optional size-conditioned peers layered on top of sector groups
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.pit import pit_chi2, pit_f
from src.common.ratio_inputs import get_canonical_sharia_ratios

log = logging.getLogger(__name__)

RIDGE_RHO: float = 0.01
MIN_PEER: int = 10
PEER_GROUP_CANDIDATES: tuple[str, ...] = ("sector", "gsector", "ggroup", "gind", "sic")

SECTOR_SUPER_GROUP_MAP: dict[str, str] = {
    "construction": "property_infra",
    "properties": "property_infra",
    "property": "property_infra",
    "utilities": "property_infra",
    "transportation and logistics": "property_infra",
    "transportation logistics": "property_infra",
    "energy": "real_assets",
    "plantation": "real_assets",
    "industrial products and services": "real_assets",
    "industrial products services": "real_assets",
    "industrial products": "real_assets",
    "trading services": "consumer_trade",
    "consumer products and services": "consumer_trade",
    "consumer products services": "consumer_trade",
    "consumer products": "consumer_trade",
    "telecommunications and media": "tech_comms",
    "telecommunications media": "tech_comms",
    "technology": "tech_comms",
    "health care": "healthcare",
    "healthcare": "healthcare",
    "financial services": "financials",
    "islamic business trust": "financials",
    "spac": "financials",
}


@dataclass(frozen=True)
class PeerReferenceContext:
    """Global peer-scoring context shared across all scored rows."""

    df: pd.DataFrame
    mask_cal: pd.Series
    ratios: pd.DataFrame
    x_cal: pd.DataFrame
    global_mu: np.ndarray
    global_inv: np.ndarray
    peer_group_cols: list[str]
    peer_references: dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int]]


def _normalize_sector_label_vectorized(series: pd.Series) -> pd.Series:
    """Normalize sector labels so closely related spellings map together.

    Vectorized via pandas ``.str`` methods instead of a per-row Python
    callback + ``re.sub`` (this ran once per Family-3 row evaluator, up to
    a few hundred times per run -- see profiling notes). NaN/None inputs are
    preserved as ``None`` outputs.
    """
    is_na = series.isna()
    normalized = series.where(~is_na, "").astype(str).str.strip().str.lower()
    normalized = normalized.str.replace("&", " and ", regex=False)
    normalized = normalized.str.replace("/", " ", regex=False)
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    return normalized.where(~is_na, None)


def _build_peer_super_group(df: pd.DataFrame) -> pd.Series:
    """Return a simple business-driven sector regrouping for thin sectors."""
    if "sector" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=object)

    normalized = _normalize_sector_label_vectorized(df["sector"])
    grouped = normalized.map(SECTOR_SUPER_GROUP_MAP)
    return grouped.astype(object)


def _available_peer_group_columns(df: pd.DataFrame) -> list[str]:
    """Return the peer-group hierarchy available on the panel."""
    available: list[str] = []
    if "sector" in df.columns and df["sector"].notna().any():
        available.append("sector")
        peer_super_group = _build_peer_super_group(df)
        if peer_super_group.notna().any():
            available.append("__peer_super_group")

    for col in PEER_GROUP_CANDIDATES:
        if col == "sector":
            continue
        if col in df.columns and df[col].notna().any():
            available.append(col)
    return available


def _shrunk_covariance(X: np.ndarray) -> np.ndarray:
    """Estimate covariance with Ledoit-Wolf shrinkage or a ridge fallback."""
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(X).covariance_
    except (ImportError, np.linalg.LinAlgError, ValueError) as exc:
        log.debug("_shrunk_covariance: falling back to ridge covariance estimate (%s).", exc)
        S = np.cov(X, rowvar=False)
        return S + RIDGE_RHO * np.eye(S.shape[0])


def _safe_inv(S: np.ndarray) -> np.ndarray | None:
    """Return a stable inverse, using the pseudo-inverse when needed."""
    try:
        return np.linalg.inv(S)
    except np.linalg.LinAlgError:
        log.warning("detect_peer: singular covariance matrix; using pseudo-inverse.")
        return np.linalg.pinv(S)


def _get_global_reference(
    df: pd.DataFrame,
    ratios: pd.DataFrame,
) -> PeerReferenceContext | None:
    """Build the global calibration reference on split ``C``."""
    df = df.copy()
    df["__peer_super_group"] = _build_peer_super_group(df)
    mask_cal = df["_split"] == "C" if "_split" in df.columns else pd.Series(True, index=df.index)
    x_cal = ratios.loc[mask_cal].dropna()
    n_global = len(x_cal)
    if n_global < MIN_PEER:
        log.warning(
            "detect_peer: calibration reference C too small (%d < %d); returning NaN.",
            n_global,
            MIN_PEER,
        )
        return None

    global_mu = x_cal.mean().to_numpy(dtype=float)
    global_cov = _shrunk_covariance(x_cal.to_numpy(dtype=float))
    global_inv = _safe_inv(global_cov)
    if global_inv is None:
        return None
    peer_group_cols = _available_peer_group_columns(df)
    peer_references = _precompute_peer_references(df, ratios, mask_cal, peer_group_cols)
    return PeerReferenceContext(
        df=df,
        mask_cal=mask_cal,
        ratios=ratios,
        x_cal=x_cal,
        global_mu=global_mu,
        global_inv=global_inv,
        peer_group_cols=peer_group_cols,
        peer_references=peer_references,
    )


def _precompute_peer_references(
    df: pd.DataFrame,
    ratios: pd.DataFrame,
    mask_cal: pd.Series,
    peer_group_cols: list[str],
) -> dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int]]:
    """Precompute one peer reference per usable group bucket."""
    references: dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int]] = {}
    for peer_group_col in peer_group_cols:
        for group_value, group_idx in df.groupby(peer_group_col, dropna=True).groups.items():
            group_mask = mask_cal & df.index.isin(group_idx)
            x_group = ratios.loc[group_mask].dropna()
            if len(x_group) < MIN_PEER:
                continue
            mu_ref = x_group.mean().to_numpy(dtype=float)
            inv_ref = _safe_inv(_shrunk_covariance(x_group.to_numpy(dtype=float)))
            if inv_ref is None:
                continue
            references[(peer_group_col, group_value)] = (mu_ref, inv_ref, len(x_group))
    return references


def _resolve_reference_for_row(
    idx: object,
    context: PeerReferenceContext,
) -> tuple[np.ndarray, np.ndarray, int, str] | None:
    """Return the mean, inverse covariance, and size of the row reference set.

    The current implementation uses a simple hierarchy:
    1. exact sector when it is large enough
    2. a business-driven super-group for thin sectors
    3. other available classification columns
    4. otherwise the global calibration cross-section
    """
    for peer_group_col in context.peer_group_cols:
        group_value = context.df.at[idx, peer_group_col]
        if pd.isna(group_value):
            continue

        cached = context.peer_references.get((peer_group_col, group_value))
        if cached is not None:
            mu_ref, inv_ref, n_group = cached
            return mu_ref, inv_ref, n_group, peer_group_col

    return context.global_mu, context.global_inv, len(context.x_cal), "global"


def _score_peer_row(
    row: np.ndarray,
    mu_ref: np.ndarray,
    inv_ref: np.ndarray,
    n_ref: int,
    n_dims: int,
) -> float:
    """Score one observation and map it to the common z-scale."""
    diff = row - mu_ref
    t7 = float(diff @ inv_ref @ diff)
    if n_ref >= 3 * n_dims:
        return float(pit_chi2(np.array([t7]), degrees_of_freedom=n_dims)[0])
    scale = (n_ref - n_dims) / (n_dims * (n_ref + 1))
    return float(pit_f(np.array([scale * t7]), dfn=n_dims, dfd=n_ref - n_dims)[0])


def detect_peer(df: pd.DataFrame) -> pd.Series:
    """Compute the ``z7`` peer-distance detector.

    Args:
        df: Input dataframe containing canonical Sharia ratios and, ideally, one
            peer-group column such as ``sector`` or ``gsector``.

    Returns:
        A series aligned on ``df.index``. Values are ``NaN`` when the reference
        set is too small or when the observation cannot be scored.

    Notes:
        The current implementation already documents the future extension path:
        thin sectors can later be mapped to grouped sectors or super-sectors
        before the final global fallback is used.
    """
    ratios = get_canonical_sharia_ratios(df)
    if ratios.empty:
        log.warning("detect_peer: no canonical ratios available; returning NaN.")
        return pd.Series(np.nan, index=df.index)

    n_dims = len(ratios.columns)
    reference = _get_global_reference(df, ratios)
    if reference is None:
        return pd.Series(np.nan, index=df.index)

    if not reference.peer_group_cols:
        log.info(
            "detect_peer: no peer-group column available; using the global cross-sectional C fallback."
        )
    else:
        log.info(
            "detect_peer: using hierarchical peer groups %s before the global fallback.",
            reference.peer_group_cols,
        )

    z7_vals = np.full(len(df), np.nan, dtype=float)
    ratios_mat = ratios.to_numpy(dtype=float)
    finite_mask = np.all(np.isfinite(ratios_mat), axis=1)

    # Pre-compute peer group assignments for all rows at once
    # to avoid per-row .at[idx, col] lookups
    row_group_map: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    if reference.peer_group_cols:
        # Build a vectorised group lookup: for each row, find which
        # cached peer reference applies
        for peer_group_col in reference.peer_group_cols:
            if peer_group_col not in reference.df.columns:
                continue
            group_vals = reference.df[peer_group_col].values
            for pos in range(len(df)):
                if pos in row_group_map or not finite_mask[pos]:
                    continue
                gv = group_vals[pos] if pos < len(group_vals) else None
                if gv is None or (isinstance(gv, float) and np.isnan(gv)):
                    continue
                cached = reference.peer_references.get((peer_group_col, gv))
                if cached is not None:
                    row_group_map[pos] = cached

    # Score all rows, batched by shared reference bucket. Rows that share a
    # peer reference are grouped by the identity of their cached (mu, inv, n)
    # tuple (the global fallback forms one batch), so the Mahalanobis distance
    # and the chi-square / F PIT run vectorised once per bucket instead of
    # paying scipy's per-call overhead once per row. This is the AnoShift
    # re-scoring hot path, where detect_peer is recomputed repeatedly.
    global_ref = (reference.global_mu, reference.global_inv, len(reference.x_cal))

    bucket_refs: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    bucket_positions: dict[int, list[int]] = {}
    for pos in range(len(df)):
        if not finite_mask[pos]:
            continue
        ref = row_group_map.get(pos, global_ref)
        key = id(ref)
        bucket_refs.setdefault(key, ref)
        bucket_positions.setdefault(key, []).append(pos)

    for key, (mu_ref, inv_ref, n_ref) in bucket_refs.items():
        idx = np.asarray(bucket_positions[key], dtype=int)
        diffs = ratios_mat[idx] - mu_ref
        # t7_i = diff_i @ inv_ref @ diff_i for the whole bucket at once.
        t7 = np.einsum("ij,jk,ik->i", diffs, inv_ref, diffs)
        if n_ref >= 3 * n_dims:
            z7_vals[idx] = pit_chi2(t7, degrees_of_freedom=n_dims)
        else:
            scale = (n_ref - n_dims) / (n_dims * (n_ref + 1))
            z7_vals[idx] = pit_f(scale * t7, dfn=n_dims, dfd=n_ref - n_dims)

    z7 = pd.Series(z7_vals, index=df.index, dtype=float)

    log.debug(
        "detect_peer: p=%d, N_global=%d, peer_group_hierarchy=%s.",
        n_dims, len(reference.x_cal), reference.peer_group_cols or ["global"],
    )
    return z7
