"""Composite z-score aggregations.

The active core uses four Working Paper composites built from detector z-scores:

- ``Z_plus``: positive-part aggregation
- ``Z_plus_renorm``: positive-part aggregation renormalized on the active set
- ``B_A``: anomaly breadth on the active set
- ``T_IUT``: intersection-union minimum statistic

This module also contains smoother extensions and bootstrap calibration helpers
kept for later stages of the project.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

N_BOOTSTRAP: int = 5_000
DETECTOR_COLS: list[str] = ["z1", "z2", "z3", "z4", "z5", "z6", "z7"]
WEIGHT_FLOOR: float = 1e-12
CHOL_REG: float = 1e-8
PSD_EIGEN_FLOOR: float = 1e-6
LOG_FLOOR: float = 1e-300
MIN_NONPARAM_CAL: int = 10


def compute_composite(
    z_df: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute the four active composite scores.

    Args:
        z_df: Dataframe containing one or more detector z-score columns.
        weights: Optional non-negative detector weights. When omitted, uniform
            weights are used over the detector columns present in ``z_df``.

    Returns:
        A dataframe aligned on ``z_df.index`` with ``Z_plus``,
        ``Z_plus_renorm``, ``B_A``, ``n_active``, and ``T_IUT``.
    """
    available = [c for c in DETECTOR_COLS if c in z_df.columns]
    if not available:
        raise ValueError("No detector z-score columns found in z_df.")

    k = len(available)
    Z = z_df[available].to_numpy(dtype=float)  # (n_obs, k)
    w = _normalize_weights(weights, k, available)

    finite_mask = np.isfinite(Z)
    positive = np.where(finite_mask & (Z > 0), Z, 0.0)
    active = finite_mask & (Z > 0)

    Z_plus = positive @ w

    w_active_sum = active @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        Z_plus_renorm = np.where(
            w_active_sum > WEIGHT_FLOOR,
            (positive * active) @ w / w_active_sum,
            np.nan,
        )

    # Keep NaN on rows where no detector can be observed at all.
    k_valid = finite_mask.sum(axis=1)
    B_A = active.sum(axis=1).astype(float) / float(k)
    B_A = np.where(k_valid > 0, B_A, np.nan)

    # T_IUT is the minimum over valid detector z-scores only (§5.2).
    Z_for_min = np.where(finite_mask, Z, np.inf)
    T_IUT = np.min(Z_for_min, axis=1)
    T_IUT = np.where(np.isinf(T_IUT), np.nan, T_IUT)

    return pd.DataFrame(
        {
            "Z_plus":        Z_plus,
            "Z_plus_renorm": Z_plus_renorm,
            "B_A":           B_A,
            "n_active":      active.sum(axis=1).astype(int),
            "T_IUT":         T_IUT,
        },
        index=z_df.index,
    )


def compute_mahalanobis(
    z_df: pd.DataFrame,
    Sigma: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute Mahalanobis-based joint scores.

    Args:
        z_df: Dataframe containing one or more detector z-score columns.
        Sigma: Optional detector covariance matrix estimated on calibration
            data. When omitted, the identity matrix is used.

    Returns:
        A dataframe aligned on ``z_df.index`` with ``Z_Mah`` and ``Z_Mah2``.
    """
    available = [c for c in DETECTOR_COLS if c in z_df.columns]
    if not available:
        raise ValueError("No detector z-score columns found in z_df.")

    Z = z_df[available].to_numpy(dtype=float)  # (n_obs, k)
    n_obs = Z.shape[0]
    if Sigma is None and np.isfinite(Z).all():
        z_mah2 = (Z ** 2).sum(axis=1)
        z_mah = np.sqrt(np.maximum(z_mah2, 0.0))
        return pd.DataFrame({"Z_Mah": z_mah, "Z_Mah2": z_mah2}, index=z_df.index)

    Z_mah = np.full(n_obs, np.nan)
    Z_mah2 = np.full(n_obs, np.nan)

    for i in range(n_obs):
        valid_idx = np.where(np.isfinite(Z[i]))[0]
        if len(valid_idx) == 0:
            continue
        z_v = Z[i, valid_idx]
        if Sigma is not None and len(valid_idx) > 1:
            Sigma_v = Sigma[np.ix_(valid_idx, valid_idx)]
            try:
                L = np.linalg.cholesky(Sigma_v + CHOL_REG * np.eye(len(valid_idx)))
                w = np.linalg.solve(L, z_v)
                mah2 = float(w @ w)
            except np.linalg.LinAlgError:
                log.warning("compute_mahalanobis: Cholesky failed; using the identity fallback.")
                mah2 = float(z_v @ z_v)
        else:
            mah2 = float(z_v @ z_v)
        Z_mah2[i] = mah2
        Z_mah[i] = np.sqrt(max(mah2, 0.0))

    return pd.DataFrame({"Z_Mah": Z_mah, "Z_Mah2": Z_mah2}, index=z_df.index)


def bootstrap_pvalues(
    z_df: pd.DataFrame,
    composite_df: pd.DataFrame,
    weights: np.ndarray | None = None,
    method: str = "parametric",
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = 42,
    df_scored: pd.DataFrame | None = None,
    beta: float = 5.0,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Bootstrap-calibrate composite p-values.

    Args:
        z_df: Detector z-scores.
        composite_df: Output of ``compute_composite`` aligned with ``z_df``.
        weights: Optional detector weights.
        method: Either ``"parametric"`` or ``"nonparametric"``.
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed.
        df_scored: Optional dataframe containing extension columns in addition to
            the base composites.
        beta: Smoothness parameter for sigmoid and softplus extensions.
        gamma: Temperature parameter for softmax-like extensions.

    Returns:
        A dataframe of bootstrap-calibrated p-values aligned on ``z_df.index``.
    """
    available = [c for c in DETECTOR_COLS if c in z_df.columns]
    k = len(available)
    w = _normalize_weights(weights, k, available)
    rng = np.random.default_rng(random_state)

    null_samples, Sigma = _generate_bootstrap_null_samples(
        z_df=z_df,
        available=available,
        method=method,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    if null_samples is None:
        return _null_pvalue_df(z_df.index)

    null_base = _compute_null_base_composites(null_samples, w)
    result = _compute_base_bootstrap_pvalues(
        composite_df=composite_df,
        null_base=null_base,
        index=z_df.index,
        n_bootstrap=n_bootstrap,
    )

    if df_scored is not None:
        _append_extension_bootstrap_pvalues(
            result=result,
            df_scored=df_scored,
            null_samples=null_samples,
            weights=w,
            beta=beta,
            gamma=gamma,
            sigma=Sigma if method == "parametric" else None,
            n_bootstrap=n_bootstrap,
        )

    return result


def iut_pvalue_exact(T_IUT: np.ndarray, k_valid: np.ndarray) -> np.ndarray:
    """Compute the exact IUT p-value under detector independence.

    Args:
        T_IUT: Observed IUT statistics.
        k_valid: Number of non-missing detectors per observation.

    Returns:
        One-sided IUT p-values aligned with the input arrays.
    """
    T = np.asarray(T_IUT, dtype=float)
    k = np.asarray(k_valid, dtype=float)
    p_single = 1.0 - norm.cdf(T)
    return np.where(np.isfinite(T) & (k > 0), p_single ** k, np.nan)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_weights(
    weights: np.ndarray | None,
    k: int,
    available: list[str],
) -> np.ndarray:
    if weights is None:
        return np.ones(k) / k
    w_full = np.asarray(weights, dtype=float)
    if len(w_full) == len(DETECTOR_COLS):
        idx = [DETECTOR_COLS.index(c) for c in available if c in DETECTOR_COLS]
        w = w_full[idx]
    else:
        if len(w_full) != k:
            raise ValueError(
                f"weights has length {len(w_full)}, expected {k} or {len(DETECTOR_COLS)}."
            )
        w = w_full
    s = w.sum()
    return w / s if s > WEIGHT_FLOOR else np.ones(k) / k


def _generate_bootstrap_null_samples(
    z_df: pd.DataFrame,
    available: list[str],
    method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Generate null detector samples for bootstrap calibration."""
    k = len(available)
    mask_cal = z_df["_split"] == "C" if "_split" in z_df.columns else pd.Series(True, index=z_df.index)
    if method == "parametric":
        z_all = z_df.loc[mask_cal, available].to_numpy(dtype=float)
        sigma = _pairwise_cov(z_all)
        if sigma is None:
            log.warning("bootstrap_pvalues: covariance is not estimable; returning NaN p-values.")
            return None, None
        null_samples = rng.multivariate_normal(np.zeros(k), sigma, size=n_bootstrap)
        return null_samples, sigma

    # Keep partially observed rows on purpose so the bootstrap matches the
    # missing-data pattern seen by the composite computations.
    z_cal = z_df.loc[mask_cal, available].dropna(how="all").to_numpy(dtype=float)
    if len(z_cal) < MIN_NONPARAM_CAL:
        log.warning("bootstrap_pvalues: calibration split C is too small; returning NaN p-values.")
        return None, None
    idx = rng.integers(0, len(z_cal), size=n_bootstrap)
    return z_cal[idx], None


def _compute_null_base_composites(
    null_samples: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute null draws for the base Working Paper composites."""
    pos_null = np.where(np.isfinite(null_samples) & (null_samples > 0), null_samples, 0.0)
    active_null = np.isfinite(null_samples) & (null_samples > 0)
    z_plus_null = pos_null @ weights
    active_weight_sum = active_null @ weights
    with np.errstate(invalid="ignore", divide="ignore"):
        z_renorm_null = np.where(
            active_weight_sum > WEIGHT_FLOOR,
            (pos_null * active_null) @ weights / active_weight_sum,
            np.nan,
        )
    z_for_min = np.where(np.isfinite(null_samples), null_samples, np.inf)
    t_iut_null = np.min(z_for_min, axis=1)
    t_iut_null = np.where(np.isinf(t_iut_null), np.nan, t_iut_null)
    k_valid_null = np.isfinite(null_samples).sum(axis=1)
    b_a_null = np.where(
        k_valid_null > 0,
        active_null.sum(axis=1) / np.maximum(null_samples.shape[1], 1),
        np.nan,
    )
    return {
        "Z_plus": z_plus_null,
        "Z_plus_renorm": z_renorm_null,
        "T_IUT": t_iut_null,
        "B_A": b_a_null,
    }


def _bootstrap_pvalue(obs_arr: np.ndarray, null_arr: np.ndarray, n_bootstrap: int) -> np.ndarray:
    """Return one-sided empirical bootstrap p-values."""
    return np.array([
        np.nan if not np.isfinite(obs)
        else (1 + (null_arr >= obs).sum()) / (n_bootstrap + 1)
        for obs in obs_arr
    ])


def _compute_base_bootstrap_pvalues(
    composite_df: pd.DataFrame,
    null_base: dict[str, np.ndarray],
    index: pd.Index,
    n_bootstrap: int,
) -> pd.DataFrame:
    """Compute bootstrap p-values for the active base composites.

    Note:
        ``p_IUT`` is bootstrapped from the full set of available detector
        columns in ``z_df``. When some observations have structurally missing
        detectors, the exact observation-specific p-value should instead come
        from ``iut_pvalue_exact`` using ``k_valid``.
    """
    result = pd.DataFrame(index=index)
    if "Z_plus" in composite_df.columns:
        result["p_Z_plus"] = _bootstrap_pvalue(composite_df["Z_plus"].values, null_base["Z_plus"], n_bootstrap)
    if "Z_plus_renorm" in composite_df.columns:
        result["p_Z_plus_renorm"] = _bootstrap_pvalue(
            composite_df["Z_plus_renorm"].values,
            null_base["Z_plus_renorm"],
            n_bootstrap,
        )
    if "T_IUT" in composite_df.columns:
        result["p_IUT"] = _bootstrap_pvalue(composite_df["T_IUT"].values, null_base["T_IUT"], n_bootstrap)
    if "B_A" in composite_df.columns:
        result["p_B_A"] = _bootstrap_pvalue(composite_df["B_A"].values, null_base["B_A"], n_bootstrap)
    return result


def _append_extension_bootstrap_pvalues(
    result: pd.DataFrame,
    df_scored: pd.DataFrame,
    null_samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    gamma: float,
    sigma: np.ndarray | None,
    n_bootstrap: int,
) -> None:
    """Append bootstrap p-values for optional extension scores in place."""
    extension_nulls = _compute_extension_null_scores(
        null_samples=null_samples,
        weights=weights,
        beta=beta,
        gamma=gamma,
        sigma=sigma,
    )

    if "Z_plus_sigmoid" in df_scored.columns:
        result["p_Z_plus_sigmoid"] = _bootstrap_pvalue(
            df_scored["Z_plus_sigmoid"].values,
            extension_nulls["Z_plus_sigmoid"],
            n_bootstrap,
        )
    if "Z_plus_softplus" in df_scored.columns:
        result["p_Z_plus_softplus"] = _bootstrap_pvalue(
            df_scored["Z_plus_softplus"].values,
            extension_nulls["Z_plus_softplus"],
            n_bootstrap,
        )
    if "Z_plus_softmax" in df_scored.columns:
        result["p_Z_plus_softmax"] = _bootstrap_pvalue(
            df_scored["Z_plus_softmax"].values,
            extension_nulls["Z_plus_softmax"],
            n_bootstrap,
        )

    if "Z_plus_orth" in df_scored.columns:
        result["p_Z_plus_orth"] = _bootstrap_pvalue(
            df_scored["Z_plus_orth"].values,
            extension_nulls["Z_plus_orth"],
            n_bootstrap,
        )

    if "Z_Mah" in df_scored.columns or "Z_Mah2" in df_scored.columns:
        if "Z_Mah" in df_scored.columns:
            result["p_Z_Mah"] = _bootstrap_pvalue(
                df_scored["Z_Mah"].values,
                extension_nulls["Z_Mah"],
                n_bootstrap,
            )
        if "Z_Mah2" in df_scored.columns:
            result["p_Z_Mah2"] = _bootstrap_pvalue(
                df_scored["Z_Mah2"].values,
                extension_nulls["Z_Mah2"],
                n_bootstrap,
            )


def _compute_extension_null_scores(
    null_samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    gamma: float,
    sigma: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Compute null draws for all optional extension scores."""
    z_fill_null = np.where(np.isfinite(null_samples), null_samples, 0.0)
    finite_null = np.isfinite(null_samples)
    z_mah_null, z_mah2_null = _compute_mahalanobis_null_scores(null_samples, sigma)
    return {
        "Z_plus_sigmoid": (_sigmoid_gate(beta * z_fill_null) * z_fill_null) @ weights,
        "Z_plus_softplus": (np.logaddexp(0.0, beta * z_fill_null) / beta) @ weights,
        "Z_plus_softmax": softmax_active(z_fill_null, finite_null, weights, gamma),
        "Z_plus_orth": orthogonal_softmax(z_fill_null, finite_null, weights, gamma, sigma),
        "Z_Mah": z_mah_null,
        "Z_Mah2": z_mah2_null,
    }


def _compute_mahalanobis_null_scores(
    null_samples: np.ndarray,
    sigma: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mahalanobis null scores using ``sigma`` when available."""
    if sigma is not None:
        try:
            chol = np.linalg.cholesky(sigma + CHOL_REG * np.eye(sigma.shape[0]))
            whitened = null_samples @ np.linalg.inv(chol).T
            z_mah2_null = (whitened ** 2).sum(axis=1)
        except np.linalg.LinAlgError:
            z_mah2_null = (null_samples ** 2).sum(axis=1)
    else:
        z_mah2_null = (null_samples ** 2).sum(axis=1)
    z_mah_null = np.sqrt(np.maximum(z_mah2_null, 0.0))
    return z_mah_null, z_mah2_null


def _pairwise_cov(Z: np.ndarray) -> np.ndarray | None:
    """Estimate covariance using pairwise-complete observations.

    Args:
        Z: Two-dimensional detector z-score matrix.

    Returns:
        A positive semi-definite covariance estimate, or ``None`` when a column
        has too few observations.
    """
    k = Z.shape[1]
    Sigma = np.eye(k)
    for i in range(k):
        for j in range(i, k):
            mask = np.isfinite(Z[:, i]) & np.isfinite(Z[:, j])
            if mask.sum() < 2:
                return None
            xi, xj = Z[mask, i], Z[mask, j]
            cov_ij = float(np.cov(xi, xj)[0, 1])
            Sigma[i, j] = cov_ij
            Sigma[j, i] = cov_ij
        Sigma[i, i] = float(np.var(Z[np.isfinite(Z[:, i]), i], ddof=1))
    # Ensure positive semi-definiteness for downstream simulations.
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += (-eigvals.min() + PSD_EIGEN_FLOOR) * np.eye(k)
    return Sigma


def _null_pvalue_df(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p_Z_plus": np.nan, "p_Z_plus_renorm": np.nan, "p_IUT": np.nan,
            "p_B_A": np.nan,
            "p_Z_plus_sigmoid": np.nan, "p_Z_plus_softplus": np.nan,
            "p_Z_plus_softmax": np.nan, "p_Z_plus_orth": np.nan,
            "p_Z_Mah": np.nan, "p_Z_Mah2": np.nan,
        },
        index=index,
    )


# ── Smooth Z⁺ extensions (extensions_Z_score.md) ─────────────────────────────

def compute_extensions(
    z_df: pd.DataFrame,
    weights: np.ndarray | None = None,
    beta: float = 5.0,
    gamma: float = 1.0,
    Sigma: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute the smooth positive-part aggregation extensions.

    Args:
        z_df: Dataframe containing one or more detector z-score columns.
        weights: Optional non-negative detector weights. When omitted, uniform
            weights are used.
        beta: Smoothness parameter for sigmoid and softplus variants.
        gamma: Temperature parameter for softmax-like variants.
        Sigma: Optional detector covariance matrix used by the orthogonal
            extension.

    Returns:
        A dataframe aligned on ``z_df.index`` with the smooth aggregation
        variants ``Z_plus_sigmoid``, ``Z_plus_softplus``, ``Z_plus_softmax``,
        and ``Z_plus_orth``.
    """
    available = [c for c in DETECTOR_COLS if c in z_df.columns]
    if not available:
        raise ValueError("No detector z-score columns found in z_df.")

    k = len(available)
    Z = z_df[available].to_numpy(dtype=float)          # (n_obs, k)
    w = _normalize_weights(weights, k, available)
    finite_mask = np.isfinite(Z)
    Z_fill = np.where(finite_mask, Z, 0.0)             # NaN -> 0 before smooth transforms

    # 1. Sigmoid extension: Z⁺_σ(β) = Σⱼ wⱼ zⱼ σ(β zⱼ)
    Z_sigmoid = (_sigmoid_gate(beta * Z_fill) * Z_fill) @ w

    # np.logaddexp keeps the softplus implementation numerically stable.
    Z_softplus = (np.logaddexp(0.0, beta * Z_fill) / beta) @ w

    # 3. Soft-max over the active set A(z) = {j : zⱼ > 0}
    Z_softmax = softmax_active(Z_fill, finite_mask, w, gamma)

    # 4. Soft-max with sequential orthogonalization (Gram-Schmidt)
    Z_orth = orthogonal_softmax(Z_fill, finite_mask, w, gamma, Sigma)

    return pd.DataFrame(
        {
            "Z_plus_sigmoid":  Z_sigmoid,
            "Z_plus_softplus": Z_softplus,
            "Z_plus_softmax":  Z_softmax,
            "Z_plus_orth":     Z_orth,
        },
        index=z_df.index,
    )


def _sigmoid_gate(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable logistic gate."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def softmax_active(
    Z: np.ndarray,
    finite_mask: np.ndarray,
    w: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute the active-set softmax aggregation.

    Args:
        Z: Detector z-score matrix with missing values already filled.
        finite_mask: Boolean mask marking finite detector scores.
        w: Detector weights aligned with ``Z`` columns.
        gamma: Softmax temperature parameter.

    Returns:
        One aggregated score per observation. When the caller used
        ``weights=None``, this matches the "no exogenous weights" extension.
    """
    active = finite_mask & (Z > 0)                            # (n_obs, k)

    # Subtract the row-wise maximum active score for numerical stability.
    Z_act = np.where(active, Z, -np.inf)
    z_max = np.max(Z_act, axis=1, keepdims=True)
    z_max = np.where(np.isneginf(z_max), 0.0, z_max)

    exp_w = np.where(active, w * np.exp(gamma * (Z - z_max)), 0.0)
    denom = exp_w.sum(axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        w_tilde = np.where(denom > WEIGHT_FLOOR, exp_w / denom, 0.0)

    score = (w_tilde * np.where(active, Z, 0.0)).sum(axis=1)
    has_active = active.any(axis=1)
    return np.where(has_active, score, np.nan)


def orthogonal_softmax(
    Z: np.ndarray,
    finite_mask: np.ndarray,
    w: np.ndarray,
    gamma: float,
    Sigma: np.ndarray | None,
) -> np.ndarray:
    """Compute the orthogonalized active-set softmax aggregation.

    Args:
        Z: Detector z-score matrix with missing values already filled.
        finite_mask: Boolean mask marking finite detector scores.
        w: Detector weights aligned with ``Z`` columns.
        gamma: Softmax temperature parameter.
        Sigma: Optional detector covariance matrix used to estimate the
            information novelty term.

    Returns:
        One orthogonalized aggregated score per observation, restricted to the
        active detector set ``A(z) = {j : z_j > 0 and finite}``.
    """
    n_obs, k = Z.shape
    result = np.full(n_obs, np.nan, dtype=float)

    for i in range(n_obs):
        valid = np.where(finite_mask[i] & (Z[i] > 0))[0]
        if len(valid) == 0:
            continue

        # Sort active detectors by descending anomaly strength.
        order = valid[np.argsort(-Z[i, valid])]
        z_ord = Z[i, order]   # sorted values
        w_ord = w[order]      # aligned weights

        # Estimate novelty terms delta over the active ordered detectors.
        delta = np.ones(len(order))
        if Sigma is not None:
            for m in range(1, len(order)):
                idx_m   = order[m]
                prev    = order[:m]
                cov_mp  = Sigma[idx_m, prev]                       # (m,)
                Sigma_p = Sigma[np.ix_(prev, prev)]                # (m, m)
                try:
                    coeff = np.linalg.solve(
                        Sigma_p + CHOL_REG * np.eye(len(prev)), cov_mp
                    )
                except np.linalg.LinAlgError:
                    continue
                proj  = coeff @ Z[i, prev]
                r     = Z[i, idx_m] - proj
                z_sq  = Z[i, idx_m] ** 2
                delta[m] = min(r ** 2 / z_sq, 1.0) if z_sq > WEIGHT_FLOOR else 0.0

        # Work on log-weights first to avoid softmax overflow.
        log_u = gamma * z_ord + np.log(np.maximum(w_ord * delta, LOG_FLOOR))
        log_u -= log_u.max()
        unnorm = np.exp(log_u)
        denom  = unnorm.sum()

        if denom > WEIGHT_FLOOR:
            result[i] = (unnorm / denom) @ z_ord

    return result
