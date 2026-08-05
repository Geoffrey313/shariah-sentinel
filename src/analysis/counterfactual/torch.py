"""Torch backend for canonical Family 3 counterfactual robustness.

This module collects the differentiable surrogate, exact replay helpers, and
Adam-based optimization adapters used by the public `torch_adam` path.
"""


from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class TorchDeviceInfo:
    device: torch.device
    dtype: torch.dtype
    cuda_available: bool
    mps_available: bool


def resolve_torch_device(prefer_gpu: bool = True) -> TorchDeviceInfo:
    """Resolve the torch device/dtype used by the whole Family 3 torch surrogate.

    MPS (Apple Silicon) is reported in ``mps_available`` for diagnostics but is
    never selected as ``device``: every numerical routine in this module (PIT,
    chi-square, Mahalanobis) is hardcoded to ``float64``, and MPS cannot
    allocate float64 tensors at all — selecting it crashes on the first tensor
    build. CUDA supports float64 and remains a valid GPU target.
    """
    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if prefer_gpu and cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return TorchDeviceInfo(
        device=device,
        dtype=torch.float64,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )


def tensor_from_array(
    values: np.ndarray | list[float],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    tensor_device = device or torch.device("cpu")
    return torch.as_tensor(np.array(values, dtype=float, copy=True), dtype=dtype, device=tensor_device)


def tensor_from_frame(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float, copy=False)
    return tensor_from_array(values, device=device, dtype=dtype)


"""Torch evaluation helpers for Family 3 exact composite rescoring."""

from dataclasses import dataclass

import numpy as np
import torch

from src.analysis.counterfactual.utils import (
    COMPOSITE_P_COLUMNS,
    COMPOSITE_VALUE_COLUMNS,
    Family3FastRowEvaluator,
    family3_column_scale_floor,
    family3_current_ratios,
    family3_reference_ratio_stats,
)


def _torch_norm_isf(p: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    return torch.special.ndtri(1.0 - clipped)


@dataclass(frozen=True)
class TorchNullContext:
    z_plus_sorted: torch.Tensor
    z_plus_renorm_sorted: torch.Tensor
    breadth_sorted: torch.Tensor
    z_mahalanobis_sorted: torch.Tensor
    t_iut_sorted: torch.Tensor
    n_replicates: int


def torch_null_context_from_score_context(score_ctx, *, device: torch.device, dtype: torch.dtype) -> TorchNullContext:
    null = score_ctx.null
    return TorchNullContext(
        z_plus_sorted=tensor_from_array(null.z_plus_sorted, device=device, dtype=dtype),
        z_plus_renorm_sorted=tensor_from_array(null.z_plus_renorm_sorted, device=device, dtype=dtype),
        breadth_sorted=tensor_from_array(null.breadth_sorted, device=device, dtype=dtype),
        z_mahalanobis_sorted=tensor_from_array(null.z_mahalanobis_sorted, device=device, dtype=dtype),
        t_iut_sorted=tensor_from_array(null.t_iut_sorted, device=device, dtype=dtype),
        n_replicates=int(null.n_replicates),
    )


def torch_upper_tail_pvalue(
    observed: torch.Tensor,
    null_sorted: torch.Tensor,
    n_replicates: int,
) -> torch.Tensor:
    if null_sorted.numel() == 0:
        return torch.full_like(observed, float("nan"))
    idx = torch.searchsorted(null_sorted, observed, right=False)
    count_ge = null_sorted.numel() - idx.to(observed.dtype)
    p = (1.0 + count_ge) / (n_replicates + 1.0)
    return torch.where(torch.isfinite(observed), p, torch.full_like(p, float("nan")))


def torch_attach_pvalues(
    composites: dict[str, torch.Tensor],
    null_ctx: TorchNullContext,
) -> dict[str, torch.Tensor]:
    return {
        "p_z_plus": torch_upper_tail_pvalue(composites["z_plus"], null_ctx.z_plus_sorted, null_ctx.n_replicates),
        "p_z_plus_renorm": torch_upper_tail_pvalue(composites["z_plus_renorm"], null_ctx.z_plus_renorm_sorted, null_ctx.n_replicates),
        "p_breadth": torch_upper_tail_pvalue(composites["breadth"], null_ctx.breadth_sorted, null_ctx.n_replicates),
        "p_z_mahalanobis_sq": torch_upper_tail_pvalue(composites["z_mahalanobis_sq"], null_ctx.z_mahalanobis_sorted, null_ctx.n_replicates),
        "p_t_iut": torch_upper_tail_pvalue(composites["t_iut"], null_ctx.t_iut_sorted, null_ctx.n_replicates),
    }


def torch_family3_target_score_z(
    *,
    target_score_name: str,
    composite_values: dict[str, torch.Tensor],
    composite_pvalues: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if target_score_name in COMPOSITE_VALUE_COLUMNS:
        if composite_pvalues is not None:
            p_col = f"p_{target_score_name}"
            if p_col in composite_pvalues:
                return _torch_norm_isf(composite_pvalues[p_col])
        return composite_values[target_score_name]
    if target_score_name in COMPOSITE_P_COLUMNS and composite_pvalues is not None:
        return _torch_norm_isf(composite_pvalues[target_score_name])
    raise KeyError(f"unsupported torch target score {target_score_name!r}")


def torch_family3_exact_cohort_eval(
    *,
    z_matrix: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    target_score_name: str,
    direction: str,
    delta_norm: torch.Tensor,
    threshold: float,
    lambda_l1: float,
    lambda_l2: float = 0.0,
    loss_name: str = "hinge_squared",
    aggregate_mode: str = "mean",
    active_threshold: float,
    min_active_for_renorm: int,
    min_active_for_iut: int,
    null_ctx: TorchNullContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Evaluate torch composites and Family 3 loss terms for one cohort state."""
    composite_values = torch_composite_columns(
        z_matrix,
        sigma,
        weights,
        active_threshold=active_threshold,
        min_active_for_renorm=min_active_for_renorm,
        min_active_for_iut=min_active_for_iut,
    )
    composite_pvalues = torch_attach_pvalues(composite_values, null_ctx) if null_ctx is not None else None
    target_score_z = torch_family3_target_score_z(
        target_score_name=target_score_name,
        composite_values=composite_values,
        composite_pvalues=composite_pvalues,
    )
    loss_total, loss_score, loss_l1, loss_l2 = torch_cohort_mean_loss(
        target_score_z=target_score_z,
        direction=direction,
        delta_norm=delta_norm,
        threshold=threshold,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        loss_name=loss_name,
        aggregate_mode=aggregate_mode,
    )
    return loss_total, loss_score, loss_l1, loss_l2, composite_values, composite_pvalues


def torch_selected_composite_inputs(score_ctx, *, device: torch.device, dtype: torch.dtype):
    active = list(score_ctx.active)
    z_np = score_ctx.zscores.loc[:, active].to_numpy(dtype=float, copy=False)
    sigma_np = np.asarray(score_ctx.sigma, dtype=float)
    weights_np = np.asarray(score_ctx.weights, dtype=float)
    return (
        tensor_from_array(z_np, device=device, dtype=dtype),
        tensor_from_array(sigma_np, device=device, dtype=dtype),
        tensor_from_array(weights_np, device=device, dtype=dtype),
        active,
    )


"""Torch kernels for Family 3 exact loss and composite calculations.

This module keeps the numerical pieces small and composable so the surrounding
engine can choose loss families without duplicating the exact-loss semantics
implemented in the numpy path.
"""

import torch


def _validate_z(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"z must be 2-D, got shape {tuple(z.shape)}")
    return z


def _validate_weights(weights: torch.Tensor, k: int) -> torch.Tensor:
    if tuple(weights.shape) != (k,):
        raise ValueError(f"weights shape {tuple(weights.shape)} != ({k},)")
    if bool((weights < 0).any().item()):
        raise ValueError("weights must be non-negative")
    weight_sum = float(weights.sum().item())
    if not torch.isfinite(weights).all():
        raise ValueError("weights must be finite")
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1 (got {weight_sum})")
    return weights


def torch_normalized_delta(
    candidate_x: torch.Tensor,
    baseline_x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    safe_scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    out = (candidate_x - baseline_x) / safe_scale
    return torch.where(scale > 1e-12, out, torch.zeros_like(out))


def _torch_family3_margin(
    *,
    target_score_z: torch.Tensor,
    direction: str,
    threshold: float,
) -> torch.Tensor:
    """Return the signed distance from the Family 3 target threshold."""
    if direction == "to_green":
        return target_score_z - threshold
    return threshold - target_score_z


def _torch_hinge_score_loss(margin: torch.Tensor) -> torch.Tensor:
    """Return the linear hinge score loss."""
    return torch.clamp(margin, min=0.0)


def _torch_hinge_squared_score_loss(margin: torch.Tensor) -> torch.Tensor:
    """Return the squared hinge score loss."""
    hinge = _torch_hinge_score_loss(margin)
    return hinge * hinge


def _torch_mse_score_loss(margin: torch.Tensor) -> torch.Tensor:
    """Return the mean-squared-error style score loss on the raw margin."""
    return margin * margin


def torch_score_loss_term(
    *,
    target_score_z: torch.Tensor,
    direction: str,
    threshold: float,
    loss_name: str = "hinge_squared",
) -> torch.Tensor:
    """Return the score-only Family 3 loss for the requested loss family."""
    margin = _torch_family3_margin(
        target_score_z=target_score_z,
        direction=direction,
        threshold=threshold,
    )
    if loss_name == "hinge_squared":
        return _torch_hinge_squared_score_loss(margin)
    if loss_name == "hinge":
        return _torch_hinge_score_loss(margin)
    if loss_name == "mse":
        return _torch_mse_score_loss(margin)
    raise ValueError(f"Unsupported Family 3 loss {loss_name!r}.")


def torch_exact_loss(
    *,
    target_score_z: torch.Tensor,
    direction: str,
    delta_norm: torch.Tensor,
    threshold: float,
    lambda_l1: float,
    lambda_l2: float = 0.0,
    loss_name: str = "hinge_squared",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row-wise Family 3 exact loss terms in the torch path."""
    loss_score = torch_score_loss_term(
        target_score_z=target_score_z,
        direction=direction,
        threshold=threshold,
        loss_name=loss_name,
    )
    loss_l1 = lambda_l1 * torch.abs(delta_norm).sum()
    loss_l2 = lambda_l2 * torch.square(delta_norm).sum()
    return loss_score + loss_l1 + loss_l2, loss_score, loss_l1, loss_l2


def torch_cohort_mean_loss(
    *,
    target_score_z: torch.Tensor,
    direction: str,
    delta_norm: torch.Tensor,
    threshold: float,
    lambda_l1: float,
    lambda_l2: float = 0.0,
    loss_name: str = "hinge_squared",
    aggregate_mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cohort-aggregated Family 3 exact loss terms in the torch path."""
    score_terms = torch_score_loss_term(
        target_score_z=target_score_z,
        direction=direction,
        threshold=threshold,
        loss_name=loss_name,
    )
    l1_terms = lambda_l1 * torch.abs(delta_norm).sum(dim=1)
    l2_terms = lambda_l2 * torch.square(delta_norm).sum(dim=1)
    if aggregate_mode == "mean":
        loss_score = torch.mean(score_terms)
        loss_l1 = torch.mean(l1_terms)
        loss_l2 = torch.mean(l2_terms)
    elif aggregate_mode == "min":
        total_terms = score_terms + l1_terms + l2_terms
        pivot = torch.argmax(total_terms)
        loss_score = score_terms[pivot]
        loss_l1 = l1_terms[pivot]
        loss_l2 = l2_terms[pivot]
    else:
        raise ValueError(f"Unsupported torch cohort aggregate mode {aggregate_mode!r}.")
    return loss_score + loss_l1 + loss_l2, loss_score, loss_l1, loss_l2


def torch_truncated_sum(z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    arr = _validate_z(z)
    w = _validate_weights(weights, arr.shape[1])
    finite = torch.isfinite(arr)
    positive = torch.where(finite, torch.clamp(arr, min=0.0), torch.zeros_like(arr))
    row_has_data = finite.any(dim=1)
    out = positive @ w
    nan_fill = torch.full_like(out, float("nan"))
    return torch.where(row_has_data, out, nan_fill)


def torch_active_set_indicator(z: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    arr = _validate_z(z)
    return torch.isfinite(arr) & (arr > threshold)


def torch_breadth(z: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    active = torch_active_set_indicator(z, threshold=threshold)
    k = active.shape[1]
    return active.sum(dim=1, dtype=z.dtype) / z.new_tensor(float(k))


def _regularized_solve(sigma: torch.Tensor, block_t: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(sigma.shape[0], dtype=sigma.dtype, device=sigma.device)
    diag_scale = torch.mean(torch.abs(torch.diagonal(sigma)))
    base_jitter = torch.clamp(diag_scale, min=torch.finfo(sigma.dtype).eps)
    for multiplier in (0.0, 1.0, 10.0, 100.0):
        matrix = sigma if multiplier == 0.0 else sigma + eye * (base_jitter * multiplier)
        try:
            solved = torch.linalg.solve(matrix, block_t)
        except RuntimeError:
            continue
        if bool(torch.isfinite(solved).all().item()):
            return solved
    return torch.linalg.lstsq(sigma, block_t).solution


def torch_renormalised_truncated_sum(
    z: torch.Tensor,
    weights: torch.Tensor,
    *,
    threshold: float = 0.0,
    min_active: int = 1,
) -> torch.Tensor:
    arr = _validate_z(z)
    w = _validate_weights(weights, arr.shape[1])
    active = torch_active_set_indicator(arr, threshold=threshold)
    n_active = active.sum(dim=1)
    numerator = torch.where(active, torch.where(torch.isfinite(arr), arr, torch.zeros_like(arr)) * w, torch.zeros_like(arr)).sum(dim=1)
    denominator = torch.where(active, w, torch.zeros_like(arr)).sum(dim=1)
    out = numerator / denominator
    nan_fill = torch.full_like(out, float("nan"))
    return torch.where(n_active >= min_active, out, nan_fill)


def torch_mahalanobis_squared(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    arr = _validate_z(z)
    if tuple(sigma.shape) != (arr.shape[1], arr.shape[1]):
        raise ValueError(f"sigma shape {tuple(sigma.shape)} != ({arr.shape[1]}, {arr.shape[1]})")
    complete = torch.isfinite(arr).all(dim=1)
    out = torch.full((arr.shape[0],), float("nan"), dtype=arr.dtype, device=arr.device)
    if bool(complete.any().item()):
        block = arr[complete]
        solved = _regularized_solve(sigma, block.T).T
        out[complete] = torch.einsum("ij,ij->i", block, solved)
    return out


def torch_iut_statistic(z: torch.Tensor, min_active: int = 1) -> torch.Tensor:
    arr = _validate_z(z)
    finite = torch.isfinite(arr)
    n_finite = finite.sum(dim=1)
    inf_fill = torch.full_like(arr, float("inf"))
    safe = torch.where(finite, arr, inf_fill)
    minimum = safe.min(dim=1).values
    nan_fill = torch.full_like(minimum, float("nan"))
    minimum = torch.where(n_finite >= min_active, minimum, nan_fill)
    return torch.where(torch.isfinite(minimum), minimum, nan_fill)


def torch_composite_columns(
    z_matrix: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    *,
    active_threshold: float,
    min_active_for_renorm: int,
    min_active_for_iut: int,
) -> dict[str, torch.Tensor]:
    return {
        "z_plus": torch_truncated_sum(z_matrix, weights),
        "z_plus_renorm": torch_renormalised_truncated_sum(
            z_matrix,
            weights,
            threshold=active_threshold,
            min_active=min_active_for_renorm,
        ),
        "breadth": torch_breadth(z_matrix, threshold=active_threshold),
        "z_mahalanobis_sq": torch_mahalanobis_squared(z_matrix, sigma),
        "t_iut": torch_iut_statistic(z_matrix, min_active=min_active_for_iut),
    }


from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TorchOptimizationTrace:
    epoch: int
    loss_total: float


@dataclass(frozen=True)
class TorchOptimizationResult:
    best_state: torch.Tensor
    best_loss: float
    executed_epochs: int
    traces: tuple[TorchOptimizationTrace, ...]
    state_trace: tuple[torch.Tensor, ...]


ObjectiveFn = Callable[[torch.Tensor], torch.Tensor]
ProjectionFn = Callable[[torch.Tensor], torch.Tensor]


def _maybe_shrink_learning_rate(
    *,
    current_step_size: float,
    min_step_size: float,
    shrink_factor: float,
    plateau_shrink_patience: int,
    epochs_without_improvement: int,
) -> tuple[float, bool]:
    """Optionally reduce the torch Adam learning rate after a plateau."""
    if plateau_shrink_patience <= 0 or shrink_factor >= 1.0:
        return current_step_size, False
    if epochs_without_improvement < plateau_shrink_patience:
        return current_step_size, False
    next_step_size = max(min_step_size, current_step_size * shrink_factor)
    if next_step_size >= current_step_size - 1e-15:
        return current_step_size, False
    return next_step_size, True


def run_torch_adam(
    *,
    initial_state: torch.Tensor,
    objective_fn: ObjectiveFn,
    projection_fn: ProjectionFn | None = None,
    max_epochs: int,
    step_size: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    early_stop_patience: int = 0,
    early_stop_loss: float | None = None,
    plateau_shrink_patience: int = 0,
    plateau_shrink_factor: float = 1.0,
    min_step_size: float = 0.0,
    restore_best_on_shrink: bool = False,
    reset_moments_on_shrink: bool = False,
) -> TorchOptimizationResult:
    state = initial_state.detach().clone()
    state.requires_grad_(True)
    optimizer = torch.optim.Adam([state], lr=step_size, betas=(beta1, beta2), eps=eps)
    current_step_size = float(step_size)

    traces: list[TorchOptimizationTrace] = []
    state_trace: list[torch.Tensor] = []
    best_state = state.detach().clone()
    with torch.no_grad():
        best_loss = float(objective_fn(state).detach().cpu().item())
    traces.append(TorchOptimizationTrace(epoch=0, loss_total=best_loss))
    state_trace.append(state.detach().clone())
    epochs_without_improvement = 0
    executed_epochs = 0

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective_fn(state)
        current_loss = float(loss.detach().cpu().item())

        # Epoch 1 still evaluates the already-recorded baseline state. Starting
        # at epoch 2, the loop's single forward is the authoritative loss for
        # the state produced by the previous optimizer step.
        if epoch > 1:
            traces.append(TorchOptimizationTrace(epoch=epoch - 1, loss_total=current_loss))
            if current_loss <= best_loss + 1e-12:
                best_loss = current_loss
                best_state = state.detach().clone()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if early_stop_loss is not None and current_loss <= early_stop_loss:
                break
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                break

        loss.backward()
        optimizer.step()
        if projection_fn is not None:
            with torch.no_grad():
                projected = projection_fn(state)
                state.copy_(projected)
        state_trace.append(state.detach().clone())
        executed_epochs = epoch
        current_step_size, shrunk = _maybe_shrink_learning_rate(
            current_step_size=current_step_size,
            min_step_size=min_step_size,
            shrink_factor=plateau_shrink_factor,
            plateau_shrink_patience=plateau_shrink_patience,
            epochs_without_improvement=epochs_without_improvement,
        )
        if shrunk:
            epochs_without_improvement = 0
            for group in optimizer.param_groups:
                group["lr"] = current_step_size
            if restore_best_on_shrink:
                with torch.no_grad():
                    state.copy_(best_state)
                    if projection_fn is not None:
                        state.copy_(projection_fn(state))
            if reset_moments_on_shrink:
                # A shrink+restore teleports the parameter state back in time.
                # Clearing Adam's full optimizer state avoids reusing stale
                # moments and step counters from the pre-restore trajectory.
                optimizer.state.clear()

    with torch.no_grad():
        final_loss = float(objective_fn(state).detach().cpu().item())
    if executed_epochs > 0:
        traces.append(TorchOptimizationTrace(epoch=executed_epochs, loss_total=final_loss))
    if final_loss <= best_loss + 1e-12:
        best_loss = final_loss
        best_state = state.detach().clone()
    return TorchOptimizationResult(
        best_state=best_state,
        best_loss=best_loss,
        executed_epochs=executed_epochs,
        traces=tuple(traces),
        state_trace=tuple(state_trace),
    )


"""Differentiable torch surrogate helpers for Family 3 research experiments.

The code here mirrors the accounting-variable attack space used by the exact
engine while keeping the objective differentiable for `torch_adam`. The exact
evaluator still decides model selection; this module only provides a smooth
search landscape and projection utilities.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from src.common.ratio_inputs import get_canonical_sharia_ratios
from src.engine.temporal import MIN_HIST as D6_MIN_HIST, Q_HIST as D6_Q_HIST
from scipy.stats import chi2 as _scipy_chi2, norm as _scipy_norm

from src.engine.mscore import _compute_mscore_raw_vectorized
from src.engine.proximity import _firm_raw_statistics
from src.common.methodology import thresholds_for_panel
from src.engine.coherence import compute_cross_statement_relations
from src.engine.peer import _get_global_reference, _resolve_reference_for_row
from src.engine.cost_of_debt import _implied_cost_of_debt, _t8_vectorized
from src.engine.pit import pit_empirical
from src.analysis.reference_sample import SPLIT_LABEL_INCLUDED
from src.analysis.torch_attack import TorchPGDResult, run_torch_projected_pgd

# Order used throughout this module for the 3 canonical Shariah ratios (see
# `family3_current_ratios`/`torch_ratio_vector`). `get_canonical_sharia_ratios`
# returns them in a *different* order (income, debt, cash) — any exact
# reference built from it (e.g. d7_peer's mu/covariance) must be permuted to
# this order before being combined with `torch_ratio_vector`'s output.
TORCH_RATIO_ORDER: tuple[str, ...] = ("ratio_debt_adj", "ratio_cash_adj", "ratio_income")


@dataclass(frozen=True)
class TorchPeerReferenceBundle:
    """Cached exact d7 peer/sector Mahalanobis reference (mu_ref, inv_ref per bucket).

    These depend only on the split-``C`` calibration sample and each row's own
    (fixed, unattacked) sector — never on the raw variables being perturbed —
    so the reference is built once per run via ``_get_global_reference`` and
    each row only performs the cheap per-row lookup (``_resolve_reference_for_row``).
    """

    peer_ctx: object
    column_perm: np.ndarray


def _build_peer_reference_bundle(base_ctx) -> TorchPeerReferenceBundle | None:
    """Build the exact d7_peer reference once, shared across every attacked row."""
    ratios = get_canonical_sharia_ratios(base_ctx.panel)
    if ratios.empty:
        return None
    peer_ctx = _get_global_reference(base_ctx.panel, ratios)
    if peer_ctx is None:
        return None
    available = list(ratios.columns)
    if not set(TORCH_RATIO_ORDER).issubset(available):
        return None
    column_perm = np.array([available.index(name) for name in TORCH_RATIO_ORDER], dtype=int)
    return TorchPeerReferenceBundle(peer_ctx=peer_ctx, column_perm=column_perm)


def _resolve_peer_reference_for_row(
    bundle: TorchPeerReferenceBundle | None,
    row_index: object,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Resolve one row's exact ``(mu_ref, inv_ref, n_ref)``, reordered to ``TORCH_RATIO_ORDER``."""
    if bundle is None:
        return None
    resolved = _resolve_reference_for_row(row_index, bundle.peer_ctx)
    if resolved is None:
        return None
    mu_ref, inv_ref, n_ref, _group_col = resolved
    perm = bundle.column_perm
    mu_reordered = np.asarray(mu_ref, dtype=float)[perm]
    inv_reordered = np.asarray(inv_ref, dtype=float)[np.ix_(perm, perm)]
    return mu_reordered, inv_reordered, int(n_ref)


def _family3_z3_reference(base_ctx) -> np.ndarray:
    """Exact raw M-score reference on split C (mirrors ``d3_mscore.detect_mscore``)."""
    panel = base_ctx.panel
    if "gvkey" not in panel.columns or "datacqtr" not in panel.columns:
        return np.array([], dtype=float)
    ratios = get_canonical_sharia_ratios(panel)
    if ratios.empty:
        return np.array([], dtype=float)
    df_sorted = panel.sort_values(["gvkey", "datacqtr"]).copy()
    for col in ratios.columns:
        df_sorted[col] = ratios.loc[df_sorted.index, col].values
    if "ratio_income_cashadj" in panel.columns:
        df_sorted["ratio_income_cashadj"] = pd.to_numeric(
            panel.loc[df_sorted.index, "ratio_income_cashadj"], errors="coerce"
        ).values
    ms_all = _compute_mscore_raw_vectorized(df_sorted)
    mask_cal = (
        df_sorted["_split"] == SPLIT_LABEL_INCLUDED
        if "_split" in df_sorted.columns
        else pd.Series(True, index=df_sorted.index)
    )
    return ms_all[mask_cal].dropna().to_numpy(dtype=float)


def _family3_z4_reference(base_ctx) -> np.ndarray:
    """Exact per-firm proximity-T reference on split C (mirrors ``d4_proximity``)."""
    panel = base_ctx.panel
    mask_cal = (
        panel["_split"] == SPLIT_LABEL_INCLUDED if "_split" in panel.columns else pd.Series(True, index=panel.index)
    )
    c_panel = panel.loc[mask_cal]
    thresholds = thresholds_for_panel(c_panel)
    ratio_cols = [c for c in thresholds if c in c_panel.columns]
    if not ratio_cols or "gvkey" not in c_panel.columns:
        return np.array([], dtype=float)
    raw_c = _firm_raw_statistics(c_panel, c_panel[ratio_cols], thresholds)
    return np.array([v for v in raw_c.values() if np.isfinite(v)], dtype=float)


def _family3_z8_reference(base_ctx) -> np.ndarray:
    """Exact per-firm cost-of-debt-break-T reference on split C (mirrors ``d8_cost_of_debt``)."""
    panel = base_ctx.panel
    if "gvkey" not in panel.columns or "datacqtr" not in panel.columns:
        return np.array([], dtype=float)
    mask_cal = (
        panel["_split"] == SPLIT_LABEL_INCLUDED if "_split" in panel.columns else pd.Series(True, index=panel.index)
    )
    c_panel = panel.loc[mask_cal]
    if c_panel.empty:
        return np.array([], dtype=float)
    # Vectorized (see profiling notes): the per-firm loop order doesn't matter
    # here since only the *set* of finite T8 values feeds the empirical PIT
    # reference, not their order.
    c_sorted = c_panel.sort_values(["gvkey", "datacqtr"])
    cod_sorted = _implied_cost_of_debt(c_sorted).to_numpy(dtype=float)
    firm_codes, _ = pd.factorize(c_sorted["gvkey"], sort=False)
    t8_sorted = _t8_vectorized(cod_sorted, firm_codes)
    return t8_sorted[np.isfinite(t8_sorted)]


def _chi2_pit_slope(raw_value: float, dof: int) -> float:
    """Exact ``d(z)/d(T)`` for ``z = Phi^-1(chi2.cdf(T, dof))`` at ``raw_value``.

    Closed-form local sensitivity of the chi-square PIT used by z5/z6/z7,
    replacing an arbitrary fixed anchor scale with the real slope of the
    calibration transform at the baseline point.
    """
    if not np.isfinite(raw_value) or dof <= 0 or raw_value < 0:
        return np.nan
    p = float(_scipy_chi2.cdf(raw_value, df=dof))
    p_clipped = min(max(p, 1e-9), 1.0 - 1e-9)
    z = float(_scipy_norm.ppf(p_clipped))
    pdf_chi2 = float(_scipy_chi2.pdf(raw_value, df=dof))
    pdf_norm = float(_scipy_norm.pdf(z))
    if pdf_norm <= 1e-300 or not np.isfinite(pdf_chi2):
        return np.nan
    return pdf_chi2 / pdf_norm


def _chi2_pit_slope_wrt_rms(rms_value: float, q_eff: int) -> float:
    """``d(z)/d(rms)`` when the anchored raw value is ``rms = sqrt(T / q_eff)``.

    The torch z5/z6 proxies aggregate as an RMS (``sqrt(mean(z_i^2))``) rather
    than the exact detector's raw sum-of-squares ``T = sum(z_i^2)``, so the
    chi-square slope must be chained through ``T = q_eff * rms^2``.
    """
    if not np.isfinite(rms_value) or q_eff <= 0:
        return np.nan
    t_stat = q_eff * rms_value**2
    dz_dt = _chi2_pit_slope(t_stat, q_eff)
    if not np.isfinite(dz_dt):
        return np.nan
    dt_drms = 2.0 * q_eff * rms_value
    return dz_dt * dt_drms


def _empirical_pit_slope(raw_value: float, reference: np.ndarray, *, eps_frac: float = 0.05) -> float:
    """Local ``d(z)/d(raw)`` for an empirical PIT, estimated by central finite
    differences against the reference sample.

    The empirical PIT is a step function (piecewise-constant empirical CDF),
    so it has no classical derivative; a finite difference with a step scaled
    to the reference sample's spread approximates the local density-based
    slope instead of guessing a fixed anchor scale.
    """
    if not np.isfinite(raw_value):
        return np.nan
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < Z5_MIN_GLOBAL_REFERENCE:
        return np.nan
    spread = float(np.std(ref))
    if not np.isfinite(spread) or spread <= 1e-9:
        return np.nan
    eps = max(spread * eps_frac, 1e-6)
    z_plus = pit_empirical(np.array([raw_value + eps]), ref)[0]
    z_minus = pit_empirical(np.array([raw_value - eps]), ref)[0]
    if not (np.isfinite(z_plus) and np.isfinite(z_minus)):
        return np.nan
    return float((z_plus - z_minus) / (2.0 * eps))


def _resolve_anchor_slope(candidate_slope: float, *, baseline_raw: float, floor: float) -> float:
    """Return a numerically safe anchor slope, falling back to the legacy heuristic.

    Falls back to ``1/max(|baseline_raw|, floor)`` (the previous fixed-scale
    behavior) whenever the estimated slope is unavailable, non-finite, or
    degenerate, and clamps the magnitude to keep the anchor numerically
    stable (a fixed-point safety net, not a methodology choice).
    """
    fallback = 1.0 / max(abs(baseline_raw), floor)
    if not np.isfinite(candidate_slope) or abs(candidate_slope) < 1e-6:
        return fallback
    sign = 1.0 if candidate_slope >= 0 else -1.0
    return sign * min(abs(candidate_slope), 50.0)

# Order must match `RELATION_SPECS` in server/detectors/d5_coherence.py so the
# torch z5 proxy stacks/standardizes the same relations in the same order as
# the exact detector.
Z5_RELATION_NAMES: tuple[str, ...] = (
    "cfo_to_revenue",
    "tata_accrual",
    "interest_to_debt",
    "delta_interest_to_debt",
    "working_capital_accrual_proxy",
    "debt_ratio_proxy",
    "cash_ratio_proxy",
    "income_ratio_proxy",
    "earnings_gap_proxy",
)
Z5_MIN_GLOBAL_REFERENCE: int = 30
_Z5_TATA_ACCRUAL_IDX = Z5_RELATION_NAMES.index("tata_accrual")
_Z5_WC_ACCRUAL_PROXY_IDX = Z5_RELATION_NAMES.index("working_capital_accrual_proxy")


def _family3_relation_reference_stats(base_ctx) -> tuple[np.ndarray, np.ndarray]:
    """Return per-relation global mean/std for the exact z5 relations.

    Mirrors ``d5_coherence._estimate_moments`` (global calibration on split
    ``C``, no sector-conditioning) so the torch z5 proxy standardizes its raw
    relations on the same scale as the exact detector instead of averaging
    heterogeneous units directly. A relation with too little reference support
    gets ``sigma=NaN`` so it drops out of the proxy the same way the exact
    detector excludes it from ``used_relations``.
    """
    panel = base_ctx.panel
    mask_cal = (
        panel["_split"] == SPLIT_LABEL_INCLUDED
        if "_split" in panel.columns
        else pd.Series(True, index=panel.index)
    )
    relation_frame = compute_cross_statement_relations(panel.loc[mask_cal])
    mu = np.zeros(len(Z5_RELATION_NAMES), dtype=float)
    sigma = np.full(len(Z5_RELATION_NAMES), np.nan, dtype=float)
    for i, name in enumerate(Z5_RELATION_NAMES):
        if name not in relation_frame.columns:
            continue
        ref = relation_frame[name].dropna()
        if len(ref) < Z5_MIN_GLOBAL_REFERENCE:
            continue
        ref_sigma = float(ref.std(ddof=1))
        if not np.isfinite(ref_sigma) or ref_sigma <= 1e-9:
            continue
        mu[i] = float(ref.mean())
        sigma[i] = ref_sigma
    return mu, sigma


def _build_fast_row_evaluator(base_ctx, row_index: object, settings):
    """Build the exact fast row evaluator without importing benchmark internals."""
    return Family3FastRowEvaluator(
        base_ctx,
        row_index,
        settings.robustness_benchmark.primary_composite,
        settings,
    )


def _current_ratios_from_row(row: pd.Series) -> np.ndarray:
    """Extract canonical ratios from a row-like object."""
    return family3_current_ratios(row)


def _reference_ratio_stats_for_base_ctx(base_ctx) -> dict[str, float]:
    """Compute reference ratio moments from the authoritative score context."""
    return family3_reference_ratio_stats(base_ctx)


def _baseline_primary_pvalue(*, base_ctx, settings, row_index: object, row: pd.Series) -> float:
    evaluator = _build_fast_row_evaluator(base_ctx, row_index, settings)
    return float(evaluator.evaluate(row))


def _finite_base_float(value: object, *, default: float) -> float:
    """Coerce a scalar value while keeping NaN-like inputs cheap."""
    numeric = float(pd.to_numeric(value, errors="coerce"))
    return numeric if np.isfinite(numeric) else default


def _safe_numpy_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(finite.mean())


def _safe_total_debt_values(*, dlttq: float, dlcq: float, ltq: float) -> float:
    components = 0.0
    has_components = False
    if np.isfinite(dlttq):
        components += max(dlttq, 0.0)
        has_components = True
    if np.isfinite(dlcq):
        components += max(dlcq, 0.0)
        has_components = True
    if has_components and components > 0.0:
        return components
    if np.isfinite(ltq):
        return max(ltq, 0.0)
    return 0.0


def _build_firm_history_panel(
    panel: pd.DataFrame, gvkey: object
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Build one firm's sorted panel slice and canonical ratios once.

    This is the expensive part of history-context construction (a full-panel
    boolean mask, sort, and ratio computation). It depends only on ``gvkey``,
    so callers iterating over several rows of the same firm (cohort/firm-level
    torch runs) should build it once via a cache and reuse it across rows
    instead of repeating it per row.
    """
    firm_panel = panel.loc[panel["gvkey"] == gvkey].sort_values("datacqtr").copy()
    if firm_panel.empty:
        return None
    if {"ratio_debt_adj", "ratio_cash_adj", "ratio_income"}.issubset(firm_panel.columns):
        ratios = firm_panel[["ratio_debt_adj", "ratio_cash_adj", "ratio_income"]].apply(
            pd.to_numeric, errors="coerce"
        )
    else:
        ratios = get_canonical_sharia_ratios(firm_panel)
    return firm_panel, ratios


_EMPTY_HISTORY_CONTEXT: dict[str, object] = {
    "prev_row_values": {},
    "prev_ratio_values": np.full(3, np.nan, dtype=float),
    "z6_hist_mu": np.full(3, np.nan, dtype=float),
    "z6_hist_sigma": np.full(3, np.nan, dtype=float),
    "z8_hist_median": np.nan,
    "z8_hist_mad": np.nan,
    "firm_ratio_history": np.empty((0, 3), dtype=float),
    "firm_row_pos": -1,
}


def _compute_history_context(
    panel: pd.DataFrame,
    row_index: object,
    *,
    firm_panel_cache: dict[object, tuple[pd.DataFrame, pd.DataFrame] | None] | None = None,
) -> dict[str, object]:
    """Compute one row's history context, optionally reusing a per-firm cache.

    Args:
        panel: Full panel (or the authoritative ``base_ctx.panel``).
        row_index: Index of the attacked row.
        firm_panel_cache: Optional cache keyed by ``gvkey``, shared across
            calls for rows of the same firm (see ``_build_firm_history_panel``).
            When omitted, the firm slice is rebuilt for this call only,
            preserving the previous single-row behavior.
    """
    if "gvkey" not in panel.columns or "datacqtr" not in panel.columns:
        return dict(_EMPTY_HISTORY_CONTEXT)

    gvkey = panel.at[row_index, "gvkey"]
    if firm_panel_cache is not None:
        if gvkey not in firm_panel_cache:
            firm_panel_cache[gvkey] = _build_firm_history_panel(panel, gvkey)
        cached = firm_panel_cache[gvkey]
    else:
        cached = _build_firm_history_panel(panel, gvkey)
    if cached is None:
        return dict(_EMPTY_HISTORY_CONTEXT)
    firm_panel, ratios = cached
    if row_index not in firm_panel.index:
        return dict(_EMPTY_HISTORY_CONTEXT)

    positions = np.flatnonzero(firm_panel.index.to_numpy() == row_index)
    pos = int(positions[0]) if positions.size else 0

    prev_row_values: dict[str, float] = {}
    prev_ratio_values = np.full(3, np.nan, dtype=float)
    if pos > 0:
        prev_row = firm_panel.iloc[pos - 1]
        prev_row_values = {
            col: _finite_base_float(prev_row.get(col), default=np.nan)
            for col in (
                "revtq",
                "iditq",
                "xintq",
                "niq",
                "oibdpq",
                "oancfq",
                "ibq",
                "atq",
                "dlttq",
                "dlcq",
                "ltq",
                "cheq",
                "actq",
                "lctq",
                "rectq",
                "invtq",
                "xsgaq",
                "ppentq",
            )
        }
        prev_ratio_values = ratios.iloc[pos - 1].to_numpy(dtype=float, copy=True)

    z6_hist_mu = np.full(3, np.nan, dtype=float)
    z6_hist_sigma = np.full(3, np.nan, dtype=float)
    if pos >= D6_MIN_HIST + 1 and len(ratios) > 1:
        hist_start = max(0, pos - D6_Q_HIST)
        deltas_hist = np.diff(ratios.iloc[hist_start:pos].to_numpy(dtype=float), axis=0)
        if len(deltas_hist) >= D6_MIN_HIST:
            with np.errstate(invalid="ignore"):
                z6_hist_mu = np.nanmean(deltas_hist, axis=0)
                z6_hist_sigma = np.nanstd(deltas_hist, axis=0, ddof=1)

    z8_hist_median = np.nan
    z8_hist_mad = np.nan
    if pos > 0:
        debt = (
            pd.to_numeric(firm_panel.get("dlttq", pd.Series(np.nan, index=firm_panel.index)), errors="coerce").fillna(0.0)
            + pd.to_numeric(firm_panel.get("dlcq", pd.Series(np.nan, index=firm_panel.index)), errors="coerce").fillna(0.0)
        )
        xintq = pd.to_numeric(firm_panel.get("xintq", pd.Series(np.nan, index=firm_panel.index)), errors="coerce")
        with np.errstate(invalid="ignore", divide="ignore"):
            cod = np.where((debt.to_numpy() > 0) & xintq.notna().to_numpy(), xintq.to_numpy(dtype=float) / debt.to_numpy(dtype=float), np.nan)
        history = cod[:pos]
        finite = history[np.isfinite(history)]
        if finite.size >= D6_MIN_HIST:
            z8_hist_median = float(np.median(finite))
            mad = 1.4826 * np.median(np.abs(finite - z8_hist_median))
            z8_hist_mad = float(mad) if np.isfinite(mad) and mad > 1e-10 else np.nan

    return {
        "prev_row_values": prev_row_values,
        "prev_ratio_values": prev_ratio_values,
        "z6_hist_mu": z6_hist_mu,
        "z6_hist_sigma": z6_hist_sigma,
        "z8_hist_median": z8_hist_median,
        "z8_hist_mad": z8_hist_mad,
        "firm_ratio_history": ratios.to_numpy(dtype=float, copy=True),
        "firm_row_pos": pos,
    }

@dataclass(frozen=True)
class TorchSurrogateContext:
    """Torch surrogate inputs tied to a single attacked row."""

    base_row: pd.Series
    x0: np.ndarray
    scale: np.ndarray
    thresholds: dict[str, float]
    baseline_ratios: np.ndarray
    baseline_primary_p: float
    modifiable_columns: tuple[str, ...]
    modifiable_positions: dict[str, int]
    device: torch.device
    dtype: torch.dtype
    baseline_active_z: torch.Tensor
    sigma: torch.Tensor
    weights: torch.Tensor
    null_ctx: object
    z5_relation_mu: torch.Tensor
    z5_relation_sigma: torch.Tensor
    active_indices: dict[str, int]
    active_threshold: float
    min_active_for_renorm: int
    min_active_for_iut: int
    prev_row_values: dict[str, float]
    prev_ratio_values: np.ndarray
    z6_hist_mu: np.ndarray
    z6_hist_sigma: np.ndarray
    z8_hist_median: float
    z8_hist_mad: float
    z4_firm_ratios: torch.Tensor
    z4_firm_row_pos: int
    z7_mu_ref: torch.Tensor
    z7_inv_ref: torch.Tensor
    baseline_z3: float
    baseline_z4: float
    baseline_z5: float
    baseline_z6: float
    baseline_z7: float
    baseline_z57: float
    baseline_z8: float
    z3_baseline_raw: float
    z4_baseline_raw: float
    z5_baseline_raw: float
    z6_baseline_raw: float
    z7_baseline_raw: float
    z8_baseline_raw: float
    z3_slope: float
    z4_slope: float
    z5_slope: float
    z6_slope: float
    z7_slope: float
    z8_slope: float


@dataclass(frozen=True)
class TorchSharedRunContext:
    """Device / null-distribution / composite inputs shared by a whole run.

    These depend only on ``base_ctx`` and the device, never on which row is
    being attacked. A cohort or firm-level torch run should build this once
    (via ``build_torch_shared_run_context``) and pass it to every
    ``build_torch_surrogate_context`` call for that run instead of letting
    each row rebuild it from scratch.
    """

    device_info: TorchDeviceInfo
    sigma: torch.Tensor
    weights: torch.Tensor
    null_ctx: TorchNullContext
    z5_relation_mu: torch.Tensor
    z5_relation_sigma: torch.Tensor
    peer_reference_bundle: TorchPeerReferenceBundle | None
    z3_reference: np.ndarray
    z4_reference: np.ndarray
    z8_reference: np.ndarray
    ratio_stats: dict[str, float]
    thresholds: dict[str, float]
    scale_floor: np.ndarray


def build_torch_shared_run_context(
    base_ctx, *, prefer_gpu: bool = True, scale_floor: np.ndarray,
) -> TorchSharedRunContext:
    """Build the per-run torch context shared across every attacked row.

    Args:
        scale_floor: Per-column normalization floor (see
            ``family3_column_scale_floor``), in the same column order used by
            every ``build_torch_surrogate_context`` call for this run. Must
            match the exact-side floor so the surrogate and exact objectives
            agree on what a given L1 budget means for each column.
    """
    device_info = resolve_torch_device(prefer_gpu=prefer_gpu)
    sigma = tensor_from_array(
        np.asarray(base_ctx.sigma, dtype=float),
        device=device_info.device,
        dtype=device_info.dtype,
    )
    weights = tensor_from_array(
        np.asarray(base_ctx.weights, dtype=float),
        device=device_info.device,
        dtype=device_info.dtype,
    )
    null_ctx = torch_null_context_from_score_context(
        base_ctx,
        device=device_info.device,
        dtype=device_info.dtype,
    )
    z5_mu_np, z5_sigma_np = _family3_relation_reference_stats(base_ctx)
    z5_relation_mu = tensor_from_array(z5_mu_np, device=device_info.device, dtype=device_info.dtype)
    z5_relation_sigma = tensor_from_array(z5_sigma_np, device=device_info.device, dtype=device_info.dtype)
    peer_reference_bundle = _build_peer_reference_bundle(base_ctx)
    return TorchSharedRunContext(
        device_info=device_info,
        sigma=sigma,
        weights=weights,
        null_ctx=null_ctx,
        z5_relation_mu=z5_relation_mu,
        z5_relation_sigma=z5_relation_sigma,
        peer_reference_bundle=peer_reference_bundle,
        z3_reference=_family3_z3_reference(base_ctx),
        z4_reference=_family3_z4_reference(base_ctx),
        z8_reference=_family3_z8_reference(base_ctx),
        ratio_stats=_reference_ratio_stats_for_base_ctx(base_ctx),
        thresholds=thresholds_for_panel(base_ctx.panel),
        scale_floor=scale_floor,
    )


def build_torch_surrogate_context(
    *,
    panel: pd.DataFrame,
    base_ctx,
    settings,
    row_index: object,
    modifiable_columns: tuple[str, ...] | None = None,
    prefer_gpu: bool = True,
    shared_ctx: TorchSharedRunContext | None = None,
    firm_panel_cache: dict[object, tuple[pd.DataFrame, pd.DataFrame] | None] | None = None,
) -> TorchSurrogateContext:
    """Build the immutable torch-side context for one attacked row.

    Args:
        shared_ctx: Optional pre-built device/null-context/sigma/weights bundle
            shared across all rows of a cohort or firm-level run (see
            ``build_torch_shared_run_context``). When omitted, these are
            rebuilt for this row only, matching the previous single-row
            behavior.
        firm_panel_cache: Optional per-firm history cache shared across rows
            of the same firm within a run (see ``_build_firm_history_panel``).
            When omitted, the firm slice is rebuilt for this row only.
    """
    mod_cols = tuple(modifiable_columns or SUPPORTED_TORCH_RAW_COLUMNS)
    unsupported = sorted(set(mod_cols) - set(SUPPORTED_TORCH_RAW_COLUMNS))
    if unsupported:
        raise ValueError(
            "torch surrogate currently supports only "
            f"{list(SUPPORTED_TORCH_RAW_COLUMNS)}, got unsupported columns {unsupported}."
        )
    base_row = panel.loc[row_index].copy()
    x0 = np.array(
        [
            _finite_base_float(base_row.get(col), default=0.0)
            if col in base_row.index else 0.0
            for col in mod_cols
        ],
        dtype=float,
    )
    # Same invariant-across-rows reasoning as `thresholds`/`ratio_stats` below:
    # the per-column floor doesn't depend on `row_index`, so a per-run
    # `shared_ctx` already has it (see the normalization-floor audit):
    # this used to be a single constant `1.0` shared by every column
    # regardless of its natural units, which made naturally small-magnitude
    # columns artificially "cheap" to move by a large relative amount).
    scale_floor = shared_ctx.scale_floor if shared_ctx is not None else family3_column_scale_floor(
        panel, mod_cols, settings.robustness_benchmark.family3_scale_floor_quantile,
    )
    scale = np.maximum(np.abs(x0), scale_floor)
    base_row_dict = base_row.to_dict()
    # `ratio_stats` depends only on `base_ctx` (never on `row_index`), so a
    # per-run `shared_ctx` already has it -- avoids re-copying/filtering the
    # whole panel on every one of the (potentially hundreds of) rows in a
    # cohort/firm-level run (see profiling notes).
    ratio_stats = shared_ctx.ratio_stats if shared_ctx is not None else _reference_ratio_stats_for_base_ctx(base_ctx)
    # Same invariant-across-rows reasoning as `ratio_stats` above: the panel's
    # methodology/country thresholds don't depend on `row_index`, so a
    # per-run `shared_ctx` already has them (see profiling notes -- this used
    # to be the hardcoded Malaysia-only `SAC_THRESHOLDS`).
    thresholds = shared_ctx.thresholds if shared_ctx is not None else thresholds_for_panel(base_ctx.panel)
    base_row_dict.update(ratio_stats)
    base_row = pd.Series(base_row_dict)
    baseline_ratios = _current_ratios_from_row(base_row)
    baseline_primary_p = _baseline_primary_pvalue(
        base_ctx=base_ctx,
        settings=settings,
        row_index=row_index,
        row=base_row,
    )
    device_info = shared_ctx.device_info if shared_ctx is not None else resolve_torch_device(prefer_gpu=prefer_gpu)
    active_names = list(base_ctx.active)
    history_ctx = _compute_history_context(
        base_ctx.panel if hasattr(base_ctx, "panel") else panel,
        row_index,
        firm_panel_cache=firm_panel_cache,
    )
    baseline_active_z = tensor_from_array(
        base_ctx.zscores.loc[row_index, active_names].to_numpy(dtype=float, copy=True),
        device=device_info.device,
        dtype=device_info.dtype,
    )
    if shared_ctx is not None:
        sigma = shared_ctx.sigma
        weights = shared_ctx.weights
        null_ctx = shared_ctx.null_ctx
        z5_relation_mu = shared_ctx.z5_relation_mu
        z5_relation_sigma = shared_ctx.z5_relation_sigma
        peer_reference_bundle = shared_ctx.peer_reference_bundle
        z3_reference = shared_ctx.z3_reference
        z4_reference = shared_ctx.z4_reference
        z8_reference = shared_ctx.z8_reference
    else:
        sigma = tensor_from_array(
            np.asarray(base_ctx.sigma, dtype=float),
            device=device_info.device,
            dtype=device_info.dtype,
        )
        weights = tensor_from_array(
            np.asarray(base_ctx.weights, dtype=float),
            device=device_info.device,
            dtype=device_info.dtype,
        )
        null_ctx = torch_null_context_from_score_context(
            base_ctx,
            device=device_info.device,
            dtype=device_info.dtype,
        )
        z5_mu_np, z5_sigma_np = _family3_relation_reference_stats(base_ctx)
        z5_relation_mu = tensor_from_array(z5_mu_np, device=device_info.device, dtype=device_info.dtype)
        z5_relation_sigma = tensor_from_array(z5_sigma_np, device=device_info.device, dtype=device_info.dtype)
        peer_reference_bundle = _build_peer_reference_bundle(base_ctx)
        z3_reference = _family3_z3_reference(base_ctx)
        z4_reference = _family3_z4_reference(base_ctx)
        z8_reference = _family3_z8_reference(base_ctx)
    resolved_peer_ref = _resolve_peer_reference_for_row(peer_reference_bundle, row_index)
    if resolved_peer_ref is not None:
        z7_mu_ref_np, z7_inv_ref_np, z7_n_ref = resolved_peer_ref
    else:
        z7_mu_ref_np = np.zeros(3, dtype=float)
        z7_inv_ref_np = np.zeros((3, 3), dtype=float)
        z7_n_ref = 0
    z7_mu_ref = tensor_from_array(z7_mu_ref_np, device=device_info.device, dtype=device_info.dtype)
    z7_inv_ref = tensor_from_array(z7_inv_ref_np, device=device_info.device, dtype=device_info.dtype)
    raw_baseline = base_ctx.raw_zscores.loc[row_index] if hasattr(base_ctx, "raw_zscores") else pd.Series(dtype=float)
    ctx = TorchSurrogateContext(
        base_row=base_row,
        x0=x0,
        scale=scale,
        thresholds=thresholds,
        baseline_ratios=baseline_ratios,
        baseline_primary_p=baseline_primary_p,
        modifiable_columns=mod_cols,
        modifiable_positions={column: pos for pos, column in enumerate(mod_cols)},
        device=device_info.device,
        dtype=device_info.dtype,
        baseline_active_z=baseline_active_z,
        sigma=sigma,
        weights=weights,
        null_ctx=null_ctx,
        z5_relation_mu=z5_relation_mu,
        z5_relation_sigma=z5_relation_sigma,
        active_indices={name: idx for idx, name in enumerate(active_names)},
        active_threshold=float(settings.composites.active_set_threshold),
        min_active_for_renorm=int(settings.composites.min_active_for_renorm),
        min_active_for_iut=int(settings.composites.min_active_for_iut),
        prev_row_values=dict(history_ctx["prev_row_values"]),
        prev_ratio_values=np.asarray(history_ctx["prev_ratio_values"], dtype=float),
        z6_hist_mu=np.asarray(history_ctx["z6_hist_mu"], dtype=float),
        z6_hist_sigma=np.asarray(history_ctx["z6_hist_sigma"], dtype=float),
        z8_hist_median=_finite_base_float(history_ctx["z8_hist_median"], default=np.nan),
        z8_hist_mad=_finite_base_float(history_ctx["z8_hist_mad"], default=np.nan),
        z4_firm_ratios=tensor_from_array(
            np.asarray(history_ctx["firm_ratio_history"], dtype=float),
            device=device_info.device,
            dtype=device_info.dtype,
        ),
        z4_firm_row_pos=int(history_ctx["firm_row_pos"]),
        z7_mu_ref=z7_mu_ref,
        z7_inv_ref=z7_inv_ref,
        baseline_z3=_finite_base_float(raw_baseline.get("z3"), default=0.0),
        baseline_z4=_finite_base_float(raw_baseline.get("z4"), default=0.0),
        baseline_z5=_finite_base_float(raw_baseline.get("z5"), default=0.0),
        baseline_z6=_finite_base_float(raw_baseline.get("z6"), default=0.0),
        baseline_z7=_finite_base_float(raw_baseline.get("z7"), default=0.0),
        baseline_z57=_finite_base_float(base_ctx.zscores.loc[row_index].get("z57"), default=0.0),
        baseline_z8=_finite_base_float(raw_baseline.get("z8"), default=0.0),
        z3_baseline_raw=0.0,
        z4_baseline_raw=0.0,
        z5_baseline_raw=0.0,
        z6_baseline_raw=0.0,
        z7_baseline_raw=0.0,
        z8_baseline_raw=0.0,
        z3_slope=1.0,
        z4_slope=1.0,
        z5_slope=1.0,
        z6_slope=1.0,
        z7_slope=1.0,
        z8_slope=1.0,
    )
    x0_tensor = _torch_tensor(ctx.x0, ctx=ctx)
    proxy_raws = _surrogate_detector_raws(x0_tensor, ctx=ctx, temperature=0.1)
    z3_baseline_raw = float(proxy_raws["z3"].detach().cpu())
    z4_baseline_raw = float(proxy_raws["z4"].detach().cpu())
    z5_baseline_raw = float(proxy_raws["z5"].detach().cpu())
    z6_baseline_raw = float(proxy_raws["z6"].detach().cpu())
    z7_baseline_raw = float(proxy_raws["z7"].detach().cpu())
    z8_baseline_raw = float(proxy_raws["z8"].detach().cpu())
    # Local slope of each detector's real PIT transform at the baseline raw
    # value, replacing a fixed floor-based anchor scale with the actual
    # sensitivity of the calibration curve (closed-form for the chi-square
    # detectors z5/z6/z7, finite-difference on the reference sample for the
    # empirical-PIT detectors z3/z4/z8). Falls back to the previous
    # 1/max(|raw|, floor) heuristic whenever the estimate is unavailable.
    z3_slope = _resolve_anchor_slope(
        _empirical_pit_slope(z3_baseline_raw, z3_reference), baseline_raw=z3_baseline_raw, floor=0.5
    )
    z4_slope = _resolve_anchor_slope(
        _empirical_pit_slope(z4_baseline_raw, z4_reference), baseline_raw=z4_baseline_raw, floor=0.25
    )
    z5_slope = _resolve_anchor_slope(
        _chi2_pit_slope_wrt_rms(z5_baseline_raw, int(proxy_raws["z5_q_eff"])),
        baseline_raw=z5_baseline_raw,
        floor=0.25,
    )
    z6_slope = _resolve_anchor_slope(
        _chi2_pit_slope_wrt_rms(z6_baseline_raw, int(proxy_raws["z6_p_eff"])),
        baseline_raw=z6_baseline_raw,
        floor=0.5,
    )
    z7_candidate_slope = _chi2_pit_slope(z7_baseline_raw, dof=3) if z7_n_ref >= 9 else np.nan
    z7_slope = _resolve_anchor_slope(z7_candidate_slope, baseline_raw=z7_baseline_raw, floor=0.5)
    z8_slope = _resolve_anchor_slope(
        _empirical_pit_slope(z8_baseline_raw, z8_reference), baseline_raw=z8_baseline_raw, floor=0.5
    )
    return replace(
        ctx,
        z3_baseline_raw=z3_baseline_raw,
        z4_baseline_raw=z4_baseline_raw,
        z5_baseline_raw=z5_baseline_raw,
        z6_baseline_raw=z6_baseline_raw,
        z7_baseline_raw=z7_baseline_raw,
        z8_baseline_raw=z8_baseline_raw,
        z3_slope=z3_slope,
        z4_slope=z4_slope,
        z5_slope=z5_slope,
        z6_slope=z6_slope,
        z7_slope=z7_slope,
        z8_slope=z8_slope,
    )


def _torch_tensor(data: np.ndarray | list[float], *, ctx: TorchSurrogateContext) -> torch.Tensor:
    """Move numeric data to the surrogate device/dtype."""
    return torch.as_tensor(np.asarray(data, dtype=float), dtype=ctx.dtype, device=ctx.device)


def torch_candidate_x_from_delta(
    delta_norm: torch.Tensor,
    *,
    ctx: TorchSurrogateContext,
    max_eps: float,
) -> torch.Tensor:
    """Project normalized deltas back to non-negative raw accounting values."""
    safe_delta = torch.clamp(delta_norm, min=-max_eps, max=max_eps)
    x0 = _torch_tensor(ctx.x0, ctx=ctx)
    scale = _torch_tensor(ctx.scale, ctx=ctx)
    candidate = x0 + safe_delta * scale
    return torch.clamp(candidate, min=0.0)


def _row_value_tensor(candidate_x: torch.Tensor, *, ctx: TorchSurrogateContext, column: str) -> torch.Tensor:
    """Read a raw value from the candidate vector or the immutable base row."""
    pos = ctx.modifiable_positions.get(column)
    if pos is not None:
        return candidate_x[pos]
    value = _finite_base_float(ctx.base_row.get(column), default=0.0)
    return torch.tensor(value, dtype=ctx.dtype, device=ctx.device)


def _ratio_reference_tensors(ctx: TorchSurrogateContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ratio centers, scales, and thresholds on the surrogate device."""
    centers = _torch_tensor(
        [
            _finite_base_float(ctx.base_row.get("ratio_center_debt", 0.0), default=0.0),
            _finite_base_float(ctx.base_row.get("ratio_center_cash", 0.0), default=0.0),
            _finite_base_float(ctx.base_row.get("ratio_center_income", 0.0), default=0.0),
        ],
        ctx=ctx,
    )
    scales = _torch_tensor(
        [
            max(_finite_base_float(ctx.base_row.get("ratio_scale_debt", 1.0), default=1.0), 1e-6),
            max(_finite_base_float(ctx.base_row.get("ratio_scale_cash", 1.0), default=1.0), 1e-6),
            max(_finite_base_float(ctx.base_row.get("ratio_scale_income", 1.0), default=1.0), 1e-6),
        ],
        ctx=ctx,
    )
    thresholds = _torch_tensor(
        [
            float(ctx.thresholds["ratio_debt_adj"]),
            float(ctx.thresholds["ratio_cash_adj"]),
            float(ctx.thresholds["ratio_income"]),
        ],
        ctx=ctx,
    )
    return centers, scales, thresholds


def _torch_safe_divide(num: torch.Tensor, den: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    valid = torch.isfinite(num) & torch.isfinite(den) & (torch.abs(den) > eps)
    out = torch.full_like(num, float("nan"))
    out[valid] = num[valid] / den[valid]
    return out


def _torch_nanmean(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.full((), float("nan"), dtype=values.dtype, device=values.device)
    return torch.where(finite, values, torch.zeros_like(values)).sum() / finite.to(values.dtype).sum()


def _total_debt_tensor(candidate_x: torch.Tensor, *, ctx: TorchSurrogateContext) -> torch.Tensor:
    dlttq = _row_value_tensor(candidate_x, ctx=ctx, column="dlttq")
    dlcq = _row_value_tensor(candidate_x, ctx=ctx, column="dlcq")
    ltq = _row_value_tensor(candidate_x, ctx=ctx, column="ltq")
    components = torch.clamp(dlttq, min=0.0) + torch.clamp(dlcq, min=0.0)
    return torch.where(components > 1e-9, components, torch.clamp(ltq, min=0.0))


def _proxy_from_raw(raw_value: torch.Tensor, *, baseline_z: float, baseline_raw: float, slope: float) -> torch.Tensor:
    """Anchor a raw detector proxy to its exact baseline z-score via a local slope.

    ``slope`` approximates ``d(z)/d(raw)`` of the real PIT transform at the
    baseline point (see ``_chi2_pit_slope``/``_empirical_pit_slope``), so the
    anchored value tracks how the exact z-score would actually move, instead
    of an arbitrary fixed scale.
    """
    anchored = baseline_z + float(slope) * (raw_value - float(baseline_raw))
    return torch.clamp(anchored, min=0.0)


def _torch_firm_proximity_statistic(
    firm_ratio_column: torch.Tensor,
    *,
    row_pos: int,
    candidate_value: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Differentiable version of ``d4_proximity._proximity_raw_statistic``.

    The exact detector computes ``T = sqrt(12*Q)*(0.5 - d_bar)`` over a firm's
    *whole* ratio history, using a hard ``[0, threshold]`` window that has no
    usable gradient. This substitutes the attacked row's candidate value into
    an otherwise-fixed firm history and replaces the hard window with a soft
    (sigmoid) membership, so the statistic responds to that one row roughly
    the way the exact aggregate would (weakly, as ``1/Q``) instead of reacting
    as if the whole firm-level score were a single-quarter signal.
    """
    if firm_ratio_column.numel() == 0 or not (0 <= row_pos < firm_ratio_column.shape[0]):
        history = candidate_value.unsqueeze(0)
    else:
        history = torch.cat(
            [firm_ratio_column[:row_pos], candidate_value.unsqueeze(0), firm_ratio_column[row_pos + 1 :]]
        )
    history = torch.where(torch.isfinite(history), history, torch.full_like(history, -1e6))
    safe_threshold = max(float(threshold), 1e-9)
    weight = torch.sigmoid(history / temperature) * torch.sigmoid((safe_threshold - history) / temperature)
    q_soft = weight.sum()
    d = (safe_threshold - history) / safe_threshold
    d_bar = (weight * d).sum() / torch.clamp(q_soft, min=1e-6)
    return torch.sqrt(torch.clamp(12.0 * q_soft, min=1e-6)) * (0.5 - d_bar)


def _torch_firm_z4_raw(
    *,
    ctx: TorchSurrogateContext,
    candidate_ratios: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Differentiable version of ``d4_proximity._firm_raw_statistics`` (max over ratios).

    Each ratio's soft-window temperature is scaled to its own SAC threshold
    (the income threshold is ~6x narrower than debt/cash) instead of reusing
    the general surrogate temperature: that one is tuned for a different
    scale and would make the window far too smooth to track the exact
    statistic (verified empirically — see the z4 unit test).
    """
    stats = torch.stack(
        [
            _torch_firm_proximity_statistic(
                ctx.z4_firm_ratios[:, i],
                row_pos=ctx.z4_firm_row_pos,
                candidate_value=candidate_ratios[i],
                threshold=float(thresholds[i].detach().cpu()),
                temperature=max(float(thresholds[i].detach().cpu()) * 0.01, 1e-5),
            )
            for i in range(candidate_ratios.shape[0])
        ]
    )
    return stats.max()


def _surrogate_detector_raws(
    candidate_x: torch.Tensor,
    *,
    ctx: TorchSurrogateContext,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    _, _, thresholds = _ratio_reference_tensors(ctx)
    ratios = torch_ratio_vector(candidate_x, ctx=ctx)

    # z4: differentiable version of the exact detector's firm-level threshold
    # proximity statistic (see d4_proximity.py) — aggregates over the firm's
    # whole ratio history with only the attacked row's position replaced by
    # the differentiable candidate, instead of scoring the current row alone.
    z4_raw = _torch_firm_z4_raw(ctx=ctx, candidate_ratios=ratios, thresholds=thresholds)
    z4_penalty = torch.nn.functional.softplus((ratios - thresholds) / temperature).sum()

    prev_ratios = _torch_tensor(ctx.prev_ratio_values, ctx=ctx)
    prev_revtq = float(ctx.prev_row_values.get("revtq", np.nan))
    prev_rectq = float(ctx.prev_row_values.get("rectq", np.nan))
    prev_ppentq = float(ctx.prev_row_values.get("ppentq", np.nan))
    prev_actq = float(ctx.prev_row_values.get("actq", np.nan))
    prev_xsgaq = float(ctx.prev_row_values.get("xsgaq", np.nan))
    income_cashadj = _finite_base_float(ctx.base_row.get("ratio_income_cashadj"), default=np.nan)

    atq = torch.clamp(_row_value_tensor(candidate_x, ctx=ctx, column="atq"), min=1e-9)
    revtq = torch.clamp(_row_value_tensor(candidate_x, ctx=ctx, column="revtq"), min=1e-9)
    rectq = _row_value_tensor(candidate_x, ctx=ctx, column="rectq")
    ppentq = _row_value_tensor(candidate_x, ctx=ctx, column="ppentq")
    actq = _row_value_tensor(candidate_x, ctx=ctx, column="actq")
    xsgaq = _row_value_tensor(candidate_x, ctx=ctx, column="xsgaq")
    ibq = _row_value_tensor(candidate_x, ctx=ctx, column="ibq")
    oancfq = _row_value_tensor(candidate_x, ctx=ctx, column="oancfq")

    prev_receivable_ratio = float(prev_rectq / prev_revtq) if np.isfinite(prev_rectq) and np.isfinite(prev_revtq) and abs(prev_revtq) > 1e-9 else np.nan
    prev_aqi_ratio = float(1.0 - (prev_ppentq + prev_actq) / float(ctx.prev_row_values.get("atq", np.nan))) if np.isfinite(prev_ppentq) and np.isfinite(prev_actq) and np.isfinite(ctx.prev_row_values.get("atq", np.nan)) and abs(float(ctx.prev_row_values.get("atq", np.nan))) > 1e-9 else np.nan
    prev_sgai_ratio = float(prev_xsgaq / prev_revtq) if np.isfinite(prev_xsgaq) and np.isfinite(prev_revtq) and abs(prev_revtq) > 1e-9 else np.nan

    dsri = _torch_safe_divide(torch.stack([rectq / revtq]), torch.tensor([prev_receivable_ratio], dtype=ctx.dtype, device=ctx.device))[0]
    aqi_num = 1.0 - (ppentq + actq) / atq
    aqi = _torch_safe_divide(torch.stack([aqi_num]), torch.tensor([prev_aqi_ratio], dtype=ctx.dtype, device=ctx.device))[0]
    sgai = _torch_safe_divide(torch.stack([xsgaq / revtq]), torch.tensor([prev_sgai_ratio], dtype=ctx.dtype, device=ctx.device))[0]
    tata = (ibq - oancfq) / atq
    income_index = _torch_safe_divide(torch.stack([ratios[2]]), prev_ratios[2:3])[0]
    debt_index = _torch_safe_divide(torch.stack([ratios[0]]), prev_ratios[0:1])[0]
    debt_shift = torch.relu(ratios[0] - prev_ratios[0]) if torch.isfinite(prev_ratios[0]) else torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
    if np.isfinite(income_cashadj):
        denom = torch.clamp(torch.abs(ratios[2]), min=1e-9)
        cash_income_mismatch = torch.abs(ratios[2] - float(income_cashadj)) / denom
    else:
        cash_income_mismatch = torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
    m_beneish = _torch_nanmean(
        torch.stack(
            (
                0.920 * dsri,
                0.404 * aqi,
                -0.172 * sgai,
                4.679 * tata,
            )
        )
    )
    m_sharia = _torch_nanmean(torch.stack((income_index, debt_index, debt_shift, cash_income_mismatch)))
    z3_raw = _torch_nanmean(torch.stack((m_beneish, m_sharia)))

    xintq = _row_value_tensor(candidate_x, ctx=ctx, column="xintq")
    niq = _row_value_tensor(candidate_x, ctx=ctx, column="niq")
    oibdpq = _row_value_tensor(candidate_x, ctx=ctx, column="oibdpq")
    cheq = _row_value_tensor(candidate_x, ctx=ctx, column="cheq")
    lctq = _row_value_tensor(candidate_x, ctx=ctx, column="lctq")
    total_debt = _total_debt_tensor(candidate_x, ctx=ctx)
    prev_total_debt = _safe_total_debt_values(
        dlttq=float(ctx.prev_row_values.get("dlttq", np.nan)),
        dlcq=float(ctx.prev_row_values.get("dlcq", np.nan)),
        ltq=float(ctx.prev_row_values.get("ltq", np.nan)),
    )
    current_non_cash_wc = (actq - cheq) - (lctq - _row_value_tensor(candidate_x, ctx=ctx, column="dlcq"))
    prev_non_cash_wc = np.nan
    if all(np.isfinite(ctx.prev_row_values.get(col, np.nan)) for col in ("actq", "cheq", "lctq", "dlcq")):
        prev_non_cash_wc = (ctx.prev_row_values["actq"] - ctx.prev_row_values["cheq"]) - (ctx.prev_row_values["lctq"] - ctx.prev_row_values["dlcq"])
    delta_non_cash_wc = current_non_cash_wc - float(prev_non_cash_wc) if np.isfinite(prev_non_cash_wc) else torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
    delta_interest_to_debt = torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
    prev_xintq = float(ctx.prev_row_values.get("xintq", np.nan))
    if np.isfinite(prev_xintq) and np.isfinite(prev_total_debt) and abs(total_debt.detach().cpu().item() - prev_total_debt) > 1e-9:
        delta_interest_to_debt = (xintq - prev_xintq) / (total_debt - prev_total_debt)
    z5_relations_raw = torch.stack(
        (
            oancfq / revtq,
            (ibq - oancfq) / atq,
            xintq / torch.clamp(total_debt, min=1e-9),
            delta_interest_to_debt,
            delta_non_cash_wc / atq,
            ratios[0],
            ratios[1],
            ratios[2],
            (niq - oibdpq) / atq,
        )
    )
    # Match the exact detector's anti-double-counting rule (d5_coherence.py):
    # drop the working-capital accrual proxy whenever the exact CFO-based
    # accrual (tata_accrual) is available for this row.
    tata_available = torch.isfinite(z5_relations_raw[_Z5_TATA_ACCRUAL_IDX])
    wc_proxy = torch.where(
        tata_available,
        torch.full_like(z5_relations_raw[_Z5_WC_ACCRUAL_PROXY_IDX], float("nan")),
        z5_relations_raw[_Z5_WC_ACCRUAL_PROXY_IDX],
    )
    z5_relations_raw = torch.cat(
        [
            z5_relations_raw[:_Z5_WC_ACCRUAL_PROXY_IDX],
            wc_proxy.unsqueeze(0),
            z5_relations_raw[_Z5_WC_ACCRUAL_PROXY_IDX + 1 :],
        ]
    )
    # Standardize each relation on the same split-C scale as the exact
    # detector before combining them, instead of averaging heterogeneous
    # units (ratios ~O(1) vs accrual ratios ~O(0.01)) directly.
    z5_relations = (z5_relations_raw - ctx.z5_relation_mu) / ctx.z5_relation_sigma
    z5_finite = torch.isfinite(z5_relations)
    z5_q_eff = int(z5_finite.sum().item())
    z5_raw = torch.sqrt(torch.clamp(_torch_nanmean(torch.square(z5_relations)), min=0.0))

    z6_mu = _torch_tensor(ctx.z6_hist_mu, ctx=ctx)
    z6_sigma = _torch_tensor(ctx.z6_hist_sigma, ctx=ctx)
    delta_ratios = ratios - prev_ratios
    valid_temporal = torch.isfinite(delta_ratios) & torch.isfinite(z6_mu) & torch.isfinite(z6_sigma) & (torch.abs(z6_sigma) > 1e-9)
    temporal_z = torch.full_like(delta_ratios, float("nan"))
    temporal_z[valid_temporal] = (delta_ratios[valid_temporal] - z6_mu[valid_temporal]) / z6_sigma[valid_temporal]
    z6_p_eff = int(valid_temporal.sum().item())
    z6_raw = torch.sqrt(torch.clamp(_torch_nanmean(torch.square(temporal_z)), min=0.0))

    # z7: exact peer/sector Mahalanobis distance (see d7_peer.py). mu_ref/inv_ref
    # depend only on the split-C calibration sample and this row's own (fixed)
    # sector, so they are precomputed exactly (numpy/sklearn, non-differentiable)
    # and reused here as constants — only the bilinear form itself is torch.
    z7_diff = ratios - ctx.z7_mu_ref
    z7_raw = z7_diff @ ctx.z7_inv_ref @ z7_diff

    if np.isfinite(ctx.z8_hist_median) and np.isfinite(ctx.z8_hist_mad) and ctx.z8_hist_mad > 1e-10:
        cod = xintq / torch.clamp(total_debt, min=1e-9)
        z8_raw = (cod - float(ctx.z8_hist_median)) / float(ctx.z8_hist_mad)
    else:
        z8_raw = torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)

    return {
        "z3": z3_raw,
        "z4": z4_raw,
        "z5": z5_raw,
        "z6": z6_raw,
        "z7": z7_raw,
        "z8": z8_raw,
        "z4_penalty": z4_penalty,
        "z5_q_eff": z5_q_eff,
        "z6_p_eff": z6_p_eff,
    }


def _surrogate_detector_proxies(
    candidate_x: torch.Tensor,
    *,
    ctx: TorchSurrogateContext,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    raws = _surrogate_detector_raws(candidate_x, ctx=ctx, temperature=temperature)
    z3_proxy = _proxy_from_raw(raws["z3"], baseline_z=ctx.baseline_z3, baseline_raw=ctx.z3_baseline_raw, slope=ctx.z3_slope)
    z4_proxy = _proxy_from_raw(raws["z4"], baseline_z=ctx.baseline_z4, baseline_raw=ctx.z4_baseline_raw, slope=ctx.z4_slope)
    z5_proxy = _proxy_from_raw(raws["z5"], baseline_z=ctx.baseline_z5, baseline_raw=ctx.z5_baseline_raw, slope=ctx.z5_slope)
    z6_proxy = _proxy_from_raw(raws["z6"], baseline_z=ctx.baseline_z6, baseline_raw=ctx.z6_baseline_raw, slope=ctx.z6_slope)
    z7_proxy = _proxy_from_raw(raws["z7"], baseline_z=ctx.baseline_z7, baseline_raw=ctx.z7_baseline_raw, slope=ctx.z7_slope)
    z8_proxy = _proxy_from_raw(raws["z8"], baseline_z=ctx.baseline_z8, baseline_raw=ctx.z8_baseline_raw, slope=ctx.z8_slope)
    merged_soft = torch.logsumexp(torch.stack((z5_proxy / temperature, z7_proxy / temperature)), dim=0) * temperature
    baseline_soft = torch.logsumexp(
        torch.tensor(
            [ctx.baseline_z5 / temperature, ctx.baseline_z7 / temperature],
            dtype=ctx.dtype,
            device=ctx.device,
        ),
        dim=0,
    ) * temperature
    z57_proxy = torch.clamp(ctx.baseline_z57 + (merged_soft - baseline_soft), min=0.0)
    return {
        "z3": z3_proxy,
        "z4": z4_proxy,
        "z5": z5_proxy,
        "z6": z6_proxy,
        "z7": z7_proxy,
        "z57": z57_proxy,
        "z8": z8_proxy,
        "z4_penalty": raws["z4_penalty"],
    }


def torch_ratio_vector(candidate_x: torch.Tensor, *, ctx: TorchSurrogateContext) -> torch.Tensor:
    """Compute canonical Shariah ratios from a torch candidate vector."""
    atq = torch.clamp(_row_value_tensor(candidate_x, ctx=ctx, column="atq"), min=1e-9)
    dlttq = _row_value_tensor(candidate_x, ctx=ctx, column="dlttq")
    dlcq = _row_value_tensor(candidate_x, ctx=ctx, column="dlcq")
    cheq = _row_value_tensor(candidate_x, ctx=ctx, column="cheq")
    iditq = _row_value_tensor(candidate_x, ctx=ctx, column="iditq")
    revtq = torch.clamp(_row_value_tensor(candidate_x, ctx=ctx, column="revtq"), min=1e-9)
    sukuk_ratio_t = _finite_base_float(ctx.base_row.get("sukuk_ratio_t"), default=0.0)
    islamic_cash_ratio_t = _finite_base_float(ctx.base_row.get("islamic_cash_ratio_t"), default=0.0)
    sukuk_keep = max(1.0 - sukuk_ratio_t, 1e-9)
    cash_keep = max(1.0 - islamic_cash_ratio_t, 1e-9)
    return torch.stack(
        (
            (dlttq * sukuk_keep + dlcq) / atq,
            (cheq * cash_keep) / atq,
            iditq / revtq,
        )
    )


def _surrogate_signal_terms(
    delta_norm: torch.Tensor,
    *,
    ctx: TorchSurrogateContext,
    max_eps: float,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw differentiable anomaly and threshold-pressure signals."""
    candidate_x = torch_candidate_x_from_delta(delta_norm, ctx=ctx, max_eps=max_eps)
    ratios = torch_ratio_vector(candidate_x, ctx=ctx)
    centers, scales, thresholds = _ratio_reference_tensors(ctx)
    standardized = torch.sqrt(torch.square((ratios - centers) / scales) + 1e-8)
    softmax_anomaly = torch.sum(torch.softmax(standardized / temperature, dim=0) * standardized)
    sc_penalty = torch.nn.functional.softplus((ratios - thresholds) / temperature).sum()
    return softmax_anomaly, sc_penalty


def torch_surrogate_loss_from_delta(
    delta_norm: torch.Tensor,
    *,
    ctx: TorchSurrogateContext,
    max_eps: float,
    direction: str = "to_green",
    penalty_lambda: float = 1.0,
    delta_l1_lambda: float = 0.0,
    temperature: float = 0.1,
    penalty_prox_lambda: float = 1e-4,
) -> torch.Tensor:
    """Evaluate the differentiable row-wise surrogate objective."""
    candidate_x = torch_candidate_x_from_delta(delta_norm, ctx=ctx, max_eps=max_eps)
    proxies = _surrogate_detector_proxies(candidate_x, ctx=ctx, temperature=temperature)
    finite_signals = [proxies[name] for name in ("z3", "z4", "z57", "z6", "z8") if bool(torch.isfinite(proxies[name]).item())]
    signal = torch.stack(finite_signals).mean() if finite_signals else torch.zeros((), dtype=ctx.dtype, device=ctx.device)
    if direction == "to_green":
        signal_loss = signal
    elif direction == "to_red":
        signal_loss = -signal
    else:
        raise ValueError(f"Unsupported torch surrogate direction {direction!r}.")
    delta_l1 = delta_l1_lambda * torch.abs(delta_norm).sum()
    delta_l2 = penalty_prox_lambda * torch.square(delta_norm).sum()
    return signal_loss + penalty_lambda * proxies["z4_penalty"] + delta_l1 + delta_l2


def torch_hybrid_exact_like_loss_from_delta(
    delta_norm: torch.Tensor,
    *,
    ctx: TorchSurrogateContext,
    max_eps: float,
    target_score_name: str,
    direction: str,
    threshold: float,
    lambda_l1: float,
    lambda_l2: float,
    loss_name: str,
    aggregate_mode: str = "mean",
    temperature: float = 0.1,
    penalty_lambda: float = 1.0,
    aux_surrogate_weight: float = 0.1,
) -> torch.Tensor:
    """Blend a differentiable z4 proxy with the exact Family-3 composite loss.

    We keep the active detector vector from the authoritative score context
    fixed except for `z4`, which is replaced by a differentiable proxy derived
    from the attacked Shariah ratios. This is still approximate, but it is much
    closer to the exact Family-3 composite objective than the legacy
    surrogate-only path.
    """
    if not ctx.active_indices:
        return torch_surrogate_loss_from_delta(
            delta_norm,
            ctx=ctx,
            max_eps=max_eps,
            direction=direction,
            penalty_lambda=penalty_lambda,
            delta_l1_lambda=lambda_l1,
            temperature=temperature,
            penalty_prox_lambda=lambda_l2,
        )

    candidate_x = torch_candidate_x_from_delta(delta_norm, ctx=ctx, max_eps=max_eps)
    proxies = _surrogate_detector_proxies(candidate_x, ctx=ctx, temperature=temperature)
    z_matrix = ctx.baseline_active_z.unsqueeze(0).clone()
    for name in ("z3", "z4", "z5", "z6", "z7", "z57", "z8"):
        idx = ctx.active_indices.get(name)
        if idx is not None and name in proxies:
            z_matrix[0, idx] = proxies[name]
    loss_total, _, _, _, _, _ = torch_family3_exact_cohort_eval(
        z_matrix=z_matrix,
        sigma=ctx.sigma,
        weights=ctx.weights,
        target_score_name=target_score_name,
        direction=direction,
        delta_norm=delta_norm.unsqueeze(0),
        threshold=threshold,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        loss_name=loss_name,
        aggregate_mode=aggregate_mode,
        active_threshold=ctx.active_threshold,
        min_active_for_renorm=ctx.min_active_for_renorm,
        min_active_for_iut=ctx.min_active_for_iut,
        null_ctx=ctx.null_ctx,
    )
    aux_sign = 1.0 if direction == "to_green" else -1.0
    aux_penalty = aux_sign * aux_surrogate_weight * penalty_lambda * proxies["z4_penalty"]
    return loss_total + aux_penalty


def candidate_x_numpy_from_delta(
    delta_norm: np.ndarray,
    *,
    ctx: TorchSurrogateContext,
    max_eps: float,
) -> np.ndarray:
    """Convert normalized deltas back to a numpy raw-value candidate."""
    delta = np.asarray(delta_norm, dtype=float)
    delta = np.clip(delta, -max_eps, max_eps)
    return np.maximum(ctx.x0 + delta * ctx.scale, 0.0)


def resolve_surrogate_knobs(
    settings,
    *,
    penalty_lambda: float | None,
    delta_l1_lambda: float | None,
    temperature: float | None,
    penalty_prox_lambda: float | None,
) -> tuple[float, float, float, float]:
    """Resolve surrogate hyperparameters from explicit args or benchmark settings."""
    rb = settings.robustness_benchmark
    return (
        float(
            penalty_lambda
            if penalty_lambda is not None
            else getattr(rb, "family3_torch_surrogate_penalty_lambda", 1.0)
        ),
        float(
            delta_l1_lambda
            if delta_l1_lambda is not None
            else getattr(rb, "family3_lambda_l1", 0.0)
        ),
        float(
            temperature
            if temperature is not None
            else getattr(rb, "family3_torch_surrogate_temperature", 0.1)
        ),
        float(
            penalty_prox_lambda
            if penalty_prox_lambda is not None
            else getattr(rb, "family3_torch_surrogate_prox_lambda", 1e-4)
        ),
    )


def run_family3_torch_surrogate(
    *,
    panel: pd.DataFrame,
    base_ctx,
    settings,
    row_index: object,
    surrogate_mode: str = "softmax_general",
    step_size: float = 0.05,
    eps_grid: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1),
    iterations: int = 25,
    penalty_lambda: float | None = None,
    delta_l1_lambda: float | None = None,
    temperature: float | None = None,
    penalty_prox_lambda: float | None = None,
    enforce_sac: bool = True,
    tolerance: float = 1e-9,
    modifiable_columns: tuple[str, ...] | None = None,
) -> tuple[TorchPGDResult, TorchSurrogateContext]:
    """Run the PGD-style torch surrogate path and replay exact scoring."""
    resolved_penalty_lambda, resolved_delta_l1_lambda, resolved_temperature, resolved_penalty_prox_lambda = resolve_surrogate_knobs(
        settings,
        penalty_lambda=penalty_lambda,
        delta_l1_lambda=delta_l1_lambda,
        temperature=temperature,
        penalty_prox_lambda=penalty_prox_lambda,
    )
    ctx = build_torch_surrogate_context(
        panel=panel,
        base_ctx=base_ctx,
        settings=settings,
        row_index=row_index,
        modifiable_columns=modifiable_columns or SUPPORTED_TORCH_RAW_COLUMNS,
    )
    evaluator = _build_fast_row_evaluator(base_ctx, row_index, settings)
    result = run_torch_projected_pgd(
        x0=ctx.x0,
        base_row=ctx.base_row.to_dict(),
        step_size=step_size,
        eps_grid=eps_grid,
        iterations=iterations,
        penalty_lambda=resolved_penalty_lambda,
        temperature=resolved_temperature,
        surrogate_mode=surrogate_mode,
        score_fn=lambda candidate: evaluator.evaluate(
            pd.Series(
                {
                    **ctx.base_row.to_dict(),
                    **{col: float(val) for col, val in zip(ctx.modifiable_columns, candidate)},
                }
            )
        ),
        tolerance=tolerance,
        penalty_prox_lambda=resolved_penalty_prox_lambda,
        enforce_sac=enforce_sac,
    )
    return result, ctx


"""Torch-specific optimizer adapters for canonical Family 3 counterfactual.

This module isolates the experimental ``torch_adam`` path from the main exact
engine so the production numpy path stays readable and stable. The adapters
defined here are intentionally narrow:

- validate that the requested raw columns are supported by the torch surrogate
- build the torch objective and projection helpers
- replay the exact evaluator over the torch trajectory to keep model selection
  anchored to the authoritative exact loss

The long-term goal is to keep the engine orchestration free of torch-specific
details while keeping the public Family 3 torch path physically co-located with
the canonical counterfactual package.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.analysis.optimizers import OptimizationEvaluation, OptimizationResult

log = logging.getLogger(__name__)

SUPPORTED_TORCH_RAW_COLUMNS: tuple[str, ...] = (
    "revtq",
    "iditq",
    "nopiq",
    "xintq",
    "niq",
    "oibdpq",
    "oancfq",
    "ibq",
    "atq",
    "dlttq",
    "dlcq",
    "ltq",
    "cheq",
    "actq",
    "lctq",
    "rectq",
    "invtq",
    "xsgaq",
    "ppentq",
)


@dataclass(frozen=True)
class TorchSurrogateKnobs:
    """Differentiable surrogate hyperparameters shared by torch adapters."""

    penalty_lambda: float
    delta_l1_lambda: float
    temperature: float
    penalty_prox_lambda: float


@dataclass(frozen=True)
class TorchExactReplayConfig:
    """Controls how much of the torch trajectory gets rescored exactly."""

    mode: str
    stride: int


@dataclass(frozen=True)
class TorchObjectiveConfig:
    """Controls which torch objective is optimized in the experimental branch."""

    mode: str
    aux_surrogate_weight: float


def _log_torch_experimental_assumptions(
    *,
    log_prefix: str,
    objective_config: TorchObjectiveConfig,
    replay_config: TorchExactReplayConfig,
) -> None:
    """Make the torch approximation boundary explicit in the logs.

    The torch branch remains research-oriented: it optimizes a differentiable
    proxy and then reselects on the exact evaluator. Surfacing that boundary in
    logs makes run interpretation safer than silently looking "exact".
    """
    log.info(
        "%s torch_adam experimental path: objective=%s, exact_replay=%s/%d. "
        "Optimization uses a differentiable proxy and exact loss remains the final authority.",
        log_prefix,
        objective_config.mode,
        replay_config.mode,
        replay_config.stride,
    )


def ensure_supported_torch_modifiable_columns(mod_cols: tuple[str, ...]) -> None:
    """Reject unsupported raw columns for the current torch surrogate."""
    unsupported = sorted(set(mod_cols) - set(SUPPORTED_TORCH_RAW_COLUMNS))
    if unsupported:
        raise ValueError(
            "optimizer_name='torch_adam' currently supports only modifiable columns "
            f"{list(SUPPORTED_TORCH_RAW_COLUMNS)}, got unsupported columns {unsupported}."
        )


def resolve_torch_surrogate_knobs(settings) -> TorchSurrogateKnobs:
    """Read surrogate hyperparameters from robustness benchmark settings."""
    rb = settings.robustness_benchmark
    return TorchSurrogateKnobs(
        penalty_lambda=float(getattr(rb, "family3_torch_surrogate_penalty_lambda", 1.0)),
        delta_l1_lambda=float(getattr(rb, "family3_lambda_l1", 0.0)),
        temperature=float(getattr(rb, "family3_torch_surrogate_temperature", 0.1)),
        penalty_prox_lambda=float(getattr(rb, "family3_torch_surrogate_prox_lambda", 1e-4)),
    )


def resolve_torch_exact_replay_config(settings) -> TorchExactReplayConfig:
    """Read exact replay policy for the experimental torch branch."""
    rb = settings.robustness_benchmark
    return TorchExactReplayConfig(
        mode=str(getattr(rb, "family3_torch_exact_replay_mode", "full")),
        stride=max(1, int(getattr(rb, "family3_torch_exact_replay_stride", 5))),
    )


def resolve_torch_objective_config(settings) -> TorchObjectiveConfig:
    """Read objective mode for the experimental torch branch."""
    rb = settings.robustness_benchmark
    return TorchObjectiveConfig(
        mode=str(getattr(rb, "family3_torch_objective_mode", "hybrid_exact_like")),
        aux_surrogate_weight=float(getattr(rb, "family3_torch_aux_surrogate_weight", 0.1)),
    )


def _record_best_exact_eval(
    *,
    candidate_state: np.ndarray,
    candidate_eval: OptimizationEvaluation,
    best_state: np.ndarray,
    best_eval: OptimizationEvaluation,
) -> tuple[np.ndarray, OptimizationEvaluation]:
    """Keep the lowest exact loss seen while replaying a torch trajectory."""
    if candidate_eval.loss_total <= best_eval.loss_total + 1e-12:
        return candidate_state.copy(), candidate_eval
    return best_state, best_eval


def _build_surrogate_objective(
    *,
    surrogate_ctx,
    max_eps: float,
    knobs: TorchSurrogateKnobs,
    objective_config: TorchObjectiveConfig,
    settings,
    target_score_name: str,
    direction: str,
):
    """Return the differentiable surrogate objective used by ``torch_adam``."""
    rb = settings.robustness_benchmark
    threshold = float(rb.family3_z_target_green if direction == "to_green" else rb.family3_z_target_red)
    lambda_l1 = float(getattr(rb, "family3_lambda_l1", 0.0))
    lambda_l2 = float(getattr(rb, "family3_lambda_l2", 0.0))
    loss_name = str(getattr(rb, "family3_loss_name", "hinge"))

    def _objective(delta_norm):
        if objective_config.mode == "hybrid_exact_like":
            return torch_hybrid_exact_like_loss_from_delta(
                delta_norm,
                ctx=surrogate_ctx,
                max_eps=max_eps,
                target_score_name=target_score_name,
                direction=direction,
                threshold=threshold,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                loss_name=loss_name,
                aggregate_mode="mean",
                temperature=knobs.temperature,
                penalty_lambda=knobs.penalty_lambda,
                aux_surrogate_weight=objective_config.aux_surrogate_weight,
            )
        return torch_surrogate_loss_from_delta(
            delta_norm,
            ctx=surrogate_ctx,
            max_eps=max_eps,
            direction=direction,
            penalty_lambda=knobs.penalty_lambda,
            delta_l1_lambda=knobs.delta_l1_lambda,
            temperature=knobs.temperature,
            penalty_prox_lambda=knobs.penalty_prox_lambda,
        )

    return _objective


def _build_projection(max_eps: float):
    """Return the box projection used after each Adam update."""
    import torch

    def _project(delta_norm):
        return torch.clamp(delta_norm, min=-max_eps, max=max_eps)

    return _project


def _replay_exact_trajectory(
    *,
    initial_state: np.ndarray,
    torch_result,
    evaluate_fn: Callable[[np.ndarray], OptimizationEvaluation],
    epoch_callback: Callable[[int, OptimizationEvaluation, OptimizationEvaluation], None],
    replay_config: TorchExactReplayConfig,
) -> tuple[np.ndarray, OptimizationEvaluation]:
    """Score every torch state with the authoritative exact evaluator."""
    best_state = np.asarray(initial_state, dtype=float).copy()
    best_eval = evaluate_fn(best_state)
    epoch_callback(0, best_eval, best_eval)

    if replay_config.mode == "best_only":
        replay_states = []
        if torch_result.executed_epochs > 0 and len(torch_result.state_trace) > 1:
            replay_states.append((torch_result.executed_epochs, torch_result.state_trace[-1]))
        replay_states.append((torch_result.executed_epochs, torch_result.best_state))
    elif replay_config.mode == "stride":
        replay_states = [
            (epoch, state_tensor)
            for epoch, state_tensor in enumerate(torch_result.state_trace[1:], start=1)
            if epoch % replay_config.stride == 0 or epoch == torch_result.executed_epochs
        ]
        replay_states.append((torch_result.executed_epochs, torch_result.best_state))
    else:
        replay_states = list(enumerate(torch_result.state_trace[1:], start=1))

    seen_keys: set[tuple[int, bytes]] = set()
    for epoch, state_tensor in replay_states:
        delta_state = np.asarray(state_tensor.detach().cpu().tolist(), dtype=float)
        state_key = (int(epoch), delta_state.tobytes())
        if state_key in seen_keys:
            continue
        seen_keys.add(state_key)
        current_eval = evaluate_fn(delta_state)
        best_state, best_eval = _record_best_exact_eval(
            candidate_state=delta_state,
            candidate_eval=current_eval,
            best_state=best_state,
            best_eval=best_eval,
        )
        epoch_callback(epoch, current_eval, best_eval)
    return best_state, best_eval


def _build_row_surrogate_loss(
    *,
    ctx,
    delta_norm,
    max_eps: float,
    knobs: TorchSurrogateKnobs,
    objective_config: TorchObjectiveConfig,
    settings,
    target_score_name: str,
    direction: str,
):
    """Return one row-level surrogate loss for a shared Family 3 target."""

    rb = settings.robustness_benchmark
    threshold = float(rb.family3_z_target_green if direction == "to_green" else rb.family3_z_target_red)
    if objective_config.mode == "hybrid_exact_like":
        return torch_hybrid_exact_like_loss_from_delta(
            delta_norm,
            ctx=ctx,
            max_eps=max_eps,
            target_score_name=target_score_name,
            direction=direction,
            threshold=threshold,
            lambda_l1=float(getattr(rb, "family3_lambda_l1", 0.0)),
            lambda_l2=float(getattr(rb, "family3_lambda_l2", 0.0)),
            loss_name=str(getattr(rb, "family3_loss_name", "hinge")),
            aggregate_mode="mean",
            temperature=knobs.temperature,
            penalty_lambda=knobs.penalty_lambda,
            aux_surrogate_weight=objective_config.aux_surrogate_weight,
        )
    return torch_surrogate_loss_from_delta(
        delta_norm,
        ctx=ctx,
        max_eps=max_eps,
        direction=direction,
        penalty_lambda=knobs.penalty_lambda,
        delta_l1_lambda=knobs.delta_l1_lambda,
        temperature=knobs.temperature,
        penalty_prox_lambda=knobs.penalty_prox_lambda,
    )


def _firm_objective_from_row_losses(losses, *, direction: str):
    """Aggregate row-level surrogate losses under the min-pvalue firm logic."""
    if direction == "to_green":
        return losses.max()
    if direction == "to_red":
        return losses.min()
    raise ValueError(f"Unsupported Family 3 direction {direction!r}.")


def run_rowwise_torch_adam(
    *,
    panel: pd.DataFrame,
    base_ctx,
    settings,
    row_index: object,
    mod_cols: tuple[str, ...],
    target_score_name: str,
    direction: str,
    max_epochs: int,
    max_eps: float,
    step_size: float,
    log_prefix: str,
    evaluate_fn: Callable[[np.ndarray], OptimizationEvaluation],
    epoch_callback: Callable[[int, OptimizationEvaluation, OptimizationEvaluation], None],
    early_stop_loss: float | None,
    early_stop_patience: int,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    plateau_shrink_patience: int,
    plateau_shrink_factor: float,
    min_step_size: float,
    restore_best_on_shrink: bool,
    reset_moments_on_shrink: bool,
    shared_ctx: TorchSharedRunContext | None = None,
    firm_panel_cache: dict[object, tuple[pd.DataFrame, pd.DataFrame] | None] | None = None,
) -> OptimizationResult:
    """Run the experimental row-wise torch Adam path and keep exact selection.

    Args:
        shared_ctx: Optional pre-built device/null-context/sigma/weights/peer
            reference bundle shared across every row of a `global_per_row`
            run (see ``build_torch_shared_run_context``). Mirrors the same
            parameter already used by ``run_cohort_torch_adam``/
            ``run_local_firm_torch_adam`` -- without it, this function
            rebuilds the whole bundle (including the ~11-12s LedoitWolf peer
            reference fit) from scratch on every single row (see profiling notes
            profiling notes).
        firm_panel_cache: Optional per-firm history cache shared across rows
            of the same firm within a run (see ``_build_firm_history_panel``).
    """
    import torch

    ensure_supported_torch_modifiable_columns(mod_cols)
    knobs = resolve_torch_surrogate_knobs(settings)
    replay_config = resolve_torch_exact_replay_config(settings)
    objective_config = resolve_torch_objective_config(settings)
    _log_torch_experimental_assumptions(
        log_prefix=log_prefix,
        objective_config=objective_config,
        replay_config=replay_config,
    )
    surrogate_ctx = build_torch_surrogate_context(
        panel=panel,
        base_ctx=base_ctx,
        settings=settings,
        row_index=row_index,
        modifiable_columns=mod_cols,
        prefer_gpu=True,
        shared_ctx=shared_ctx,
        firm_panel_cache=firm_panel_cache,
    )
    initial_state = torch.zeros(len(mod_cols), dtype=surrogate_ctx.dtype, device=surrogate_ctx.device)

    torch_result = run_torch_adam(
        initial_state=initial_state,
        objective_fn=_build_surrogate_objective(
            surrogate_ctx=surrogate_ctx,
            max_eps=max_eps,
            knobs=knobs,
            objective_config=objective_config,
            settings=settings,
            target_score_name=target_score_name,
            direction=direction,
        ),
        projection_fn=_build_projection(max_eps),
        max_epochs=max_epochs,
        step_size=step_size,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        early_stop_patience=early_stop_patience,
        early_stop_loss=early_stop_loss,
        plateau_shrink_patience=plateau_shrink_patience,
        plateau_shrink_factor=plateau_shrink_factor,
        min_step_size=min_step_size,
        restore_best_on_shrink=restore_best_on_shrink,
        reset_moments_on_shrink=reset_moments_on_shrink,
    )
    best_state, best_eval = _replay_exact_trajectory(
        initial_state=np.zeros(len(mod_cols), dtype=float),
        torch_result=torch_result,
        evaluate_fn=evaluate_fn,
        epoch_callback=epoch_callback,
        replay_config=replay_config,
    )
    log.info(
        "%s torch_adam finished device=%s executed_epochs=%d objective=%s surrogate_best_loss=%.6f exact_best_loss=%.6f replay=(mode=%s,stride=%d) knobs=(threshold_lambda=%.4g,delta_l1=%.4g,temp=%.4g,delta_l2=%.4g)",
        log_prefix,
        surrogate_ctx.device,
        torch_result.executed_epochs,
        objective_config.mode,
        torch_result.best_loss,
        best_eval.loss_total,
        replay_config.mode,
        replay_config.stride,
        knobs.penalty_lambda,
        knobs.delta_l1_lambda,
        knobs.temperature,
        knobs.penalty_prox_lambda,
    )
    return OptimizationResult(
        best_state=best_state,
        best_evaluation=best_eval,
        executed_epochs=torch_result.executed_epochs,
    )


def run_cohort_torch_adam(
    *,
    panel: pd.DataFrame,
    base_ctx,
    settings,
    row_indices: list[object],
    mod_cols: tuple[str, ...],
    target_score_name: str,
    direction: str,
    max_epochs: int,
    max_eps: float,
    step_size: float,
    log_prefix: str,
    evaluate_fn: Callable[[np.ndarray], OptimizationEvaluation],
    epoch_callback: Callable[[int, OptimizationEvaluation, OptimizationEvaluation], None],
    early_stop_loss: float | None,
    early_stop_patience: int,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    plateau_shrink_patience: int,
    plateau_shrink_factor: float,
    min_step_size: float,
    restore_best_on_shrink: bool,
    reset_moments_on_shrink: bool,
) -> OptimizationResult:
    """Run the experimental cohort torch Adam path with shared cohort deltas."""
    import torch

    ensure_supported_torch_modifiable_columns(mod_cols)
    knobs = resolve_torch_surrogate_knobs(settings)
    replay_config = resolve_torch_exact_replay_config(settings)
    objective_config = resolve_torch_objective_config(settings)
    _log_torch_experimental_assumptions(
        log_prefix=log_prefix,
        objective_config=objective_config,
        replay_config=replay_config,
    )
    scale_floor = family3_column_scale_floor(panel, mod_cols, settings.robustness_benchmark.family3_scale_floor_quantile)
    shared_ctx = build_torch_shared_run_context(base_ctx, prefer_gpu=True, scale_floor=scale_floor)
    firm_panel_cache: dict[object, tuple[pd.DataFrame, pd.DataFrame] | None] = {}
    surrogate_contexts = [
        build_torch_surrogate_context(
            panel=panel,
            base_ctx=base_ctx,
            settings=settings,
            row_index=row_index,
            modifiable_columns=mod_cols,
            shared_ctx=shared_ctx,
            firm_panel_cache=firm_panel_cache,
        )
        for row_index in row_indices
    ]
    initial_state = torch.zeros(len(mod_cols), dtype=surrogate_contexts[0].dtype, device=surrogate_contexts[0].device)
    aggregate_mode = str(getattr(settings.robustness_benchmark, "family3_cohort_loss_mode", "mean"))

    def _objective(delta_norm):
        """Aggregate the row-wise surrogate over the whole cohort."""
        rb = settings.robustness_benchmark
        threshold = float(rb.family3_z_target_green if direction == "to_green" else rb.family3_z_target_red)

        losses = torch.stack(
            [
                (
                    torch_hybrid_exact_like_loss_from_delta(
                        delta_norm,
                        ctx=ctx,
                        max_eps=max_eps,
                        target_score_name=target_score_name,
                        direction=direction,
                        threshold=threshold,
                        lambda_l1=float(getattr(rb, "family3_lambda_l1", 0.0)),
                        lambda_l2=float(getattr(rb, "family3_lambda_l2", 0.0)),
                        loss_name=str(getattr(rb, "family3_loss_name", "hinge")),
                        aggregate_mode="mean",
                        temperature=knobs.temperature,
                        penalty_lambda=knobs.penalty_lambda,
                        aux_surrogate_weight=objective_config.aux_surrogate_weight,
                    )
                    if objective_config.mode == "hybrid_exact_like"
                    else torch_surrogate_loss_from_delta(
                        delta_norm,
                        ctx=ctx,
                        max_eps=max_eps,
                        direction=direction,
                        penalty_lambda=knobs.penalty_lambda,
                        delta_l1_lambda=knobs.delta_l1_lambda,
                        temperature=knobs.temperature,
                        penalty_prox_lambda=knobs.penalty_prox_lambda,
                    )
                )
                for ctx in surrogate_contexts
            ]
        )
        if aggregate_mode == "mean":
            return losses.mean()
        if aggregate_mode == "min":
            return losses.max()
        raise ValueError(f"Unsupported torch cohort aggregate mode {aggregate_mode!r}.")

    torch_result = run_torch_adam(
        initial_state=initial_state,
        objective_fn=_objective,
        projection_fn=_build_projection(max_eps),
        max_epochs=max_epochs,
        step_size=step_size,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        early_stop_patience=early_stop_patience,
        early_stop_loss=early_stop_loss,
        plateau_shrink_patience=plateau_shrink_patience,
        plateau_shrink_factor=plateau_shrink_factor,
        min_step_size=min_step_size,
        restore_best_on_shrink=restore_best_on_shrink,
        reset_moments_on_shrink=reset_moments_on_shrink,
    )
    best_state, best_eval = _replay_exact_trajectory(
        initial_state=np.zeros(len(mod_cols), dtype=float),
        torch_result=torch_result,
        evaluate_fn=evaluate_fn,
        epoch_callback=epoch_callback,
        replay_config=replay_config,
    )
    log.info(
        "%s torch_adam finished device=%s executed_epochs=%d objective=%s surrogate_best_loss=%.6f exact_best_loss=%.6f replay=(mode=%s,stride=%d) knobs=(threshold_lambda=%.4g,delta_l1=%.4g,temp=%.4g,delta_l2=%.4g)",
        log_prefix,
        surrogate_contexts[0].device,
        torch_result.executed_epochs,
        objective_config.mode,
        torch_result.best_loss,
        best_eval.loss_total,
        replay_config.mode,
        replay_config.stride,
        knobs.penalty_lambda,
        knobs.delta_l1_lambda,
        knobs.temperature,
        knobs.penalty_prox_lambda,
    )
    return OptimizationResult(
        best_state=best_state,
        best_evaluation=best_eval,
        executed_epochs=torch_result.executed_epochs,
    )


def run_local_firm_torch_adam(
    *,
    panel: pd.DataFrame,
    base_ctx,
    settings,
    row_indices: list[object],
    mod_cols: tuple[str, ...],
    target_score_name: str,
    direction: str,
    max_epochs: int,
    max_eps: float,
    step_size: float,
    log_prefix: str,
    evaluate_fn: Callable[[np.ndarray], OptimizationEvaluation],
    epoch_callback: Callable[[int, OptimizationEvaluation, OptimizationEvaluation], None],
    early_stop_loss: float | None,
    early_stop_patience: int,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    plateau_shrink_patience: int,
    plateau_shrink_factor: float,
    min_step_size: float,
    restore_best_on_shrink: bool,
    reset_moments_on_shrink: bool,
) -> OptimizationResult:
    """Run the experimental firm-level torch Adam path over all firm quarters."""
    import torch

    ensure_supported_torch_modifiable_columns(mod_cols)
    knobs = resolve_torch_surrogate_knobs(settings)
    replay_config = resolve_torch_exact_replay_config(settings)
    objective_config = resolve_torch_objective_config(settings)
    _log_torch_experimental_assumptions(
        log_prefix=log_prefix,
        objective_config=objective_config,
        replay_config=replay_config,
    )
    scale_floor = family3_column_scale_floor(panel, mod_cols, settings.robustness_benchmark.family3_scale_floor_quantile)
    shared_ctx = build_torch_shared_run_context(base_ctx, prefer_gpu=True, scale_floor=scale_floor)
    firm_panel_cache: dict[object, tuple[pd.DataFrame, pd.DataFrame] | None] = {}
    surrogate_contexts = [
        build_torch_surrogate_context(
            panel=panel,
            base_ctx=base_ctx,
            settings=settings,
            row_index=row_index,
            modifiable_columns=mod_cols,
            shared_ctx=shared_ctx,
            firm_panel_cache=firm_panel_cache,
        )
        for row_index in row_indices
    ]
    first_ctx = surrogate_contexts[0]
    state_dim = len(row_indices) * len(mod_cols)
    initial_state = torch.zeros(state_dim, dtype=first_ctx.dtype, device=first_ctx.device)

    def _objective(flat_delta_norm):
        delta_matrix = flat_delta_norm.reshape(len(row_indices), len(mod_cols))
        losses = torch.stack(
            [
                _build_row_surrogate_loss(
                    ctx=ctx,
                    delta_norm=delta_matrix[pos],
                    max_eps=max_eps,
                    knobs=knobs,
                    objective_config=objective_config,
                    settings=settings,
                    target_score_name=target_score_name,
                    direction=direction,
                )
                for pos, ctx in enumerate(surrogate_contexts)
            ]
        )
        return _firm_objective_from_row_losses(losses, direction=direction)

    torch_result = run_torch_adam(
        initial_state=initial_state,
        objective_fn=_objective,
        projection_fn=_build_projection(max_eps),
        max_epochs=max_epochs,
        step_size=step_size,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        early_stop_patience=early_stop_patience,
        early_stop_loss=early_stop_loss,
        plateau_shrink_patience=plateau_shrink_patience,
        plateau_shrink_factor=plateau_shrink_factor,
        min_step_size=min_step_size,
        restore_best_on_shrink=restore_best_on_shrink,
        reset_moments_on_shrink=reset_moments_on_shrink,
    )
    best_state, best_eval = _replay_exact_trajectory(
        initial_state=np.zeros(state_dim, dtype=float),
        torch_result=torch_result,
        evaluate_fn=evaluate_fn,
        epoch_callback=epoch_callback,
        replay_config=replay_config,
    )
    log.info(
        "%s torch_adam finished device=%s executed_epochs=%d objective=%s surrogate_best_loss=%.6f exact_best_loss=%.6f replay=(mode=%s,stride=%d) knobs=(threshold_lambda=%.4g,delta_l1=%.4g,temp=%.4g,delta_l2=%.4g)",
        log_prefix,
        first_ctx.device,
        torch_result.executed_epochs,
        objective_config.mode,
        torch_result.best_loss,
        best_eval.loss_total,
        replay_config.mode,
        replay_config.stride,
        knobs.penalty_lambda,
        knobs.delta_l1_lambda,
        knobs.temperature,
        knobs.penalty_prox_lambda,
    )
    return OptimizationResult(
        best_state=best_state,
        best_evaluation=best_eval,
        executed_epochs=torch_result.executed_epochs,
    )
