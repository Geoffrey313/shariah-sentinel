"""Reusable black-box optimizers for exact robustness attacks.

This module isolates the small numerical core from the much larger benchmark
orchestration code. Callers provide an evaluation function that scores one
candidate state and returns an opaque artifact/payload; the optimizer only
handles state updates, early stopping, and best-state tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationEvaluation:
    """Scalar objective plus caller-owned artifacts for one candidate state."""

    loss_total: float
    loss_score: float
    loss_regularizer: float
    threshold: float
    artifact: Any
    payload: np.ndarray
    loss_regularizer_l1: float = 0.0
    loss_regularizer_l2: float = 0.0


@dataclass(frozen=True)
class OptimizationResult:
    """Best state reached by one optimizer run."""

    best_state: np.ndarray
    best_evaluation: OptimizationEvaluation
    executed_epochs: int


EvaluateFn = Callable[[np.ndarray], OptimizationEvaluation]
EpochCallback = Callable[[int, OptimizationEvaluation, OptimizationEvaluation], None]


def _record_if_improved(
    *,
    candidate_state: np.ndarray,
    candidate_eval: OptimizationEvaluation,
    best_state: np.ndarray,
    best_eval: OptimizationEvaluation,
) -> tuple[np.ndarray, OptimizationEvaluation, bool]:
    if candidate_eval.loss_total > best_eval.loss_total + 1e-12:
        return best_state, best_eval, False
    return candidate_state.copy(), candidate_eval, True


def _should_stop_early(
    *,
    current_eval: OptimizationEvaluation,
    epoch: int,
    log_prefix: str,
    early_stop_loss: float | None,
    early_stop_patience: int,
    epochs_without_improvement: int,
) -> bool:
    if early_stop_loss is not None and current_eval.loss_total <= early_stop_loss:
        log.info(
            "%s early stop on loss threshold at epoch=%d (loss=%.6f <= %.6f)",
            log_prefix,
            epoch,
            current_eval.loss_total,
            early_stop_loss,
        )
        return True
    if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
        log.info(
            "%s early stop on patience at epoch=%d (%d epochs without improvement)",
            log_prefix,
            epoch,
            epochs_without_improvement,
        )
        return True
    return False


def _maybe_shrink_step_size(
    *,
    current_step_size: float,
    min_step_size: float,
    shrink_factor: float,
    plateau_shrink_patience: int,
    epochs_without_improvement: int,
    log_prefix: str,
    epoch: int,
) -> tuple[float, bool]:
    """Optionally reduce the optimizer step size after a plateau.

    Returns the possibly updated step size plus a boolean indicating whether a
    shrink event was applied on this epoch.
    """
    if plateau_shrink_patience <= 0 or shrink_factor >= 1.0:
        return current_step_size, False
    if epochs_without_improvement < plateau_shrink_patience:
        return current_step_size, False
    next_step_size = max(min_step_size, current_step_size * shrink_factor)
    if next_step_size >= current_step_size - 1e-15:
        return current_step_size, False
    log.info(
        "%s shrinking step size at epoch=%d from %.6f to %.6f after %d non-improving epoch(s)",
        log_prefix,
        epoch,
        current_step_size,
        next_step_size,
        epochs_without_improvement,
    )
    return next_step_size, True


def _estimate_spsa_gradient(
    *,
    current_state: np.ndarray,
    fd_step: float,
    max_eps: float,
    rng: np.random.Generator,
    evaluate_fn: EvaluateFn,
) -> np.ndarray:
    perturb = rng.choice(np.array([-1.0, 1.0], dtype=float), size=current_state.shape[0])
    plus_state = np.clip(current_state + fd_step * perturb, -max_eps, max_eps)
    minus_state = np.clip(current_state - fd_step * perturb, -max_eps, max_eps)
    plus_eval = evaluate_fn(plus_state)
    minus_eval = evaluate_fn(minus_state)
    scale = max(2.0 * fd_step, 1e-9)
    return ((plus_eval.loss_total - minus_eval.loss_total) / scale) * perturb


def _estimate_central_gradient(
    *,
    current_state: np.ndarray,
    fd_step: float,
    max_eps: float,
    evaluate_fn: EvaluateFn,
) -> np.ndarray:
    grad = np.zeros_like(current_state, dtype=float)
    for dim in range(current_state.shape[0]):
        plus_state = current_state.copy()
        minus_state = current_state.copy()
        plus_state[dim] = min(max_eps, plus_state[dim] + fd_step)
        minus_state[dim] = max(-max_eps, minus_state[dim] - fd_step)
        plus_eval = evaluate_fn(plus_state)
        minus_eval = evaluate_fn(minus_state)
        grad[dim] = (plus_eval.loss_total - minus_eval.loss_total) / max(2.0 * fd_step, 1e-9)
    return grad


def _run_spsa_sign_descent(
    *,
    state_dim: int,
    max_epochs: int,
    max_eps: float,
    step_size: float,
    fd_step: float,
    random_seed: int,
    log_prefix: str,
    evaluate_fn: EvaluateFn,
    epoch_callback: EpochCallback,
    early_stop_loss: float | None,
    early_stop_patience: int,
    plateau_shrink_patience: int,
    plateau_shrink_factor: float,
    min_step_size: float,
    restore_best_on_shrink: bool,
) -> OptimizationResult:
    rng = np.random.default_rng(random_seed)
    current_state = np.zeros(state_dim, dtype=float)
    current_step_size = float(step_size)
    best_eval = evaluate_fn(current_state)
    best_state = current_state.copy()
    epoch_callback(0, best_eval, best_eval)
    executed_epochs = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        log.info("%s epoch=%d/%d", log_prefix, epoch, max_epochs)
        grad = _estimate_spsa_gradient(
            current_state=current_state,
            fd_step=fd_step,
            max_eps=max_eps,
            rng=rng,
            evaluate_fn=evaluate_fn,
        )
        current_state = np.clip(current_state - current_step_size * np.sign(grad), -max_eps, max_eps)
        current_eval = evaluate_fn(current_state)
        executed_epochs = epoch
        best_state, best_eval, improved = _record_if_improved(
            candidate_state=current_state,
            candidate_eval=current_eval,
            best_state=best_state,
            best_eval=best_eval,
        )
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        epoch_callback(epoch, current_eval, best_eval)
        current_step_size, shrunk = _maybe_shrink_step_size(
            current_step_size=current_step_size,
            min_step_size=min_step_size,
            shrink_factor=plateau_shrink_factor,
            plateau_shrink_patience=plateau_shrink_patience,
            epochs_without_improvement=epochs_without_improvement,
            log_prefix=log_prefix,
            epoch=epoch,
        )
        if shrunk:
            epochs_without_improvement = 0
            if restore_best_on_shrink:
                current_state = best_state.copy()
        if _should_stop_early(
            current_eval=current_eval,
            epoch=epoch,
            log_prefix=log_prefix,
            early_stop_loss=early_stop_loss,
            early_stop_patience=early_stop_patience,
            epochs_without_improvement=epochs_without_improvement,
        ):
            break
    return OptimizationResult(best_state=best_state, best_evaluation=best_eval, executed_epochs=executed_epochs)


def _run_finite_diff_adam(
    *,
    state_dim: int,
    max_epochs: int,
    max_eps: float,
    step_size: float,
    fd_step: float,
    log_prefix: str,
    evaluate_fn: EvaluateFn,
    epoch_callback: EpochCallback,
    early_stop_loss: float | None,
    early_stop_patience: int,
    beta1: float,
    beta2: float,
    eps: float,
    plateau_shrink_patience: int,
    plateau_shrink_factor: float,
    min_step_size: float,
    restore_best_on_shrink: bool,
    reset_moments_on_shrink: bool,
) -> OptimizationResult:
    current_state = np.zeros(state_dim, dtype=float)
    current_step_size = float(step_size)
    best_eval = evaluate_fn(current_state)
    best_state = current_state.copy()
    first_moment = np.zeros(state_dim, dtype=float)
    second_moment = np.zeros(state_dim, dtype=float)
    epoch_callback(0, best_eval, best_eval)
    executed_epochs = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        log.info("%s epoch=%d/%d", log_prefix, epoch, max_epochs)
        grad = _estimate_central_gradient(
            current_state=current_state,
            fd_step=fd_step,
            max_eps=max_eps,
            evaluate_fn=evaluate_fn,
        )
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(grad)
        first_hat = first_moment / max(1.0 - beta1**epoch, 1e-12)
        second_hat = second_moment / max(1.0 - beta2**epoch, 1e-12)
        current_state = np.clip(
            current_state - current_step_size * first_hat / (np.sqrt(second_hat) + eps),
            -max_eps,
            max_eps,
        )
        current_eval = evaluate_fn(current_state)
        executed_epochs = epoch
        best_state, best_eval, improved = _record_if_improved(
            candidate_state=current_state,
            candidate_eval=current_eval,
            best_state=best_state,
            best_eval=best_eval,
        )
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        epoch_callback(epoch, current_eval, best_eval)
        current_step_size, shrunk = _maybe_shrink_step_size(
            current_step_size=current_step_size,
            min_step_size=min_step_size,
            shrink_factor=plateau_shrink_factor,
            plateau_shrink_patience=plateau_shrink_patience,
            epochs_without_improvement=epochs_without_improvement,
            log_prefix=log_prefix,
            epoch=epoch,
        )
        if shrunk:
            epochs_without_improvement = 0
            if restore_best_on_shrink:
                current_state = best_state.copy()
            if reset_moments_on_shrink:
                first_moment.fill(0.0)
                second_moment.fill(0.0)
        if _should_stop_early(
            current_eval=current_eval,
            epoch=epoch,
            log_prefix=log_prefix,
            early_stop_loss=early_stop_loss,
            early_stop_patience=early_stop_patience,
            epochs_without_improvement=epochs_without_improvement,
        ):
            break
    return OptimizationResult(best_state=best_state, best_evaluation=best_eval, executed_epochs=executed_epochs)


def run_optimizer(
    *,
    optimizer_name: str,
    state_dim: int,
    max_epochs: int,
    max_eps: float,
    step_size: float,
    fd_step: float,
    random_seed: int,
    log_prefix: str,
    evaluate_fn: EvaluateFn,
    epoch_callback: EpochCallback,
    early_stop_loss: float | None = None,
    early_stop_patience: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    plateau_shrink_patience: int = 0,
    plateau_shrink_factor: float = 1.0,
    min_step_size: float = 0.0,
    restore_best_on_shrink: bool = False,
    reset_moments_on_shrink: bool = False,
) -> OptimizationResult:
    """Run one black-box optimizer.

    Args:
        optimizer_name: Supported values are ``spsa_sign``, ``finite_diff_adam``,
            and engine-managed row-wise ``torch_adam``.
    """
    if optimizer_name == "spsa_sign":
        return _run_spsa_sign_descent(
            state_dim=state_dim,
            max_epochs=max_epochs,
            max_eps=max_eps,
            step_size=step_size,
            fd_step=fd_step,
            random_seed=random_seed,
            log_prefix=log_prefix,
            evaluate_fn=evaluate_fn,
            epoch_callback=epoch_callback,
            early_stop_loss=early_stop_loss,
            early_stop_patience=early_stop_patience,
            plateau_shrink_patience=plateau_shrink_patience,
            plateau_shrink_factor=plateau_shrink_factor,
            min_step_size=min_step_size,
            restore_best_on_shrink=restore_best_on_shrink,
        )
    if optimizer_name == "finite_diff_adam":
        return _run_finite_diff_adam(
            state_dim=state_dim,
            max_epochs=max_epochs,
            max_eps=max_eps,
            step_size=step_size,
            fd_step=fd_step,
            log_prefix=log_prefix,
            evaluate_fn=evaluate_fn,
            epoch_callback=epoch_callback,
            early_stop_loss=early_stop_loss,
            early_stop_patience=early_stop_patience,
            beta1=adam_beta1,
            beta2=adam_beta2,
            eps=adam_eps,
            plateau_shrink_patience=plateau_shrink_patience,
            plateau_shrink_factor=plateau_shrink_factor,
            min_step_size=min_step_size,
            restore_best_on_shrink=restore_best_on_shrink,
            reset_moments_on_shrink=reset_moments_on_shrink,
        )
    if optimizer_name == "torch_adam":
        raise ValueError(
            "optimizer_name='torch_adam' is handled directly by the Family 3 row-wise engine "
            "and should not be routed through run_optimizer()."
        )
    raise ValueError(
        "optimizer_name must be one of {'spsa_sign', 'finite_diff_adam', 'torch_adam'}, "
        f"got {optimizer_name!r}."
    )
