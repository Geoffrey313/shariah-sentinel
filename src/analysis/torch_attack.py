"""Projected-gradient (PGD) adversarial-evasion attack.

Implements the PGD evasion in numpy, projecting every step back onto the SAC
ratio constraints so candidates stay inside the compliance region. Backs the
Family-3 SAC-projected evasion result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.common.methodology import SAC_MY


@dataclass(frozen=True)
class TorchPGDResult:
    candidate: np.ndarray
    baseline_score: float
    final_score: float
    baseline_loss: float
    final_loss: float
    baseline_ratios: tuple[float, float, float]
    final_ratios: tuple[float, float, float]
    loss_history: tuple[float, ...]
    ratio_history: tuple[tuple[float, float, float], ...]
    raw_history: tuple[tuple[float, float, float, float], ...]
    final_eps: float
    iterations: int
    converged: bool


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _surrogate_constants(base_row: dict[str, float]) -> dict[str, float]:
    return {
        "atq": _finite_float(base_row.get("atq"), default=0.0),
        "revtq": _finite_float(base_row.get("revtq"), default=0.0),
        "sukuk_ratio_t": _finite_float(base_row.get("sukuk_ratio_t"), default=0.0),
        "islamic_cash_ratio_t": _finite_float(base_row.get("islamic_cash_ratio_t"), default=0.0),
        "ratio_center_debt": _finite_float(base_row.get("ratio_center_debt"), default=0.0),
        "ratio_center_cash": _finite_float(base_row.get("ratio_center_cash"), default=0.0),
        "ratio_center_income": _finite_float(base_row.get("ratio_center_income"), default=0.0),
        "ratio_scale_debt": max(_finite_float(base_row.get("ratio_scale_debt"), default=1.0), 1e-6),
        "ratio_scale_cash": max(_finite_float(base_row.get("ratio_scale_cash"), default=1.0), 1e-6),
        "ratio_scale_income": max(_finite_float(base_row.get("ratio_scale_income"), default=1.0), 1e-6),
        "threshold_debt": _finite_float(base_row.get("threshold_debt"), default=SAC_MY.threshold_debt),
        "threshold_cash": _finite_float(base_row.get("threshold_cash"), default=SAC_MY.threshold_cash),
        "threshold_income": _finite_float(base_row.get("threshold_income"), default=SAC_MY.threshold_income),
    }


def _np_softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _np_project(
    x: np.ndarray,
    x0: np.ndarray,
    scale: np.ndarray,
    eps_frac: float,
    const: dict[str, float],
    *,
    enforce_sac: bool = True,
) -> np.ndarray:
    lower = np.maximum(x0 - eps_frac * scale, 0.0)
    upper = np.maximum(x0 + eps_frac * scale, 0.0)
    projected = np.clip(np.asarray(x, dtype=float), lower, upper)

    if enforce_sac:
        atq = max(const["atq"], 1e-9)
        revtq = max(const["revtq"], 1e-9)
        sukuk_keep = max(1.0 - const["sukuk_ratio_t"], 1e-9)
        cash_keep = max(1.0 - const["islamic_cash_ratio_t"], 1e-9)

        debt_adj = projected[0] * sukuk_keep + projected[1]
        debt_cap = const["threshold_debt"] * atq
        debt_scale = min(debt_cap / max(debt_adj, 1e-9), 1.0)
        projected = np.array(
            [
                projected[0] * debt_scale,
                projected[1] * debt_scale,
                projected[2],
                projected[3],
            ],
            dtype=float,
        )

        cash_cap = const["threshold_cash"] * atq / cash_keep
        income_cap = const["threshold_income"] * revtq
        projected[2] = np.clip(projected[2], 0.0, cash_cap)
        projected[3] = np.clip(projected[3], 0.0, income_cap)
    return projected


def _np_ratio_vector(x: np.ndarray, const: dict[str, float]) -> np.ndarray:
    atq = max(const["atq"], 1e-9)
    revtq = max(const["revtq"], 1e-9)
    sukuk_keep = max(1.0 - const["sukuk_ratio_t"], 1e-9)
    cash_keep = max(1.0 - const["islamic_cash_ratio_t"], 1e-9)
    return np.array(
        [
            (x[0] * sukuk_keep + x[1]) / atq,
            (x[2] * cash_keep) / atq,
            x[3] / revtq,
        ],
        dtype=float,
    )


def _np_surrogate_loss(
    x: np.ndarray,
    x0: np.ndarray,
    scale: np.ndarray,
    const: dict[str, float],
    *,
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
) -> float:
    ratios = _np_ratio_vector(x, const)
    if surrogate_mode == "threshold":
        thresholds = np.array(
            [const["threshold_debt"], const["threshold_cash"], const["threshold_income"]],
            dtype=float,
        )
        smooth_excess = _np_softplus((ratios - thresholds) / temperature).sum()
        prox = penalty_lambda * np.square((x - x0) / scale).sum()
        return float(smooth_excess + prox)
    if surrogate_mode == "softmax_general":
        centers = np.array(
            [const["ratio_center_debt"], const["ratio_center_cash"], const["ratio_center_income"]],
            dtype=float,
        )
        scales = np.array(
            [const["ratio_scale_debt"], const["ratio_scale_cash"], const["ratio_scale_income"]],
            dtype=float,
        )
        standardized = np.sqrt(np.square((ratios - centers) / scales) + 1e-8)
        logits = standardized / temperature
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.clip(weights.sum(), 1e-9, np.inf)
        softmax_anomaly = float(np.sum(weights * standardized))
        thresholds = np.array(
            [const["threshold_debt"], const["threshold_cash"], const["threshold_income"]],
            dtype=float,
        )
        sc_penalty = penalty_lambda * _np_softplus((ratios - thresholds) / temperature).sum() if enforce_sac else 0.0
        prox = penalty_prox_lambda * np.square((x - x0) / scale).sum()
        return float(softmax_anomaly + sc_penalty + prox)
    raise ValueError(f"Unknown surrogate_mode: {surrogate_mode}")


def _np_numeric_grad(
    x: np.ndarray,
    *,
    x0: np.ndarray,
    scale: np.ndarray,
    eps_frac: float,
    const: dict[str, float],
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
) -> tuple[np.ndarray, float]:
    projected = _np_project(x, x0, scale, eps_frac, const, enforce_sac=enforce_sac)
    base_loss = _np_surrogate_loss(
        projected,
        x0,
        scale,
        const,
        penalty_lambda=penalty_lambda,
        temperature=temperature,
        surrogate_mode=surrogate_mode,
        penalty_prox_lambda=penalty_prox_lambda,
        enforce_sac=enforce_sac,
    )
    grad = np.zeros_like(projected)
    for i in range(projected.size):
        step = max(1e-5 * scale[i], 1e-6)
        x_plus = projected.copy()
        x_minus = projected.copy()
        x_plus[i] += step
        x_minus[i] -= step
        proj_plus = _np_project(x_plus, x0, scale, eps_frac, const, enforce_sac=enforce_sac)
        proj_minus = _np_project(x_minus, x0, scale, eps_frac, const, enforce_sac=enforce_sac)
        f_plus = _np_surrogate_loss(
            proj_plus,
            x0,
            scale,
            const,
            penalty_lambda=penalty_lambda,
            temperature=temperature,
            surrogate_mode=surrogate_mode,
            penalty_prox_lambda=penalty_prox_lambda,
            enforce_sac=enforce_sac,
        )
        f_minus = _np_surrogate_loss(
            proj_minus,
            x0,
            scale,
            const,
            penalty_lambda=penalty_lambda,
            temperature=temperature,
            surrogate_mode=surrogate_mode,
            penalty_prox_lambda=penalty_prox_lambda,
            enforce_sac=enforce_sac,
        )
        grad[i] = (f_plus - f_minus) / (2.0 * step)
    return grad, base_loss


def _run_numpy_projected_pgd(
    *,
    x0: np.ndarray,
    base_row: dict[str, float],
    step_size: float,
    eps_grid: tuple[float, ...],
    iterations: int,
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str,
    score_fn: Callable[[np.ndarray], float] | None,
    tolerance: float,
    penalty_prox_lambda: float,
    enforce_sac: bool = True,
) -> TorchPGDResult:
    x0_np = np.asarray(x0, dtype=float)
    scale = np.maximum(np.abs(x0_np), 1.0)
    const = _surrogate_constants(base_row)

    baseline_projected = _np_project(x0_np, x0_np, scale, 0.0, const, enforce_sac=enforce_sac)
    baseline_ratios = tuple(float(v) for v in _np_ratio_vector(baseline_projected, const))
    baseline_loss = _np_surrogate_loss(
        baseline_projected,
        x0_np,
        scale,
        const,
        penalty_lambda=penalty_lambda,
        temperature=temperature,
        surrogate_mode=surrogate_mode,
        penalty_prox_lambda=penalty_prox_lambda,
        enforce_sac=enforce_sac,
    )
    baseline_score = float(score_fn(baseline_projected)) if score_fn is not None else float("nan")

    best_candidate = baseline_projected.copy()
    best_score = baseline_score
    best_loss = baseline_loss
    best_ratios = baseline_ratios
    best_eps = 0.0
    best_iter = 0
    converged = False
    best_loss_history: tuple[float, ...] = ()
    best_ratio_history: tuple[tuple[float, float, float], ...] = ()
    best_raw_history: tuple[tuple[float, float, float, float], ...] = ()

    for eps_frac in eps_grid:
        x = baseline_projected.copy()
        local_best_loss = float("inf")
        local_best_score = best_score
        local_best_candidate = best_candidate.copy()
        local_loss_history: list[float] = []
        local_ratio_history: list[tuple[float, float, float]] = []
        local_raw_history: list[tuple[float, float, float, float]] = []
        stagnant = 0

        for iteration in range(1, iterations + 1):
            grad, current_loss = _np_numeric_grad(
                x,
                x0=x0_np,
                scale=scale,
                eps_frac=float(eps_frac),
                const=const,
                penalty_lambda=penalty_lambda,
                temperature=temperature,
                surrogate_mode=surrogate_mode,
                penalty_prox_lambda=penalty_prox_lambda,
                enforce_sac=enforce_sac,
            )
            direction = -scale * grad
            if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-12:
                break

            alpha = float(step_size)
            accepted = x.copy()
            accepted_loss = current_loss
            improved = False
            for _ in range(10):
                trial = _np_project(x + alpha * direction, x0_np, scale, float(eps_frac), const, enforce_sac=enforce_sac)
                trial_loss = _np_surrogate_loss(
                    trial,
                    x0_np,
                    scale,
                    const,
                    penalty_lambda=penalty_lambda,
                    temperature=temperature,
                    surrogate_mode=surrogate_mode,
                    penalty_prox_lambda=penalty_prox_lambda,
                    enforce_sac=enforce_sac,
                )
                if trial_loss + tolerance < accepted_loss:
                    accepted = trial
                    accepted_loss = trial_loss
                    improved = True
                    break
                alpha *= 0.5

            x = accepted
            local_loss_history.append(float(accepted_loss))
            local_ratio_history.append(tuple(float(v) for v in _np_ratio_vector(x, const)))
            local_raw_history.append(tuple(float(v) for v in x))

            if accepted_loss + tolerance < local_best_loss:
                local_best_loss = accepted_loss
                local_best_candidate = x.copy()
                stagnant = 0
            else:
                stagnant += 1
            if not improved or stagnant >= 5:
                break

        if score_fn is not None:
            local_best_score = float(score_fn(local_best_candidate))
        if local_best_loss + tolerance < best_loss:
            best_loss = local_best_loss
            best_candidate = local_best_candidate.copy()
            best_ratios = tuple(float(v) for v in _np_ratio_vector(local_best_candidate, const))
            best_loss_history = tuple(local_loss_history)
            best_ratio_history = tuple(local_ratio_history)
            best_raw_history = tuple(local_raw_history)
            best_eps = float(eps_frac)
            best_iter = iteration
            if score_fn is not None:
                best_score = local_best_score
        elif score_fn is not None and local_best_score > best_score + tolerance:
            best_score = local_best_score
            best_candidate = local_best_candidate.copy()
            best_ratios = tuple(float(v) for v in _np_ratio_vector(local_best_candidate, const))
            best_loss_history = tuple(local_loss_history)
            best_ratio_history = tuple(local_ratio_history)
            best_raw_history = tuple(local_raw_history)
            best_eps = float(eps_frac)
            best_iter = iteration
        if score_fn is not None and best_score > baseline_score + tolerance:
            converged = True

    return TorchPGDResult(
        candidate=np.asarray(best_candidate, dtype=float),
        baseline_score=baseline_score,
        final_score=best_score if score_fn is not None else float("nan"),
        baseline_loss=float(baseline_loss),
        final_loss=float(best_loss),
        baseline_ratios=baseline_ratios,
        final_ratios=best_ratios,
        loss_history=best_loss_history,
        ratio_history=best_ratio_history,
        raw_history=best_raw_history,
        final_eps=best_eps,
        iterations=best_iter,
        converged=converged,
    )


def _torch_project(x, x0, scale, eps_frac, const, torch, *, enforce_sac: bool = True):
    lower = torch.clamp(x0 - eps_frac * scale, min=0.0)
    upper = torch.clamp(x0 + eps_frac * scale, min=0.0)
    projected = torch.maximum(torch.minimum(x, upper), lower)

    if enforce_sac:
        atq = max(const["atq"], 1e-9)
        revtq = max(const["revtq"], 1e-9)
        sukuk_keep = max(1.0 - const["sukuk_ratio_t"], 1e-9)
        cash_keep = max(1.0 - const["islamic_cash_ratio_t"], 1e-9)

        debt_adj = projected[0] * sukuk_keep + projected[1]
        debt_cap = const["threshold_debt"] * atq
        debt_scale = torch.clamp(debt_cap / torch.clamp(debt_adj, min=1e-9), max=1.0)
        projected = torch.stack(
            (
                projected[0] * debt_scale,
                projected[1] * debt_scale,
                projected[2],
                projected[3],
            )
        )

        cash_cap = const["threshold_cash"] * atq / cash_keep
        income_cap = const["threshold_income"] * revtq
        projected = torch.stack(
            (
                projected[0],
                projected[1],
                torch.clamp(projected[2], min=0.0, max=cash_cap),
                torch.clamp(projected[3], min=0.0, max=income_cap),
            )
        )
    return projected


def _torch_ratio_vector(x, const, torch):
    atq = max(const["atq"], 1e-9)
    revtq = max(const["revtq"], 1e-9)
    sukuk_keep = max(1.0 - const["sukuk_ratio_t"], 1e-9)
    cash_keep = max(1.0 - const["islamic_cash_ratio_t"], 1e-9)
    return torch.stack(
        (
            (x[0] * sukuk_keep + x[1]) / atq,
            (x[2] * cash_keep) / atq,
            x[3] / revtq,
        )
    )


def _surrogate_loss_threshold(
    x,
    x0,
    scale,
    const,
    penalty_lambda: float,
    temperature: float,
    torch,
    enforce_sac: bool = True,
):
    ratios = _torch_ratio_vector(x, const, torch)
    thresholds = torch.tensor(
        [const["threshold_debt"], const["threshold_cash"], const["threshold_income"]],
        dtype=x.dtype,
        device=x.device,
    )
    smooth_excess = torch.nn.functional.softplus((ratios - thresholds) / temperature).sum()
    prox = penalty_lambda * torch.square((x - x0) / scale).sum()
    return smooth_excess + prox


def _surrogate_loss_softmax_general(
    x,
    x0,
    scale,
    const,
    penalty_lambda: float,
    temperature: float,
    torch,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
):
    ratios = _torch_ratio_vector(x, const, torch)
    centers = torch.tensor(
        [const["ratio_center_debt"], const["ratio_center_cash"], const["ratio_center_income"]],
        dtype=x.dtype,
        device=x.device,
    )
    scales = torch.tensor(
        [const["ratio_scale_debt"], const["ratio_scale_cash"], const["ratio_scale_income"]],
        dtype=x.dtype,
        device=x.device,
    )
    standardized = torch.sqrt(torch.square((ratios - centers) / scales) + 1e-8)
    weights = torch.softmax(standardized / temperature, dim=0)
    softmax_anomaly = torch.sum(weights * standardized)

    # SC violation penalty: penalise ratios that exceed Shariah thresholds.
    # penalty_lambda acts as λ_sc — the primary feasibility cost.
    if enforce_sac:
        thresholds = torch.tensor(
            [const["threshold_debt"], const["threshold_cash"], const["threshold_income"]],
            dtype=x.dtype,
            device=x.device,
        )
        sc_penalty = penalty_lambda * torch.nn.functional.softplus(
            (ratios - thresholds) / temperature
        ).sum()
    else:
        sc_penalty = torch.zeros((), dtype=x.dtype, device=x.device)

    # Tiny proximity regularisation to break ties and stabilise gradients.
    # penalty_prox_lambda should be << penalty_lambda (e.g. 1e-4).
    prox = penalty_prox_lambda * torch.square((x - x0) / scale).sum()

    return softmax_anomaly + sc_penalty + prox


def _surrogate_loss(
    x,
    x0,
    scale,
    const,
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str,
    torch,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
):
    if surrogate_mode == "threshold":
        return _surrogate_loss_threshold(
            x,
            x0,
            scale,
            const,
            penalty_lambda=penalty_lambda,
            temperature=temperature,
            torch=torch,
            enforce_sac=enforce_sac,
        )
    if surrogate_mode == "softmax_general":
        return _surrogate_loss_softmax_general(
            x,
            x0,
            scale,
            const,
            penalty_lambda=penalty_lambda,
            temperature=temperature,
            torch=torch,
            penalty_prox_lambda=penalty_prox_lambda,
            enforce_sac=enforce_sac,
        )
    raise ValueError(f"Unknown surrogate_mode: {surrogate_mode}")


def _projected_surrogate_eval(
    x,
    *,
    x0,
    scale,
    eps_frac: float,
    const,
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str,
    torch,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
):
    projected = _torch_project(x, x0, scale, eps_frac, const, torch, enforce_sac=enforce_sac)
    loss = _surrogate_loss(
        projected,
        x0,
        scale,
        const,
        penalty_lambda=penalty_lambda,
        temperature=temperature,
        surrogate_mode=surrogate_mode,
        torch=torch,
        penalty_prox_lambda=penalty_prox_lambda,
        enforce_sac=enforce_sac,
    )
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    return projected, loss, grad


def _wolfe_projected_step(
    x,
    *,
    x0,
    scale,
    eps_frac: float,
    const,
    step_size: float,
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str,
    torch,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_backtracks: int = 10,
):
    x_current = x.detach().clone().requires_grad_(True)
    projected, loss, grad = _projected_surrogate_eval(
        x_current,
        x0=x0,
        scale=scale,
        eps_frac=eps_frac,
        const=const,
        penalty_lambda=penalty_lambda,
        temperature=temperature,
        surrogate_mode=surrogate_mode,
        torch=torch,
        penalty_prox_lambda=penalty_prox_lambda,
        enforce_sac=enforce_sac,
    )
    direction = -scale * grad
    dir_deriv0 = float(torch.dot(grad.reshape(-1), direction.reshape(-1)).detach().cpu())
    if not np.isfinite(dir_deriv0) or dir_deriv0 >= 0.0:
        # Fallback to steepest feasible sign step if the gradient direction is degenerate.
        direction = -scale * torch.sign(grad)
        dir_deriv0 = float(torch.dot(grad.reshape(-1), direction.reshape(-1)).detach().cpu())

    current_loss = float(loss.detach().cpu())
    accepted_x = projected.detach().clone()
    accepted_loss = current_loss
    accepted_alpha = 0.0
    accepted_grad = grad.detach().clone()
    best_x = projected.detach().clone()
    best_loss = current_loss

    alpha = float(step_size)
    for _ in range(max_backtracks):
        trial_var = (x_current.detach() + alpha * direction).requires_grad_(True)
        trial_projected, trial_loss_t, trial_grad = _projected_surrogate_eval(
            trial_var,
            x0=x0,
            scale=scale,
            eps_frac=eps_frac,
            const=const,
            penalty_lambda=penalty_lambda,
            temperature=temperature,
            surrogate_mode=surrogate_mode,
            torch=torch,
            penalty_prox_lambda=penalty_prox_lambda,
            enforce_sac=enforce_sac,
        )
        trial_loss = float(trial_loss_t.detach().cpu())
        if trial_loss < best_loss:
            best_loss = trial_loss
            best_x = trial_projected.detach().clone()

        armijo_ok = trial_loss <= current_loss + c1 * alpha * dir_deriv0
        dir_deriv_trial = float(torch.dot(trial_grad.reshape(-1), direction.reshape(-1)).detach().cpu())
        curvature_ok = dir_deriv_trial >= c2 * dir_deriv0
        if armijo_ok and curvature_ok:
            accepted_x = trial_projected.detach().clone()
            accepted_loss = trial_loss
            accepted_alpha = alpha
            accepted_grad = trial_grad.detach().clone()
            break
        alpha *= 0.5

    if accepted_alpha == 0.0 and best_loss + 1e-12 < current_loss:
        accepted_x = best_x
        accepted_loss = best_loss

    return accepted_x, accepted_loss, accepted_alpha, accepted_grad.detach().clone()


def run_torch_projected_pgd(
    *,
    x0: np.ndarray,
    base_row: dict[str, float],
    step_size: float,
    eps_grid: tuple[float, ...],
    iterations: int,
    penalty_lambda: float,
    temperature: float,
    surrogate_mode: str = "threshold",
    score_fn: Callable[[np.ndarray], float] | None = None,
    tolerance: float,
    penalty_prox_lambda: float = 0.0,
    enforce_sac: bool = True,
) -> TorchPGDResult:
    try:
        import torch
    except Exception:
        return _run_numpy_projected_pgd(
            x0=x0,
            base_row=base_row,
            step_size=step_size,
            eps_grid=eps_grid,
            iterations=iterations,
            penalty_lambda=penalty_lambda,
            temperature=temperature,
            surrogate_mode=surrogate_mode,
            score_fn=score_fn,
            tolerance=tolerance,
            penalty_prox_lambda=penalty_prox_lambda,
            enforce_sac=enforce_sac,
        )

    dtype = torch.float64
    device = torch.device("cpu")
    x0_t = torch.tensor(np.asarray(x0, dtype=float), dtype=dtype, device=device)
    scale_t = torch.tensor(np.maximum(np.abs(x0), 1.0), dtype=dtype, device=device)
    const = _surrogate_constants(base_row)
    
    def _to_numpy(tensor) -> np.ndarray:
        return np.asarray(tensor.detach().cpu().tolist(), dtype=float)

    def _ratios_from_numpy(candidate: np.ndarray) -> tuple[float, float, float]:
        tensor = torch.tensor(np.asarray(candidate, dtype=float), dtype=dtype, device=device)
        ratios = _torch_ratio_vector(tensor, const, torch)
        return tuple(float(v) for v in _to_numpy(ratios))

    baseline_projected = _torch_project(x0_t, x0_t, scale_t, 0.0, const, torch, enforce_sac=enforce_sac)
    baseline_ratios_t = _torch_ratio_vector(baseline_projected, const, torch)
    baseline_loss_t = _surrogate_loss(
        baseline_projected,
        x0_t,
        scale_t,
        const,
        penalty_lambda=penalty_lambda,
        temperature=temperature,
        surrogate_mode=surrogate_mode,
        torch=torch,
        penalty_prox_lambda=penalty_prox_lambda,
        enforce_sac=enforce_sac,
    )
    baseline_score = float(score_fn(_to_numpy(baseline_projected))) if score_fn is not None else float("nan")

    best_candidate = _to_numpy(baseline_projected)
    best_score = baseline_score
    best_loss = float(baseline_loss_t.detach().cpu())
    best_ratios = tuple(float(v) for v in _to_numpy(baseline_ratios_t))
    best_eps = 0.0
    best_iter = 0
    converged = False
    best_loss_history: tuple[float, ...] = ()
    best_ratio_history: tuple[tuple[float, float, float], ...] = ()
    best_raw_history: tuple[tuple[float, float, float, float], ...] = ()

    for eps_frac in eps_grid:
        x = baseline_projected.detach().clone()
        local_best_score = best_score
        local_best_loss = float("inf")
        local_best_candidate = best_candidate.copy()
        local_loss_history: list[float] = []
        local_ratio_history: list[tuple[float, float, float]] = []
        local_raw_history: list[tuple[float, float, float, float]] = []
        stagnant = 0

        for iteration in range(1, iterations + 1):
            x, loss_value, accepted_alpha, _ = _wolfe_projected_step(
                x,
                x0=x0_t,
                scale=scale_t,
                eps_frac=float(eps_frac),
                const=const,
                step_size=step_size,
                penalty_lambda=penalty_lambda,
                temperature=temperature,
                surrogate_mode=surrogate_mode,
                torch=torch,
                penalty_prox_lambda=penalty_prox_lambda,
                enforce_sac=enforce_sac,
            )
            local_loss_history.append(loss_value)
            local_ratio_history.append(_ratios_from_numpy(_to_numpy(x)))
            local_raw_history.append(tuple(float(v) for v in _to_numpy(x)))
            if loss_value + tolerance < local_best_loss:
                local_best_loss = loss_value
                local_best_candidate = _to_numpy(x)
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= 5:
                break

        if score_fn is not None:
            local_best_score = float(score_fn(local_best_candidate))
        if local_best_loss + tolerance < best_loss:
            best_loss = local_best_loss
            best_candidate = local_best_candidate.copy()
            best_ratios = _ratios_from_numpy(local_best_candidate)
            best_loss_history = tuple(local_loss_history)
            best_ratio_history = tuple(local_ratio_history)
            best_raw_history = tuple(local_raw_history)
            best_eps = float(eps_frac)
            best_iter = iteration
            if score_fn is not None:
                best_score = local_best_score
        elif score_fn is not None and local_best_score > best_score + tolerance:
            best_score = local_best_score
            best_candidate = local_best_candidate.copy()
            best_ratios = _ratios_from_numpy(local_best_candidate)
            best_loss_history = tuple(local_loss_history)
            best_ratio_history = tuple(local_ratio_history)
            best_raw_history = tuple(local_raw_history)
            best_eps = float(eps_frac)
            best_iter = iteration
        if score_fn is not None and best_score > baseline_score + tolerance:
            converged = True

    return TorchPGDResult(
        candidate=np.asarray(best_candidate, dtype=float),
        baseline_score=baseline_score,
        final_score=best_score if score_fn is not None else float("nan"),
        baseline_loss=float(baseline_loss_t.detach().cpu()),
        final_loss=best_loss,
        baseline_ratios=tuple(float(v) for v in _to_numpy(baseline_ratios_t)),
        final_ratios=best_ratios,
        loss_history=best_loss_history,
        ratio_history=best_ratio_history,
        raw_history=best_raw_history,
        final_eps=best_eps,
        iterations=best_iter,
        converged=converged,
    )
