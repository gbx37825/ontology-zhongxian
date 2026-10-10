"""Adaptive, model-independent steady initialization for bundled SuperLink."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math
import time
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class WarmupConfig:
    initial_dt: float = 30.0
    min_dt: float = 1.0
    max_dt: float = 3600.0
    growth_factor: float = 2.0
    shrink_factor: float = 0.5
    deterioration_factor: float = 1.5
    check_interval: int = 100
    consecutive_checks: int = 3
    max_steps: int = 50000
    max_wall_seconds: float | None = None
    verification_steps: int = 1000
    head_tolerance_m: float = 0.00001
    flow_tolerance_m3s: float = 0.01
    storage_tolerance_m3s: float = 0.01
    balance_tolerance_m3s: float = 0.001
    internal_method: str = "lsq"
    num_iter: int = 1

    def __post_init__(self):
        positive = ("initial_dt", "min_dt", "max_dt", "growth_factor",
                    "shrink_factor", "deterioration_factor", "head_tolerance_m",
                    "flow_tolerance_m3s", "storage_tolerance_m3s", "balance_tolerance_m3s")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("check_interval", "consecutive_checks", "max_steps", "verification_steps", "num_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "max_steps" else 1):
                raise ValueError(f"{name} has an invalid integer value")
        if not self.min_dt <= self.initial_dt <= self.max_dt:
            raise ValueError("Require min_dt <= initial_dt <= max_dt")
        if self.growth_factor <= 1 or not 0 < self.shrink_factor < 1 or self.deterioration_factor <= 1:
            raise ValueError("Invalid adaptive step factors")
        if self.max_wall_seconds is not None:
            if isinstance(self.max_wall_seconds, bool) or not isinstance(self.max_wall_seconds, (int, float)) or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
                raise ValueError("max_wall_seconds must be null or finite and positive")
        if self.internal_method not in {"b", "f", "lsq"}:
            raise ValueError("internal_method must be b, f or lsq")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WarmupConfig":
        if not isinstance(values, Mapping):
            raise ValueError("Warmup configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown warmup settings: {sorted(unknown)}")
        return cls(**values)


def _snapshot(model):
    # Include derived numeric arrays, not only public hydraulic states.
    state = {name: value.copy() if isinstance(value, np.ndarray) else value
             for name, value in vars(model).items()
             if isinstance(value, (np.ndarray, int, float, bool, np.number))}
    state["states"] = deepcopy(model.states)
    return state


def _restore(model, state):
    for name, value in state.items():
        setattr(model, name, deepcopy(value) if name == "states" else
                value.copy() if isinstance(value, np.ndarray) else value)


def _finite(model):
    arrays = [model.H_j, model.h_Ik, model.Q_ik, model.Q_uk, model.Q_dk]
    arrays += [getattr(model, name) for name in ("Q_o", "Q_w", "Q_p")]
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("Non-finite hydraulic state")
    if (model.H_j < model._z_inv_j - 1e-8).any() or (model.h_Ik < -1e-8).any():
        raise FloatingPointError("Negative hydraulic depth")


def _flows(engine):
    m = engine.model
    arrays = (m.Q_ik, m.Q_uk, m.Q_dk, m.Q_o, m.Q_w, m.Q_p)
    return np.concatenate(arrays)


def _metric(value, location, tolerance, unit):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite convergence metric")
    ratio = value / tolerance
    capped = not math.isfinite(ratio)
    return {"value": value, "threshold": tolerance, "excess": max(value - tolerance, 0.0),
            "ratio": float(np.finfo(float).max) if capped else ratio,
            "ratio_is_lower_bound": capped, "location": location, "unit": unit}


def _worst(values, locations):
    if not len(values):
        return 0.0, None
    index = int(np.argmax(np.abs(values)))
    return float(abs(values[index])), locations[index]


def _measure(engine, previous, q_in, dt, reference_dt, config, labels):
    m = engine.model
    old_H, old_h, old_Q = previous
    head_changes = np.r_[m.H_j - old_H, m.h_Ik - old_h]
    flow_changes = _flows(engine) - old_Q
    head, head_location = _worst(head_changes, labels["heads"])
    flow, flow_location = _worst(flow_changes, labels["flows"])

    net_sj = q_in.copy()
    np.add.at(net_sj, m._J_uk, -m.Q_uk)
    np.add.at(net_sj, m._J_dk, m.Q_dk)
    for up, down, flows in ((m._J_uo, m._J_do, m.Q_o),
                            (m._J_uw, m._J_dw, m.Q_w),
                            (m._J_up, m._J_dp, m.Q_p)):
        np.add.at(net_sj, up, -flows)
        np.add.at(net_sj, down, flows)
    internal_inflow = m._Q_0Ik
    net_j = np.zeros(m.NIk) if internal_inflow is None else np.array(internal_inflow, copy=True)
    np.add.at(net_j, m._Ik, -m.Q_ik)
    np.add.at(net_j, m._Ip1k, m.Q_ik)
    np.add.at(net_j, m._I_1k, m.Q_uk)
    np.add.at(net_j, m._I_Np1k, -m.Q_dk)
    storage_sj = m._A_sj * (m.H_j - old_H) / dt
    storage_j = m._E_Ik * (m.h_Ik - old_h)
    residuals = np.r_[(net_sj - storage_sj)[~m.bc], net_j - storage_j]
    storage = np.r_[storage_sj[~m.bc], storage_j]
    net = np.r_[net_sj[~m.bc], net_j]
    names = labels["free_nodes"]
    residual, residual_location = _worst(residuals, names)
    imbalance, imbalance_location = _worst(net, names)
    _, storage_location = _worst(storage, names)
    # L1 storage avoids cancellation between a filling pool and a draining pool.
    metrics = {
        "head_change_reference_m": _metric(head * reference_dt / dt, head_location, config.head_tolerance_m, "m"),
        "flow_change_reference_m3s": _metric(flow * reference_dt / dt, flow_location, config.flow_tolerance_m3s, "m3/s"),
        "storage_abs_sum_m3s": _metric(np.abs(storage).sum(), storage_location, config.storage_tolerance_m3s, "m3/s"),
        "steady_flow_imbalance_m3s": _metric(imbalance, imbalance_location, config.flow_tolerance_m3s, "m3/s"),
        "max_continuity_residual_m3s": _metric(residual, residual_location, config.balance_tolerance_m3s, "m3/s"),
        "global_continuity_residual_m3s": _metric(abs(residuals.sum()), "network", config.balance_tolerance_m3s, "m3/s"),
    }
    return {"metrics": metrics, "head_change_actual_m": head, "flow_change_actual_m3s": flow,
            "storage_change_m3s": float(storage.sum()),
            "net_external_inflow_m3s": float(q_in.sum() + (0 if internal_inflow is None else np.sum(internal_inflow))),
            "boundary_outflow_m3s": float((net_sj - storage_sj)[m.bc].sum()),
            "location_values": (net, storage, residuals)}


def _peak(current, sample):
    result = deepcopy(current)
    for name, metric in sample["metrics"].items():
        if name not in result or metric["value"] > result[name]["value"]:
            result[name] = metric.copy()
    return result


def _score(metrics):
    return max((metric["ratio"] for metric in metrics.values()), default=math.inf)


def adaptive_warmup(engine, config: WarmupConfig, *, boundary_heads=None, node_inflows=None,
                    gate_openings=None, pump_controls=None,
                    progress: Callable[[dict], None] | None = None) -> dict:
    """Keep input conditions fixed; return diagnostics even when convergence fails.

    max_steps includes attempted/rejected steps and native verification. The
    orifice guard uses engine.dt during pseudo-time, then its original method
    is restored. No model names, observational targets or gate budgets enter.
    """
    m = engine.model
    reference_dt = float(engine.dt)
    if not math.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("The transient calculation timestep must be finite and positive")
    h = engine._input_array(boundary_heads, engine.node_names, np.where(m.bc, m.H_j, np.nan))
    q = engine._input_array(node_inflows, engine.node_names, 0.0)
    u = engine._input_array(gate_openings, engine.gate_names, 1.0) if m.n_o else None
    p = engine._input_array(pump_controls, engine.pump_names, 1.0) if m.n_p else None
    if not np.isfinite(h[m.bc]).all() or not np.isfinite(q).all():
        raise ValueError("Boundary heads and inflows must be finite")
    for control in (u, p):
        if control is not None and (not np.isfinite(control).all() or (control < 0).any() or (control > 1).any()):
            raise ValueError("Device controls must be finite and between zero and one")
    _finite(m)
    m._method = config.internal_method
    internal_names = [f"{engine.channel_names[int(k)]}/junction:{i}"
                      for i, k in enumerate(m._kI)]
    link_names = [f"{engine.channel_names[int(k)]}/link:{i}" for i, k in enumerate(m._ki)]
    labels = {"heads": list(engine.node_names) + internal_names,
              "free_nodes": [name for name, fixed in zip(engine.node_names, m.bc) if not fixed] + internal_names,
              "flows": link_names + [f"{name}/upstream" for name in engine.channel_names]
              + [f"{name}/downstream" for name in engine.channel_names] + list(engine.gate_names)
              + [f"weir:{i}" for i in range(m.n_w)] + list(engine.pump_names)}
    started = time.perf_counter()
    original_time, original_iterations = m.t, m.iter_count
    original_orifice = m.solve_orifice_flows
    had_override = "solve_orifice_flows" in vars(m)
    history, errors = [], []
    attempts = accepted = rejected = verification_run = 0
    pseudo_seconds = 0.0
    last_sample = None
    final_metrics = {}
    stop_reason = "step_budget"
    verify_count = min(config.verification_steps, config.max_steps)
    adaptive_budget = config.max_steps - verify_count
    dt, previous_score, stable_checks = config.initial_dt, None, 0
    growth_wait = 0

    def timed_out():
        return config.max_wall_seconds is not None and time.perf_counter() - started >= config.max_wall_seconds

    def advance(step_dt):
        before = (m.H_j.copy(), m.h_Ik.copy(), _flows(engine).copy())
        try:
            m.step(H_bc=h, Q_in=q, u_o=u, u_p=p, dt=step_dt, num_iter=config.num_iter)
        except ValueError as exc:
            # SciPy's check_finite uses ValueError, not FloatingPointError.
            if "array must not contain infs or NaNs" not in str(exc):
                raise
            raise FloatingPointError("Non-finite solver matrix") from exc
        _finite(m)
        return _measure(engine, before, q, step_dt, reference_dt, config, labels)

    def emit(phase, step_dt, action, metrics, error=None):
        row = {"phase": phase, "attempted_steps": attempts, "accepted_steps": accepted,
               "rejected_steps": rejected, "dt_s": float(step_dt), "action": action,
               "pseudo_seconds": pseudo_seconds, "wall_seconds": time.perf_counter() - started,
               "score": None if not metrics else _score(metrics), "error": error}
        for name, metric in metrics.items():
            row[name] = metric["value"]
        history.append(row)
        if progress:
            progress(row.copy())

    def restore_orifice():
        if had_override:
            m.solve_orifice_flows = original_orifice
        else:
            vars(m).pop("solve_orifice_flows", None)

    try:
        m.solve_orifice_flows = lambda dt, u=None: original_orifice(dt=reference_dt, u=u)
        while attempts < adaptive_budget and not timed_out():
            checkpoint = _snapshot(m)
            previous_sample = last_sample
            count, peaks, error = 0, {}, None
            for _ in range(min(config.check_interval, adaptive_budget - attempts)):
                if timed_out():
                    break
                attempts += 1
                count += 1
                try:
                    sample = advance(dt)
                    peaks = _peak(peaks, sample)
                    last_sample = sample
                except (FloatingPointError, np.linalg.LinAlgError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(error)
                    break
            score = _score(peaks)
            worsening = previous_score is not None and score > previous_score * config.deterioration_factor and score > 1
            if error or (worsening and dt > config.min_dt):
                _restore(m, checkpoint)
                last_sample = previous_sample
                rejected += count
                emit("adaptive", dt, "rollback_shrink", peaks, error)
                stable_checks = 0
                if dt <= config.min_dt:
                    stop_reason = "numerical_failure_at_min_dt"
                    break
                dt = max(config.min_dt, dt * config.shrink_factor)
                previous_score = None
                growth_wait = config.consecutive_checks
                continue
            accepted += count
            pseudo_seconds += count * dt
            stable_checks = stable_checks + 1 if score <= 1 else 0
            if stable_checks >= config.consecutive_checks:
                stop_reason = "candidate_converged"
                emit("adaptive", dt, "verify", peaks)
                break
            next_dt = dt
            if growth_wait > 0:
                growth_wait -= 1
            elif previous_score is None or score <= previous_score * 1.05:
                next_dt = min(config.max_dt, dt * config.growth_factor)
            action = "grow" if next_dt > dt else "hold"
            emit("adaptive", dt, action, peaks)
            previous_score, dt = score, next_dt

        restore_orifice()
        for _ in range(verify_count):
            if timed_out():
                stop_reason = "wall_time_budget"
                break
            checkpoint = _snapshot(m)
            attempts += 1
            try:
                sample = advance(reference_dt)
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                _restore(m, checkpoint)
                rejected += 1
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                stop_reason = "verification_numerical_failure"
                emit("verification", reference_dt, "rollback", final_metrics, error)
                break
            accepted += 1
            verification_run += 1
            last_sample = sample
            final_metrics = _peak(final_metrics, sample)
            if verification_run % config.check_interval == 0 or verification_run == verify_count:
                emit("verification", reference_dt, "check", sample["metrics"])
        if timed_out():
            stop_reason = "wall_time_budget"
    finally:
        restore_orifice()
        m.t, m.iter_count = original_time, original_iterations

    converged = verification_run == config.verification_steps and _score(final_metrics) <= 1
    if converged:
        status = "CONVERGED"
    elif config.max_steps == 0:
        status = "SKIPPED"
        stop_reason = "disabled_by_zero_budget"
    else:
        status = "NOT_CONVERGED_CONTINUING"
    diagnostics = final_metrics or (last_sample["metrics"] if last_sample else {})
    locations = []
    if last_sample:
        net, storage, residuals = last_sample["location_values"]
        locations = [{"object": name, "net_inflow_m3s": float(inflow), "storage_change_m3s": float(change),
                      "continuity_residual_m3s": float(error)}
                     for name, inflow, change, error in zip(labels["free_nodes"], net, storage, residuals)]
    return {"status": status, "converged": converged, "continue_simulation": True,
            "stop_reason": stop_reason, "config": asdict(config),
            "reference_dt_s": reference_dt, "orifice_guard_reference_dt_s": reference_dt,
            "steps_run": attempts, "accepted_steps": accepted, "rejected_steps": rejected,
            "adaptive_steps": attempts - verification_run - (1 if stop_reason == "verification_numerical_failure" else 0),
            "verification_steps_run": verification_run, "verification_steps_requested": config.verification_steps,
            "verification_steps_shortfall": config.verification_steps - verification_run,
            "effective_verification_steps": verify_count, "pseudo_seconds": pseudo_seconds,
            "verification_seconds": verification_run * reference_dt,
            "wall_seconds": time.perf_counter() - started, "method": m._method,
            "native_orifice_method_restored": vars(m).get("solve_orifice_flows") is original_orifice if had_override else "solve_orifice_flows" not in vars(m),
            "metrics": diagnostics, "last_step_metrics": last_sample["metrics"] if last_sample else {},
            "last_storage_change_m3s": last_sample["storage_change_m3s"] if last_sample else None,
            "last_net_external_inflow_m3s": last_sample["net_external_inflow_m3s"] if last_sample else None,
            "last_boundary_outflow_m3s": last_sample["boundary_outflow_m3s"] if last_sample else None,
            "unmet_metrics": [name for name, metric in diagnostics.items() if metric["ratio"] > 1],
            "locations": locations, "history": history,
            "numerical_errors": errors,
            "warning": None if converged else "Initial state was not certified steady; transient results retain this warning."}
