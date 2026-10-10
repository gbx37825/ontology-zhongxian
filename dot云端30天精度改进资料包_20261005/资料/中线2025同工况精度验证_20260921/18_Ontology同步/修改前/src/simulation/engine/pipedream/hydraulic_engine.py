"""把 Pipedream 数值求解器包装成 HydroOS 可调用的水动力引擎。"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pipedream_solver.superlink import SuperLink

from .wxq_to_superlink import parse_wxq_json

logger = logging.getLogger(__name__)


def _diagnostic_enabled() -> bool:
    return os.getenv("HYDROS_PIPEDREAM_DIAGNOSTIC", "false").lower() in {"1", "true", "yes", "on"}


def _diagnostic_interval() -> int:
    try:
        return max(1, int(os.getenv("HYDROS_PIPEDREAM_DIAGNOSTIC_INTERVAL", "25")))
    except ValueError:
        return 25


def _array_summary(values: Any) -> dict[str, Any]:
    try:
        array = np.asarray(values, dtype=float)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    flat = array.ravel()
    finite = np.isfinite(flat)
    summary: dict[str, Any] = {
        "available": True,
        "shape": list(array.shape),
        "count": int(flat.size),
        "finite": int(finite.sum()),
        "nonfinite": int(flat.size - finite.sum()),
    }
    if finite.any():
        finite_values = flat[finite]
        summary.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "abs_max": float(np.max(np.abs(finite_values))),
            }
        )
    if summary["nonfinite"]:
        summary["nonfinite_indices_sample"] = np.flatnonzero(~finite)[:8].astype(int).tolist()
    return summary


def _model_summary(model: Any) -> dict[str, Any]:
    names = ("H_j", "Q_ik", "Q_uk", "Q_dk", "Q_o", "h_Ik", "A", "b", "O", "W", "P")
    return {
        "t": float(getattr(model, "t", 0.0) or 0.0),
        "iter_count": int(getattr(model, "iter_count", 0) or 0),
        "arrays": {
            name: _array_summary(getattr(model, name))
            for name in names
            if hasattr(model, name) and getattr(model, name) is not None
        },
    }


class PipedreamHydraulicEngine:
    """只负责模型转换、边界输入、单步求解和状态读取。"""

    def __init__(self, converted_model: dict[str, Any], dt: float = 60.0):
        self.converted_model = converted_model
        self.dt = float(dt)

        superjunctions = converted_model["superjunctions"].copy().set_index("id", drop=False)
        superlinks = converted_model["superlinks"].copy().set_index("id", drop=False)
        orifices = converted_model.get("orifices")
        if orifices is not None:
            orifices = orifices.copy().set_index("id", drop=False)
        pumps = converted_model.get("pumps_df")
        if pumps is not None:
            pumps = pumps.copy().set_index("id", drop=False)

        self.node_names = list(superjunctions["name"])
        self.channel_names = list(superlinks["name"])
        self.gate_names = list(orifices["name"]) if orifices is not None and "name" in orifices else []
        self.pump_names = list(pumps["name"]) if pumps is not None and "name" in pumps else []
        self.node_index = {name: index for index, name in enumerate(self.node_names)}
        self.channel_index = {name: index for index, name in enumerate(self.channel_names)}
        self.gate_index = {name: index for index, name in enumerate(self.gate_names)}
        self.pump_index = {name: index for index, name in enumerate(self.pump_names)}

        self.model = SuperLink(
            superlinks=superlinks,
            superjunctions=superjunctions,
            links=converted_model["links"],
            junctions=converted_model["junctions"],
            orifices=orifices,
            pumps=pumps,
            dt=self.dt,
            mobile_elements=False,
        )

    @classmethod
    def from_wxq(
        cls,
        source: Path | str | dict,
        *,
        dt: float = 60.0,
        gate_max_openings: dict[str, float] | None = None,
    ) -> "PipedreamHydraulicEngine":
        return cls(parse_wxq_json(source, gate_max_openings=gate_max_openings), dt=dt)

    @staticmethod
    def _input_array(
        values: Mapping[str, float] | Sequence[float] | np.ndarray | None,
        names: list[str],
        default: float | np.ndarray,
    ) -> np.ndarray:
        if values is None:
            if np.isscalar(default):
                return np.full(len(names), float(default), dtype=float)
            return np.asarray(default, dtype=float).copy()
        if isinstance(values, Mapping):
            result = PipedreamHydraulicEngine._input_array(None, names, default)
            index = {name: i for i, name in enumerate(names)}
            for name, value in values.items():
                if name not in index:
                    raise KeyError(f"模型中不存在对象: {name}")
                result[index[name]] = float(value)
            return result
        result = np.asarray(values, dtype=float)
        if result.shape != (len(names),):
            raise ValueError(f"输入长度应为 {len(names)}，实际为 {result.size}")
        return result

    def step(
        self,
        *,
        dt: float | None = None,
        boundary_heads: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        node_inflows: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        gate_openings: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        pump_controls: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        num_iter: int = 1,
        head_tol: float = 0.0015,
    ) -> dict[str, Any]:
        """推进一个计算步；映射输入允许只覆盖指定对象。"""
        step_index = int(round(float(self.model.t) / float(dt or self.dt))) + 1
        diagnostic_enabled = _diagnostic_enabled()
        diagnostic_interval = _diagnostic_interval()
        default_heads = np.where(self.model.bc, self.model.H_j, np.nan)
        h_bc = self._input_array(boundary_heads, self.node_names, default_heads)
        q_in = self._input_array(node_inflows, self.node_names, 0.0)
        u_o = self._input_array(gate_openings, self.gate_names, 1.0) if self.gate_names else None
        if u_o is not None:
            u_o = np.clip(u_o, 0.0, 1.0)
        u_p = None
        if self.model.n_p:
            u_p = self._input_array(pump_controls, self.pump_names, 1.0)
            u_p = np.clip(u_p, 0.0, 1.0)
        try:
            self.model.step(
                H_bc=h_bc,
                Q_in=q_in,
                u_o=u_o,
                u_p=u_p,
                dt=float(dt or self.dt),
                num_iter=max(int(num_iter), 1),
                head_tol=float(head_tol),
            )
        except Exception:
            logger.exception(
                "Pipedream hydraulic step failed | step=%s dt=%s model_summary=%s "
                "input_summary=%s",
                step_index,
                float(dt or self.dt),
                _model_summary(self.model),
                {
                    "H_bc": _array_summary(h_bc),
                    "Q_in": _array_summary(q_in),
                    "u_o": _array_summary(u_o) if u_o is not None else None,
                    "u_p": _array_summary(u_p) if u_p is not None else None,
                },
            )
            raise
        if diagnostic_enabled and (step_index <= 5 or step_index % diagnostic_interval == 0):
            logger.info(
                "Pipedream hydraulic step diagnostic | step=%s dt=%s model_summary=%s",
                step_index,
                float(dt or self.dt),
                _model_summary(self.model),
            )
        return self.state()

    def stabilize(
        self,
        *,
        boundary_heads: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        node_inflows: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        gate_openings: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        pump_controls: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        dt: float | None = None,
        max_steps: int = 1000,
        min_steps: int = 5,
        consecutive_steps: int = 5,
        head_change_tol: float = 0.0015,
        flow_change_tol: float = 0.01,
        num_iter: int = 1,
        step_head_tol: float = 0.0015,
    ) -> dict[str, Any]:
        """用固定的 0 秒真实边界推进，直到状态变化连续满足收敛阈值。"""
        stable_count = 0
        converged = False
        max_head_change = float("inf")
        max_flow_change = float("inf")
        steps_run = 0
        started_at = float(self.model.t)

        for step_index in range(1, max(int(max_steps), 0) + 1):
            previous_heads = self.model.H_j.copy()
            previous_flows = self.model.Q_ik.copy()
            self.step(
                dt=dt,
                boundary_heads=boundary_heads,
                node_inflows=node_inflows,
                gate_openings=gate_openings,
                pump_controls=pump_controls,
                num_iter=num_iter,
                head_tol=step_head_tol,
            )
            steps_run = step_index
            max_head_change = float(np.max(np.abs(self.model.H_j - previous_heads)))
            max_flow_change = float(np.max(np.abs(self.model.Q_ik - previous_flows)))
            if step_index >= max(int(min_steps), 1) and max_head_change <= head_change_tol and max_flow_change <= flow_change_tol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= max(int(consecutive_steps), 1):
                converged = True
                break

        simulated_seconds = float(self.model.t) - started_at
        self.model.t = 0.0
        self.model.iter_count = 0
        return {
            "converged": converged,
            "steps_run": steps_run,
            "simulated_seconds": simulated_seconds,
            "max_head_change_m": max_head_change,
            "max_flow_change_m3s": max_flow_change,
            "stable_consecutive_steps": stable_count,
            "head_change_tol_m": float(head_change_tol),
            "flow_change_tol_m3s": float(flow_change_tol),
        }

    def get_restart_state(self) -> dict[str, Any]:
        """Dump numeric SuperLink state after warmup so it can be restored later."""
        arrays: dict[str, Any] = {}
        scalars: dict[str, Any] = {}
        for name, value in vars(self.model).items():
            if isinstance(value, np.ndarray) and value.dtype.kind in {"b", "i", "u", "f"}:
                arrays[name] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "values": value.tolist(),
                }
            elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                scalars[name] = value.item() if hasattr(value, "item") else value
        return {
            "schema_version": "pipedream-superlink-restart-state-v1",
            "arrays": arrays,
            "scalars": scalars,
        }

    def load_restart_state(self, restart_state: Mapping[str, Any]) -> None:
        """Restore a state produced by get_restart_state()."""
        if restart_state.get("schema_version") != "pipedream-superlink-restart-state-v1":
            raise ValueError("不支持的 Pipedream 预热缓存格式")
        for name, payload in (restart_state.get("arrays") or {}).items():
            array = np.asarray(payload["values"], dtype=np.dtype(payload["dtype"]))
            expected_shape = tuple(payload.get("shape") or array.shape)
            if array.shape != expected_shape:
                array = np.reshape(array, expected_shape)
            setattr(self.model, name, array)
        for name, value in (restart_state.get("scalars") or {}).items():
            setattr(self.model, name, value)
        self.model.t = 0.0
        self.model.iter_count = 0

    def state(self) -> dict[str, Any]:
        """返回名称化状态，供上层发布 MQTT、写 OPC UA 或落盘。"""
        return {
            "time": float(self.model.t),
            "nodes": {
                name: {
                    "head": float(self.model.H_j[index]),
                    "depth": float(self.model.H_j[index] - self.model._z_inv_j[index]),
                }
                for index, name in enumerate(self.node_names)
            },
            "channels": {
                name: {
                    "upstream_flow": float(self.model.Q_uk[index]),
                    "downstream_flow": float(self.model.Q_dk[index]),
                }
                for index, name in enumerate(self.channel_names)
            },
            "gate_flows": {
                name: float(self.model.Q_o[index])
                for index, name in enumerate(self.gate_names)
            },
        }


__all__ = ["PipedreamHydraulicEngine"]
