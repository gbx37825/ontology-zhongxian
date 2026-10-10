"""Standalone wrapper around the bundled Pipedream hydraulic solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .backends import DEFAULT_BACKEND, COMPATIBILITY_REVISIONS, RESTART_GUARD_ARRAYS, create_solver

from .wxq_to_superlink import parse_wxq_json


class PipedreamHydraulicEngine:
    """只负责模型转换、边界输入、单步求解和状态读取。"""

    def __init__(self, converted_model: dict[str, Any], dt: float = 60.0, *, backend: str = DEFAULT_BACKEND):
        self.converted_model = converted_model
        self.dt = float(dt)
        self.backend = backend

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

        self.model, self.backend_diagnostics = create_solver(
            backend=backend,
            superlinks=superlinks,
            superjunctions=superjunctions,
            links=converted_model["links"],
            junctions=converted_model["junctions"],
            transects=converted_model.get("transects", {}),
            orifices=orifices,
            pumps=pumps,
            dt=self.dt,
            mobile_elements=False,
        )
        self.solver_identity = {
            "backend": backend,
            "class": type(self.model).__module__ + "." + type(self.model).__name__,
            "compatibility_revision": COMPATIBILITY_REVISIONS[backend],
        }
        self._restart_guards = {name: getattr(self.model, name).copy() for name in RESTART_GUARD_ARRAYS}
        self._restart_sizes = {name: getattr(self.model, name) for name in ("M", "NK", "Nik", "NIk", "n_o", "n_w", "n_p")}

    @classmethod
    def from_wxq(
        cls,
        source: Path | str | dict,
        *,
        dt: float = 60.0,
        backend: str = DEFAULT_BACKEND,
        gate_max_openings: dict[str, float] | None = None,
        section_geometry_mode: str = "safe",
        minimum_initial_depth_m: float = 0.01,
    ) -> "PipedreamHydraulicEngine":
        return cls(
            parse_wxq_json(
                source,
                gate_max_openings=gate_max_openings,
                section_geometry_mode=section_geometry_mode,
                minimum_initial_depth_m=minimum_initial_depth_m,
            ),
            dt=dt,
            backend=backend,
        )

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
        self.model.step(
            H_bc=h_bc,
            Q_in=q_in,
            u_o=u_o,
            u_p=u_p,
            dt=float(dt or self.dt),
            num_iter=max(int(num_iter), 1),
            head_tol=float(head_tol),
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
            "schema_version": "pipedream-superlink-restart-state-v2",
            "solver": dict(self.solver_identity),
            "reference_dt_s": self.dt,
            "internal_method": self.model._method,
            "arrays": arrays,
            "scalars": scalars,
        }

    def load_restart_state(self, restart_state: Mapping[str, Any], *, allow_legacy: bool = False) -> None:
        """Restore a state produced by get_restart_state()."""
        version = restart_state.get("schema_version")
        legacy = version == "pipedream-superlink-restart-state-v1"
        if legacy:
            if self.backend != "standard" or not allow_legacy:
                raise ValueError("旧预热缓存仅允许 --backend standard --warmup-mode fixed；Numba请重新预热")
        elif version != "pipedream-superlink-restart-state-v2":
            raise ValueError("不支持的 Pipedream 预热缓存格式")
        elif restart_state.get("solver") != self.solver_identity or restart_state.get("reference_dt_s") != self.dt:
            raise ValueError("预热缓存的求解器、参数约定或正式步长不一致，拒绝混用；切换后端请重新预热")
        method = restart_state.get("internal_method", self.model._method if legacy else None)
        if method not in {"b", "f", "lsq", "nnls"}:
            raise ValueError("预热缓存的内部断面求解方法无效")
        arrays = {}
        for name, payload in (restart_state.get("arrays") or {}).items():
            array = np.asarray(payload["values"], dtype=np.dtype(payload["dtype"]))
            expected_shape = tuple(payload.get("shape") or array.shape)
            if array.shape != expected_shape:
                array = np.reshape(array, expected_shape)
            current = getattr(self.model, name, None)
            if current is None and name in {"_Q_in", "_Q_0Ik"}:
                size = self.model.M if name == "_Q_in" else self.model.NIk
                current = np.empty(size, dtype=np.float64)
            if not isinstance(current, np.ndarray) or array.shape != current.shape or array.dtype != current.dtype:
                raise ValueError(f"预热缓存数组与当前模型不一致: {name}")
            if name not in {"max_depth", "max_depth_k"} and not np.isfinite(array).all():
                raise ValueError(f"预热缓存数组存在无效数值: {name}")
            if name in {"max_depth", "max_depth_k"} and (np.isnan(array).any() or np.isneginf(array).any()):
                raise ValueError(f"预热缓存水深上限无效: {name}")
            arrays[name] = array
        for name, expected in self._restart_guards.items():
            if name not in arrays or not np.array_equal(arrays[name], expected):
                raise ValueError(f"预热缓存拓扑或水力参数不一致，拒绝混用: {name}")
        for name in ("H_j", "_h_Ik", "_Q_ik", "_Qo"):
            if name not in arrays:
                raise ValueError(f"预热缓存缺少水力状态: {name}")
        scalars = restart_state.get("scalars") or {}
        numeric = (bool, int, float, np.bool_, np.integer, np.floating)
        for name, value in scalars.items():
            if not isinstance(getattr(self.model, name, None), numeric) or not isinstance(value, numeric) or not np.isfinite(value):
                raise ValueError(f"预热缓存标量无效: {name}")
        for name, expected in self._restart_sizes.items():
            if scalars.get(name) != expected:
                raise ValueError(f"预热缓存拓扑规模不一致: {name}")
        # Validate the whole payload before mutating the model. Coefficients are already adapted.
        for name, array in arrays.items():
            setattr(self.model, name, array.copy())
        for name, value in scalars.items():
            setattr(self.model, name, value)
        self.model._method = method
        self.model.t = 0.0
        self.model.iter_count = 0

    def state(self) -> dict[str, Any]:
        """Return solver state keyed by WXQ object names."""
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
