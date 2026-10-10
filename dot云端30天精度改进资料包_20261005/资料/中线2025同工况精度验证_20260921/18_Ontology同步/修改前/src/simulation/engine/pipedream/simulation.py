"""Pipedream 后端在 HydroOS 任务生命周期中的运行适配器。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import time
from typing import Any

import numpy as np

from src.simulation.engine.hydro1dWxqSimulation import (
    Hydro1dWXQSimulation,
    should_pause_after_step,
)
from src.simulation.engine.hydraulic.base import HydraulicInput, HydraulicResult
from src.simulation.engine.hydraulic.step_diagnostics import (
    capture_hydraulic_compute_snapshot,
    hydraulic_compute_steps_per_output,
    hydraulic_output_dump_enabled,
    write_hydraulic_output_snapshot,
)
from src.simulation.engine.sim_control import pause_simulation
from src.simulation.disturbance.boundary_event_timing import (
    boundary_event_value,
    interval_start_elapsed_seconds,
)
from src.simulation.disturbance.motor_fault_policy import MOTOR_FAULT_ENGINE_TYPES, is_motor_faulted

from .hydraulic_engine import PipedreamHydraulicEngine
from .wxq_to_superlink import turbine_discharge_from_curve

logger = logging.getLogger(__name__)

PIPEDREAM_ADAPTER_VERSION = "0.4.0"
PIPEDREAM_RESTART_SCHEMA_VERSION = "pipedream-superlink-restart-state-v1"
WARM_CACHE_ROOT = os.path.join("cache", "steady_state")


def _stable_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(data) -> str:
    return hashlib.sha256(_stable_json(data).encode("utf-8")).hexdigest()


def _use_steady_state_cache_if_available() -> bool:
    value = os.getenv("HYDROS_USE_STEADY_STATE_CACHE_IF_AVAILABLE", "true")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _warm_cache_paths(job_state, fingerprint: str) -> tuple[str, str, str]:
    simulation_mode = str(getattr(job_state, "simulation_mode", None) or "unknown").strip() or "unknown"
    cache_dir = os.path.join(WARM_CACHE_ROOT, str(job_state.biz_scenario_id), simulation_mode)
    cache_file_name = f"{fingerprint}.json"
    manifest_path = os.path.join(cache_dir, f"{fingerprint}.manifest.json")
    return cache_dir, cache_file_name, manifest_path


def _write_warm_cache_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)


def _load_warm_cache_manifest(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _is_steady_state_profiling_enabled() -> bool:
    value = os.getenv("HYDROS_STEADY_STATE_PROFILING", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _record_steady_state_phase(job_state, phase: str, seconds: float) -> None:
    if not hasattr(job_state, "steady_state_profile") or job_state.steady_state_profile is None:
        job_state.steady_state_profile = {}
    job_state.steady_state_profile[phase] = seconds
    if _is_steady_state_profiling_enabled():
        logger.info(
            "[steady_state_profile] job=%s scenario=%s phase=%s seconds=%.6f",
            job_state.job_instance_id,
            job_state.biz_scenario_id,
            phase,
            seconds,
        )


def _write_steady_state_profile(job_state) -> None:
    if not _is_steady_state_profiling_enabled():
        return
    profile_dir = os.path.join("cache", "profiling")
    os.makedirs(profile_dir, exist_ok=True)
    profile_path = os.path.join(profile_dir, f"{job_state.job_instance_id}.json")
    payload = {
        "job_instance_id": job_state.job_instance_id,
        "biz_scenario_id": job_state.biz_scenario_id,
        "status": job_state.job_instance_status,
        "steady_state_start_mode": getattr(job_state, "steady_state_start_mode", "unknown"),
        "steady_state_warm_start_source": getattr(job_state, "steady_state_warm_start_source", "unknown"),
        "steady_state_cache_key": getattr(job_state, "steady_state_cache_key", None),
        "steady_state_cache_fingerprint": getattr(job_state, "steady_state_cache_fingerprint", None),
        "steady_state_cache_file": getattr(job_state, "steady_state_cache_file", None),
        "steady_state_cache_reason": getattr(job_state, "steady_state_cache_reason", None),
        "phases": getattr(job_state, "steady_state_profile", {}),
    }
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("[steady_state_profile] job=%s profile_file=%s", job_state.job_instance_id, profile_path)


class PipedreamSimulation(Hydro1dWXQSimulation):
    """把 Pipedream 求解器接入现有启动、控制和结果发布链路。"""

    source_id = "PipedreamSimulation"

    def __init__(self, job_state):
        super().__init__(job_state)
        self.engine: PipedreamHydraulicEngine | None = None
        self.base_data: dict[str, Any] = {}
        self._normalized_gate_openings: dict[str, float] = {}
        self._pump_controls: dict[str, float] = {}
        self._turbine_outputs: dict[str, float] = {}
        self._turbine_flows: dict[str, float] = {}
        self._bleeder_flows: dict[str, float] = {}
        self._last_logged_gate: dict[str, float] = {}
        self._last_logged_bleeder: dict[str, float] = {}
        self._last_logged_boundary: dict[str, float] = {}

    def _runtime_model(self) -> dict[str, Any]:
        model = copy.deepcopy(self.job_state.wxq_model_data)
        simulate = self.job_state.wxq_model_inputs_simulate_data or {}
        for key in ("initialData", "timeControl", "timeSeries", "pollutions"):
            if key in simulate:
                model[key] = copy.deepcopy(simulate[key])
        return model

    def _warmup_enabled(self) -> bool:
        enabled = os.getenv("HYDROS_PIPEDREAM_WARMUP_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        allow_turbines = os.getenv("HYDROS_PIPEDREAM_WARMUP_WITH_TURBINES", "").lower() in {"1", "true", "yes", "on"}
        if enabled and (self.base_data.get("turbines") or {}) and not allow_turbines:
            logger.info("[%s] 模型包含水轮机，跳过 Pipedream 稳态预热并保留调度初始水位。", self.job_instance_id)
            return False
        return enabled

    def _warmup_config(self) -> dict[str, Any]:
        configured_dt = float(os.getenv("HYDROS_PIPEDREAM_WARMUP_DT", self.job_state.sim_step_size))
        return {
            "dt": min(configured_dt, float(self.job_state.sim_step_size)),
            "max_steps": int(os.getenv("HYDROS_PIPEDREAM_WARMUP_MAX_STEPS", "1000")),
            "min_steps": int(os.getenv("HYDROS_PIPEDREAM_WARMUP_MIN_STEPS", "5")),
            "consecutive_steps": int(os.getenv("HYDROS_PIPEDREAM_WARMUP_CONSECUTIVE_STEPS", "5")),
            "head_change_tol": float(os.getenv("HYDROS_PIPEDREAM_WARMUP_HEAD_TOL", "0.0015")),
            "flow_change_tol": float(os.getenv("HYDROS_PIPEDREAM_WARMUP_FLOW_TOL", "0.01")),
        }

    def _build_warm_cache_manifest(self, runtime_model: dict[str, Any], gate_max_openings: dict[str, float]) -> dict:
        hydro_dyn_config_data = getattr(self.job_state, "wxq_raw_model_data", None) or runtime_model
        warmup_config = self._warmup_config()
        fingerprint_payload = {
            "hydro_dyn_config_sha256": _sha256_json(hydro_dyn_config_data),
            "hydro_model_input_sha256": _sha256_json(runtime_model),
            "gate_max_openings_sha256": _sha256_json(gate_max_openings),
            "request": {
                "biz_scenario_id": self.job_state.biz_scenario_id,
                "sim_step_size": self.job_state.sim_step_size,
                "output_step_size": self.job_state.output_step_size,
                "simulation_mode": self.job_state.simulation_mode,
            },
            "warmup_config": warmup_config,
            "pipedream_adapter_version": PIPEDREAM_ADAPTER_VERSION,
            "restart_schema_version": PIPEDREAM_RESTART_SCHEMA_VERSION,
        }
        fingerprint = hashlib.sha256(_stable_json(fingerprint_payload).encode("utf-8")).hexdigest()
        return {
            "cache_fingerprint": fingerprint,
            "cache_key": self.job_state.biz_scenario_id,
            "biz_scenario_id": self.job_state.biz_scenario_id,
            "sim_step_size": self.job_state.sim_step_size,
            "output_step_size": self.job_state.output_step_size,
            "simulation_mode": self.job_state.simulation_mode,
            "hydro_dyn_config_sha256": fingerprint_payload["hydro_dyn_config_sha256"],
            "hydro_model_input_sha256": fingerprint_payload["hydro_model_input_sha256"],
            "gate_max_openings_sha256": fingerprint_payload["gate_max_openings_sha256"],
            "pipedream_adapter_version": PIPEDREAM_ADAPTER_VERSION,
            "restart_schema_version": PIPEDREAM_RESTART_SCHEMA_VERSION,
            "warmup_config": warmup_config,
            "fingerprint_payload": fingerprint_payload,
        }

    async def _prepare_warm_start(self, runtime_model: dict[str, Any], gate_max_openings: dict[str, float]) -> None:
        assert self.engine is not None
        manifest = self._build_warm_cache_manifest(runtime_model, gate_max_openings)
        cache_fingerprint = manifest["cache_fingerprint"]
        cache_dir, cache_file_name, manifest_path = _warm_cache_paths(self.job_state, cache_fingerprint)
        cache_path = os.path.join(cache_dir, cache_file_name)
        cache_label = f"{self.job_state.biz_scenario_id}/{cache_fingerprint}"

        os.makedirs(cache_dir, exist_ok=True)
        self.job_state.steady_state_cache_key = self.job_state.biz_scenario_id
        self.job_state.steady_state_cache_fingerprint = cache_fingerprint
        self.job_state.steady_state_cache_file = cache_path
        self.job_state.steady_state_cache_manifest_file = manifest_path
        self.job_state.steady_state_cache_read_enabled = _use_steady_state_cache_if_available()
        self.job_state.steady_state_memory_cache_enabled = False
        self.job_state.steady_state_cache_reason = "not_checked"
        self.job_state.steady_state_cache_valid = False

        if not self._warmup_enabled():
            self.job_state.steady_state_start_mode = "cold_start"
            self.job_state.steady_state_warm_start_source = "none"
            self.job_state.steady_state_cache_reason = "pipedream_warmup_disabled"
            return

        cached_manifest = (
            _load_warm_cache_manifest(manifest_path)
            if self.job_state.steady_state_cache_read_enabled
            else None
        )
        manifest_matches = cached_manifest is not None and cached_manifest.get("cache_fingerprint") == cache_fingerprint
        if self.job_state.steady_state_cache_read_enabled and os.path.exists(cache_path) and manifest_matches:
            logger.info("[%s] 在本地文件发现 Pipedream 预热缓存 %s，准备加载...", self.job_instance_id, cache_label)
            started = time.perf_counter()
            with open(cache_path, "r", encoding="utf-8") as file:
                restart_state = json.load(file)
            await asyncio.to_thread(self.engine.load_restart_state, restart_state)
            _record_steady_state_phase(self.job_state, "restart_load_seconds", time.perf_counter() - started)
            self.job_state.steady_state_start_mode = "warm_start"
            self.job_state.steady_state_warm_start_source = "file"
            self.job_state.steady_state_cache_valid = True
            self.job_state.steady_state_cache_reason = "file_fingerprint_matched"
            logger.info("[%s] Pipedream 预热缓存 %s 加载成功。", self.job_instance_id, cache_label)
            return

        if not self.job_state.steady_state_cache_read_enabled:
            self.job_state.steady_state_cache_reason = "cache_read_disabled"
        elif os.path.exists(cache_path) and not manifest_matches:
            self.job_state.steady_state_cache_reason = "manifest_fingerprint_mismatch"
        else:
            self.job_state.steady_state_cache_reason = "cache_miss"

        boundary_heads, node_inflows, gate_openings, pump_controls = await self._step_inputs(0)
        warmup_config = self._warmup_config()
        started = time.perf_counter()
        warmup_summary = await asyncio.to_thread(
            self.engine.stabilize,
            boundary_heads=boundary_heads,
            node_inflows=node_inflows,
            gate_openings=gate_openings,
            pump_controls=pump_controls,
            dt=warmup_config["dt"],
            max_steps=warmup_config["max_steps"],
            min_steps=warmup_config["min_steps"],
            consecutive_steps=warmup_config["consecutive_steps"],
            head_change_tol=warmup_config["head_change_tol"],
            flow_change_tol=warmup_config["flow_change_tol"],
        )
        _record_steady_state_phase(self.job_state, "steady_state_initialize_seconds", time.perf_counter() - started)
        logger.info("[%s] Pipedream 真实边界预热结果: %s", self.job_instance_id, warmup_summary)

        self.job_state.steady_state_start_mode = "cold_start"
        self.job_state.steady_state_warm_start_source = "none"
        started = time.perf_counter()
        restart_state = await asyncio.to_thread(self.engine.get_restart_state)
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(restart_state, file, ensure_ascii=False, separators=(",", ":"))
        _write_warm_cache_manifest(manifest_path, manifest)
        _record_steady_state_phase(self.job_state, "restart_save_seconds", time.perf_counter() - started)
        self.job_state.steady_state_cache_valid = True
        self.job_state.steady_state_cache_reason = "cache_created"
        logger.info("[%s] Pipedream 预热结果已保存至：%s。", self.job_instance_id, cache_path)

    def _object_id(self, name: str) -> int | None:
        record = self.job_state.resolver.resolve_raw(name)
        return None if record is None or record.object_id is None else int(record.object_id)

    def _gate_max_openings_by_name(self, gates: dict[str, Any]) -> dict[str, float]:
        result = {}
        for name, gate in gates.items():
            object_id = self._object_id(name)
            if object_id is not None and object_id in self.job_state.gate_max_opening:
                result[name] = float(self.job_state.gate_max_opening[object_id])
            else:
                result[name] = float(gate.get("b", 1.0) or 1.0)
        return result

    @staticmethod
    def _series_value(series: dict[str, Any], elapsed_seconds: float) -> float:
        """按计算时刻读取启动层已经展开好的时间序列。"""
        values_by_time = series["values_by_time"]
        timestamp = float(elapsed_seconds)
        if timestamp in values_by_time:
            return float(values_by_time[timestamp])

        # 单点常量或序列结束后保持最近值；正常展开序列会命中上面的精确时间点。
        previous_index = int(np.searchsorted(series["times_s"], timestamp, side="right")) - 1
        if previous_index >= 0:
            return float(series["values"][previous_index])
        return float(series["values"][0])

    def _disturbance_value(self, object_name: str, step_counter: int) -> float | None:
        object_id = self._object_id(object_name)
        info = self.job_state.weather_forecast.get(object_id)
        if not info:
            info = self.job_state.weather_forecast.get(object_name)
        if not info:
            return None
        return boundary_event_value(
            info,
            compute_step=step_counter,
            steps_per_output=self.job_state.steps_per_output,
        )

    async def _hydraulic_input_for_step(self, step_counter: int) -> HydraulicInput:
        assert self.engine is not None
        converted = self.engine.converted_model
        base_bleeder_flows = {
            name: float(value)
            for name, value in (
                converted.get("initial_data", {}).get("bleeders", {}) or {}
            ).items()
        }
        elapsed = interval_start_elapsed_seconds(
            step_counter,
            self.job_state.sim_step_size,
        )
        node_types = {
            name: int(node.get("nodeType", 0) or 0)
            for name, node in (self.base_data.get("nodes") or {}).items()
        }
        time_series = list(converted.get("time_series", {}).values())
        flow_series_nodes = {
            series["target_name"]
            for series in time_series
            if series["target_type"] == "nodes"
            and node_types.get(series["target_name"]) != 1
        }
        boundary_heads: dict[str, float] = {}
        node_inflows = {name: 0.0 for name in self.engine.node_names}
        for name, value in (converted.get("boundaries") or {}).items():
            if name not in self.engine.node_index:
                continue
            if node_types.get(name) == 1:
                boundary_heads[name] = float(value)
            elif name not in flow_series_nodes:
                node_inflows[name] += float(value)

        channel_inflows: dict[str, float] = {}
        gate_openings = dict(self._normalized_gate_openings)
        pump_controls = dict(self._pump_controls)
        turbine_outputs = dict(self._turbine_outputs)
        for series in time_series:
            target_type = series["target_type"]
            target_name = series["target_name"]
            value = self._series_value(series, elapsed)
            if target_type == "nodes":
                if node_types.get(target_name) == 1:
                    boundary_heads[target_name] = value
                else:
                    node_inflows[target_name] = node_inflows.get(target_name, 0.0) + value
            elif target_type == "gates" and target_name in self.engine.gate_index:
                object_id = self._object_id(target_name)
                if object_id is not None and (
                    is_motor_faulted(self.job_state, "gates", object_id)
                    or not self.job_state.use_gate_time_series.get(object_id, True)
                ):
                    continue
                row = converted["orifices"].iloc[self.engine.gate_index[target_name]]
                gate_openings[target_name] = value / max(float(row["y_max"]), 1e-9)
            elif target_type == "bleeders":
                base_bleeder_flows[target_name] = value
            elif target_type == "channels":
                channel_inflows[target_name] = value
            elif target_type == "pumps" and target_name in self.engine.pump_index:
                object_id = self._object_id(target_name)
                if object_id is not None and (
                    is_motor_faulted(self.job_state, "pumps", object_id)
                    or not self.job_state.use_pump_time_series.get(object_id, True)
                ):
                    continue
                pump_controls[target_name] = value
            elif target_type == "turbines":
                object_id = self._object_id(target_name)
                if object_id is not None and (
                    is_motor_faulted(self.job_state, "turbines", object_id)
                    or not self.job_state.use_turbine_time_series.get(object_id, True)
                ):
                    continue
                turbine_outputs[target_name] = value

        for pump_name in self.engine.pump_names:
            object_id = self._object_id(pump_name)
            if object_id is not None and not self.job_state.use_pump_time_series.get(object_id, True):
                value = self.job_state.pump_target_angle.get(object_id)
                if value is not None:
                    pump_controls[pump_name] = float(value)

        for turbine in converted.get("turbines", []):
            turbine_name = turbine["name"]
            object_id = self._object_id(turbine_name)
            if object_id is not None and not self.job_state.use_turbine_time_series.get(object_id, True):
                value = self.job_state.turbine_target_para.get(object_id)
                if value is not None:
                    turbine_outputs[turbine_name] = float(value)

        initial_turbines = converted.get("initial_data", {}).get("turbines", {}) or {}
        turbine_flows: dict[str, float] = {}
        for turbine in converted.get("turbines", []):
            name = turbine["name"]
            operating_value = float(turbine_outputs.get(name, initial_turbines.get(name, 0.0)))
            upstream_index = self.engine.node_index[turbine["node1"]]
            downstream_index = self.engine.node_index[turbine["node2"]]
            head = max(
                float(self.engine.model.H_j[upstream_index] - self.engine.model.H_j[downstream_index]),
                0.0,
            )
            flow = turbine_discharge_from_curve(turbine["curve_data"], operating_value, head)
            turbine_flows[name] = flow
            node_inflows[turbine["node1"]] = node_inflows.get(turbine["node1"], 0.0) - flow
            node_inflows[turbine["node2"]] = node_inflows.get(turbine["node2"], 0.0) + flow

        for bleeder in converted.get("bleeders", []):
            name, node_name = bleeder["name"], bleeder.get("node")
            value = self._disturbance_value(name, step_counter)
            if name in self.job_state.modify_disturbancenode_value:
                value = float(self.job_state.modify_disturbancenode_value[name])
            if value is None:
                value = base_bleeder_flows.get(name, 0.0)
            self._bleeder_flows[name] = value
            if node_name:
                node_inflows[node_name] = node_inflows.get(node_name, 0.0) - value

        channels = self.base_data.get("channels") or {}
        for channel_name, channel in channels.items():
            value = self._disturbance_value(channel_name, step_counter)
            if value is not None:
                channel_inflows[channel_name] = value
            if channel_name in channel_inflows and channel.get("node1"):
                node_name = channel["node1"]
                node_inflows[node_name] = node_inflows.get(node_name, 0.0) + channel_inflows[channel_name]

        for object_key, info in self.job_state.weather_forecast.items():
            object_name = info.get("engine_object_name") or object_key
            object_type = info.get("object_type")
            object_id = info.get("object_id")
            if object_id is None:
                object_id = self._object_id(object_name)
            if (
                object_type in MOTOR_FAULT_ENGINE_TYPES
                and object_id is not None
                and is_motor_faulted(self.job_state, object_type, object_id)
            ):
                continue
            value = self._disturbance_value(object_name, step_counter)
            if value is None:
                continue
            if object_type == "nodes" and object_name in self.engine.node_index:
                if node_types.get(object_name) == 1:
                    boundary_heads[object_name] = value
                else:
                    node_inflows[object_name] = value
            elif object_type == "gates" and object_name in self.engine.gate_index:
                row = converted["orifices"].iloc[self.engine.gate_index[object_name]]
                gate_openings[object_name] = value / max(float(row["y_max"]), 1e-9)
            elif object_type == "pumps" and object_name in self.engine.pump_index:
                pump_controls[object_name] = value

        if self.job_state.actuator_acceleration_enabled:
            await self._process_gate_control_requests(step_counter)
            accelerated = await self._get_gate_openings_for_step(step_counter)
        else:
            accelerated = {}
        for gate_name in self.engine.gate_names:
            object_id = self._object_id(gate_name)
            if object_id is None or self.job_state.use_gate_time_series.get(object_id, True):
                continue
            row = converted["orifices"].iloc[self.engine.gate_index[gate_name]]
            max_opening = max(float(row["y_max"]), 1e-9)
            if is_motor_faulted(self.job_state, "gates", object_id):
                hold_opening = self.job_state.gate_fault_hold_opening.get(object_id)
                if hold_opening is not None:
                    gate_openings[gate_name] = float(hold_opening) / max_opening
                continue
            value = accelerated.get(object_id, -1.0) if self.job_state.actuator_acceleration_enabled else self.job_state.gate_target_opening.get(object_id)
            if value is not None and value != -1.0:
                gate_openings[gate_name] = float(value) / max_opening
                self.job_state.gate_actual_opening[object_id] = float(value)

        self._normalized_gate_openings = gate_openings
        self._pump_controls = pump_controls
        self._turbine_outputs = turbine_outputs
        self._turbine_flows = turbine_flows
        return HydraulicInput(
            step_index=step_counter,
            boundary_heads=boundary_heads,
            node_inflows=node_inflows,
            gate_openings=gate_openings,
            pump_controls=pump_controls,
            turbine_controls=dict(self._turbine_outputs),
            bleeder_flows=dict(self._bleeder_flows),
        )

    async def _step_inputs(self, step_counter: int) -> tuple[dict, dict, dict, dict]:
        hydraulic_input = await self._hydraulic_input_for_step(step_counter)
        return (
            hydraulic_input.boundary_heads,
            hydraulic_input.node_inflows,
            hydraulic_input.gate_openings,
            hydraulic_input.pump_controls,
        )

    def _objects_data(self) -> list[dict[str, Any]]:
        assert self.engine is not None
        converted = self.engine.converted_model
        model = self.engine.model
        objects: list[dict[str, Any]] = []

        for channel_name in (self.base_data.get("channels") or {}):
            spec = converted["channel_discretization"].get(channel_name)
            if spec is None:
                continue
            children = []
            for section in spec["sections"]:
                if not section.get("name"):
                    continue
                junction_id = int(section["junction_id"])
                flow_link_id = int(section["flow_link_id"])
                flow_sign = float(section.get("flow_sign", spec.get("flow_sign", 1.0)))
                children.append({
                    "name": section["name"],
                    "type": "CrossSection",
                    "water_level": float(model._z_inv_Ik[junction_id] + model.h_Ik[junction_id]),
                    "discharge": float(model.Q_ik[flow_link_id]) * flow_sign,
                })
            internal_ids = np.asarray(spec["internal_link_ids"], dtype=int)
            volume = float(np.sum(model._A_ik[internal_ids] * model._dx_ik[internal_ids]))
            objects.append({"name": channel_name, "type": "Channel", "children": children, "volume": volume})

        orifices = converted.get("orifices")
        if orifices is not None:
            for gate_name in (self.base_data.get("gates") or {}):
                if gate_name not in self.engine.gate_index:
                    continue
                index = self.engine.gate_index[gate_name]
                row = orifices.iloc[index]
                front_head = float(model.H_j[int(row["sj_0"])])
                back_head = float(model.H_j[int(row["sj_1"])])
                flow = float(model.Q_o[index])
                objects.append({
                    "name": gate_name,
                    "type": "GateStation",
                    "gate_opening": (
                        float(self._normalized_gate_openings.get(gate_name, 1.0))
                        * float(row["y_max"])
                    ),
                    "front_water_level": front_head,
                    "back_water_level": back_head,
                    "front_discharge": flow,
                    "back_discharge": flow,
                })


        for turbine in converted.get("turbines", []):
            upstream_index = self.engine.node_index[turbine["node1"]]
            downstream_index = self.engine.node_index[turbine["node2"]]
            flow = float(self._turbine_flows.get(turbine["name"], 0.0))
            objects.append({
                "name": turbine["name"],
                "type": "Turbine",
                "front_water_level": float(model.H_j[upstream_index]),
                "back_water_level": float(model.H_j[downstream_index]),
                "front_discharge": flow,
                "back_discharge": flow,
            })
        for siphon in converted.get("siphons_info", []):
            superlink_id = int(siphon["superlink_id"])
            flow_sign = float(siphon.get("flow_sign", 1.0))
            upstream_flow = float(model.Q_uk[superlink_id])
            downstream_flow = float(model.Q_dk[superlink_id])
            if flow_sign < 0:
                node1_flow = downstream_flow * flow_sign
                node2_flow = upstream_flow * flow_sign
            else:
                node1_flow = upstream_flow * flow_sign
                node2_flow = downstream_flow * flow_sign
            objects.append({
                "name": siphon["name"],
                "type": "Siphons",
                "children": [
                    {"name": siphon["node1"], "type": "CrossSection", "water_level": float(model.H_j[int(siphon["sj_0"])]), "discharge": node1_flow},
                    {"name": siphon["node2"], "type": "CrossSection", "water_level": float(model.H_j[int(siphon["sj_1"])]), "discharge": node2_flow},
                ],
            })

        for bleeder in converted.get("bleeders", []):
            if int(bleeder.get("sj", -1)) < 0:
                continue
            objects.append({
                "name": bleeder["name"],
                "type": "Bleeder",
                "children": [{
                    "name": bleeder["node"],
                    "type": "CrossSection",
                    "water_level": float(model.H_j[int(bleeder["sj"])]),
                    "discharge": float(self._bleeder_flows.get(bleeder["name"], 0.0)),
                }],
            })

        for pump in converted.get("pumps", []):
            name = pump["name"]
            if name not in self.engine.pump_index:
                continue
            index = self.engine.pump_index[name]
            flow = float(model.Q_p[index])
            objects.append({
                "name": name,
                "type": "Pump",
                "pump_status": float(self._pump_controls.get(name, 1.0)),
                "children": [
                    {"name": pump["node1"], "type": "CrossSection", "water_level": float(model.H_j[int(pump["sj_0"])]), "discharge": flow},
                    {"name": pump["node2"], "type": "CrossSection", "water_level": float(model.H_j[int(pump["sj_1"])]), "discharge": flow},
                ],
            })
        return objects

    def _hydraulic_result_for_step(self, step_index: int, current_time_ms: int) -> HydraulicResult:
        return HydraulicResult(
            step_index=step_index,
            current_time_ms=current_time_ms,
            objects_data=self._objects_data(),
        )

    async def _publish_hydraulic_result(self, result: HydraulicResult) -> dict[str, Any]:
        return await self.publish_step_results(
            result.step_index,
            self.job_state.objects_id,
            result.objects_data,
            result.current_time_ms,
            data_type="simulation",
        )

    def _log_hydraulic_input_changes(self, hydraulic_input: HydraulicInput) -> None:
        step_counter = hydraulic_input.step_index
        for name, val in hydraulic_input.gate_openings.items():
            prev = self._last_logged_gate.get(name)
            if prev is None or abs(val - prev) > 1e-6:
                self._last_logged_gate[name] = val
                logger.info('step=%s gate=%s opening=%.6f', step_counter, name, val)
        for name, val in hydraulic_input.bleeder_flows.items():
            prev = self._last_logged_bleeder.get(name)
            if prev is None or abs(val - prev) > 1e-6:
                self._last_logged_bleeder[name] = val
                logger.info('step=%s bleeder=%s flow=%.6f', step_counter, name, val)
        for name, val in hydraulic_input.boundary_heads.items():
            prev = self._last_logged_boundary.get(name)
            if prev is None or abs(val - prev) > 1e-6:
                self._last_logged_boundary[name] = val
                logger.info('step=%s boundary=%s head=%.6f', step_counter, name, val)

    async def _execute_hydraulic_step(self, hydraulic_input: HydraulicInput) -> dict[str, Any]:
        assert self.engine is not None
        self._log_hydraulic_input_changes(hydraulic_input)
        return await asyncio.to_thread(
            self.engine.step,
            dt=self.job_state.sim_step_size,
            boundary_heads=hydraulic_input.boundary_heads,
            node_inflows=hydraulic_input.node_inflows,
            gate_openings=hydraulic_input.gate_openings,
            pump_controls=hydraulic_input.pump_controls,
        )

    async def _initialize_hydraulic_engine(self) -> tuple[dict[str, Any], dict[str, float]]:
        runtime_model = self._runtime_model()
        self.base_data = runtime_model.get("baseData", {}) or {}
        gate_max_openings = self._gate_max_openings_by_name(self.base_data.get("gates", {}) or {})
        self.engine = await asyncio.to_thread(
            PipedreamHydraulicEngine.from_wxq,
            runtime_model,
            dt=self.job_state.sim_step_size,
            gate_max_openings=gate_max_openings,
        )
        repairs = self.engine.converted_model.get("topology_repairs", [])
        if repairs:
            logger.warning("[%s] Pipedream 自动修复 %s 个拓扑高程字段: %s", self.job_instance_id, len(repairs), repairs)

        initial_gates = self.engine.converted_model.get("initial_data", {}).get("gates", {}) or {}
        for gate_name in self.engine.gate_names:
            index = self.engine.gate_index[gate_name]
            row = self.engine.converted_model["orifices"].iloc[index]
            opening_m = float(initial_gates.get(gate_name, row["y_max"]) or 0.0)
            self._normalized_gate_openings[gate_name] = opening_m / max(float(row["y_max"]), 1e-9)
        self._bleeder_flows.update({
            name: float(value) for name, value in (
                self.engine.converted_model.get("initial_data", {}).get("bleeders", {}) or {}
            ).items()
        })

        initial_pumps = self.engine.converted_model.get('initial_data', {}).get('pumps', {}) or {}
        for pump_name, value in initial_pumps.items():
            if pump_name in self.engine.pump_index:
                self._pump_controls[pump_name] = float(value)
        initial_turbines = self.engine.converted_model.get('initial_data', {}).get('turbines', {}) or {}
        self._turbine_outputs.update({name: float(value) for name, value in initial_turbines.items()})

        return runtime_model, gate_max_openings

    async def run(self):
        if self.job_state.environment_type in {"HIL", "XIL"}:
            raise ValueError("Pipedream 第一阶段仅支持 NORMAL/SIL，暂不支持 HIL/XIL")

        self.job_state.job_instance_status = "STEADY_STATE_PREPARING"
        runtime_model, gate_max_openings = await self._initialize_hydraulic_engine()
        await self._prepare_warm_start(runtime_model, gate_max_openings)
        _write_steady_state_profile(self.job_state)

        if self.job_state.cancellation_event.is_set():
            self.job_state.job_instance_status = "CANCELLED"
            return
        self.job_state.job_instance_status = "RUNNING"
        await asyncio.sleep(1.5)
        if self.job_state.mqtt_manager is not None:
            await self.job_state.mqtt_manager.publish_status(
                self.job_instance_id, self.job_state.cluster_id, self.job_state.edge_node_code,
                "RUNNING", self.job_state.start_time,
            )
        initial_result = self._hydraulic_result_for_step(0, self.job_state.start_time)
        if hydraulic_output_dump_enabled():
            capture_hydraulic_compute_snapshot(
                self.job_state,
                compute_step=0,
                inputs={"initial_state": True},
                result=initial_result.objects_data,
            )
        initial_summary = await self._publish_hydraulic_result(initial_result)
        self.job_state.latest_compute_result = {
            "job_instance_id": self.job_instance_id,
            "driven_mode": self.job_state.driven_mode,
            "compute_step": 0,
            "output_step": 0,
            "sim_step_size": self.job_state.sim_step_size,
            "output_step_size": self.job_state.output_step_size,
            "source_timestamp_ms": initial_result.current_time_ms,
            "result": initial_summary.get("result_data"),
            "metrics": initial_summary.get("metrics", []),
            "metrics_count": {
                "mqtt": initial_summary.get("mqtt_metrics_count", 0),
                "opcua": initial_summary.get("opcua_metrics_count", 0),
            },
            "result_kind": "output_boundary",
        }
        await pause_simulation(
            0 in self.job_state.pause_at_steps or self.job_state.driven_by_coordinator,
            self.job_state,
            self.job_instance_id,
        )

        self._transient_output_step_started_at = time.monotonic()
        step_counter = 1
        is_infinite_run = self.job_state.total_steps == -1
        steps_per_output = hydraulic_compute_steps_per_output(self.job_state)
        while not self.job_state.cancellation_event.is_set():
            if not is_infinite_run and step_counter > self.job_state.total_steps:
                break
            self.job_state.current_step = step_counter
            hydraulic_input = await self._hydraulic_input_for_step(step_counter)
            hydraulic_state = await self._execute_hydraulic_step(hydraulic_input)
            if hydraulic_output_dump_enabled():
                capture_hydraulic_compute_snapshot(
                    self.job_state,
                    compute_step=step_counter,
                    inputs=hydraulic_input,
                    result={
                        "state": hydraulic_state,
                        "objects_data": self._objects_data(),
                    },
                )
            logger.info("Pipedream sim done | scenario=%s step=%s", self.job_state.biz_scenario_id, step_counter)

            should_publish = step_counter % steps_per_output == 0
            if should_publish:
                output_step = step_counter // steps_per_output
                timestamp = self.job_state.start_time + step_counter * self.job_state.sim_step_size * 1000
                result = self._hydraulic_result_for_step(output_step, timestamp)
                if hydraulic_output_dump_enabled():
                    await asyncio.to_thread(
                        write_hydraulic_output_snapshot,
                        self.job_state,
                        engine="hydro1d_pipedream",
                        output_step=output_step,
                        compute_step=step_counter,
                        inputs=hydraulic_input,
                        result={
                            "state": hydraulic_state,
                            "objects_data": result.objects_data,
                        },
                    )
                summary = await self._publish_hydraulic_result(result)
                self.job_state.latest_compute_result = {
                    "job_instance_id": self.job_instance_id,
                    "driven_mode": self.job_state.driven_mode,
                    "compute_step": step_counter,
                    "output_step": output_step,
                    "sim_step_size": self.job_state.sim_step_size,
                    "output_step_size": self.job_state.output_step_size,
                    "source_timestamp_ms": timestamp,
                    "result": summary.get("result_data"),
                    "metrics": summary.get("metrics", []),
                    "metrics_count": {
                        "mqtt": summary.get("mqtt_metrics_count", 0),
                        "opcua": summary.get("opcua_metrics_count", 0),
                    },
                    "result_kind": "output_boundary",
                }
                self.job_state.record_transient_output_step(
                    output_step=output_step,
                    elapsed_seconds=time.monotonic() - self._transient_output_step_started_at,
                    mqtt_metrics_count=summary.get("mqtt_metrics_count", 0),
                )

            await pause_simulation(
                should_pause_after_step(
                    step_counter, self.job_state.pause_at_steps,
                    self.job_state.total_steps, is_infinite_run,
                ),
                self.job_state,
                self.job_instance_id,
            )
            if should_publish:
                self._transient_output_step_started_at = time.monotonic()
            step_counter += 1
            if not self.job_state.driven_by_coordinator:
                await asyncio.sleep(self.job_state.step_interval)


__all__ = ["PipedreamSimulation"]
