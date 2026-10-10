from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine
from wxq_pipedream.backends import BACKENDS, DEFAULT_BACKEND
from wxq_pipedream.wxq_to_superlink import parse_wxq_json
from wxq_pipedream.warmup import adaptive_warmup
from wxq_pipedream.warmup_io import resolve_config, write_report


def load_wxq(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "baseData" in raw:
        return raw
    if not raw:
        raise ValueError("WXQ JSON 为空")
    model_name, model = next(iter(raw.items()))
    if not isinstance(model, dict) or "baseData" not in model:
        raise ValueError("无法识别 WXQ JSON 的 baseData")
    model = dict(model)
    model.setdefault("model_id", model_name)
    return model


def series_value(series: dict[str, Any], elapsed: float) -> float:
    times = np.asarray(series["times_s"], dtype=float)
    values = np.asarray(series["values"], dtype=float)
    if not len(times):
        return 0.0
    if len(times) == 1:
        return float(values[0])
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    return float(np.interp(float(elapsed), times, values))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def contains_none(value: Any) -> bool:
    if isinstance(value, list):
        return any(contains_none(item) for item in value)
    return value is None


def restart_contract(model_path, converted, engine):
    tables = {}
    for name in ("superjunctions", "superlinks", "junctions", "links", "orifices", "pumps_df"):
        frame = converted.get(name)
        tables[name] = None if frame is None else hashlib.sha256(
            frame.to_json(orient="split", double_precision=15).encode()).hexdigest()
    return {"input_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "tables": tables,
            "solver": dict(engine.solver_identity), "compute_dt_s": engine.dt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                        help="水动力求解器，默认numba；standard为普通版")
    parser.add_argument("--warmup-mode", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--warmup-config", type=Path)
    parser.add_argument("--warmup-max-steps", type=int)
    parser.add_argument("--warmup-checkpoints")
    parser.add_argument("--warmup-dt", type=float)
    parser.add_argument("--warmup-head-tol", type=float)
    parser.add_argument("--warmup-flow-tol", type=float)
    parser.add_argument("--internal-method", choices=("b", "f", "lsq"))
    parser.add_argument("--gate-coefficients", type=Path)
    parser.add_argument("--initial-depth", type=float)
    parser.add_argument("--minimum-initial-depth", type=float)
    parser.add_argument("--all-superjunction-depth", type=float)
    parser.add_argument("--max-compute-steps", type=int)
    parser.add_argument("--restart-state", type=Path)
    parser.add_argument("--restart-warmup-steps", type=int, default=0)
    parser.add_argument("--compute-dt", type=float)
    parser.add_argument("--fixed-inputs-at-zero", action="store_true")
    parser.add_argument("--num-iter", type=int)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--superjunctions-csv", type=Path)
    parser.add_argument("--section-geometry-mode", choices=("legacy", "safe"), default="safe")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("输出目录非空，请使用新目录，避免覆盖已有试验结果")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = load_wxq(args.model)
    base_data = model.get("baseData", {}) or {}
    time_control = model.get("timeControl", {}) or {}
    dt = float(args.compute_dt if args.compute_dt is not None else time_control.get("calculateStep", 120.0))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("计算步长必须为有限正数")
    steady_options = (model.get("initialData", {}) or {}).get("options", {}) or {}
    warmup_config = None
    if args.warmup_mode == "adaptive":
        if args.warmup_checkpoints or args.restart_warmup_steps:
            raise ValueError("分档检查点和restart-warmup-steps仅用于fixed模式；自适应用warmup-config设置总上限")
        warmup_config = resolve_config(steady_options, dt, args.warmup_config, {
            "initial_dt": args.warmup_dt, "max_steps": args.warmup_max_steps,
            "head_tolerance_m": args.warmup_head_tol, "flow_tolerance_m3s": args.warmup_flow_tol,
            "internal_method": args.internal_method,
            "num_iter": args.num_iter,
        })
    elif args.warmup_config is not None:
        raise ValueError("warmup-config只用于adaptive模式")
    args.warmup_max_steps = args.warmup_max_steps if args.warmup_max_steps is not None else 1000
    args.warmup_head_tol = args.warmup_head_tol if args.warmup_head_tol is not None else 0.0015
    args.warmup_flow_tol = args.warmup_flow_tol if args.warmup_flow_tol is not None else 0.01
    warmup_dt = float(args.warmup_dt or steady_options.get("timeStep", dt) or dt)
    if warmup_config:
        warmup_dt = warmup_config.initial_dt
    args.num_iter = warmup_config.num_iter if warmup_config else (args.num_iter if args.num_iter is not None else 1)
    if args.num_iter < 1:
        raise ValueError("num-iter必须为正整数")
    output_step = float(time_control.get("outputStep", 7200.0))
    end_time = float(args.end_time if args.end_time is not None else time_control.get("endTime", output_step))
    if any(not math.isfinite(value) or value <= 0 for value in (output_step, end_time, warmup_dt)):
        raise ValueError("输出间隔、结束时间和预热步长必须为有限正数")
    if not math.isclose(output_step / dt, round(output_step / dt), abs_tol=1e-9):
        raise ValueError("当前独立入口要求输出间隔为计算步长的整倍数，不能静默改变记录时刻")
    if args.max_compute_steps is not None and args.max_compute_steps < 0:
        raise ValueError("max-compute-steps不能为负数")
    total_steps = int(math.ceil(end_time / dt))
    if args.max_compute_steps is not None:
        total_steps = min(total_steps, max(args.max_compute_steps, 0))
    steps_per_output = max(int(round(output_step / dt)), 1)

    gates = base_data.get("gates", {}) or {}
    gate_max_openings = {
        name: float(gate.get("b", 1.0) or 1.0)
        for name, gate in gates.items()
        if isinstance(gate, dict)
    }
    converted = parse_wxq_json(
        model,
        gate_max_openings=gate_max_openings,
        section_geometry_mode=args.section_geometry_mode,
        minimum_initial_depth_m=args.minimum_initial_depth if args.minimum_initial_depth is not None else
        (7.0 if args.warmup_mode == "fixed" else 0.01),
    )
    if args.gate_coefficients:
        coefficients = json.loads(args.gate_coefficients.read_text(encoding="utf-8"))
        if not isinstance(coefficients, dict):
            raise ValueError("闸流系数文件应为闸孔名称到系数的JSON对象")
        orifices = converted.get("orifices")
        names = set(orifices["name"]) if orifices is not None else set()
        for name, value in coefficients.items():
            if name not in names or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"无效闸孔名称或流量系数: {name}={value}")
            orifices.loc[orifices["name"] == name, "C"] = float(value)
    if args.superjunctions_csv is not None and args.all_superjunction_depth is not None:
        raise ValueError("--superjunctions-csv 与 --all-superjunction-depth 不能同时使用")
    if args.superjunctions_csv is not None:
        override = pd.read_csv(args.superjunctions_csv)
        required_columns = {"id", "name", "h_0"}
        if not required_columns.issubset(override.columns):
            raise ValueError(f"SuperJunction覆盖CSV缺少字段: {sorted(required_columns - set(override.columns))}")
        if override["id"].duplicated().any() or override["name"].duplicated().any():
            raise ValueError("SuperJunction覆盖CSV存在重复id或name")
        source = converted["superjunctions"].copy()
        if set(override["name"]) != set(source["name"]):
            missing = sorted(set(source["name"]) - set(override["name"]))
            extra = sorted(set(override["name"]) - set(source["name"]))
            raise ValueError(f"SuperJunction覆盖CSV拓扑不一致: missing={missing}, extra={extra}")
        override_by_name = override.set_index("name")
        for row in source.itertuples():
            override_id = int(override_by_name.at[row.name, "id"])
            if override_id != int(row.id):
                raise ValueError(f"SuperJunction id不一致: {row.name}, {row.id} != {override_id}")
        h_0 = source["name"].map(override_by_name["h_0"]).to_numpy(dtype=float)
        if not np.isfinite(h_0).all() or (h_0 <= 0.0).any():
            raise ValueError("SuperJunction覆盖CSV的h_0必须全部为有限正数")
        source.loc[:, "h_0"] = h_0
        source.loc[:, "max_depth"] = np.maximum(
            source["max_depth"].to_numpy(dtype=float),
            h_0,
        )
        converted["superjunctions"] = source
    if args.all_superjunction_depth is not None:
        depth = float(args.all_superjunction_depth)
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("--all-superjunction-depth 必须是有限正数")
        converted["superjunctions"].loc[:, "h_0"] = depth
        converted["superjunctions"].loc[:, "max_depth"] = np.maximum(
            converted["superjunctions"]["max_depth"].to_numpy(dtype=float),
            depth,
        )
    if args.initial_depth is not None:
        initial_depth = max(float(args.initial_depth), 0.01)
        converted["superjunctions"].loc[:, "h_0"] = np.maximum(
            converted["superjunctions"]["h_0"].to_numpy(dtype=float),
            initial_depth,
        )
        for index, row in converted["superjunctions"].iterrows():
            node_name = row["name"]
            if node_types := (base_data.get("nodes", {}) or {}).get(node_name):
                if int(node_types.get("nodeType", 0) or 0) == 1:
                    boundary_level = (converted.get("boundaries", {}) or {}).get(node_name)
                    if boundary_level is not None:
                        converted["superjunctions"].at[index, "h_0"] = max(
                            float(boundary_level) - float(row["z_inv"]),
                            0.01,
                        )
        converted["superjunctions"].loc[:, "max_depth"] = np.maximum(
            converted["superjunctions"]["max_depth"].to_numpy(dtype=float),
            converted["superjunctions"]["h_0"].to_numpy(dtype=float),
        )
        converted["junctions"].loc[:, "h_0"] = np.maximum(
            converted["junctions"]["h_0"].to_numpy(dtype=float),
            initial_depth,
        )
    engine = PipedreamHydraulicEngine(converted, dt=dt, backend=args.backend)
    state_contract = restart_contract(args.model, converted, engine)
    print("SOLVER " + json.dumps(engine.solver_identity, ensure_ascii=False), flush=True)
    if args.internal_method:
        engine.model._method = args.internal_method
    conversion_seconds = time.perf_counter() - started
    model_limitations = []
    if converted.get("turbines"):
        model_limitations.append("水轮机仅有转换元数据，当前独立入口未求解水轮机流量，不能据此验收完整水电模型")
    if any(series.get("target_type") in {"pumps", "turbines"}
           for series in converted.get("time_series", {}).values()):
        model_limitations.append("当前独立入口未驱动泵或水轮机的时变运行曲线")
    for limitation in model_limitations:
        print("MODEL_WARNING " + limitation, flush=True)
    converted["superjunctions"].to_csv(
        args.output / "实际使用的superjunctions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if converted.get("orifices") is not None:
        converted["orifices"].to_csv(
            args.output / "实际使用的orifices.csv",
            index=False,
            encoding="utf-8-sig",
        )
    converted["junctions"].to_csv(
        args.output / "实际使用的junctions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    node_types = {
        name: int(node.get("nodeType", 0) or 0)
        for name, node in (base_data.get("nodes", {}) or {}).items()
    }
    time_series = list((converted.get("time_series", {}) or {}).values())
    flow_series_nodes = {
        series["target_name"]
        for series in time_series
        if series["target_type"] == "nodes"
        and node_types.get(series["target_name"]) != 1
    }
    initial_data = converted.get("initial_data", {}) or {}
    initial_gates = initial_data.get("gates", {}) or {}
    gate_openings: dict[str, float] = {}
    if converted.get("orifices") is not None:
        for gate_name in engine.gate_names:
            row = converted["orifices"].iloc[engine.gate_index[gate_name]]
            opening_m = float(initial_gates.get(gate_name, row["y_max"]) or 0.0)
            gate_openings[gate_name] = opening_m / max(float(row["y_max"]), 1e-9)
    bleeder_flows = {
        name: float(value)
        for name, value in (initial_data.get("bleeders", {}) or {}).items()
    }

    def inputs_at(elapsed: float) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        boundary_heads: dict[str, float] = {}
        node_inflows = {name: 0.0 for name in engine.node_names}
        for name, value in (converted.get("boundaries", {}) or {}).items():
            if name not in engine.node_index:
                continue
            if node_types.get(name) == 1:
                boundary_heads[name] = float(value)
            elif name not in flow_series_nodes:
                node_inflows[name] += float(value)

        channel_inflows: dict[str, float] = {}
        current_gates = dict(gate_openings)
        current_bleeders = dict(bleeder_flows)
        for series in time_series:
            target_type = series["target_type"]
            target_name = series["target_name"]
            value = series_value(series, elapsed)
            if target_type == "nodes" and target_name in engine.node_index:
                if node_types.get(target_name) == 1:
                    boundary_heads[target_name] = value
                else:
                    node_inflows[target_name] += value
            elif target_type == "gates" and target_name in engine.gate_index:
                row = converted["orifices"].iloc[engine.gate_index[target_name]]
                current_gates[target_name] = value / max(float(row["y_max"]), 1e-9)
            elif target_type == "bleeders":
                current_bleeders[target_name] = value
            elif target_type == "channels":
                channel_inflows[target_name] = value

        for bleeder in converted.get("bleeders", []):
            node_name = bleeder.get("node")
            if node_name in node_inflows:
                node_inflows[node_name] -= float(current_bleeders.get(bleeder["name"], 0.0))
        for channel_name, value in channel_inflows.items():
            channel = (base_data.get("channels", {}) or {}).get(channel_name, {})
            node_name = channel.get("node1")
            if node_name in node_inflows:
                node_inflows[node_name] += float(value)
        return boundary_heads, node_inflows, current_gates

    boundary_heads, node_inflows, current_gates = inputs_at(0.0)
    print(
        "INPUTS "
        f"boundary_heads={sorted(boundary_heads.items())} "
        f"node_inflow_sum={sum(node_inflows.values()):.12f} "
        f"node_inflow_nonzero={sum(abs(value) > 0 for value in node_inflows.values())} "
        f"node_inflow_min={min(node_inflows.values()):.12f} "
        f"node_inflow_max={max(node_inflows.values()):.12f} "
        f"gate_sum={sum(current_gates.values()):.12f} "
        f"gate_min={min(current_gates.values()) if current_gates else 0.0:.12f} "
        f"gate_max={max(current_gates.values()) if current_gates else 0.0:.12f}",
        flush=True,
    )
    restart_arrays_skipped: list[str] = []
    if args.restart_state is not None:
        with args.restart_state.open("r", encoding="utf-8") as handle:
            restart_state = json.load(handle)
        stored_contract = restart_state.get("restart_contract")
        allow_legacy = args.warmup_mode == "fixed" and args.backend == "standard"
        expected_contract = state_contract
        if allow_legacy and restart_state.get("schema_version") == "pipedream-superlink-restart-state-v1":
            expected_contract = {key: state_contract[key] for key in ("input_model_sha256", "tables")}
        if stored_contract is not None and stored_contract != expected_contract:
            raise ValueError("预热末态与当前模型、几何、初始配置、闸流系数或求解器不一致，拒绝混用")
        if args.warmup_mode == "adaptive" and stored_contract is None:
            raise ValueError("旧末态缺少模型核对信息，请从模型重新预热；fixed模式仅保留旧缓存兼容入口")
        arrays = restart_state.get("arrays") or {}
        for name in list(arrays):
            if not contains_none(arrays[name].get("values")):
                continue
            if name not in {"max_depth", "max_depth_k"}:
                raise ValueError(f"预热末态动态数组含null，拒绝恢复: {name}")
            arrays.pop(name)
            restart_arrays_skipped.append(name)
        engine.load_restart_state(restart_state, allow_legacy=allow_legacy)
        if args.internal_method:
            engine.model._method = args.internal_method
        if args.warmup_mode == "adaptive":
            warmup = None
        elif args.restart_warmup_steps > 0:
            warmup_started = time.perf_counter()
            warmup = engine.stabilize(
                boundary_heads=boundary_heads,
                node_inflows=node_inflows,
                gate_openings=current_gates,
                dt=warmup_dt,
                max_steps=args.restart_warmup_steps,
                min_steps=5,
                consecutive_steps=5,
                head_change_tol=args.warmup_head_tol,
                flow_change_tol=args.warmup_flow_tol,
            )
            warmup = {
                **warmup,
                "resumed_from": str(args.restart_state),
                "additional_step_budget": args.restart_warmup_steps,
            }
            warmup_seconds = time.perf_counter() - warmup_started
            print(
                "RESTART_WARMUP "
                f"steps={warmup['steps_run']} converged={warmup['converged']} "
                f"head_change={warmup['max_head_change_m']:.12f} "
                f"flow_change={warmup['max_flow_change_m3s']:.12f} "
                f"stable={warmup['stable_consecutive_steps']}",
                flush=True,
            )
        else:
            warmup = {
                "skipped": True,
                "source": str(args.restart_state),
            }
            warmup_seconds = 0.0
    elif args.warmup_mode == "fixed":
        warmup_started = time.perf_counter()
        checkpoints = []
        if args.warmup_checkpoints:
            checkpoints = [int(value.strip()) for value in args.warmup_checkpoints.split(",") if value.strip()]
            if not checkpoints or any(value <= 0 for value in checkpoints):
                raise ValueError("预热检查点必须是正整数")
            if checkpoints != sorted(set(checkpoints)):
                raise ValueError("预热检查点必须严格递增且不能重复")
        else:
            checkpoints = [max(int(args.warmup_max_steps), 0)]

        attempts = []
        cumulative_steps = 0
        warmup = None
        for checkpoint in checkpoints:
            additional_steps = checkpoint - cumulative_steps
            if additional_steps <= 0:
                continue
            attempt = engine.stabilize(
                boundary_heads=boundary_heads,
                node_inflows=node_inflows,
                gate_openings=current_gates,
                dt=warmup_dt,
                max_steps=additional_steps,
                min_steps=5,
                consecutive_steps=5,
                head_change_tol=args.warmup_head_tol,
                flow_change_tol=args.warmup_flow_tol,
            )
            cumulative_steps += int(attempt["steps_run"])
            attempt = {
                **attempt,
                "checkpoint_steps": checkpoint,
                "additional_step_budget": additional_steps,
                "cumulative_steps_run": cumulative_steps,
            }
            attempts.append(attempt)
            print(
                "WARMUP "
                f"checkpoint={checkpoint} cumulative={cumulative_steps} "
                f"converged={attempt['converged']} "
                f"head_change={attempt['max_head_change_m']:.12f} "
                f"flow_change={attempt['max_flow_change_m3s']:.12f} "
                f"stable={attempt['stable_consecutive_steps']} "
                f"head_tol={args.warmup_head_tol:.12f} "
                f"flow_tol={args.warmup_flow_tol:.12f}",
                flush=True,
            )
            warmup = attempt
            if attempt["converged"]:
                break
        if warmup is None:
            warmup = {
                "converged": False,
                "steps_run": 0,
                "simulated_seconds": 0.0,
                "max_head_change_m": None,
                "max_flow_change_m3s": None,
                "stable_consecutive_steps": 0,
                "head_change_tol_m": args.warmup_head_tol,
                "flow_change_tol_m3s": args.warmup_flow_tol,
                "checkpoint_steps": 0,
                "additional_step_budget": 0,
                "cumulative_steps_run": 0,
            }
        else:
            warmup = dict(warmup)
        warmup["attempts"] = attempts
        warmup["requested_checkpoints"] = checkpoints
        warmup["cumulative_simulated_seconds"] = cumulative_steps * warmup_dt
        warmup_seconds = time.perf_counter() - warmup_started

    if args.warmup_mode == "adaptive":
        def progress(row):
            if row["attempted_steps"] % 1000 == 0 or row["action"] in {"rollback_shrink", "verify", "rollback"}:
                print("WARMUP " + json.dumps(row, ensure_ascii=False), flush=True)

        result = adaptive_warmup(engine, warmup_config, boundary_heads=boundary_heads,
                                 node_inflows=node_inflows, gate_openings=current_gates, progress=progress)
        warmup = write_report(args.output, result)
        warmup_seconds = result["wall_seconds"]
        print("WARMUP_RESULT " + json.dumps(warmup, ensure_ascii=False), flush=True)
        if not warmup["converged"]:
            print("WARNING 预热未收敛，按要求继续仿真；超出量和位置见预热说明.md", flush=True)

    with (args.output / "预热阶梯结果.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(warmup), handle, ensure_ascii=False, indent=2, allow_nan=False)

    with (args.output / "预热末态.json").open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe({**engine.get_restart_state(), "warmup_validation": warmup, "restart_contract": state_contract}),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failure: str | None = None
    completed_steps = 0
    run_started = time.perf_counter()

    try:
        for step_index in range(total_steps):
            elapsed = 0.0 if args.fixed_inputs_at_zero else step_index * dt
            step_dt = min(dt, end_time - step_index * dt)
            boundary_heads, node_inflows, current_gates = inputs_at(elapsed)
            state = engine.step(
                boundary_heads=boundary_heads,
                node_inflows=node_inflows,
                gate_openings=current_gates,
                dt=step_dt,
                num_iter=max(int(args.num_iter), 1),
            )
            completed_steps = step_index + 1
            arrays = [engine.model.H_j, engine.model.Q_uk, engine.model.Q_dk]
            if engine.model.n_o:
                arrays.append(engine.model.Q_o)
            if not all(np.isfinite(values).all() for values in arrays):
                raise FloatingPointError(f"第 {completed_steps} 个计算步出现 NaN 或 Inf")

            should_output = completed_steps % steps_per_output == 0 or completed_steps == total_steps
            if not should_output:
                continue

            output_index = int(math.ceil(completed_steps / steps_per_output))
            output_time = float(state["time"])
            node_heads = np.asarray([item["head"] for item in state["nodes"].values()], dtype=float)
            node_depths = np.asarray([item["depth"] for item in state["nodes"].values()], dtype=float)
            channel_up = np.asarray([item["upstream_flow"] for item in state["channels"].values()], dtype=float)
            channel_down = np.asarray([item["downstream_flow"] for item in state["channels"].values()], dtype=float)
            gate_flows = np.asarray(list(state["gate_flows"].values()), dtype=float)

            summary_rows.append({
                "output_index": output_index,
                "time_s": output_time,
                "min_water_level_m": float(node_heads.min()),
                "max_water_level_m": float(node_heads.max()),
                "min_depth_m": float(node_depths.min()),
                "max_depth_m": float(node_depths.max()),
                "negative_depth_count": int((node_depths < -1e-9).sum()),
                "min_channel_upstream_flow_m3s": float(channel_up.min()),
                "max_channel_upstream_flow_m3s": float(channel_up.max()),
                "max_abs_channel_flow_m3s": float(max(np.abs(channel_up).max(), np.abs(channel_down).max())),
                "negative_channel_upstream_flow_count": int((channel_up < 0.0).sum()),
                "min_gate_flow_m3s": float(gate_flows.min()) if len(gate_flows) else 0.0,
                "max_gate_flow_m3s": float(gate_flows.max()) if len(gate_flows) else 0.0,
            })

            for name, values in state["nodes"].items():
                raw_rows.append({
                    "output_index": output_index,
                    "time_s": output_time,
                    "object_type": "SuperJunction",
                    "object_name": name,
                    "water_level_m": values["head"],
                    "water_depth_m": values["depth"],
                    "upstream_flow_m3s": "",
                    "downstream_flow_m3s": "",
                    "gate_flow_m3s": "",
                    "gate_opening_ratio": "",
                })
            for name, values in state["channels"].items():
                raw_rows.append({
                    "output_index": output_index,
                    "time_s": output_time,
                    "object_type": "SuperLink",
                    "object_name": name,
                    "water_level_m": "",
                    "water_depth_m": "",
                    "upstream_flow_m3s": values["upstream_flow"],
                    "downstream_flow_m3s": values["downstream_flow"],
                    "gate_flow_m3s": "",
                    "gate_opening_ratio": "",
                })
            for name, flow in state["gate_flows"].items():
                raw_rows.append({
                    "output_index": output_index,
                    "time_s": output_time,
                    "object_type": "Orifice",
                    "object_name": name,
                    "water_level_m": "",
                    "water_depth_m": "",
                    "upstream_flow_m3s": "",
                    "downstream_flow_m3s": "",
                    "gate_flow_m3s": flow,
                    "gate_opening_ratio": current_gates.get(name, ""),
                })
            print(f"OUTPUT {output_index}/{math.ceil(total_steps / steps_per_output)} time_s={output_time:.0f} "
                  f"head=[{node_heads.min():.3f},{node_heads.max():.3f}] "
                  f"max_abs_Q={max(np.abs(channel_up).max(), np.abs(channel_down).max()):.3f}", flush=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    run_seconds = time.perf_counter() - run_started
    initial_state_status = warmup.get("status", "CONVERGED" if warmup.get("converged") else "NOT_CERTIFIED")
    for row in raw_rows + summary_rows:
        row["initial_state_status"] = initial_state_status
        row["model_coverage"] = "INCOMPLETE" if model_limitations else "SUPPORTED_COMPONENTS"
    write_csv(
        args.output / "Pipedream原始输出.csv",
        raw_rows,
        [
            "output_index", "time_s", "object_type", "object_name",
            "water_level_m", "water_depth_m", "upstream_flow_m3s",
            "downstream_flow_m3s", "gate_flow_m3s", "gate_opening_ratio", "initial_state_status", "model_coverage",
        ],
    )
    write_csv(
        args.output / "输出步汇总.csv",
        summary_rows,
        [
            "output_index", "time_s", "min_water_level_m", "max_water_level_m",
            "min_depth_m", "max_depth_m", "negative_depth_count",
            "min_channel_upstream_flow_m3s", "max_channel_upstream_flow_m3s",
            "max_abs_channel_flow_m3s", "negative_channel_upstream_flow_count",
            "min_gate_flow_m3s", "max_gate_flow_m3s", "initial_state_status", "model_coverage",
        ],
    )

    links = converted["links"]
    conversion_summary = {
        "solver": engine.solver_identity,
        "backend_adaptation": engine.backend_diagnostics,
        "model_name": converted.get("model_name"),
        "section_geometry_mode": converted.get("section_geometry_mode"),
        "model_path": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "calculate_step_s": dt,
        "output_step_s": output_step,
        "configured_end_time_s": end_time,
        "configured_compute_steps": int(math.ceil(end_time / dt)),
        "requested_compute_steps": total_steps,
        "superjunctions": len(converted["superjunctions"]),
        "superlinks": len(converted["superlinks"]),
        "internal_junctions": len(converted["junctions"]),
        "internal_links": len(links),
        "irregular_internal_links": int((links["shape"] == "irregular").sum()),
        "shape_counts": links["shape"].value_counts().to_dict(),
        "transects": len(converted.get("transects", {})),
        "geometry_audit_rows": len(converted.get("section_geometry_audit", [])),
        "geometry_fallbacks": len(converted.get("section_geometry_fallbacks", [])),
        "geometry_issues": len(converted.get("section_geometry_issues", [])),
        "orifices": 0 if converted.get("orifices") is None else len(converted["orifices"]),
        "bleeders": len(converted.get("bleeders", [])),
        "siphons": len(converted.get("siphons_info", [])),
        "pumps": len(converted.get("pumps", [])),
        "turbines": len(converted.get("turbines", [])),
        "time_series": len(converted.get("time_series", {})),
        "topology_repairs": converted.get("topology_repairs", []),
        "conversion_seconds": conversion_seconds,
    }
    run_summary = {
        "solver": engine.solver_identity,
        "backend_adaptation": engine.backend_diagnostics,
        "status": ("COMPLETED_WITH_MODEL_LIMITATIONS" if model_limitations else "COMPLETED")
        if failure is None and completed_steps == total_steps else "FAILED",
        "failure": failure,
        "completed_compute_steps": completed_steps,
        "expected_compute_steps": total_steps,
        "completed_output_steps": len(summary_rows),
        "expected_output_steps": int(math.ceil(total_steps / steps_per_output)),
        "warmup": warmup,
        "warmup_mode": args.warmup_mode,
        "initial_state_status": initial_state_status,
        "model_limitations": model_limitations,
        "physical_model_complete": not model_limitations,
        "minimum_initial_depth_m": args.minimum_initial_depth if args.minimum_initial_depth is not None else
        (7.0 if args.warmup_mode == "fixed" else 0.01),
        "internal_method": engine.model._method,
        "gate_coefficients": str(args.gate_coefficients) if args.gate_coefficients else None,
        "warmup_dt_s": warmup_dt,
        "initial_depth_override_m": args.initial_depth,
        "all_superjunction_depth_m": args.all_superjunction_depth,
        "superjunctions_csv": str(args.superjunctions_csv) if args.superjunctions_csv else None,
        "superjunction_h0_min_m": float(converted["superjunctions"]["h_0"].min()),
        "superjunction_h0_max_m": float(converted["superjunctions"]["h_0"].max()),
        "restart_state": str(args.restart_state) if args.restart_state else None,
        "restart_warmup_steps": max(int(args.restart_warmup_steps), 0),
        "restart_static_arrays_skipped": restart_arrays_skipped,
        "fixed_inputs_at_zero": bool(args.fixed_inputs_at_zero),
        "num_iter": max(int(args.num_iter), 1),
        "warmup_wall_seconds": warmup_seconds,
        "simulation_wall_seconds": run_seconds,
        "raw_output_rows": len(raw_rows),
    }
    with (args.output / "转换摘要.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(conversion_summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (args.output / "运行摘要.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(run_summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(json_safe(run_summary), ensure_ascii=False), flush=True)
    return 0 if failure is None and completed_steps == total_steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
