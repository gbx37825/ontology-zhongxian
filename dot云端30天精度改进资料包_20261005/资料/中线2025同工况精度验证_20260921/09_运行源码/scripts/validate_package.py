from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def execute(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)


def check_quick_conversion(summary: dict) -> None:
    expected = {
        "section_geometry_mode": "safe",
        "superjunctions": 2,
        "superlinks": 1,
        "internal_junctions": 3,
        "internal_links": 2,
        "orifices": 0,
        "transects": 0,
        "geometry_audit_rows": 2,
        "geometry_fallbacks": 0,
        "geometry_issues": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError(f"示例转换不一致: {key}={summary.get(key)!r}, expected={value!r}")
    if summary["shape_counts"] != {"rect_open": 2}:
        raise AssertionError(f"示例断面形状异常: {summary['shape_counts']}")


def check_middle_conversion(summary: dict, manifest: dict) -> None:
    expected = manifest["latest_internal_baseline"]["conversion"]
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError(f"中线转换基线不一致: {key}={summary.get(key)!r}, expected={value!r}")


def compare_tables(reference: pd.DataFrame, actual: pd.DataFrame, *,
                   keys: list[str], tolerances: dict[str, float],
                   allow_missing: bool = False, expected_rows: int | None = None) -> dict[str, float]:
    if len(reference) != len(actual) or not len(actual):
        raise AssertionError("基线和本次结果的行数不一致或没有数据")
    if expected_rows is not None and len(actual) != expected_rows:
        raise AssertionError(f"结果行数应为{expected_rows}，实际为{len(actual)}")
    for frame in (reference, actual):
        if not set(keys + list(tolerances)).issubset(frame.columns):
            raise AssertionError("结果缺少时间、对象标识或比较字段")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise AssertionError("结果存在空标识或重复的时间与对象记录")
    reference = reference.set_index(keys).sort_index()
    actual = actual.set_index(keys).sort_index()
    if not actual.index.equals(reference.index):
        raise AssertionError("输出时间轴或对象标识与基线不一致")
    differences = {}
    for column, tolerance in tolerances.items():
        expected = reference[column].to_numpy(dtype=float)
        values = actual[column].to_numpy(dtype=float)
        missing = np.isnan(expected)
        if not np.array_equal(missing, np.isnan(values)) or (missing.any() and not allow_missing):
            raise AssertionError(f"{column}出现缺失数据")
        valid = ~missing
        if not np.isfinite(expected[valid]).all() or not np.isfinite(values[valid]).all():
            raise AssertionError(f"{column}出现非有限值")
        difference = float(np.max(np.abs(values[valid] - expected[valid]))) if valid.any() else 0.0
        differences[column] = difference
        if difference > tolerance:
            raise AssertionError(f"{column}基线差值{difference}超过容差{tolerance}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate standalone WXQ Pipedream package")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--reference-raw", type=Path)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs" / "validation")
    args = parser.parse_args()
    manifest = json.loads((PACKAGE_ROOT / "baseline" / "基线清单.json").read_text(encoding="utf-8"))
    model = args.model.resolve() if args.model else PACKAGE_ROOT / "examples" / "示例模型.json"
    if args.mode == "full" and args.model is None:
        raise ValueError("完整验证必须通过--model提供待验证WXQ JSON")
    args.output.mkdir(parents=True, exist_ok=True)

    conversion_output = args.output / "conversion"
    execute([
        sys.executable, "scripts/convert_wxq.py", str(model), str(conversion_output),
        "--section-geometry-mode", "safe", "--all-superjunction-depth", "7",
        "--minimum-initial-depth", "7",
    ])
    conversion = json.loads((conversion_output / "conversion_summary.json").read_text(encoding="utf-8"))
    if conversion["superjunction_h0_min_m"] != 7.0 or conversion["superjunction_h0_max_m"] != 7.0:
        raise AssertionError("SuperJunction h0没有全部设为7m")
    if args.mode == "quick":
        check_quick_conversion(conversion)
    else:
        expected_hash = manifest["latest_internal_baseline"]["model_sha256"]
        actual_hash = hashlib.sha256(model.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            check_middle_conversion(conversion, manifest)

    simulation_output = args.output / ("full_run" if args.mode == "full" else "quick_run")
    command = [
        sys.executable, "scripts/run_simulation.py", str(model), str(simulation_output),
        "--compute-dt", "15", "--warmup-dt", "15",
        "--warmup-mode", "fixed", "--backend", "standard",
        "--minimum-initial-depth", "7",
        "--section-geometry-mode", "safe", "--all-superjunction-depth", "7",
    ]
    if args.mode == "quick":
        command.extend(["--warmup-max-steps", "100", "--max-compute-steps", "1", "--end-time", "15"])
    else:
        command.extend([
            "--warmup-head-tol", "0.00001", "--warmup-flow-tol", "0.01",
            "--warmup-checkpoints", "40000", "--end-time", "259200",
        ])
    execute(command)

    simulation = json.loads((simulation_output / "运行摘要.json").read_text(encoding="utf-8"))
    if simulation["status"] != "COMPLETED":
        raise AssertionError(f"仿真未完成: {simulation}")
    if simulation["superjunction_h0_min_m"] != 7.0 or simulation["superjunction_h0_max_m"] != 7.0:
        raise AssertionError("仿真实际SuperJunction h0不是全7m")

    result = {
        "mode": args.mode,
        "passed": True,
        "model_name": model.name,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "conversion": {key: value for key, value in conversion.items() if key != "model_path"},
        "simulation": simulation,
    }
    if args.mode == "full" and args.reference_summary:
        reference = pd.read_csv(args.reference_summary)
        actual = pd.read_csv(simulation_output / "输出步汇总.csv")
        tolerances = {
            "min_water_level_m": 1e-4,
            "max_water_level_m": 1e-4,
            "min_depth_m": 1e-4,
            "max_depth_m": 1e-4,
            "max_abs_channel_flow_m3s": 1e-3,
            "min_gate_flow_m3s": 1e-3,
            "max_gate_flow_m3s": 1e-3,
            "min_channel_upstream_flow_m3s": 1e-3,
            "max_channel_upstream_flow_m3s": 1e-3,
            "negative_depth_count": 0,
            "negative_channel_upstream_flow_count": 0,
        }
        result["reference_differences"] = compare_tables(
            reference, actual, keys=["output_index", "time_s"],
            tolerances=tolerances, expected_rows=36,
        )
    if args.mode == "full" and args.reference_raw:
        reference = pd.read_csv(args.reference_raw)
        actual = pd.read_csv(simulation_output / "Pipedream原始输出.csv")
        result["reference_raw_differences"] = compare_tables(
            reference, actual, keys=["output_index", "time_s", "object_type", "object_name"],
            tolerances={
                "water_level_m": 1e-4, "water_depth_m": 1e-4,
                "upstream_flow_m3s": 1e-3, "downstream_flow_m3s": 1e-3,
                "gate_flow_m3s": 1e-3, "gate_opening_ratio": 1e-12,
            }, allow_missing=True,
        )
        result["reference_raw_rows"] = len(actual)

    report = args.output / "package_validation.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "mode": args.mode, "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
