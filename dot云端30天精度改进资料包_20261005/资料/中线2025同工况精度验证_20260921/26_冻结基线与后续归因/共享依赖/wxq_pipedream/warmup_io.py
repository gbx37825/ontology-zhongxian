"""Configuration precedence and diagnostics for standalone warmup."""

from dataclasses import asdict
import csv
import json
from pathlib import Path

from .warmup import WarmupConfig


def resolve_config(options, compute_dt, path=None, overrides=None):
    values = asdict(WarmupConfig())
    if options.get("ignoreSteadyState") is True:
        values["max_steps"] = 0
    for source, target in (("timeStep", "initial_dt"), ("nStepMax", "max_steps"),
                           ("thresholdZ", "head_tolerance_m")):
        if source in options:
            values[target] = options[source]
    if "timeStep" not in options:
        values["initial_dt"] = compute_dt
    file_values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(file_values, dict):
        raise ValueError("Warmup configuration must be a JSON object")
    values.update(file_values)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return WarmupConfig.from_mapping(values)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, result):
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "预热收敛轨迹.csv", result["history"])
    _write_csv(output / "预热节点水量诊断.csv", result["locations"])
    summary = {key: value for key, value in result.items() if key not in {"history", "locations"}}
    (output / "预热诊断.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    labels = {"head_change_reference_m": "最大水位变化", "flow_change_reference_m3s": "最大流量变化",
              "storage_abs_sum_m3s": "各节点蓄量变化率绝对值之和",
              "steady_flow_imbalance_m3s": "最大节点进出流量差",
              "max_continuity_residual_m3s": "最大局部水量不闭合量",
              "global_continuity_residual_m3s": "全网水量不闭合量"}
    status = "已收敛" if result["converged"] else "未确认收敛，继续仿真"
    lines = ["# 预热结果", "", f"本次状态为 **{status}**。", "",
             f"共尝试{result['steps_run']}步，其中{result['rejected_steps']}步被回退。"
             f"实际耗时{result['wall_seconds']:.3f}秒。", "",
             f"正常计算步长为{result['reference_dt_s']:g}秒，末态复核完成"
             f"{result['verification_steps_run']}步。加速推进只用于寻找初态，不代表工程实际历时。", "",
             "下表优先列出正常步长复核期间的最差值。若未执行复核，则列最后一次有效试算的诊断，"
             "不能把它当成已经通过复核。水位和流量变化按正常计算步长折算。", "",
             "| 指标 | 检查值 | 门槛 | 超出量 | 单位 | 最差位置 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, metric in result["metrics"].items():
        location = str(metric["location"] or "无").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {labels[name]} | {metric['value']:.8g} | {metric['threshold']:.8g} | "
                     f"{metric['excess']:.8g} | {metric['unit']} | {location} |")
    if not result["metrics"]:
        lines += ["", "没有完成有效预热计算，收敛程度尚未测得。"]
    if result["verification_steps_shortfall"]:
        lines += ["", f"距要求的正常步长复核还差{result['verification_steps_shortfall']}步，不能确认初态已经稳定。"]
    lines += ["", "蓄量变化率为正表示存水增加，为负表示存水减少。各节点的正负变化不会在绝对值总和中相互抵消。",
              "", "汇总量标出最大贡献节点；它的数值仍代表全网汇总，不能当作该单个节点的值。",
              "", "详细位置见同目录的预热节点水量诊断.csv，分档步长和回退记录见预热收敛轨迹.csv。",
              "", "未收敛时程序按要求继续计算，后续输出保留未收敛标记。"
              "这不等于水位已符合实测，也不能据此宣称精度达标。"]
    (output / "预热说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
