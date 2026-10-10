"""只汇总已完成产物与文件指纹，不导入或重跑水力模型。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import shutil

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parent
FROZEN = ROOT / "26_冻结基线与后续归因"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


summary = read(OUT / "条件性在线对照汇总.json")
assert summary["completed_count"] == summary["audited_completed_count"] == 35
assert summary["failed_terminal_count"] == 0
assert summary["artifact_audit_failures"] == []
assert not summary["baseline_replaced"] and not summary["engineering_acceptance"]
issues = summary["common_completed_issue_times_s"]
assert len(issues) == 7

# 同一来源文件只重新读一次；逐分支与自己的冻结哈希比较。
current_hashes = {}
case_audits = []
for issue in issues:
    paired = []
    for method in "FPWSH":
        p = RUN / "06_配对运行" / f"起点_{issue}s" / method
        result = read(p / "结果.json")
        progress = read(p / "进度.json")
        config = read(p / "运行配置.json")
        assert result["status"] == progress["status"] == "COMPLETED"
        assert result["completed_adaptation_cycles"] == 12
        assert result["forecast_rows"] == 1296
        assert not result["future_observations_used_by_algorithm"]
        assert not result["future_parameter_updates"]
        assert not result["future_state_updates"]
        assert result["source_hashes_unchanged"]
        for path, expected in config["source_hashes"].items():
            if path not in current_hashes:
                current_hashes[path] = sha(Path(path))
            assert current_hashes[path] == expected, path
        for module in config["runtime_identity"]["modules"]:
            path = module["file"]
            if path not in current_hashes:
                current_hashes[path] = sha(Path(path))
            assert current_hashes[path] == module["sha256"], path
            assert "/water/" not in path and "corelib" not in path.lower()
        assert sha(Path(config["common_state_path"])) == config["common_state_sha256"]
        paired.append({k: config[k] for k in [
            "common_state_sha256", "initial_theta_sha256", "initial_states_sha256"]})
        case_audits.append({"issue_s": issue, "method": method,
                            "status": "COMPLETED", "current_sources_match": True,
                            "result_sha256": sha(p / "结果.json")})
    assert all(x == paired[0] for x in paired)

manifest_path = FROZEN / "冻结文件校验.json"
manifest = read(manifest_path)
freeze_checks = []
for relative, expected in manifest.items():
    path = FROZEN / relative
    actual = sha(path)
    assert actual == expected["sha256"] and path.stat().st_size == expected["bytes"], relative
    freeze_checks.append({"path": relative, "sha256": actual, "unchanged": True})
assert len(freeze_checks) == 107
save(OUT / "冻结基线终检.json", {
    "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "status": "UNCHANGED", "file_count": len(freeze_checks),
    "manifest_sha256": sha(manifest_path), "files": freeze_checks})

with (OUT / "冻结更新预测分组误差.csv").open() as f:
    metrics = list(csv.DictReader(f))
metrics = [x for x in metrics if x["population"] == "全部共同完成起点"]


def score(method, lead, group="旧留出9", track="frozen_at_issue", metric="MAE"):
    row, = [x for x in metrics if x["method"] == method and int(x["lead_s"]) == lead
            and x["score_group"] == group and x["track"] == track]
    return float(row[metric])


labels = {"F": "固定参数集合", "P": "逐次参数微调", "W": "窗口参数更新",
          "S": "仅校正状态", "H": "参数与状态联合更新"}
table = ["|方法|30分钟 MAE|1小时 MAE|3小时 MAE|6小时 MAE|6小时 RMSE|",
         "|---|---:|---:|---:|---:|---:|"]
for method in "FPWSH":
    values = [score(method, lead) for lead in [1800, 3600, 10800, 21600]]
    values.append(score(method, 21600, metric="RMSE"))
    table.append("|" + f"{method} {labels[method]}|" + "|".join(f"{v:.6f}" for v in values) + "|")
table = "\n".join(table)
gain = score("F", 21600) - score("S", 21600)
flow_f = score("F", 21600, "兰河测流Q16")
flow_s = score("S", 21600, "兰河测流Q16")
primary_check = read(OUT / "主进程主要成绩重算.json")
assert abs(primary_check["S_6h_MAE_improvement_m"] - gain) < 1e-12
independent = read(OUT / "独立主结论终检.json")
assert independent["W_windows"]["total_windows"] == 42
assert independent["W_windows"]["all_planned_and_actual_one_hour"]

report = f"""# 条件性在线方法对照结论

## 结论

**状态澄清：完成的是本轮条件性方法对照，基础水动力模型的精度修复目标尚未完成。原模型问题未全部解决，不能将本报告或目标工具此前的完成标记解释为模型已经搞准。**

本轮预定的7个起点、5种方法共35组试验全部完成，35组产物核验通过，未剔除起点或失败分支。仅校正状态的S组在旧9个留出测点上表现最好，6小时水位MAE由{score('F',21600):.6f}米降到{score('S',21600):.6f}米，降低{gain:.6f}米，约{gain/score('F',21600)*100:.2f}%。P、W、H三组的留出测点平均MAE均高于F。

这是用户授权的条件性在线方法比较。F是本轮相同初态、相同参数集合的固定参数对照，不是26目录冻结基线的替换版本。原38轮全线参数转移和39轮水量核验未通过的结论保留。冻结基线107个文件未变，未替换基线、修改正式链或部署。

## 留出测点的预测成绩

主轨在起报时刻固定外部水位、流量、闸门开度和分水；起报后不再使用观测、不再更新参数或状态。以下均为7个共同起点的集合均值预测误差，每个提前量63个误差样本，单位为米。

{table}

按起点分别配对，两个输入轨道、四个提前量的留出MAE改善数均为S 7/7、P 0/7、W 0/7、H 1/7。已知历史输入的次轨6小时MAE依次为F {score('F',21600,track='known_historical_inputs'):.6f}、P {score('P',21600,track='known_historical_inputs'):.6f}、W {score('W',21600,track='known_historical_inputs'):.6f}、S {score('S',21600,track='known_historical_inputs'):.6f}、H {score('H',21600,track='known_historical_inputs'):.6f}米，MAE优劣方向相同。

不能将H的平均MAE较高写成所有指标都差。H在30分钟和1小时的RMSE优于F，在全部提前量的最大绝对误差也优于F。S在旧9点的MAE、RMSE、最大绝对误差三个指标上均优于F。

![五方法预测提前量对照](07_结果汇总/五方法预测提前量对照.png)

## 全线和测流检验

主轨6小时，按43个安装位置合并后，水位MAE由F的{score('F',21600,'安装位置43'):.6f}米变为S的{score('S',21600,'安装位置43'):.6f}米、H的{score('H',21600,'安装位置43'):.6f}米。H在该合并指标上优于S，但包含34个参与更新的测点，不能据此推翻预定留出9点的结论。

S也没有改善所有位置。按每个位置7个起点的6小时MAE比较，43个安装位置中26个改善超过1厘米，1个变差超过1厘米，16个变化在1厘米内；变差位置为JZZ45闸前，增加约1.048厘米。按118个闸节点口径，48个改善、2个变差、68个变化在1厘米内，其中JZZ14闸前增加约2.624厘米。这两个空间口径不能相加当作独立测点数。

兰河300091米测流断面未参与在线观测更新。其6小时流量MAE由{flow_f:.6f}增至{flow_s:.6f}立方米/秒，增加{flow_s-flow_f:.6f}立方米/秒；RMSE从{score('F',21600,'兰河测流Q16',metric='RMSE'):.6f}降至{score('S',21600,'兰河测流Q16',metric='RMSE'):.6f}立方米/秒，最大绝对误差也降低。每个提前量仅7个单站流量样本，结论应保留这些指标差异。

## 状态校正的等效蓄量

每个独立6小时适应期，S组状态更新产生的集合平均等效蓄量增量为约+53.54万至+89.04万立方米；H为约+49.27万至+77.10万立方米。F、P、W没有直接状态校正蓄量。这说明水位改善同时伴随模型内部水量修正，尚不能解释为物理参数已经率定正确。

上述数值按各次独立起算分别计账，不相加称作整月补水，不等同现场补水量。图中误差线为50个成员范围，不是置信区间。

![状态等效水量与兰河流量检验](07_结果汇总/状态等效水量与兰河流量检验.png)

## 试验条件和未覆盖范围

五组统一50个成员、119个参数、相同随机种子和各起点的共同初态。每次先适应6小时，再克隆两个6小时预测分支。参数中心仅在POOL16糙率和JZZ16闸流系数采用38轮的条件性倍率；其余保持原参数中心。参数允许域为原物理界与中心±5%区间的交集，分析增量与补散度分账。

H由同一个先验集合、同一组扰动观测同时计算参数和状态增量，不把同一观测依次交给两个独立滤波器。S/H的状态限幅为每成员整批次水位2厘米、流量3立方米/秒，不等同于旧B01的逐观测处理。

W的42个实际窗口全部为1小时。预定的过去输入缓变判据未选到3小时或6小时窗口，不能据此评价较长窗口的效果。6/3/1小时窗口选择逻辑已自检，逻辑通过不代表真实长窗试验已覆盖。

本轮是7个已见历史起点的短期共同起算比较，不是连续30天在线稳定性验证，也不是新盲测。主轨采用起报时刻输入持值这一明确假设；次轨使用已知历史输入，不代表未知未来输入预报。

原始断面输入轮廓被水位超出的现象仍存在，采样检查的最大超出约2.622米。输入轮廓尚不能直接等同真实渠顶，不能把它解释为现场漫顶或把数值完成当作工程适用性通过。原求解器失败减步规则保留；35组完成表示无终态失败，不表示过程中没有减步重试。

## 完成核验与交付

全部45,360条预测评分记录已经汇总；主进程从实际预测CSV重算两轨、四个提前量、五方法的40行主要成绩，另有独立复核与NPZ成员均值/离散度核验。各分支来源指纹、起点配对、参数有效域、观测消费、防止未来观测进入算法以及状态等效蓄量账均已检查。

首轮定位坐标断言失败的5个启动案例均为0个适应周期，另行归档。40实验层统一参数和状态定位坐标后重新运行，未修改冻结B01/B02、水力几何、共同初态或方法阈值。该启动问题未作为某种方法的预测失败计分。

原T01至T11已逐项回查。在本轮现有资料条件性试验范围内，没有遗漏的必做试验，正结论、负结论和前提不满足时的停止记录均保留。T10现场真参数与全线替换资格、T11原工程前提仍未通过。这里的完成只指方法试验及结果交付；先修复基础模型、在不依赖在线状态校正时达到水位和流量精度要求的主目标仍未完成。后续应回到基础模型问题，不以新增同化试验替代修复。

当前结论支持将S保留为后续研究候选，不支持用P/W/H替换本轮固定参数对照，也不支持把S直接作为工程验收通过结果。本轮不再根据已见成绩追调参数。

主要文件如下。

- [试验定义与复现入口](试验定义与复现入口.md)
- [全部分组评分](07_结果汇总/冻结更新预测分组误差.csv)
- [配对误差变化](07_结果汇总/相对F配对误差变化.csv)
- [各起点状态等效蓄量](07_结果汇总/各起点独立六小时同化等效水量.json)
- [主进程成绩重算](07_结果汇总/主进程主要成绩重算.json)
- [独立主结论终检](07_结果汇总/独立主结论终检.json)
- [最终来源与完成核验](07_结果汇总/主进程完成核验.json)
- [原测试清单完成审计](07_结果汇总/原测试清单完成审计.json)
- [冻结基线终检](07_结果汇总/冻结基线终检.json)
- [预测图矢量版](07_结果汇总/五方法预测提前量对照.svg)
- [水量与流量图矢量版](07_结果汇总/状态等效水量与兰河流量检验.svg)
"""
(RUN / "条件性在线对照结论.md").write_text(report)

removed = []
for path in [Path("/private/tmp/hydro40_mpl"), Path("/private/tmp/hydro40_cache")]:
    if path.exists():
        shutil.rmtree(path)
        removed.append(str(path))

audited_paths = [OUT / name for name in ["条件性在线对照汇总.json", "主进程主要成绩重算.json",
                 "独立主结论终检.json", "五方法预测提前量对照.png", "五方法预测提前量对照.svg",
                 "状态等效水量与兰河流量检验.png", "状态等效水量与兰河流量检验.svg",
                 "冻结基线终检.json", "原测试清单完成审计.json"]] + [RUN / "条件性在线对照结论.md"]
save(OUT / "主进程完成核验.json", {
    "status": "VERIFIED", "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "current_conditional_case_scope": "7 origins x 5 methods; each 6h adaptation plus two frozen-update 6h forecasts",
    "cases_expected": 35, "cases_completed": 35, "terminal_failures": 0,
    "all_7_origins_retained": True, "paired_initial_states_and_parameters_match": True,
    "case_checks": case_audits, "distinct_current_source_hashes": current_hashes,
    "frozen_107_files_unchanged": True,
    "primary_40_metric_rows_independently_recomputed": primary_check["primary_metrics_match"],
    "total_forecast_rows": 45360, "future_observation_or_analysis_update_after_issue": False,
    "no_old_water_or_corelib_hydraulic_module_loaded": True,
    "figures_visually_checked": {"五方法预测提前量对照.png": "Final figure: text and legends legible, no overlap or clipping",
                                "状态等效水量与兰河流量检验.png": "Final figure: text and legends legible, no overlap or clipping"},
    "removed_temporary_paths": removed,
    "deliverable_sha256": {str(x.relative_to(RUN)): sha(x) for x in audited_paths},
    "baseline_replaced": False, "engineering_acceptance_passed": False,
    "base_hydraulic_model_accuracy_objective_complete": False,
    "continuous_30day_online_run_completed_this_round": False,
    "scope_of_goal_completion": "T01-T11 under the user's latest authorization for conditional tests with existing evidence; not resolution of all engineering issues.",
    "requested_test_scope_complete": read(OUT / "原测试清单完成审计.json")["requested_test_scope_complete"],
    "remaining_required_work_in_authorized_test_scope": read(OUT / "原测试清单完成审计.json")["remaining_required_work"]
})
print(json.dumps({"status": "VERIFIED", "cases": len(case_audits),
                  "source_files": len(current_hashes), "frozen_files": len(freeze_checks),
                  "report": str(RUN / "条件性在线对照结论.md"), "removed_temp": removed}, ensure_ascii=False))
