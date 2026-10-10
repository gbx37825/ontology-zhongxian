"""纯转换、已保存状态及原函数代数核验；不构建求解模型，不调用step。"""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
B = R / "26_冻结基线与后续归因"
D = B / "共享依赖"
END = R / "41_基础模型深入检查与修复/10_端面几何锁参验证/endpoint"
sys.dont_write_bytecode = True
sys.path.insert(0, str(D))
from wxq_pipedream.wxq_to_superlink import parse_wxq_json

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(path, names, context):
    definitions = [n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions} == set(names)
    for node in definitions:
        node.decorator_list = []
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), context)
    return context


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    sources = [B / "共享输入/C07.json", B / "共享输入/Pipedream闸流系数.json",
               B / "共享输入/2025开度归一化上限.json",
               D / "wxq_pipedream/wxq_to_superlink.py", D / "pipedream_solver/superlink.py",
               D / "pipedream_solver/nsuperlink.py", D / "pipedream_solver/gate_hydraulics.py",
               D / "并联结构物理修复.py", D / "水力运行脚本/试验公共.py",
               END / "半小时水位流量.csv", END / "实际初始化_0s.npz", END / "末态.npz",
               HERE / "梅河闸底原表独立复核.json", HERE / "梅河四孔闸底隔离候选.py"]
    before = {str(p): sha(p) for p in sources}
    source = json.loads((B / "共享输入/C07.json").read_text())["中线全线"]
    gate_max = {k: float(v.get("b", 1) or 1) for k, v in source["baseData"]["gates"].items()}
    gate_max.update(json.loads((B / "共享输入/2025开度归一化上限.json").read_text()))
    c = parse_wxq_json(source, gate_max_openings=gate_max,
                       section_geometry_mode="safe", minimum_initial_depth_m=7.)
    coefficient = json.loads((B / "共享输入/Pipedream闸流系数.json").read_text())
    c["orifices"].loc[:, "C"] = c["orifices"].name.map(coefficient).fillna(c["orifices"].C)
    sj, ori = c["superjunctions"].set_index("id", drop=False), c["orifices"]

    # 检查所有实际端部的绝对床高，只加一次offset，包括solver反向。
    end_rows = []
    for name, meta in c["channel_discretization"].items():
        k = int(meta["superlink_id"])
        sl = c["superlinks"].loc[c["superlinks"].id == k].iloc[0]
        for side, point, node, offset in (("u", meta["sections"][0], sl.sj_0, sl.in_offset),
                                          ("d", meta["sections"][-1], sl.sj_1, sl.out_offset)):
            effective = float(sj.loc[int(node), "z_inv"] + offset)
            interior = float(c["junctions"].loc[int(point["junction_id"]), "z_inv"])
            end_rows.append(dict(name=name, side=side, superlink_id=k,
                                 flow_sign=float(meta["flow_sign"]), node_id=int(node),
                                 offset_m=float(offset), effective_bed_m=effective,
                                 source_endpoint_bed_m=float(point["z_inv"]),
                                 internal_endpoint_bed_m=interior,
                                 difference_m=effective - float(point["z_inv"])))
    pd.DataFrame(end_rows).to_csv(HERE / "全部端部床高组合核验.csv", index=False)
    assert max(abs(r["difference_m"]) for r in end_rows) < 1e-12
    assert all(r["effective_bed_m"] == r["internal_endpoint_bed_m"] for r in end_rows)
    gate_rows = []
    for row in ori.itertuples():
        raw = source["baseData"]["gates"][row.name]
        effective = float(sj.loc[row.sj_0, "z_inv"] + row.z_o)
        gate_rows.append(dict(name=row.name, id=int(row.id), source_gate_bottom_m=float(raw["zb"]),
                              upstream_node_bottom_m=float(sj.loc[row.sj_0, "z_inv"]),
                              z_o_m=float(row.z_o), effective_gate_bottom_m=effective,
                              effective_minus_source_m=effective - float(raw["zb"])))
    pd.DataFrame(gate_rows).to_csv(HERE / "全部门孔闸底组合核验.csv", index=False)
    mismatches = [r for r in gate_rows if abs(r["effective_minus_source_m"]) > 1e-12]
    assert len(mismatches) == 4 and all(r["name"].startswith("JZZ20-") for r in mismatches)

    # 装载原闸流函数体。仅移除JIT装饰，算式原样；没有solver实例或水力时间步。
    gate = bind(D / "pipedream_solver/gate_hydraulics.py",
                ["rectangular_gate_discharge", "rectangular_gate_terms"], {"np": np})
    discharge, terms = gate["rectangular_gate_discharge"], gate["rectangular_gate_terms"]
    targets = ori.loc[ori.name.str.startswith("JZZ20-")]
    assert len(targets) == 4
    input_series = {}
    for g in targets.itertuples():
        series = [v for v in c["time_series"].values()
                  if v["target_type"] == "gates" and v["target_name"] == g.name]
        assert len(series) == 1
        v = series[0]
        input_series[g.name] = (np.asarray(v["times_s"], float),
                                np.clip(np.asarray(v["values"], float) / g.y_max, 0., 1.))
    times = input_series[targets.iloc[0]["name"]][0]
    assert len(times) == 1441 and all(np.array_equal(t, times) for t, u in input_series.values())
    df = pd.read_csv(END / "半小时水位流量.csv")
    df = df.loc[df.station == "JZZ20"].copy()
    assert len(df) == 1440 and np.array_equal(df.time_s, times[1:])
    step_start = {}
    chunk_files = sorted((END / "逐步残差分块").glob("*.npz"))
    for p in chunk_files:
        with np.load(p) as z:
            selected = np.isin(z["time_s"], df.time_s)
            step_start.update(zip(z["time_s"][selected], z["interval_start_s"][selected]))
    assert len(step_start) == 1440
    initial = np.load(END / "实际初始化_0s.npz")
    up, down = int(targets.iloc[0].sj_0), int(targets.iloc[0].sj_1)

    def algebra(t, forcing_t, hu, hd, native_q, category):
        q0 = q1 = 0.
        margins, openings = [], []
        for g in targets.itertuples():
            ts, us = input_series[g.name]
            opening = float(g.y_max * np.interp(forcing_t, ts, us))
            q0 += discharge(hu - 116.723, hd - 116.723, opening, g.g2, g.C)
            q1 += discharge(hu - 115.521, hd - 115.521, opening, g.g2, g.C)
            margins.append(min(hu, hd) - 116.723 - opening)
            openings.append(opening)
        return dict(time_s=float(t), forcing_time_s=float(forcing_t), category=category,
                    upstream_head_m=float(hu), downstream_head_m=float(hd),
                    max_opening_m=max(openings), minimum_old_submergence_margin_m=min(margins),
                    original_algebra_flow_m3s=float(q0), corrected_algebra_flow_m3s=float(q1),
                    corrected_minus_original_flow_m3s=float(q1-q0),
                    saved_native_flow_m3s=None if native_q is None else float(native_q),
                    original_minus_saved_flow_m3s=None if native_q is None else float(q0-native_q))

    actual = [algebra(r.time_s, step_start[r.time_s], r.forebay_m, r.afterbay_m,
                      r.gate_flow_m3s, "accepted_halfhour_end_with_actual_last_step_t0")
              for r in df.itertuples()]
    grid = [algebra(0., 0., initial["H_j"][up], initial["H_j"][down],
                    None, "input_grid_time_with_saved_head")]
    grid.extend(algebra(r.time_s, r.time_s, r.forebay_m, r.afterbay_m,
                        None, "input_grid_time_with_saved_head") for r in df.itertuples())
    pd.DataFrame(actual).to_csv(HERE / "梅河1440末步同源闸流代数核查.csv", index=False)
    pd.DataFrame(grid).to_csv(HERE / "梅河1441输入时刻闸流代数核查.csv", index=False)
    assert max(abs(r["original_minus_saved_flow_m3s"]) for r in actual) < 1e-9
    assert all(r["corrected_minus_original_flow_m3s"] == 0. for r in actual + grid)
    assert min(r["minimum_old_submergence_margin_m"] for r in actual + grid) > 0

    # API容许负offset，检验实际terms中的bed+offset及线性化，不只手写公式。
    examples = []
    n = len(ori)
    for case, h_up, h_down, opening in [("submerged", 122., 121., 3.),
                                      ("partly_submerged", 119., 118., 3.),
                                      ("reverse_partly_submerged", 118., 119., 3.)]:
        outputs = []
        for bottom in [116.723, 115.521]:
            heads = np.full(len(sj), 120.); heads[up] = h_up; heads[down] = h_down
            offsets = ori.z_o.to_numpy(float).copy()
            ids = targets.id.to_numpy(int)
            offsets[ids] = bottom - sj.loc[up, "z_inv"]
            controls = np.zeros(n); controls[ids] = opening / targets.y_max.to_numpy()
            q, a, b, chi = (np.zeros(n) for _ in range(4))
            terms(ids, heads, ori.sj_0.to_numpy(int), ori.sj_1.to_numpy(int),
                  sj.z_inv.to_numpy(float), offsets, ori.y_max.to_numpy(float),
                  ori.g2.to_numpy(float), ori.C.to_numpy(float), np.zeros(n, bool), controls,
                  q, a, b, chi, True)
            assert np.isfinite(np.r_[q, a, b, chi]).all()
            assert abs(q[ids].sum() - 4*discharge(h_up-bottom, h_down-bottom, opening,
                                                 7., float(targets.iloc[0].C))) < 1e-10
            residual = a[ids]*h_up + b[ids]*h_down + chi[ids] - q[ids]
            assert abs(residual).max() < 1e-10
            outputs.append(dict(bottom_m=bottom, offset_m=float(offsets[ids[0]]),
                                total_Q_m3s=float(q[ids].sum()),
                                linearization_identity_max_error_m3s=float(abs(residual).max())))
        examples.append(dict(case=case, upstream_head_m=h_up, downstream_head_m=h_down,
                             opening_m=opening, original=outputs[0], candidate=outputs[1],
                             delta_Q_m3s=outputs[1]["total_Q_m3s"]-outputs[0]["total_Q_m3s"]))
    assert examples[0]["delta_Q_m3s"] == 0 and examples[1]["delta_Q_m3s"] > 0
    assert abs(examples[1]["delta_Q_m3s"] + examples[2]["delta_Q_m3s"]) < 1e-10

    # 隔离候选用无求解能力的命名空间验证，仅四孔z_o变化。
    spec = importlib.util.spec_from_file_location("_meihe_isolated_candidate", HERE / "梅河四孔闸底隔离候选.py")
    hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
    cc = copy.deepcopy(c)
    fake = types.SimpleNamespace(_z_o=ori.z_o.to_numpy(float).copy(),
                orifices=ori.copy(deep=True), _J_uo=ori.sj_0.to_numpy(int).copy(),
                _J_do=ori.sj_1.to_numpy(int).copy(), _z_inv_j=sj.z_inv.to_numpy(float).copy(),
                _Co=ori.C.to_numpy(float).copy(), _n_ik=c["links"].n.to_numpy(float).copy())
    for key in initial.files:
        setattr(fake, key, initial[key].copy())
    saved = copy.deepcopy(vars(fake))
    manifest = hook.restore_meihe_gate_bottom(fake, cc, source)
    changed_ids = np.flatnonzero(saved["_z_o"] != fake._z_o).tolist()
    assert changed_ids == targets.id.tolist()
    for key in saved:
        if key not in ["_z_o", "orifices"]:
            assert np.array_equal(saved[key], getattr(fake, key))
    assert ori.drop(columns="z_o").equals(cc["orifices"].drop(columns="z_o"))
    for key in ["superjunctions", "superlinks", "links", "junctions"]:
        assert c[key].equals(cc[key])
    bad_source = copy.deepcopy(source)
    bad_source["baseData"]["gates"][hook.GATE_NAMES[-1]]["zb"] += .01
    try:
        hook._changes(c, bad_source)
        raise AssertionError("来源变化未拦截")
    except ValueError:
        pass

    # 原端面损失及并联摩阻对正反流的符号，直接原函数纯代数核验。
    ns = bind(D / "pipedream_solver/nsuperlink.py", ["gamma_uk", "gamma_dk"],
              {"np": np, "safe_divide_vec": lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)})
    signed = []
    for q in [-100., 100.]:
        a, loss = 60., .20151121
        gu = float(ns["gamma_uk"](np.array([q]), np.array([loss]), np.array([a]))[0])
        gd = float(ns["gamma_dk"](np.array([q]), np.array([loss]), np.array([a]))[0])
        drop = loss * q * abs(q)/(2*9.81*a*a)
        assert abs(-gu*q-drop) < 1e-14 and abs(gd*q-drop) < 1e-14
        signed.append(dict(Q_m3s=q, upstream_Hsuper_minus_Hinternal_m=-gu*q,
                           downstream_Hinternal_minus_Hsuper_m=gd*q,
                           signed_head_loss_m=drop, dissipation_sign_correct=q*drop >= 0))
    parallel = bind(D / "并联结构物理修复.py", ["_correct_section", "_link_coeffs"], {"np": np})
    radius_max_error = loss_max_error = friction_max_error = 0.
    for row in c["siphons_info"]:
        count, width, height = row["n_orifice"], row["width"], row["height"]
        assert len(c["links"].loc[c["links"].k == row["superlink_id"]]) == 1
        for fraction in [.5, 1.]:
            y = fraction*height; area = np.array([count*width*y]); p = np.zeros(1); rh = np.zeros(1)
            parallel["_correct_section"](None, area, p, rh, np.array([0]),
                                          np.array([count]), np.array([width]), np.array([height]))
            single_p = width+2*y if fraction < 1 else 2*(width+height)
            single_r = width*y/single_p
            radius_max_error = max(radius_max_error, abs(float(rh[0])-single_r))
            for q in [-100., 100.]:
                fake_loss = types.SimpleNamespace(_parallel_native_class=types.SimpleNamespace(link_coeffs=lambda *a, **kw: None),
                              _parallel_i=np.array([0]), _A_ik=area, _Q_ik=np.array([q]),
                              _parallel_Ki=np.array([row["loss_coeff"]]), _b_ik=np.zeros(1))
                parallel["_link_coeffs"](fake_loss)
                actual_loss = fake_loss._b_ik[0]*q/(9.81*area[0])
                single_q, single_a = q/count, width*y
                exact_loss = row["loss_coeff"]*single_q*abs(single_q)/(2*9.81*single_a**2)
                loss_max_error = max(loss_max_error, abs(float(actual_loss)-exact_loss))
                total_f = row["nc"]**2*row["physical_length"]*q*abs(q)/(area[0]**2*rh[0]**(4/3))
                single_f = row["nc"]**2*row["physical_length"]*single_q*abs(single_q)/(single_a**2*single_r**(4/3))
                friction_max_error = max(friction_max_error, abs(float(total_f)-single_f))
                assert q*actual_loss >= 0 and q*total_f >= 0
    assert max(radius_max_error, loss_max_error, friction_max_error) < 1e-12

    result = dict(status="PURE_CONVERSION_AND_ALGEBRA_VERIFIED_NO_HYDRAULIC_RUN",
        scope=dict(engine_instances_created=0, hydraulic_steps=0, assimilation=False,
                   source_or_26_or_41_modified=False, baseline_replaced=False),
        bed_composition=dict(superlinks=len(c["superlinks"]), endpoint_checks=len(end_rows),
                reversed_superlinks=sum(float(m["flow_sign"]) < 0 for m in c["channel_discretization"].values()),
                max_absolute_endpoint_bed_error_m=max(abs(r["difference_m"]) for r in end_rows),
                orifices=len(ori), gate_bottom_mismatches=mismatches,
                note="运行端床高=主节点床高+对应offset；内部h为距内部床高的水深；全部高程/水深单位m，不在此更改节点储水基准"),
        meihe_bug=dict(classification="SOURCE_CONFIRMED_CONVERSION_BUG_NEGATIVE_GATE_OFFSET_CLIPPED",
                source_verification=str(HERE/"梅河闸底原表独立复核.json"),
                cause="原转换max(zb_gate-zb_upstream_node,0)把−1.202m截为0，抬高有效闸底1.202m",
                negative_offset_supported_by_active_gate_terms=True, candidate_manifest=manifest,
                changed_runtime_indices=changed_ids, other_states_parameters_geometry_preserved=True,
                hypothetical_partial_submergence_examples=examples),
        actual_forcing_and_saved_heads=dict(
                accepted_halfhour_count=len(actual), input_time_count=len(grid),
                maximum_original_formula_minus_saved_Q_m3s=max(abs(r["original_minus_saved_flow_m3s"]) for r in actual),
                maximum_Q_change_at_accepted_halfhour_m3s=max(abs(r["corrected_minus_original_flow_m3s"]) for r in actual),
                maximum_Q_change_at_input_times_m3s=max(abs(r["corrected_minus_original_flow_m3s"]) for r in grid),
                minimum_old_full_submergence_margin_m=min(r["minimum_old_submergence_margin_m"] for r in actual+grid),
                actual_last_step_start_times_read_from_saved_chunks=True,
                input_normalization="原Forcing顺序：各输入时刻开度/y_max，clip到[0,1]，再按实际t0线性插值",
                interpretation="全部检查点闸孔完全淹没，降低闸底不改变同水位同开度流量；源错误成立，但当前点集不支持解释30天水位偏差",
                limits="只检已保存1441水位时刻和1440真实末步forcing；未保存梅河全子步/非线性迭代水位，不能声称每个过程子步均不触发。input_grid的开度t不同于接受末步t0，不与该时刻原Q混比"),
        end_energy_sign=dict(active_bc_method="z", interpretation="上游端Hsuper-Hinternal与下游端Hinternal-Hsuper均为K*Q*abs(Q)/(2gA²)，正反流耗能方向正确；未发现重复底高程或mm/m混用", examples=signed),
        parallel_geometry_and_loss=dict(structures=len(c["siphons_info"]), all_one_internal_link=True,
                max_parallel_vs_single_barrel_radius_error_m=radius_max_error,
                max_parallel_vs_single_barrel_local_headloss_error_m=loss_max_error,
                max_parallel_vs_single_barrel_friction_headloss_error_m=friction_max_error,
                interpretation="N孔总面积配N孔湿周，R与单孔相同；显式K只加一次，正反流符号正确。未修改或补录未知源K"),
        retained_discretization_limits=[
            "端部内部节点对流速度采用相邻内部连接Q/A外推；端部水头损失用端面Q/A。二者是不同离散位置，单凭不同不能判bug，也不擅加Bernoulli动能项",
            "非活动bc_method=b的kappa含端部摩阻项，单位/符号需另案审查；本线活动bc_method=z且端长为0，不作为当前偏差原因或本轮修改",
            "parse中zb_abs<=0回退0是另一一般性限制；本次正高程C07不触发，不扩展本候选",
            "q20s1桩号转录另案，不与闸底恢复混合；断面域外和源资料缺项仍保留"],
        sources=[dict(path=p,sha256=h) for p,h in before.items()],
        saved_substep_sources=[dict(path=str(p),sha256=sha(p)) for p in chunk_files],
        source_hashes_unchanged=all(sha(p)==h for p,h in before.items()))
    assert result["source_hashes_unchanged"]
    dump("剩余方程一致性审查.json", result)
    print(json.dumps({k:result[k] for k in ["status","bed_composition","actual_forcing_and_saved_heads","parallel_geometry_and_loss"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
