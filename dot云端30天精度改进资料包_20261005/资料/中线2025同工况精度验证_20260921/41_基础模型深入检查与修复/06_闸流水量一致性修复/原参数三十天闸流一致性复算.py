"""仅收紧flow_tol的G00无同化全月复算；逐成功步记录原控制体及全部主节点收支。"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
B = R / "26_冻结基线与后续归因"
DEPS = B / "共享依赖"
SOURCE_HELPER = HERE.parent / "01_求解器核查/面积储水与闸流最小复现.py"
STATE_KEYS = ("H_j", "h_Ik", "Q_ik", "Q_uk", "Q_dk", "Q_o")
PARAM_KEYS = ("_n_ik", "_n_uk", "_n_dk", "_Co", "_C_uk", "_C_dk")
OUT = HERE / "G00_更严闸流一致性"
END_S = 2592000.
PERSIST_CYCLES = 24
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime_identity():
    from numba import config
    from pipedream_solver import nsuperlink
    from pipedream_solver.gate_hydraulics import rectangular_gate_terms
    modules = []
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if "corelib" in name.lower() or (file and "corelib" in file.lower()):
            raise RuntimeError("不允许加载Corelib")
        if file and name.startswith(("pipedream_solver", "wxq_pipedream")):
            path = Path(file).resolve()
            assert path.is_relative_to(DEPS.resolve()), path
            modules.append(dict(name=name, path=str(path), sha256=sha(path)))
    return dict(modules=modules, interpreter=sys.executable,
                numba_jit_disabled=bool(config.DISABLE_JIT),
                momentum_compiled_signatures=len(nsuperlink.numba_b_ik.signatures),
                gate_compiled_signatures=len(rectangular_gate_terms.signatures),
                no_old_water_hydraulic_modules=True, no_Corelib=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "逐步残差分块").mkdir()
    started = time.time()
    status = dict(status="INITIALIZING", pid=os.getpid(), completed_cycles=0,
                  completed_successful_substeps=0, rejected_attempts=0,
                  model_time_s=0., full_30day_completed=False, baseline_replaced=False)
    dump(OUT / "进度.json", status)
    print(json.dumps(status), flush=True)
    shared = None
    m = None
    scalar_buffer, node_buffer, chunks = [], [], []
    node_rows, mapped_rows, half_hour_ledger = [], [], []
    chunk_start_step = 0
    success_total = rejected_total = 0
    maximum_cv = maximum_node = maximum_internal = 0.
    cv_exceeded_steps = node_exceeded_steps = node_exceeded_point_steps = 0
    max_iterations = 0
    iteration_counts = {}
    rejected_events = OUT / "减步失败尝试.jsonl"
    rejected_handle = rejected_events.open("w")

    def persist_chunk():
        nonlocal scalar_buffer, node_buffer, chunk_start_step
        if not scalar_buffer:
            return
        scalar = pd.DataFrame(scalar_buffer)
        path = OUT / "逐步残差分块" / f"分块_{len(chunks)+1:03d}.npz"
        arrays = {key: scalar[key].to_numpy() for key in scalar}
        arrays["nonboundary_node_residual_m3s"] = np.stack(node_buffer)
        np.savez_compressed(path, **arrays)
        chunks.append(dict(file=str(path), sha256=sha(path), rows=len(scalar),
                           first_step=chunk_start_step,
                           last_step_exclusive=chunk_start_step+len(scalar),
                           start_s=float(scalar.interval_start_s.iloc[0]),
                           end_s=float(scalar.time_s.iloc[-1])))
        chunk_start_step += len(scalar)
        scalar_buffer, node_buffer = [], []
        dump(OUT / "逐成功子步文件索引.json", dict(chunks=chunks, total_rows=chunk_start_step,
             nonboundary_node_metadata="非定水位主节点索引.json", scalar_and_node_rows_aligned=True))

    def persist_tables():
        if node_rows:
            pd.DataFrame(node_rows).to_csv(OUT / "半小时水位流量.csv", index=False)
        if mapped_rows:
            pd.DataFrame(mapped_rows).to_csv(OUT / "实际位置水位.csv", index=False)
        if half_hour_ledger:
            pd.DataFrame(half_hour_ledger).to_csv(OUT / "半小时测流至闸水量账.csv", index=False)

    try:
        protocol_path = HERE / "运行前约定.json"
        protocol = json.loads(protocol_path.read_text())
        assert protocol["flow_tol_m3s"] == 1e-9 and protocol["head_tol_m"] == .0015
        reference = R / "38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数"
        trace_reference = R / "39_局部与全线边界归因/03_接口与水量补录/G00_冻结原参数"
        original_config_path = reference / "运行配置.json"
        old = json.loads(original_config_path.read_text())
        assert old["factors"] == {"n": 1., "C": 1.}
        old_trace = json.loads((trace_reference / "进度.json").read_text())
        assert old_trace["status"] == "COMPLETED" and old_trace["exact_node_trajectory"] and old_trace["exact_six_final_states"]
        assert not old_trace["control_volume_gate_pass"]
        source_paths = [Path(__file__), SOURCE_HELPER, protocol_path, original_config_path,
                        trace_reference / "补录配置.json", trace_reference / "进度.json",
                        R / "39_局部与全线边界归因/03_接口与水量补录/原求解过程补录.py",
                        R / "30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy",
                        R / "30_分通道方法逐项测试/02_输入与测点核查/输入与算子核验.json"]
        sources = {**old["source_hashes"], **{str(p): sha(p) for p in source_paths}}
        sources.update(protocol["source_sha256"])
        assert all(sha(p) == h for p, h in sources.items())
        spec = importlib.util.spec_from_file_location("_41_single_step_helper", SOURCE_HELPER)
        shared = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shared)
        common = shared.bind(DEPS / "水力运行脚本/试验公共.py",
             ["Forcing", "meshes", "weights_at", "flow_operator"],
             dict(np=np, pd=pd, re=shared.re, Path=Path, R=R), "_41_S03_forcing")
        physics = shared.bind(DEPS / "并联结构物理修复.py",
             ["_correct_section", "_link_geometry", "_up_geometry", "_down_geometry", "_link_coeffs", "attach_parallel_physics", "build_engine_physics"],
             dict(np=np, types=shared.types, copy=copy, hashlib=hashlib, Path=Path, R=R,
                  PipedreamHydraulicEngine=shared.PipedreamHydraulicEngine,
                  _legacy_build=shared.legacy_full_build), "_41_S03_physics")
        advance = shared.bind(DEPS / "参数并行运行脚本/常驻水力进程.py", ["advance"],
                              dict(np=np, STATE_KEYS=STATE_KEYS), "_41_S03_original_advance").advance
        restore = shared.bind(DEPS / "参数并行运行脚本/局地参数EnKF.py", ["restore"], {}, "_41_S03_restore").restore
        e, source, c = physics.build_engine_physics(gate_max_normalization=json.loads((B / "共享输入/2025开度归一化上限.json").read_text()))
        m = e.model
        with np.load(B / "共享输入/初始化后状态_0s.npz") as z:
            restore(m, {key: z[key].copy() for key in STATE_KEYS})
        m.t = 0.; m.iter_count = 0
        initial_hashes = {key: arrsha(getattr(m, key)) for key in STATE_KEYS}
        assert initial_hashes == old["initial_state_sha256"]
        fixed = {key: arrsha(getattr(m, key)) for key in PARAM_KEYS}
        assert fixed == old["parameter_sha256"]
        forcing = common.Forcing(e, source, c)
        assert {key: arrsha(getattr(forcing, key)) for key in ["times", "h", "q", "q0", "u"]} == old["forcing_array_sha256"]
        assert m.n_w == 0 and m.n_p == 0
        trace_config = json.loads((trace_reference / "补录配置.json").read_text())
        qop = trace_config["flow_operator"]
        assert common.flow_operator(e, source, c, "JZZ16", 300091.) == qop
        qids, qw = np.asarray(qop["indices"]), np.asarray(qop["weights"])
        gids = np.asarray(old["C_indices"])
        W = np.load(R / "30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy")
        assert W.shape == (m.M + m.NIk,)
        for field, up, down in [("Q_ik", m._Ik+m.M, m._Ip1k+m.M), ("Q_uk", m._J_uk, m._I_1k+m.M),
                                ("Q_dk", m._I_Np1k+m.M, m._J_dk), ("Q_o", m._J_uo, m._J_do)]:
            expected = np.zeros(len(up))
            if field == "Q_ik": expected[qids] = qw
            if field == "Q_o": expected[gids] = -1.
            assert np.max(abs((W[down]-W[up])-expected)) < 1e-12
        assert np.max(abs(W[:m.M][m.bc])) < 1e-12
        jids, iids = np.flatnonzero(W[:m.M]), np.flatnonzero(W[m.M:])
        wj, wi = W[jids], W[m.M+iids]
        node_ids = np.flatnonzero(~m.bc)
        dump(OUT / "非定水位主节点索引.json", dict(ids=node_ids.tolist(),
             names=[e.node_names[i] for i in node_ids], count=len(node_ids), units="m3/s",
             residual_definition="Q_in + incident_superlink_and_reported_gate_flux - native_A_sj_dH_dt - native_half_end_xi_dH"))
        ops = json.loads((B / "B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json").read_text())
        assert len(ops) == 43
        groups = {st: (int(g.iloc[0].sj_0), int(g.iloc[0].sj_1), g.id.to_numpy(int))
                  for st, g in c["orifices"].assign(st=c["orifices"].name.str.split("-").str[0]).groupby("st")}
        assert len(groups) == 59
        identity = runtime_identity()
        assert not identity["numba_jit_disabled"] and identity["momentum_compiled_signatures"] > 0 and identity["gate_compiled_signatures"] > 0
        dump(OUT / "运行配置.json", dict(case="G00_更严闸流一致性", source_hashes=sources,
             protocol=protocol, reference=str(reference), original_trace_reference=str(trace_reference),
             initial_state_sha256=initial_hashes, initial_state_exactly_matches_original=True,
             parameter_sha256=fixed, parameter_arrays_exactly_match_original=True,
             forcing_array_sha256=old["forcing_array_sha256"], forcing_arrays_exactly_match_original=True,
             flow_operator=qop, control_volume_weights_sha256=arrsha(W),
             source_reference_original_gate_pass=False, source_reference_gate_status_unchanged=True,
             runtime_identity=identity, assimilation=False, state_corrections=0))
        np.savez_compressed(OUT / "实际初始化_0s.npz", **{key:getattr(m,key) for key in STATE_KEYS})
        status.update(status="RUNNING", initialization_and_provenance_verified=True)
        dump(OUT / "进度.json", status)
        original_step = m.step
        cycle_integrals = dict(duration=0., meter=0., gate=0., offtake=0., storage=0.)

        def recorded_step(**kwargs):
            nonlocal success_total, rejected_total, maximum_cv, maximum_node, maximum_internal
            nonlocal cv_exceeded_steps, node_exceeded_steps, node_exceeded_point_steps, max_iterations
            t0, dt = float(m.t), float(kwargs["dt"])
            before = {key:getattr(m,key).copy() for key in STATE_KEYS}
            kwargs.update(flow_tol=1e-9, head_tol=.0015)
            assert kwargs["num_iter"] == 20 and kwargs["strict"] is True
            try:
                original_step(**kwargs)
            except BaseException as exc:
                rejected_total += 1
                exact = all(np.array_equal(getattr(m,key),before[key]) for key in STATE_KEYS)
                event = dict(attempt=rejected_total, start_s=t0, dt_s=dt, error=repr(exc),
                    rolled_back_time_s=float(m.t), six_state_rollback_exact=exact,
                    iterations=int(getattr(m,"iter_elapsed",0)),
                    last_head_residual_m=float(getattr(m,"last_head_residual",0)),
                    last_gate_residual_m3s=float(getattr(m,"last_gate_flow_residual",0)))
                for key in ["last_head_residual_m", "last_gate_residual_m3s"]:
                    if not np.isfinite(event[key]):event[key]=str(event[key])
                rejected_handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+"\n");rejected_handle.flush()
                dump(OUT / "最近失败尝试.json", event)
                if not exact or m.t != t0:
                    raise RuntimeError("原求解器失败后的六态/时钟回退不一致") from exc
                raise
            assert m.t == t0 + dt and all(np.isfinite(getattr(m,key)).all() for key in STATE_KEYS)
            dH, dh = m.H_j-before["H_j"], m.h_Ik-before["h_Ik"]
            dv = float(wj @ (m._A_sj[jids]*dH[jids]) + wi @ (m._E_Ik[iids]*dh[iids]*dt))
            take = -float(wj @ kwargs["Q_in"][jids] + wi @ kwargs["Q_0Ik"][iids])*dt
            qm, qg = float(m.Q_ik[qids] @ qw), float(m.Q_o[gids].sum())
            cv = qm-qg-take/dt-dv/dt
            rj, ri = kwargs["Q_in"].copy(), kwargs["Q_0Ik"].copy()
            for net, nodes, values in [(rj,m._J_uk,-m.Q_uk),(rj,m._J_dk,m.Q_dk),
                    (rj,m._J_uo,-m.Q_o),(rj,m._J_do,m.Q_o),(ri,m._I_1k,m.Q_uk),
                    (ri,m._I_Np1k,-m.Q_dk),(ri,m._Ik,-m.Q_ik),(ri,m._Ip1k,m.Q_ik)]:
                np.add.at(net,nodes,values)
            rj -= m._A_sj*dH/dt;ri -= m._E_Ik*dh
            endpoint_storage = np.zeros(m.M)
            for nodes,width,length,theta in [(m._J_uk,m._B_uk,m._dx_uk,m._theta_uk),(m._J_dk,m._B_dk,m._dx_dk,m._theta_dk)]:
                np.add.at(endpoint_storage,nodes,length*width*theta/(2*dt)*dH[nodes])
            rj -= endpoint_storage
            # 原39控制体账不含xi；本冻结模型所有端段长度为0，此处仍显式检查。
            assert np.max(abs(endpoint_storage)) == 0
            assert abs(cv - float(W[:m.M]@rj + W[m.M:]@ri)) < 1e-9
            nr = rj[node_ids]
            mxnode, mxinternal = float(abs(nr).max()), float(abs(ri).max())
            maximum_cv=max(maximum_cv,abs(cv));maximum_node=max(maximum_node,mxnode);maximum_internal=max(maximum_internal,mxinternal)
            cv_exceeded_steps += int(abs(cv)>1e-6);node_exceeded_steps += int(mxnode>1e-6)
            node_exceeded_point_steps += int((abs(nr)>1e-6).sum())
            iterations=int(m.iter_elapsed);max_iterations=max(max_iterations,iterations)
            iteration_counts[str(iterations)]=iteration_counts.get(str(iterations),0)+1
            success_total += 1
            scalar_buffer.append(dict(interval_start_s=t0,time_s=float(m.t),dt_s=dt,
                Q16_cut_m3s=qm,Q16_gate_m3s=qg,offtake_volume_m3=take,native_storage_change_m3=dv,
                continuity_residual_m3s=cv,max_nonboundary_node_residual_m3s=mxnode,
                max_internal_residual_m3s=mxinternal,iterations=iterations,
                head_iteration_residual_m=float(m.last_head_residual),gate_iteration_residual_m3s=float(m.last_gate_flow_residual)))
            node_buffer.append(nr.copy())
            cycle_integrals["duration"]+=dt;cycle_integrals["meter"]+=qm*dt;cycle_integrals["gate"]+=qg*dt
            cycle_integrals["offtake"]+=take;cycle_integrals["storage"]+=dv

        m.step = recorded_step
        retries = 0
        for cycle in range(1440):
            t0, t1 = cycle*1800., (cycle+1)*1800.
            cycle_integrals.update(duration=0.,meter=0.,gate=0.,offtake=0.,storage=0.)
            retry,_,_ = advance(m,forcing,t0,t1)
            retries += retry
            assert cycle_integrals["duration"] == 1800. and m.t == t1
            for station,(ju,jd,ids) in groups.items():
                node_rows.append(dict(time_s=t1,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[ids].sum())))
            for op in ops:
                mapped_rows.append(dict(time_s=t1,station=op["station"],metric=op["metric"],
                     mapped_sim=float(getattr(m,op["state"])[op["indices"]] @ np.asarray(op["weights"])+op.get("offset",0)),
                     group="旧参与同化测点" if op["assimilated"] else "旧留出检验测点",chainage_m=op["chainage"]))
            half_hour_ledger.append(dict(time_s=t1,duration_s=1800.,
                 model_meter_mean_m3s=cycle_integrals["meter"]/1800,
                 model_gate_mean_m3s=cycle_integrals["gate"]/1800,
                 offtake_mean_m3s=cycle_integrals["offtake"]/1800,
                 native_storage_rate_m3s=cycle_integrals["storage"]/1800,
                 continuity_residual_m3s=(cycle_integrals["meter"]-cycle_integrals["gate"]-cycle_integrals["offtake"]-cycle_integrals["storage"])/1800))
            status.update(completed_cycles=cycle+1,completed_successful_substeps=success_total,
                 rejected_attempts=rejected_total,retries=retries,model_time_s=t1,
                 maximum_control_volume_residual_m3s=maximum_cv,
                 maximum_nonboundary_node_residual_m3s=maximum_node,
                 maximum_internal_residual_m3s=maximum_internal,
                 control_volume_exceeded_step_count=cv_exceeded_steps,
                 nonboundary_node_exceeded_step_count=node_exceeded_steps,
                 nonboundary_node_exceeded_point_steps=node_exceeded_point_steps,
                 maximum_iterations=max_iterations,iteration_counts=iteration_counts,wall_seconds=time.time()-started)
            if (cycle+1)%PERSIST_CYCLES==0 or cycle==1439:
                assert all(arrsha(getattr(m,key))==h for key,h in fixed.items())
                persist_chunk();persist_tables()
                np.savez_compressed(OUT/"最新成功状态.npz",**{key:getattr(m,key) for key in STATE_KEYS},time_s=np.array(m.t))
                dump(OUT/"进度.json",status);print(json.dumps(status),flush=True)
        assert retries == rejected_total
        assert len(node_rows)==84960 and len(mapped_rows)==61920 and len(half_hour_ledger)==1440
        assert success_total==sum(x["rows"] for x in chunks) and chunks[0]["start_s"]==0 and chunks[-1]["end_s"]==END_S
        assert all(sha(p)==h for p,h in sources.items())
        identity=runtime_identity()
        assert not identity["numba_jit_disabled"] and identity["momentum_compiled_signatures"]>0 and identity["gate_compiled_signatures"]>0
        np.savez_compressed(OUT/"末态.npz",**{key:getattr(m,key) for key in STATE_KEYS})
        status.update(status="COMPLETED",full_30day_completed=True,
             original_39_control_volume_threshold_m3s=1e-6,
             all_successful_steps_control_volume_gate_pass=cv_exceeded_steps==0,
             all_successful_steps_all_nonboundary_nodes_under_1e6=node_exceeded_steps==0,
             parameters_unchanged=True,source_hashes_unchanged=True,assimilation=False,
             runtime_identity=identity,old_39_failure_unchanged=True,baseline_replaced=False,
             full_line_accuracy_improvement_claimed=False,wall_seconds=time.time()-started)
    except BaseException as exc:
        status.update(status="FAILED",error=repr(exc),traceback=traceback.format_exc(),
             full_30day_completed=False,model_time_s=float(m.t) if m is not None else 0.,
             completed_successful_substeps=success_total,rejected_attempts=rejected_total,
             maximum_control_volume_residual_m3s=maximum_cv,
             maximum_nonboundary_node_residual_m3s=maximum_node,
             control_volume_exceeded_step_count=cv_exceeded_steps,
             nonboundary_node_exceeded_step_count=node_exceeded_steps,
             wall_seconds=time.time()-started)
        persist_chunk();persist_tables()
        if m is not None:np.savez_compressed(OUT/"停止状态.npz",**{key:getattr(m,key) for key in STATE_KEYS},time_s=np.array(m.t))
        raise
    finally:
        rejected_handle.close()
        dump(OUT/"结果.json",status);dump(OUT/"进度.json",status)
        if shared is not None:shutil.rmtree(shared.CACHE,ignore_errors=True)


if __name__=="__main__":main()
