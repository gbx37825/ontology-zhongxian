"""41 的有限实现核查：几何代数例，以及保存的39检查点各一个15秒步。

不改变26源码、工程输入、参数或历史结果；不推进整月。
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
B = R / "26_冻结基线与后续归因"
DEPS = B / "共享依赖"
sys.dont_write_bytecode = True
sys.path.insert(0, str(DEPS))
CACHE = tempfile.mkdtemp(prefix="hydros41_solveraudit_")
os.environ["NUMBA_CACHE_DIR"] = CACHE
from wxq_pipedream.wxq_to_superlink import parse_wxq_json
from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine
from pipedream_solver import ngeometry as ng, nsuperlink

STATE_KEYS = ("H_j", "h_Ik", "Q_ik", "Q_uk", "Q_dk", "Q_o")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(path, names, context, module_name):
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in names]
    assert {n.name for n in definitions} == set(names)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(context)
    sys.modules[module_name] = module
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def secant_storage_width(y0, y1, g1, b, side_slope):
    """仅在给定开敞轮廓内成立的储水割线宽；不改变摩阻用真实瞬时顶宽。"""
    if min(y0, y1) < 0 or max(y0, y1) > g1:
        raise ValueError("开敞水深超给定轮廓，不能借储水修正猜测上部工程几何")
    # A(y)=b*y+m*y*y，因此[A(y1)-A(y0)]/(y1-y0)=b+m*(y0+y1)。
    return b + side_slope * (y0 + y1)


def geometry_cases():
    result = []
    cases = [
        ("rect_inside", 9.1, 20., 0., 8.6, 8.61),
        ("rect_crosses_g1", 9.1, 20., 0., 9.095, 9.105),
        ("rect_above_g1", 9.1, 20., 0., 10.1, 10.11),
        ("trapezoid_inside", 8., 20., 2., 4., 4.01),
        ("trapezoid_inside_larger_increment", 8., 20., 2., 4., 4.1),
        ("trapezoid_above_g1", 8., 20., 2., 9., 9.01),
    ]
    for name, g1, b, slope, y0, y1 in cases:
        area = ng.Rect_Open_A_ik if slope == 0 else ng.Trapezoidal_A_ik
        width = ng.Rect_Open_B_ik if slope == 0 else ng.Trapezoidal_B_ik
        perimeter = ng.Rect_Open_Pe_ik if slope == 0 else ng.Trapezoidal_Pe_ik
        args = (g1, b) if slope == 0 else (g1, b, slope)
        A0, A1 = float(area(y0, y0, *args)), float(area(y1, y1, *args))
        B1 = float(width(y1, y1, *args))
        L, dt = 100., 15.
        # 同一连接两端节点的原E，各分得B*L/2；总和与33定义完全一致。
        Eleft = float(nsuperlink.E_Ik(B1, L, 0., 0., 0., dt))
        Eright = float(nsuperlink.E_Ik(0., 0., B1, L, 0., dt))
        geometric = (A1 - A0) * L
        native = (Eleft + Eright) * dt * (y1 - y0)
        eps = 1e-6
        numerical_dA = float((area(y1+eps, y1+eps, *args) - area(y1-eps, y1-eps, *args))/(2*eps))
        row = dict(case=name, g1_m=g1, width_m=b, side_slope=slope, old_depth_m=y0,
                   new_depth_m=y1, A0_m2=A0, A1_m2=A1, final_B_m=B1,
                   final_Pe_m=float(perimeter(y1,y1,*args)), dA_dy_finite_difference_m=numerical_dA,
                   length_m=L, dt_s=dt, geometric_change_m3=geometric,
                   native_E_storage_change_m3=native, native_minus_geometry_m3=native-geometric)
        try:
            Bs = secant_storage_width(y0, y1, g1, b, slope)
            candidate = (float(nsuperlink.E_Ik(Bs, L, 0., 0., 0., dt))
                         + float(nsuperlink.E_Ik(0., 0., Bs, L, 0., dt))) * dt * (y1-y0)
            row.update(candidate_storage_width_m=Bs, candidate_storage_change_m3=candidate,
                       candidate_minus_geometry_m3=candidate-geometric, candidate_domain_pass=True)
            assert abs(candidate-geometric) < 1e-9
        except ValueError as exc:
            row.update(candidate_domain_pass=False, candidate_stop_reason=str(exc))
        result.append(row)
    # 非均匀两端增量也应按平均水深的面积割线闭合全连接几何蓄变。
    h0=np.array([3.,5.]);h1=np.array([3.08,4.97]);g1=8.;b=20.;slope=2.;L=100.;dt=15.
    Bs=secant_storage_width(float(h0.mean()),float(h1.mean()),g1,b,slope)
    E=float(nsuperlink.E_Ik(Bs,L,0.,0.,0.,dt))
    volume=(ng.Trapezoidal_A_ik(*h1,g1,b,slope)-ng.Trapezoidal_A_ik(*h0,g1,b,slope))*L
    residual=float(E*dt*(h1-h0).sum()-volume)
    assert abs(residual)<1e-9
    return dict(cases=result,unequal_endpoint_increment_conservative_residual_m3=residual,
                proof_scope="给定轮廓内、固定连接几何及中点面积定义下的全连接时间积分恒等式；不等于已完成全网离散改造")


def legacy_full_build(*args, gate_max_normalization=None, **kwargs):
    source = json.loads((B/"共享输入/C07.json").read_text())["中线全线"]
    gate_max = {k:float(v.get("b",1) or 1) for k,v in source["baseData"]["gates"].items()}
    if gate_max_normalization:
        gate_max.update(gate_max_normalization)
    c = parse_wxq_json(source,gate_max_openings=gate_max,section_geometry_mode="safe",minimum_initial_depth_m=7.)
    c["superjunctions"].loc[:,"h_0"]=7.
    c["superjunctions"].loc[:,"max_depth"]=np.maximum(c["superjunctions"].max_depth.to_numpy(),7.)
    coeff=json.loads((B/"共享输入/Pipedream闸流系数.json").read_text())
    c["orifices"].loc[:,"C"]=c["orifices"].name.map(coeff).fillna(c["orifices"].C)
    engine=PipedreamHydraulicEngine(c,dt=15.,backend="numba")
    engine.model._method="lsq"
    return engine,source,c


def mass_audit(m, H0, h0, kw, dt, W):
    qlinear=m._alpha_o*m.H_j[m._J_uo]+m._beta_o*m.H_j[m._J_do]+m._chi_o
    rj=kw["Q_in"].copy();ri=kw["Q_0Ik"].copy()
    for net,indices,values in [(rj,m._J_uk,-m.Q_uk),(rj,m._J_dk,m.Q_dk),
            (rj,m._J_uo,-m.Q_o),(rj,m._J_do,m.Q_o),(ri,m._I_1k,m.Q_uk),
            (ri,m._I_Np1k,-m.Q_dk),(ri,m._Ik,-m.Q_ik),(ri,m._Ip1k,m.Q_ik)]:
        np.add.at(net,indices,values)
    sj=m._A_sj*(m.H_j-H0)/dt;si=m._E_Ik*(m.h_Ik-h0)
    end=np.zeros(m.M)
    for nodes,Bend,dx,theta in [(m._J_uk,m._B_uk,m._dx_uk,m._theta_uk),(m._J_dk,m._B_dk,m._dx_dk,m._theta_dk)]:
        np.add.at(end,nodes,dx*Bend*theta/(2*dt)*(m.H_j[nodes]-H0[nodes]))
    residual_j=rj-sj-end;residual_i=ri-si
    nonboundary=~m.bc
    return dict(control_volume_reported_residual_m3s=float(W[:m.M]@residual_j+W[m.M:]@residual_i),
                max_nonboundary_node_residual_m3s=float(abs(residual_j[nonboundary]).max()),
                max_internal_residual_m3s=float(abs(residual_i).max()),
                gate_formula_vs_linear_max_m3s=float(abs(m.Q_o-qlinear).max()),
                gate_formula_vs_linear_sum_abs_m3s=float(abs(m.Q_o-qlinear).sum()),
                head_iteration_residual_m=float(m.last_head_residual),iterations=int(m.iter_elapsed),
                minimum_depth_m=float(m.h_Ik.min()),all_states_finite=all(np.isfinite(getattr(m,k)).all() for k in STATE_KEYS))


def gate_checkpoint_cases():
    common=bind(DEPS/"水力运行脚本/试验公共.py",["Forcing","meshes","weights_at"],
                dict(np=np,pd=pd,re=re,Path=Path,R=R),"_41_forcing_definitions")
    physical=bind(DEPS/"并联结构物理修复.py",["_correct_section","_link_geometry","_up_geometry","_down_geometry","_link_coeffs","attach_parallel_physics","build_engine_physics"],
                  dict(np=np,types=types,copy=copy,hashlib=hashlib,Path=Path,R=R,
                       PipedreamHydraulicEngine=PipedreamHydraulicEngine,_legacy_build=legacy_full_build),"_41_physical_definitions")
    e,source,c=physical.build_engine_physics(gate_max_normalization=json.loads((B/"共享输入/2025开度归一化上限.json").read_text()))
    forcing=common.Forcing(e,source,c)
    W=np.load(R/"30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy")
    rows=[]
    for case,label in [("G00","G00_冻结原参数"),("G16","G16_联合参数转移")]:
        prior_dir=R/"39_局部与全线边界归因/05_连续性残差定位"/case
        previous=json.loads((prior_dir/"最坏子步残差分解.json").read_text())
        cfg=json.loads((R/"38_测点假设下的联合辨识/06_全线联合参数转移"/label/"运行配置.json").read_text())
        with np.load(prior_dir/"最坏步前状态及几何缓存.npz") as z:initial={k:z[k].copy() for k in z.files}
        base=copy.deepcopy(e.model)
        base.load_state(initial,compute_hydraulic_geometries=False)
        base.t=previous["start_s"];base.iter_count=0
        base._n_ik[cfg["n_indices"]]*=cfg["factors"]["n"]
        base._n_uk[:]=base._n_ik[base._i_1k];base._n_dk[:]=base._n_ik[base._i_nk]
        base._Co[cfg["C_indices"]]*=cfg["factors"]["C"]
        fixed=[getattr(base,k).copy() for k in ("_n_ik","_Co","_g1_ik","_g2_ik","_g3_ik")]
        models=[];records=[]
        for name,tolerance in [("原门槛",.01),("更严闸流一致性",1e-9)]:
            m=copy.deepcopy(base);kw=forcing.at(m.t);kw.update(dt=15.,num_iter=20,strict=True,flow_tol=tolerance)
            m.step(**kw)
            record=mass_audit(m,base.H_j,base.h_Ik,kw,15.,W)
            record.update(branch=name,flow_tolerance_m3s=tolerance,time_s=float(m.t),
                          parameters_and_geometry_unchanged=all(np.array_equal(getattr(m,k),v) for k,v in zip(("_n_ik","_Co","_g1_ik","_g2_ik","_g3_ik"),fixed)))
            assert record["parameters_and_geometry_unchanged"]
            np.savez_compressed(HERE/f"{case}_{name}_单步末态.npz",**{k:getattr(m,k) for k in STATE_KEYS})
            models.append(m);records.append(record)
        with np.load(prior_dir/"最坏步后状态.npz") as z:
            exact=all(np.array_equal(getattr(models[0],k),z[k]) for k in STATE_KEYS)
        assert exact,"原门槛单步未逐值复现39，候选比较不成立"
        assert abs(records[0]["control_volume_reported_residual_m3s"]-previous["recorded_total_residual_m3s"])<1e-10
        rows.append(dict(case=case,start_s=float(base.t),dt_s=15.,original_six_end_states_exact=exact,
                         records=records,max_candidate_minus_original={k:float(abs(getattr(models[1],k)-getattr(models[0],k)).max()) for k in STATE_KEYS}))
        print(case,json.dumps(rows[-1],ensure_ascii=False),flush=True)
    return dict(cases=rows,accepted_physical_steps=4,full_month_simulations=0,
                construction_microsteps_not_scored=True,
                interpretation="只验证提高既有孔口非线性一致性精度的单步数值效果；不修正几何、不称实测精度改善或整月水量验收通过")


def main():
    sources=[Path(__file__),DEPS/"pipedream_solver/ngeometry.py",DEPS/"pipedream_solver/nsuperlink.py",
             DEPS/"pipedream_solver/superlink.py",DEPS/"pipedream_solver/gate_hydraulics.py",
             DEPS/"并联结构物理修复.py",DEPS/"水力运行脚本/试验公共.py",
             B/"共享输入/C07.json",B/"共享输入/Pipedream闸流系数.json",B/"共享输入/2025开度归一化上限.json"]
    for case,label in [("G00","G00_冻结原参数"),("G16","G16_联合参数转移")]:
        sources += [R/"39_局部与全线边界归因/05_连续性残差定位"/case/name
                    for name in ["最坏步前状态及几何缓存.npz","最坏步后状态.npz","最坏子步残差分解.json"]]
        sources.append(R/"38_测点假设下的联合辨识/06_全线联合参数转移"/label/"运行配置.json")
    sources.append(R/"30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy")
    hashes={str(p):sha(p) for p in sources}
    result=dict(scope="41独立最小验证，无同化，无全月重跑",geometry=geometry_cases(),gate_single_step=gate_checkpoint_cases())
    modules=[]
    for name,mod in sorted(sys.modules.items()):
        path=getattr(mod,"__file__",None)
        if name.startswith(("wxq_pipedream","pipedream_solver")) and path:
            p=Path(path).resolve();assert p.is_relative_to(DEPS.resolve())
            modules.append(dict(name=name,path=str(p),sha256=sha(p)))
        assert not ("corelib" in name.lower() or (path and "corelib" in path.lower()))
    assert all(sha(p)==h for p,h in hashes.items())
    result.update(status="COMPLETED_MINIMAL_REPRODUCTIONS",source_sha256=hashes,source_hashes_unchanged=True,
                  runtime_modules=modules,interpreter=sys.executable,old_water_implementation_loaded=False,
                  temporary_numba_cache_removed_on_exit=True)
    (HERE/"最小复现结果.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print("saved",HERE/"最小复现结果.json",flush=True)


if __name__=="__main__":
    try:main()
    finally:shutil.rmtree(CACHE,ignore_errors=True)
