"""未集成storage_B候选的反例与系数核验；零水力推进、零实测拟合。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
DEPS = R/"26_冻结基线与后续归因/共享依赖"
sys.dont_write_bytecode = True
sys.path.insert(0, str(DEPS))
CACHE = tempfile.mkdtemp(prefix="hydros41_storage_coeff_")
os.environ["NUMBA_CACHE_DIR"] = CACHE
from pipedream_solver import ngeometry as ng, nsuperlink as ns

spec = importlib.util.spec_from_file_location("storage_candidate_review", HERE/"域内梯形储水候选_未集成.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scalar(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, dict):return {k:scalar(v) for k,v in value.items()}
    if isinstance(value, list):return [scalar(v) for v in value]
    return value


def fixture(h0, h1, tops=(8.,8.), g1=8., b=20., slope=2., code=5,
            upstream_length=0., downstream_length=0.):
    h0, h1 = np.asarray(h0, float), np.asarray(h1, float)
    if code == 5:B = ng.Trapezoidal_B_ik(*h1,g1,b,slope)
    elif code == 3:B = ng.Rect_Open_B_ik(*h1,g1,b)
    elif code == 2:B = ng.Rect_Closed_B_ik(*h1,g1,b,.001)
    else:raise ValueError(code)
    geometry = dict(left=np.array([0]),right=np.array([1]),g1=np.array([g1]),
         bottom_width=np.array([b]),side_slope=np.array([slope]),geom_codes=np.array([code]),
         raw_node_heights=np.asarray(tops,float))
    proposal = candidate.propose_storage_B(h0,h1,**geometry,native_B=np.array([B]))
    args = dict(_D_Ik=np.zeros(2),_E_Ik=np.zeros(2),_Q_0Ik=np.zeros(2),_B_ik=np.array([B]),
         _h_Ik=h0.copy(),_dx_ik=np.array([100.]),_A_SIk=np.zeros(2),
         _B_uk=np.array([21.5]),_B_dk=np.array([22.5]),
         _dx_uk=np.array([upstream_length]),_dx_dk=np.array([downstream_length]),
         _kI=np.zeros(2,dtype=int),_dt=15.,_forward_I_i=np.array([0,-1]),
         _backward_I_i=np.array([-1,0]),_is_start=np.array([True,False]),_is_end=np.array([False,True]))
    original = candidate.node_coefficients_for_review(ns.numba_node_coeffs,args,
                 {**proposal,"storage_B_ik":args["_B_ik"].copy()})
    before = {k:v.copy() for k,v in args.items() if isinstance(v,np.ndarray)}
    used = candidate.node_coefficients_for_review(ns.numba_node_coeffs,args,proposal)
    assert all(np.array_equal(args[k],v) for k,v in before.items())
    return geometry, proposal, args, original, used


def midpoint_area(h, b=20., slope=2., g1=8.):
    return float(ng.Trapezoidal_A_ik(float(h[0]),float(h[1]),g1,b,slope))


def linear_depth_integrated_volume(h, start_fraction=0., end_fraction=1., b=20., slope=2., L=100.):
    # 恒截面、连接内水深线性时的解析积分，用于说明空间集总局限；不是现场实测体积。
    a,c = start_fraction,end_fraction
    hL,dh = float(h[0]),float(h[1]-h[0])
    first = hL*(c-a)+dh*(c*c-a*a)/2
    second = hL*hL*(c-a)+hL*dh*(c*c-a*a)+dh*dh*(c**3-a**3)/3
    return L*(b*first+slope*second)


def interior_examples():
    examples=[]
    changes=[("same_rise",[3.,5.],[3.01,5.01]),("same_fall",[3.,5.],[2.99,4.99]),
             ("different_end_changes",[3.,5.],[3.08,4.97]),
             ("constant_mean_changed_slope",[3.,5.],[3.1,4.9]),
             ("near_zero_mean_change",[3.,5.],[3.+1e-12,5.+1e-12])]
    for name,h0,h1 in changes:
        h0,h1=np.array(h0),np.array(h1)
        geometry,proposal,args,original,used=fixture(h0,h1)
        A0,A1=midpoint_area(h0),midpoint_area(h1)
        storage=float(used["E_Ik"]@(h1-h0)*args["_dt"])
        geometric=(A1-A0)*100.
        error=storage-geometric
        roundoff_bound=64*np.finfo(float).eps*max(abs(A0*100),abs(A1*100),1.)
        assert abs(error)<=roundoff_bound
        D_rearrangement_error=float(abs(used["D_Ik"]-args["_Q_0Ik"]-used["E_Ik"]*h0).max())
        D_roundoff_bound=64*np.finfo(float).eps*max(float(abs(used["D_Ik"]).max()),1.)
        assert D_rearrangement_error <= D_roundoff_bound
        examples.append(dict(case=name,h0=h0,h1=h1,native_B=proposal["native_B_ik"],
             storage_B=proposal["storage_B_ik"],used_mask=proposal["used_mask"],
             actual_E_used=used["E_Ik"],actual_D_used=used["D_Ik"],
             D_rearrangement_max_abs_m3s=D_rearrangement_error,D_roundoff_bound_m3s=D_roundoff_bound,
             full_link_lumped_storage_change_m3=storage,midpoint_geometry_change_m3=geometric,
             residual_m3=error,roundoff_bound_m3=roundoff_bound,
             linear_depth_integrated_full_volume_change_m3=linear_depth_integrated_volume(h1)-linear_depth_integrated_volume(h0),
             analytical_half_link_volume_changes_m3=[linear_depth_integrated_volume(h1,0,.5)-linear_depth_integrated_volume(h0,0,.5),
                 linear_depth_integrated_volume(h1,.5,1)-linear_depth_integrated_volume(h0,.5,1)],
             lumped_node_storage_changes_m3=used["E_Ik"]*(h1-h0)*args["_dt"]))
    return examples


def epsilon_switch_examples():
    rows=[]
    for boundary in ["mean_and_node_g1_boundary","one_real_node_boundary_mean_stays_valid"]:
        for eps in [1e-2,1e-4,1e-6,1e-8,1e-10,1e-12]:
            pair=[]
            for side in [-1,1]:
                if boundary=="mean_and_node_g1_boundary":
                    h0=[3.99,3.99];h1=[4.+side*eps,4.+side*eps];tops=[4.,4.];g1=4.
                else:
                    h0=[3.99,3.99];h1=[4.+side*eps,3.99];tops=[4.,12.];g1=8.
                g,p,a,n,u=fixture(h0,h1,tops=tops,g1=g1)
                rate=float(u["E_Ik"]@(np.array(h1)-h0))
                native_rate=float(n["E_Ik"]@(np.array(h1)-h0))
                row=dict(boundary=boundary,epsilon_m=eps,side=side,h0=h0,h_assembly=h1,
                    connection_mean_old=float(np.mean(h0)),connection_mean_assembly=float(np.mean(h1)),
                    used_mask=p["used_mask"],endpoints_inside=p["assembly_endpoints_inside"],
                    connection_means_inside=p["connection_means_inside"],native_B_m=p["native_B_ik"],
                    storage_B_m=p["storage_B_ik"],storage_B_minus_native_B_m=p["storage_B_ik"]-p["native_B_ik"],
                    E_used=u["E_Ik"],D_used=u["D_Ik"],
                    storage_rate_m3s=rate,native_storage_rate_m3s=native_rate,
                    storage_rate_difference_m3s=rate-native_rate)
                rows.append(row);pair.append(row)
            pair[0]["right_minus_left_storage_B_m"]=float(pair[1]["storage_B_m"][0]-pair[0]["storage_B_m"][0])
            pair[0]["right_minus_left_storage_rate_m3s"]=pair[1]["storage_rate_m3s"]-pair[0]["storage_rate_m3s"]
    limits=[]
    for boundary,width,rate in [("mean_and_node_g1_boundary",.02,2*100*.01**2/15),
                                 ("one_real_node_boundary_mean_stays_valid",.01,2*100*.005**2/15)]:
        left=next(x for x in rows if x["boundary"]==boundary and x["epsilon_m"]==1e-12 and x["side"]==-1)
        assert abs(left["right_minus_left_storage_B_m"]-width)<1e-10
        assert abs(left["right_minus_left_storage_rate_m3s"]-rate)<1e-8
        limits.append(dict(boundary=boundary,nonzero_limit_B_jump_m=width,
                          nonzero_limit_storage_rate_jump_m3s=rate,
                          epsilon_1e12_observed_B_jump_m=left["right_minus_left_storage_B_m"],
                          epsilon_1e12_observed_rate_jump_m3s=left["right_minus_left_storage_rate_m3s"]))
    return dict(rows=rows,nonvanishing_limits=limits,smoothed=False,domain_threshold_buffer_m=0.)


def exclusions_and_last_assembly():
    checks=[]
    for label,code,h0,h1,tops,g1 in [
            ("rectangular_open_unchanged",3,[4.,4.],[4.01,4.01],[8.,8.],8.),
            ("closed_pressurized_slot_unchanged",2,[9.,9.],[9.01,9.01],[8.,8.],8.),
            ("outside_trapezoid_unchanged",5,[9.,9.],[9.01,9.01],[8.,8.],8.)]:
        g,p,a,n,u=fixture(h0,h1,tops=tops,g1=g1,code=code)
        assert not p["used_mask"].any() and np.array_equal(p["storage_B_ik"],p["native_B_ik"])
        assert np.array_equal(u["E_Ik"],n["E_Ik"]) and np.array_equal(u["D_Ik"],n["D_Ik"])
        checks.append(dict(case=label,native_B=p["native_B_ik"],storage_B=p["storage_B_ik"],used_mask=p["used_mask"],E_D_exactly_unchanged=True))
    g,p,a,n,u=fixture([3.,5.],[3.01,5.01],upstream_length=10.,downstream_length=12.)
    expected=(p["storage_B_ik"][0]-p["native_B_ik"][0])*100/(2*15)
    assert np.allclose(u["E_Ik"]-n["E_Ik"],expected,rtol=0,atol=1e-13)
    checks.append(dict(case="boundary_B_and_lengths_unchanged",B_up=a["_B_uk"],B_down=a["_B_dk"],
                       lengths=[10.,12.],only_internal_link_E_delta=u["E_Ik"]-n["E_Ik"],expected_internal_delta=expected))
    # 最后组装时在域内，但被解出的最终状态越界，必须停；实际用过的B不覆盖。
    g,p,a,n,u=fixture([3.99,3.99],[3.999999,3.999999],tops=[4.,4.],g1=4.)
    used_B=u["storage_B_ik"].copy()
    final_check=candidate.check_accepted_domain(u,np.array([4.000001,4.000001]),g)
    assert final_check["stop_candidate"] and np.array_equal(u["storage_B_ik"],used_B)
    # 即使两者均在域内，最后组装与接受状态不同也会留下真实迭代误差，不能伪装理想割线。
    old=np.array([3.,5.]);assembly=np.array([3.01,5.01]);final=np.array([3.01001,5.01001])
    g,p,a,n,u=fixture(old,assembly)
    accepted_check=candidate.check_accepted_domain(u,final,g)
    actual_volume=float(u["E_Ik"]@(final-old)*15.)
    geometry_volume=(midpoint_area(final)-midpoint_area(old))*100.
    iteration_residual=actual_volume-geometry_volume
    expected_residual=2*100*(float(final.mean()-old.mean()))*(float(assembly.mean()-final.mean()))
    assert abs(iteration_residual-expected_residual)<1e-9 and not accepted_check["stop_candidate"]
    return dict(exclusion_checks=checks,assembled_inside_final_outside_stop=final_check,
        inside_but_not_fully_iterated=dict(h0=old,h_assembly=assembly,h_accepted=final,
            actual_B_used=u["storage_B_ik"],ideal_final_B_for_diagnostic_only=accepted_check["final_state_ideal_storage_B_ik"],
            actual_assembled_storage_change_m3=actual_volume,accepted_midpoint_geometry_change_m3=geometry_volume,
            actual_not_ideal_storage_residual_m3=iteration_residual,expected_residual_m3=expected_residual))


def main():
    paths=[Path(__file__),HERE/"域内梯形储水候选_未集成.py",HERE/"最小例预定条件.json",DEPS/"pipedream_solver/nsuperlink.py",
           DEPS/"pipedream_solver/ngeometry.py",DEPS/"pipedream_solver/superlink.py"]
    source={str(p):sha(p) for p in paths}
    interior=interior_examples();switches=epsilon_switch_examples();checks=exclusions_and_last_assembly()
    assert candidate.NOT_APPROVED_FOR_HYDRAULIC_RUN
    modules=[]
    for name,mod in sorted(sys.modules.items()):
        file=getattr(mod,"__file__",None)
        if name.startswith("pipedream_solver") and file:
            p=Path(file).resolve();assert p.is_relative_to(DEPS.resolve())
            modules.append(dict(name=name,file=str(p),sha256=sha(p)))
        assert not("corelib" in name.lower() or (file and "corelib" in file.lower()))
    assert all(sha(p)==v for p,v in source.items())
    result=dict(status="STOP_BEFORE_HYDRAULIC_INTEGRATION_NONVANISHING_DOMAIN_SWITCH",not_approved_for_hydraulic_run=True,
        hydraulic_models_constructed=0,hydraulic_steps=0,three_day_run_started=False,
        scope="26真实几何/内部节点系数函数的制造最小例；非实测精度验证。",
        interior_examples=interior,epsilon_switch_examples=switches,last_assembly_and_exclusions=checks,
        source_sha256=source,runtime_modules=modules,
        native_numba_node_coeffs_compiled_signatures=len(ns.numba_node_coeffs.signatures),
        conclusion="域内全连接中点体积恒等式成立，但动态域mask和保留原域外路径之间存在非零跳变；不批准集成或3天推进。")
    (HERE/"最小例核验结果.json").write_text(json.dumps(scalar(result),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],not_approved_for_hydraulic_run=True,
          nonvanishing_limits=switches["nonvanishing_limits"],hydraulic_steps=0),ensure_ascii=False))


if __name__=="__main__":
    try:main()
    finally:shutil.rmtree(CACHE,ignore_errors=True)
