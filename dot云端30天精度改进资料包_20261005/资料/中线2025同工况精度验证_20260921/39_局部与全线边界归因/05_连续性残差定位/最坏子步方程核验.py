"""原推进至已定位子步，仅读矩阵/重构/孔口量拆分残差，不改变收敛阈值。"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
R39=HERE.parent;R=R39.parent;R38=R/'38_测点假设下的联合辨识'
spec=importlib.util.spec_from_file_location('full38',R38/'06_全线联合参数转移/全线固定参数转移.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full);api=full.api
def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['G00','G16']);case=p.parse_args().case
    out=HERE/case;out.mkdir(parents=True,exist_ok=False);started=time.time()
    label={'G00':'G00_冻结原参数','G16':'G16_联合参数转移'}[case]
    cfg=json.loads((R38/'06_全线联合参数转移'/label/'运行配置.json').read_text())
    trace_dir=R39/'03_接口与水量补录'/label
    with np.load(trace_dir/'逐成功子步接口及水量.npz') as z:
        idx=int(np.argmax(abs(z['continuity_residual_m3s'])))
        expected={k:float(z[k][idx]) for k in z.files}
    t0=expected['interval_start_s'];dt=expected['dt_s'];assert dt==15.
    frozen=api.DEPS/'参数并行运行脚本'
    advance=api._bind_definitions(frozen/'常驻水力进程.py',['advance'],dict(np=np,STATE_KEYS=api.STATE_KEYS),'_38_point_advance').advance
    restore=api._bind_definitions(frozen/'局地参数EnKF.py',['restore'],{},'_38_point_restore').restore
    api._physics._legacy_build=full.frozen_full_build
    e,s,c=api._physics.build_engine_physics(gate_max_normalization=api.data()[2]);m=e.model
    with np.load(full.B/'共享输入/初始化后状态_0s.npz') as z:restore(m,{k:z[k].copy() for k in api.STATE_KEYS})
    m.t=0.;m.iter_count=0
    m._n_ik[cfg['n_indices']]*=cfg['factors']['n'];m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m._Co[cfg['C_indices']]*=cfg['factors']['C']
    f=api.Forcing(e,s,c);cursor=0.;retries=0
    while cursor<t0:
        end=min(cursor+86400,t0);retry,_,_=advance(m,f,cursor,end);retries+=retry;cursor=end
        full.dump(out/'进度.json',dict(status='RUNNING',time_s=cursor,target_s=t0,wall_seconds=time.time()-started,retries=retries))
        print(case,cursor,flush=True)
    W=np.load(R/'30_分通道方法逐项测试/02_输入与测点核查/兰河测流到闸门控制体权重.npy');wj=W[:m.M];wi=W[m.M:]
    qop=json.loads((trace_dir/'补录配置.json').read_text())['flow_operator'];gids=np.asarray(cfg['C_indices'])
    H=m.H_j.copy();h=m.h_Ik.copy()
    np.savez_compressed(out/'最坏步前状态及几何缓存.npz',**{k:getattr(m,k) for k in [*api.STATE_KEYS,'A_ik','A_uk','A_dk','A_sj','V_j','x_Ik']})
    kw=f.at(t0);kw.update(dt=dt,num_iter=20,strict=True);m.step(**kw)
    qm=float(m.Q_ik[qop['indices']]@np.asarray(qop['weights']));qg=float(m.Q_o[gids].sum())
    linear=m._alpha_o*m.H_j[m._J_uo]+m._beta_o*m.H_j[m._J_do]+m._chi_o
    raw_j=kw['Q_in'].copy();raw_i=kw['Q_0Ik'].copy()
    for net,indices,values in [(raw_j,m._J_uk,-m.Q_uk),(raw_j,m._J_dk,m.Q_dk),(raw_j,m._J_uo,-m.Q_o),(raw_j,m._J_do,m.Q_o),(raw_i,m._I_1k,m.Q_uk),(raw_i,m._I_Np1k,-m.Q_dk),(raw_i,m._Ik,-m.Q_ik),(raw_i,m._Ip1k,m.Q_ik)]:np.add.at(net,indices,values)
    store_j=m._A_sj*(m.H_j-H)/dt;store_i=m._E_Ik*(m.h_Ik-h)
    # Include the native half-end storage explicitly; default full model may have zero end lengths.
    xi_u=m._dx_uk*m._B_uk*m._theta_uk/(2*dt);xi_d=m._dx_dk*m._B_dk*m._theta_dk/(2*dt)
    endpoint_store=np.zeros(m.M)
    np.add.at(endpoint_store,m._J_uk,xi_u*(m.H_j[m._J_uk]-H[m._J_uk]))
    np.add.at(endpoint_store,m._J_dk,xi_d*(m.H_j[m._J_dk]-H[m._J_dk]))
    linear_j=raw_j.copy()
    np.add.at(linear_j,m._J_uo,m.Q_o-linear);np.add.at(linear_j,m._J_do,linear-m.Q_o)
    cut_total=qm-qg+float(wj@kw['Q_in']+wi@kw['Q_0Ik'])-float(wj@store_j+wi@store_i)
    node_raw=float(wj@(raw_j-store_j));node_linear=float(wj@(linear_j-store_j-endpoint_store));internal=float(wi@(raw_i-store_i));gate_gap=float(wj@(raw_j-linear_j));endpart=float(wj@endpoint_store)
    reconstructed=node_linear+internal+gate_gap+endpart
    assert abs(cut_total-reconstructed)<1e-10
    assert abs(cut_total-expected['continuity_residual_m3s'])<1e-10
    assert qm==expected['Q16_cut_m3s'] and qg==expected['Q16_gate_m3s']
    support_j=np.flatnonzero(wj);support_i=np.flatnonzero(wi)
    # This is a decomposition of the existing solver result, not a corrected hydraulic run.
    result=dict(status='COMPLETED',case=case,start_s=t0,end_s=float(m.t),dt_s=dt,
        same_step_Qcut_and_Qgate_exactly_reproduced=True,recorded_total_residual_m3s=cut_total,
        weighted_superjunction_raw_residual_m3s=node_raw,weighted_superjunction_linear_residual_m3s=node_linear,
        weighted_internal_reconstruction_residual_m3s=internal,weighted_reported_vs_linear_gate_contribution_m3s=gate_gap,
        weighted_endpoint_storage_omitted_by_old_formula_m3s=endpart,decomposition_sum_m3s=reconstructed,
        decomposition_error_m3s=cut_total-reconstructed,selected_gate_linear_Q_m3s=linear[gids].tolist(),selected_gate_reported_Q_m3s=m.Q_o[gids].tolist(),
        superjunction_support=[dict(index=int(i),weight=float(wj[i]),raw_residual=float(raw_j[i]-store_j[i]),linear_residual=float(linear_j[i]-store_j[i]-endpoint_store[i])) for i in support_j],
        internal_support=[dict(index=int(i),weight=float(wi[i]),residual=float(raw_i[i]-store_i[i]),D_definition_error_m3s=float(m._D_Ik[i]-kw['Q_0Ik'][i]-m._E_Ik[i]*h[i])) for i in support_i],
        max_end_lengths_m=[float(m._dx_uk.max()),float(m._dx_dk.max())],last_head_residual_m=float(m.last_head_residual),last_gate_flow_residual_m3s=float(m.last_gate_flow_residual),iter_elapsed=int(m.iter_elapsed),
        original_1e6_gate_pass=abs(cut_total)<=1e-6,threshold_unchanged=True,hydraulic_equations_unchanged=True,
        runtime_identity=api.runtime_identity(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_seconds=time.time()-started)
    full.dump(out/'最坏子步残差分解.json',result);np.savez_compressed(out/'最坏步后状态.npz',**{k:getattr(m,k) for k in api.STATE_KEYS})
    full.dump(out/'进度.json',dict(status='COMPLETED',time_s=float(m.t),wall_seconds=time.time()-started))
    print(json.dumps({k:result[k] for k in ['case','recorded_total_residual_m3s','weighted_superjunction_linear_residual_m3s','weighted_internal_reconstruction_residual_m3s','weighted_reported_vs_linear_gate_contribution_m3s','weighted_endpoint_storage_omitted_by_old_formula_m3s']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
