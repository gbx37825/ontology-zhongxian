"""同C00同4h状态的联合欠松弛固定点；不改变冻结物理hook/主runner。
仅以未松弛F(x)-x判收敛；每步起始states与时间固定。失败步仅保留原生候选作诊断。
"""
from pathlib import Path
import argparse,json,hashlib,time,copy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from 试验公共 import R,Forcing,flow_operators,control_volumes
from 并联结构物理修复 import build_engine_physics
KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
def state(m):return {k:getattr(m,k).copy() for k in KEYS}
def flux(m,kw):
    M=m.M;balance=np.r_[kw['Q_in'],kw['Q_0Ik']].copy()
    for key,u,v in [('Q_ik',m._Ik+M,m._Ip1k+M),('Q_uk',m._J_uk,m._I_1k+M),('Q_dk',m._I_Np1k+M,m._J_dk),('Q_o',m._J_uo,m._J_do)]:
        q=getattr(m,key);np.add.at(balance,u,-q);np.add.at(balance,v,q)
    return balance

def damped_step(m,kw,omega,max_iter,head_tol,flow_tol):
    dt=kw['dt'];t0=float(m.t);count0=m.iter_count;m.save_state();frozen=state(m);m.states['t']=t0
    raw={};converged=False
    call={**kw,'first_iter':False,'first_time':False,'implicit':True,'banded':m.banded};call.pop('num_iter',None);call.pop('head_tol',None)
    for it in range(1,max_iter+1):
        trial=state(m);m.t=t0;m.iter_count=count0
        m._setup_step(**call);mass_area=m._A_sj.copy();internal_E=m._E_Ik.copy();m._solve_step(**call);candidate=state(m)
        raw={k:float(np.max(abs(candidate[k]-trial[k]))) for k in KEYS}
        finite=all(np.isfinite(v).all() for v in candidate.values())
        converged=finite and max(raw['H_j'],raw['h_Ik'])<head_tol and max(raw[k] for k in KEYS[2:])<flow_tol
        if converged or not finite or it==max_iter:break
        mixed={k:trial[k]+omega*(candidate[k]-trial[k]) for k in KEYS}
        mixed['H_j'][m.bc]=kw['H_bc'][m.bc]
        m.load_state(mixed,compute_hydraulic_geometries=True)
    assert all(np.array_equal(m.states[k],frozen[k]) for k in KEYS)
    assert m.states['t']==t0
    assert abs(m.t-(t0+dt))<1e-9
    assert m.iter_count==count0+1
    m.iter_elapsed=it
    # 最后原生候选，使用其所对应的离散存储系数，未将不收敛候选当正式可接受解。
    storage=np.r_[mass_area*(m.H_j-frozen['H_j'])/dt,internal_E*(m.h_Ik-frozen['h_Ik'])]
    continuity=flux(m,kw)-storage
    return dict(iterations=it,converged=bool(converged),finite=finite,at_limit=it==max_iter,**{'raw_'+k:val for k,val in raw.items()}),continuity

def main():
    p=argparse.ArgumentParser();p.add_argument('--omega',type=float,default=.5);p.add_argument('--dt',type=float,default=15.);p.add_argument('--max-iter',type=int,default=30);p.add_argument('--name',default='PHV03_C00联合欠松弛05');p.add_argument('--head-tol',type=float,default=.0015);p.add_argument('--flow-tol',type=float,default=.01);a=p.parse_args();assert 0<a.omega<=1
    out=R/'06_结果'/a.name;out.mkdir(exist_ok=False);base=R/'06_结果/P00_并联物理修复稳态6小时';start=time.time()
    with np.load(base/'内部断面状态.npz') as d:
        j=int(np.flatnonzero(d['time_s']==14400)[0]);initial={k:d[k][j].copy() for k in KEYS}
    norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());e,s,c=build_engine_physics(gate_max_normalization=norm);m=e.model;f=Forcing(e,s,c,positioned=True);m.load_state({**initial,'t':14400.});ops,_=flow_operators(e,s,c);W,cv=control_volumes(e,ops);W=csr_matrix(W)
    gg=c['orifices'][c['orifices'].name.str.startswith('JZZ42-')];up=int(gg.iloc[0].sj_0);down=int(gg.iloc[0].sj_1);ids=gg.id.to_numpy(int);dynamic=np.r_[~m.bc,np.ones(m.NIk,dtype=bool)];rows=[];ledger=[]
    for i in range(int(round(7200/a.dt))):
        t=14400+i*a.dt;kw=f.at(t);kw['dt']=a.dt;diag,res=damped_step(m,kw,a.omega,a.max_iter,a.head_tol,a.flow_tol);now=t+a.dt
        cr=np.asarray(W@res).ravel()
        rows.append(dict(time_s=now,model_t=m.t,forebay_m=float(m.H_j[up]),afterbay_m=float(m.H_j[down]),gate_flow=float(m.Q_o[ids].sum()),max_control_volume_residual_m3s=float(abs(cr).max()),max_dynamic_node_continuity_residual_m3s=float(abs(res[dynamic]).max()),**diag))
        for j,x in enumerate(cv):ledger.append(dict(time_s=now,station_code=x['station'],continuity_residual_m3s=float(cr[j])))
        if not diag['finite']:break
    df=pd.DataFrame(rows);df.to_csv(out/'逐步未松弛残差与JZZ42.csv',index=False);pd.DataFrame(ledger).to_csv(out/'逐步测流至闸门连续残差.csv',index=False)
    summ=dict(status='PASS' if len(df)==round(7200/a.dt) and df.converged.all() else 'FAILED_NUMERICAL_GATE',omega=a.omega,dt=a.dt,max_iter=a.max_iter,head_tol=a.head_tol,flow_tol=a.flow_tol,flow_tol_basis='沿用既有预热流量容差0.01m3/s，另列H/h/Q逐项残差',steps=len(df),at_limit_count=int(df.at_limit.sum()),unconverged_count=int((~df.converged).sum()),node_head_unconverged_count=int((df.raw_H_j>=a.head_tol).sum()),mean_iterations=float(df.iterations.mean()),max_raw_residuals={k:float(df['raw_'+k].max()) for k in KEYS},max_control_volume_residual_m3s=float(df.max_control_volume_residual_m3s.max()),max_dynamic_node_continuity_residual_m3s=float(df.max_dynamic_node_continuity_residual_m3s.max()),min_Q42=float(df.gate_flow.min()),max_Q42=float(df.gate_flow.max()),min_H42up=float(df.forebay_m.min()),max_H42up=float(df.forebay_m.max()),point_5p5h=df[df.time_s==19800].to_dict('records'),elapsed_seconds=time.time()-start,start_states_frozen_verified=True,model_time_advances_once_per_step_verified=True,failed_step_policy='retain last unrelaxed native candidate only to complete diagnostic; not a production accepted state',input='C00 unchanged',hook_sha256=hashlib.sha256((R/'04_脚本/并联结构物理修复.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'欠松弛核验.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2));print(json.dumps(summ,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
