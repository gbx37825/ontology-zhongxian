"""同一4h物理状态/C00输入，只改变fixed-point设置；不改已冻结物理hook。"""
from pathlib import Path
import argparse,json,types,hashlib,time
import numpy as np
import pandas as pd
from 试验公共 import R,Forcing
from 并联结构物理修复 import build_engine_physics

def track_solve(self,*args,**kwargs):
    result=self._diag_original_class._solve_step(self,*args,**kwargs)
    self._diag_heads.append(self.H_j.copy())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-name',default='PHV02_实际开度非线性迭代窗');p.add_argument('--dt',type=float,default=15.);p.add_argument('--iterations',default='5,10,20');p.add_argument('--head-tol',type=float,default=.0015);a=p.parse_args()
    out=R/'06_结果'/a.out_name;out.mkdir(exist_ok=False)
    base=R/'06_结果/P00_并联物理修复稳态6小时'
    with np.load(base/'内部断面状态.npz') as d:
        j=int(np.flatnonzero(d['time_s']==14400)[0]);initial={k:d[k][j].copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']}
    norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());results=[]
    for niter in map(int,a.iterations.split(',')):
        started=time.time();e,s,c=build_engine_physics(gate_max_normalization=norm);m=e.model;f=Forcing(e,s,c,positioned=True);m.load_state({**initial,'t':14400.})
        m._diag_original_class=type(m);m._solve_step=types.MethodType(track_solve,m)
        gg=c['orifices'][c['orifices'].name.str.startswith('JZZ42-')];up=int(gg.iloc[0].sj_0);down=int(gg.iloc[0].sj_1);ids=gg.id.to_numpy(int);rows=[];failure=None
        for i in range(int(round(7200/a.dt))):
            t=14400+i*a.dt;kw=f.at(t);kw.update(dt=a.dt,num_iter=niter,head_tol=a.head_tol);before=m.H_j.copy();m._diag_heads=[]
            try:m.step(**kw)
            except Exception as err:failure=repr(err);break
            now=t+a.dt;finite=bool(np.isfinite(m.H_j).all() and np.isfinite(m.Q_ik).all() and np.isfinite(m.Q_o).all())
            prev=m._diag_heads[-2] if len(m._diag_heads)>=2 else before
            residual=float(abs(m.H_j-prev).max());assert len(m._diag_heads)==m.iter_elapsed
            rows.append(dict(time_s=now,forebay_m=float(m.H_j[up]),afterbay_m=float(m.H_j[down]),gate_flow=float(m.Q_o[ids].sum()),iterations=int(m.iter_elapsed),last_head_residual_m=residual,at_iteration_limit=bool(m.iter_elapsed==niter),unconverged=bool(residual>=a.head_tol),max_all_node_step_change_m=float(abs(m.H_j-before).max()),finite=finite))
            if not finite:failure=f'nonfinite at {now}';break
        df=pd.DataFrame(rows);df.to_csv(out/f'iter_{niter}_dt_{a.dt:g}_tol_{a.head_tol:g}.csv',index=False)
        result=dict(num_iter=niter,dt=a.dt,head_tol=a.head_tol,status='FAILED' if failure else 'COMPLETED',failure=failure,steps=len(df),wall_seconds=time.time()-started,at_limit_count=int(df.at_iteration_limit.sum()),unconverged_count=int(df.unconverged.sum()),mean_iterations=float(df.iterations.mean()),max_last_head_residual_m=float(df.last_head_residual_m.max()),min_gate_flow=float(df.gate_flow.min()),max_gate_flow=float(df.gate_flow.max()),min_forebay_m=float(df.forebay_m.min()),max_forebay_m=float(df.forebay_m.max()),max_all_node_step_change_m=float(df.max_all_node_step_change_m.max()),point_5p5h=df[df.time_s==19800].to_dict('records'))
        results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        (out/'迭代核验.json').write_text(json.dumps(dict(start_h=4,end_h=6,model='C00 unchanged',initial='same physical dt15 state at4h',cases=results,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hook_sha256=hashlib.sha256((R/'04_脚本/并联结构物理修复.py').read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
