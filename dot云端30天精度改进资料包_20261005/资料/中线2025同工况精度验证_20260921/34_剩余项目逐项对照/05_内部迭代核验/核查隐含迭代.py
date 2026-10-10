from pathlib import Path
import sys,json,copy,types,time,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];B=R/'26_冻结基线与后续归因';O=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
from 试验公共 import Forcing
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import STATE_KEYS,restore
e,s,c=build_engine_physics(B/'共享输入/C07.json',gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()));m=e.model;f=Forcing(e,s,c)
with np.load(B/'共享输入/初始化后状态_0s.npz') as z:restore(m,{k:z[k] for k in STATE_KEYS})
m.t=0.;m.iter_count=0;native=type(m)._solve_step
# A bound wrapper records iteration changes only; it does not change convergence decisions.
def observed_solve(self,**kwargs):
 h=self.h_Ik.copy();q=self.Q_ik.copy();result=native(self,**kwargs);self._iteration_audit.append(dict(h_change=float(abs(self.h_Ik-h).max()),Q_change=float(abs(self.Q_ik-q).max())));return result
m._solve_step=types.MethodType(observed_solve,m);rows=[];outputs=[];retries=0;beg=time.time()
for k in range(144):
 for ts in np.arange(k*1800,(k+1)*1800,15.):
  pending=[(ts,15.)]
  while pending:
   t,dt=pending.pop(0);kw=f.at(t);kw.update(dt=dt,num_iter=20,strict=True);m._iteration_audit=[]
   try:m.step(**kw)
   except RuntimeError as ex:
    if 'did not converge' not in str(ex) or dt<=1.875:raise
    retries+=1;pending[:0]=[(t,dt/2),(t+dt/2,dt/2)];continue
   last=m._iteration_audit[-1]
   if m.iter_elapsed>1 and (last['h_change']>=.0015 or last['Q_change']>=.01):rows.append(dict(time_s=m.t,iterations=m.iter_elapsed,head_residual=m.last_head_residual,gate_residual=m.last_gate_flow_residual,internal_h_last_iter=last['h_change'],internal_Q_last_iter=last['Q_change']))
 # On a copy, make one more fixed-point iteration at the SAME physical time step.
 shadow=copy.deepcopy(m);shadow.t-=dt;shadow.iter_count-=1;arguments={k:v for k,v in kw.items() if k not in ['num_iter','strict']};arguments['first_iter']=False
 shadow._setup_step(**arguments);shadow._solve_step(**arguments)
 outputs.append(dict(time_s=m.t,accepted_iterations=m.iter_elapsed,max_extra_H_change=float(abs(shadow.H_j-m.H_j).max()),max_extra_h_change=float(abs(shadow.h_Ik-m.h_Ik).max()),max_extra_Q_change=float(abs(shadow.Q_ik-m.Q_ik).max()),max_extra_gate_Q_change=float(abs(shadow.Q_o-m.Q_o).max())))
 if (k+1)%48==0:print('day',(k+1)//48,flush=True)
pd.DataFrame(outputs).to_csv(O/'每半小时追加一次同时间迭代.csv',index=False);pd.DataFrame(rows).to_csv(O/'主节点通过但内部变化仍大.csv',index=False)
with np.load(R/'34_剩余项目逐项对照/03_初态与步长/I00_原初态15秒/末态.npz') as z:assert all(np.array_equal(getattr(m,k),z[k]) for k in STATE_KEYS)
r=dict(status='COMPLETED',steps=17280,periods=144,non_mutating_trace_final_exact=True,strict_pass_but_large_internal_last_iter_count=len(rows),shadow_max_H_change=max(r['max_extra_H_change'] for r in outputs),shadow_max_h_change=max(r['max_extra_h_change'] for r in outputs),shadow_max_Q_change=max(r['max_extra_Q_change'] for r in outputs),retries=retries,wall_seconds=time.time()-beg,note='只读迭代审计；额外迭代只在副本上执行，未回灌轨迹。内部改变量是非线性迭代收敛诊断，不是与实测误差。')
(O/'内部迭代核验.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False),flush=True)
