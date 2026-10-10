"""保持水面、参数、输入一致，仅检查初始代数闸流一致化和时间步减半。"""
from pathlib import Path
import sys,json,time,hashlib,argparse,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];B=R/'26_冻结基线与后续归因';sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
from 试验公共 import Forcing
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import STATE_KEYS,restore
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dt',type=float,default=15);p.add_argument('--sync-gate-flow',action='store_true');a=p.parse_args();O=Path(__file__).resolve().parent/a.name;O.mkdir(exist_ok=False);beg=time.time()
e,s,c=build_engine_physics(B/'共享输入/C07.json',gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()));m=e.model;f=Forcing(e,s,c)
with np.load(B/'共享输入/初始化后状态_0s.npz') as z:initial={k:z[k].copy() for k in STATE_KEYS}
restore(m,initial);m.t=0.;m.iter_count=0
before={k:getattr(m,k).copy() for k in STATE_KEYS}
if a.sync_gate_flow:m.solve_orifice_flows(dt=15.,u=f.at(0)['u_o'])
assert all(np.array_equal(before[k],getattr(m,k)) for k in STATE_KEYS if k!='Q_o')
config=dict(name=a.name,dt=a.dt,sync_gate_flow=a.sync_gate_flow,initial_Qo_max_change=float(abs(m.Q_o-before['Q_o']).max()),initial_non_Qo_states_equal=True,input_sha256=hashlib.sha256((B/'共享输入/C07.json').read_bytes()).hexdigest(),initial_sha256=hashlib.sha256((B/'共享输入/初始化后状态_0s.npz').read_bytes()).hexdigest(),note='只调整初始代数闸流，不等同全渠道初始水位流量场已率定。非恒定渠内水面与流量不满足稳态关系，本身不能认定为错误。无同化，原糙率/闸系数/输入不变，15s与7.5s均用相同严格收敛门槛。')
(O/'运行配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')};rows=[];checks=[];retries=0;maxhead=0.;maxgate=0.
try:
 for k in range(144):
  t0=k*1800
  for ts in np.arange(t0,t0+1800,a.dt):
   pending=[(ts,a.dt)]
   while pending:
    t,dt=pending.pop(0);kw=f.at(t);kw.update(dt=dt,num_iter=20,strict=True)
    try:m.step(**kw)
    except RuntimeError as ex:
     if 'did not converge' not in str(ex) or dt<=1.875:raise
     retries+=1;pending[:0]=[(t,dt/2),(t+dt/2,dt/2)];continue
    assert all(np.isfinite(getattr(m,key)).all() for key in STATE_KEYS);maxhead=max(maxhead,m.last_head_residual);maxgate=max(maxgate,m.last_gate_flow_residual)
    if abs(m.t-15.)<1e-6:
     checks.append(dict(time_s=m.t,max_H_change=float(abs(m.H_j-before['H_j']).max()),max_internal_Q_change=float(abs(m.Q_ik-before['Q_ik']).max())))
  for st,(ju,jd) in groups.items():rows.append(dict(time_s=t0+1800,station=st,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd])))
  if (k+1)%48==0:print('DAY',(k+1)//48,flush=True)
 pd.DataFrame(rows).to_csv(O/'水位.csv',index=False);np.savez_compressed(O/'末态.npz',**{key:getattr(m,key) for key in STATE_KEYS});result=dict(status='COMPLETED',periods=144,days=3,retries=retries,max_head_residual=maxhead,max_gate_residual=maxgate,wall_seconds=time.time()-beg,first_15s=checks)
except Exception as ex:
 result=dict(status='FAILED',error=repr(ex),periods=len(rows)/59,wall_seconds=time.time()-beg);raise
finally:(O/'结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
