"""固定最早出现补值差异的窗口，保留共同历史推进，不按误差选窗。"""
from pathlib import Path
import sys,json,hashlib,time,copy,traceback
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';sys.path[:0]=[str(R/'30_分通道方法逐项测试/03_稳态与分通道试验')]
from 试验水力工具 import build,advance,STATE_KEYS,restore
from 常驻水力进程 import jit_evidence

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shaarr(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
audit=pd.read_csv(O/'I03_袁庄补值前值保持三天/袁庄已补值点改动.csv');different=audit[audit.delta_m3s.abs()>1e-12];start_s=int(different.time_s.min())-1800;end_s=start_s+259200
assert start_s==1231200
origin=pd.Timestamp('2025-07-22 12:00:00')
dump(O/'I03_袁庄补值前值保持三天/无差异预检.json',dict(status='NO_CHANGE',first_three_day_fill_points=int((audit.time_s<=259200).sum()),changed_points=0,reason='首三天106个已补点两种补值都为0.4m3/s，按输入恒等可复用原输入三天结果；不重复水力计算',first_different_sample_s=int(start_s+1800),next_window_start_s=start_s,next_window_end_s=end_s,selection='全月最早发生补值差异样本前半小时，固定推进三天，与水位误差无关'))
e,s,c,f,params=build();m=e.model;beg=time.time();retry=0;hh=qq=0.;prefix=O/'袁庄共同历史推进';prefix.mkdir(exist_ok=False)
try:
 for k in range(start_s//1800):
  st=advance(m,f,k*1800,(k+1)*1800);retry+=st[0];hh=max(hh,st[1]);qq=max(qq,st[2])
  if (k+1)%48==0:
   p=dict(status='RUNNING',completed_cycles=k+1,total_cycles=start_s//1800,time_s=float(m.t),retries=retry,wall_seconds=time.time()-beg);dump(prefix/'进度.json',p);print('共同历史',p,flush=True)
 np.savez_compressed(prefix/'分叉起点.npz',**{key:getattr(m,key) for key in STATE_KEYS});dump(prefix/'进度.json',dict(status='COMPLETED',completed_cycles=start_s//1800,total_cycles=start_s//1800,time_s=float(m.t),retries=retry,wall_seconds=time.time()-beg,max_head_residual_m=float(hh),max_gate_residual_m3s=float(qq),numba_actual=jit_evidence(),purpose='仅建立两支共同历史状态，无同化，不在此段调参或评分选窗'))
except BaseException as ex:dump(prefix/'失败.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
statehash={k:shaarr(getattr(m,k)) for k in STATE_KEYS};paramhash={k:shaarr(getattr(m,k)) for k in ['_n_ik','_Co','_C_uk','_C_dk']};groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')}
for name,hold in [('I04_袁庄最早差异窗原插值三天',False),('I05_袁庄最早差异窗前值保持三天',True)]:
 d=O/name;d.mkdir(exist_ok=False);member=copy.deepcopy(m);forcing=copy.copy(f);forcing.q0=f.q0.copy();rows=[];started=time.time();retries=0;maxh=maxq=0.;delta=np.zeros(len(f.times))
 if hold:
  delta[audit.time_s.to_numpy(int)//1800]=audit.delta_m3s.to_numpy();route=next(x for x in f.routing if x['target']=='FSK63-1-pool33');forcing.q0[:,np.array(route['junction_ids'])]-=delta[:,None]*np.array(route['weights'])
 assert all(np.array_equal(getattr(member,key),getattr(m,key)) for key in STATE_KEYS)
 scope=(f.times>=start_s)&(f.times<=end_s);net=-delta
 cfg=dict(case=name,days=3,cycles=144,members=1,assimilation=False,start_s=start_s,end_s=end_s,start_time=str(origin+pd.Timedelta(seconds=start_s)),end_time=str(origin+pd.Timedelta(seconds=end_s)),input_sha256=sha(B/'共享输入/C07.json'),initial_sha256=sha(B/'共享输入/初始化后状态_0s.npz'),initial_state_sha256=statehash,parameter_sha256=paramhash,common_history_checkpoint_sha256=sha(prefix/'分叉起点.npz'),common_history_seconds=start_s,window_selection='最早有插值差异样本前半小时到三天后，不按误差选择',input_net_delta_m3=float(np.trapezoid(net[scope],f.times[scope])),input_net_delta_mean_m3s=float(np.trapezoid(net[scope],f.times[scope])/259200),changed_window_grid_points=int((abs(delta[scope])>1e-12).sum()),changed_month_grid_points=int((abs(delta)>1e-12).sum()),max_abs_window_flow_delta_m3s=float(abs(delta[scope]).max()),changed_only_documented_fills=True,all_original_valid_samples_preserved=True,strict=True,max_iterations=20,min_failed_step_s=1.875,numerical_interpolation='两组半小时网格间同用原Forcing线性插值；差异只在已补值半小时点')
 dump(d/'运行配置.json',cfg)
 try:
  for k in range(144):
   t=start_s+k*1800;st=advance(member,forcing,t,t+1800);retries+=st[0];maxh=max(maxh,st[1]);maxq=max(maxq,st[2])
   for station,(ju,jd,gi) in groups.items():rows.append(dict(time_s=t+1800,station=station,forebay_m=float(member.H_j[ju]),afterbay_m=float(member.H_j[jd]),gate_flow_m3s=float(member.Q_o[gi].sum())))
   if (k+1)%12==0:
    p=dict(status='RUNNING',completed_cycles=k+1,total_cycles=144,wall_seconds=time.time()-started,retries=retries);dump(d/'进度.json',p);print(name,p,flush=True)
  assert all(paramhash[key]==shaarr(getattr(member,key)) for key in paramhash)
  pd.DataFrame(rows).to_csv(d/'半小时水位流量.csv',index=False);dump(d/'进度.json',dict(status='COMPLETED',completed_cycles=144,total_cycles=144,days=3,finite_state=all(np.isfinite(getattr(member,key)).all() for key in STATE_KEYS),parameters_unchanged=True,wall_seconds=time.time()-started,retries=retries,max_head_residual_m=float(maxh),max_gate_flow_residual_m3s=float(maxq),numba_actual=jit_evidence(),net_input_delta_m3=cfg['input_net_delta_m3']))
 except BaseException as ex:dump(d/'失败.json',dict(error=repr(ex),traceback=traceback.format_exc(),completed_cycles=len(rows)//59));raise
