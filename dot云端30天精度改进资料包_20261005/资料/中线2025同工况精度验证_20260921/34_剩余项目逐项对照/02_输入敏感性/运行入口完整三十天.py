"""补齐事先固定入口幅度的后27天保留检验；保持三天证据。"""
from pathlib import Path
import sys,json,hashlib,time,os,argparse,traceback,csv
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';sys.path[:0]=[str(R/'30_分通道方法逐项测试/03_稳态与分通道试验')]
from 试验水力工具 import build,advance,STATE_KEYS
from 常驻水力进程 import jit_evidence

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shaar(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(p)
p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['M00','M01','M02']);a=p.parse_args();lookup={'M00':('M00_原输入完整30天','I00_原输入三天',0.),'M01':('M01_陶岔增加完整30天','I01_陶岔增加三天',1.),'M02':('M02_陶岔减少完整30天','I02_陶岔减少三天',-1.)};name,prior,sign=lookup[a.case];d=O/name;d.mkdir(exist_ok=False);started=time.time()
reference=pd.read_csv(R/'07_对比分析/2025水量闭合冻结候选与保留验证.csv');amp=abs(float(reference.loc[reference.candidate=='陶岔至湍河合并控制体的入口小偏移假设','frozen_correction_m3s'].iloc[0]));assert abs(amp-.5528845617844347)<1e-12;offset=sign*amp
try:
 e,s,c,f,params=build();m=e.model;statehash={k:shaar(getattr(m,k)) for k in STATE_KEYS};phash={k:shaar(getattr(m,k)) for k in ['_n_ik','_Co','_C_uk','_C_dk']};oldcfg=json.loads((O/prior/'运行配置.json').read_text());assert oldcfg['initial_state_sha256']==statehash and oldcfg['parameter_sha256']==phash
 before=f.q.sum(axis=1)+f.q0.sum(axis=1)
 if offset:f.q[:,e.node_index['NODE-main-0']]+=offset
 delta=f.q.sum(axis=1)+f.q0.sum(axis=1)-before
 cfg=dict(case=name,prior_3day=prior,days=30,total_cycles=1440,members=1,assimilation=False,pid=os.getpid(),start_s=0,end_s=2592000,start_time='2025-07-22 12:00:00',end_time='2025-08-21 12:00:00',training_end_s=259200,heldout_start_exclusive_s=259200,input_sha256=sha(B/'共享输入/C07.json'),initial_sha256=sha(B/'共享输入/初始化后状态_0s.npz'),initial_state_sha256=statehash,parameter_sha256=phash,runner_sha256=sha(Path(__file__)),inlet_delta_m3s=offset,amplitude_source='既有陶岔至湍河联合控制体首72h诊断；正负幅度预先固定，非真实流量修正',net_input_delta_m3=float(np.trapezoid(delta,f.times)),strict=True,max_iterations=20,min_failed_step_s=1.875,output_step_s=1800,note='已知历史边界，无同化单成员诊断，不是50成员B02精度')
 dump(d/'运行配置.json',cfg);groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')};retries=0;maxh=maxq=0.;prefix=[];completed=0
 dump(d/'进度.json',dict(status='RUNNING',pid=os.getpid(),completed_cycles=0,total_cycles=1440,wall_seconds=time.time()-started))
 with (d/'半小时水位流量.csv').open('w',newline='') as handle:
  w=csv.DictWriter(handle,fieldnames=['time_s','station','forebay_m','afterbay_m','gate_flow_m3s']);w.writeheader()
  for k in range(1440):
   st=advance(m,f,k*1800,(k+1)*1800);retries+=st[0];maxh=max(maxh,st[1]);maxq=max(maxq,st[2])
   for station,(ju,jd,gi) in groups.items():
    row=dict(time_s=(k+1)*1800,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[gi].sum()));w.writerow(row)
    if k<144:prefix.append(row)
   completed=k+1
   if completed==144:
    old=pd.read_csv(O/prior/'半小时水位流量.csv').sort_values(['station','time_s']).reset_index(drop=True);new=pd.DataFrame(prefix).sort_values(['station','time_s']).reset_index(drop=True);assert new[['station','time_s']].equals(old[['station','time_s']]);cols=['forebay_m','afterbay_m','gate_flow_m3s'];diff=float(abs(new[cols].to_numpy()-old[cols].to_numpy()).max());assert diff<1e-12,diff;dump(d/'前三天复现核验.json',dict(samples=len(old),max_abs_difference=diff,exact_float64_before_csv=bool(np.array_equal(new[cols].to_numpy(),old[cols].to_numpy())),tolerance=1e-12,reason='CSV解析的末位舍入允许1e-12，水力输入参数初态完全相同'))
   if completed%12==0:
    handle.flush();dump(d/'Numba实际执行证据.json',jit_evidence());status=dict(status='RUNNING',pid=os.getpid(),completed_cycles=completed,total_cycles=1440,days=completed/48,wall_seconds=time.time()-started,retries=retries,max_head_residual_m=float(maxh),max_gate_flow_residual_m3s=float(maxq));dump(d/'进度.json',status);print(json.dumps(status),flush=True)
 assert all(phash[key]==shaar(getattr(m,key)) for key in phash);assert sha(B/'共享输入/C07.json')==cfg['input_sha256'];assert sha(B/'共享输入/初始化后状态_0s.npz')==cfg['initial_sha256'];np.savez_compressed(d/'末态.npz',**{key:getattr(m,key) for key in STATE_KEYS});status.update(status='COMPLETED',finite_state=all(np.isfinite(getattr(m,key)).all() for key in STATE_KEYS),parameters_unchanged=True,source_unchanged=True,wall_seconds=time.time()-started);dump(d/'进度.json',status);print(json.dumps(status),flush=True)
except BaseException as ex:
 dump(d/'失败.json',dict(error=repr(ex),traceback=traceback.format_exc(),completed_cycles=locals().get('completed',0),pid=os.getpid()));raise
