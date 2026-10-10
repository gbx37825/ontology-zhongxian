"""在冻结起点上逐个推进单成员三天，复用严格求解器，不改输入真值。"""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因'
sys.path[:0]=[str(R/'30_分通道方法逐项测试/03_稳态与分通道试验')]
from 试验水力工具 import build,advance,STATE_KEYS
from 常驻水力进程 import jit_evidence

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shaarr(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))

# 固定首三天，不根据评分选窗口。保持有效实测，只有初版补值记录中袁庄点可变。
raw=pd.read_csv(R/'02_输入/2025分退水原始窗口.csv',parse_dates=['时间']);raw=raw[raw['闸站名称']=='袁庄分水口'].set_index('时间').sort_index()['总瞬时流量']
raw_good=raw.where(np.isfinite(raw)&(raw>-100));held=raw_good.ffill()
fix=pd.read_csv(R/'03_运行参数/2025输入补值逐点记录.csv',parse_dates=['timestamp']);fix=fix[(fix.kind=='分退水')&(fix.name=='袁庄分水口')]
start=pd.Timestamp('2025-07-22 12:00:00');grid=pd.date_range(start,periods=1441,freq='30min');missing=grid.isin(fix.timestamp)
assert np.array_equal(missing,raw_good.reindex(grid).isna().to_numpy())
source=B/'共享输入/C07.json';initial=B/'共享输入/初始化后状态_0s.npz'
reference=json.loads((R/'07_对比分析/2025水量闭合冻结候选与保留验证.csv').read_text()) if False else pd.read_csv(R/'07_对比分析/2025水量闭合冻结候选与保留验证.csv')
amplitude=abs(float(reference.loc[reference.candidate=='陶岔至湍河合并控制体的入口小偏移假设','frozen_correction_m3s'].iloc[0]))
assert abs(amplitude-.5528845617844347)<1e-12
cases=[('I00_原输入三天',0.,False),('I01_陶岔增加三天',amplitude,False),('I02_陶岔减少三天',-amplitude,False),('I03_袁庄补值前值保持三天',0.,True)]
locked=None;finished=[]
for name,offset,hold in cases:
 d=O/name;d.mkdir(exist_ok=False);started=time.time();e,s,c,f,params=build();m=e.model
 statehash={k:shaarr(getattr(m,k)) for k in STATE_KEYS};phash={k:shaarr(getattr(m,k)) for k in ['_n_ik','_Co','_C_uk','_C_dk']}
 if locked is None:locked=(statehash,phash)
 assert locked==(statehash,phash)
 input_before_q=f.q.sum(axis=1)+f.q0.sum(axis=1)
 if offset:f.q[:,e.node_index['NODE-main-0']]+=offset
 changeinfo={}
 if hold:
  target='FSK63-1-pool33';old=f.bleeder_values[target].copy();new=old.copy();new[missing]=held.reindex(grid).to_numpy()[missing]
  assert np.array_equal(old[~missing],new[~missing]);assert np.isfinite(new).all()
  route=next(x for x in f.routing if x['target']==target);assert route['mode']=='按模型真实桩号向相邻内部节点守恒分配'
  ii=np.asarray(route['junction_ids']);w=np.asarray(route['weights']);delta=new-old;f.q0[:,ii]-=delta[:,None]*w;f.bleeder_values[target]=new
  audit=pd.DataFrame(dict(timestamp=grid,time_s=f.times,allowed_fill=missing,original=old,candidate=new,delta_m3s=delta));audit=audit[audit.allowed_fill];audit.to_csv(d/'袁庄已补值点改动.csv',index=False)
  changeinfo=dict(target=target,only_original_missing_changed=True,valid_samples_unchanged=True,permitted_month_points=int(missing.sum()),changed_month_points=int(np.count_nonzero(delta)),changed_first_3day_points=int(np.count_nonzero(delta[f.times<=259200])),month_max_abs_delta_m3s=float(abs(delta).max()),first_3day_max_abs_delta_m3s=float(abs(delta[f.times<=259200]).max()),same_runtime_linear_interpolation=True)
  assert changeinfo['changed_first_3day_points']>0
 input_delta=f.q.sum(axis=1)+f.q0.sum(axis=1)-input_before_q
 mask=f.times<=259200;config=dict(case=name,days=3,cycles=144,start_time=str(start),end_time=str(start+pd.Timedelta(days=3)),members=1,assimilation=False,initial_source=str(initial),initial_sha256=sha(initial),input_sha256=sha(source),runner_sha256=sha(Path(__file__)),initial_state_sha256=statehash,parameter_sha256=phash,inlet_delta_m3s=offset,amplitude_source='07_对比分析/2025水量闭合冻结候选与保留验证.csv 合并控制体诊断，不是真值',yuan_fill=changeinfo,input_net_delta_m3=float(np.trapezoid(input_delta[mask],f.times[mask])),input_net_delta_mean_m3s=float(np.trapezoid(input_delta[mask],f.times[mask])/259200),output_step_s=1800,solver_step_s=15,strict=True,max_iterations=20,min_failed_step_s=1.875)
 dump(d/'运行配置.json',config);groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')};rows=[];retries=0;maxh=maxq=0.;status={}
 try:
  for k in range(144):
   stats=advance(m,f,k*1800,(k+1)*1800);retries+=stats[0];maxh=max(maxh,stats[1]);maxq=max(maxq,stats[2])
   for st,(ju,jd,gi) in groups.items():rows.append(dict(time_s=(k+1)*1800,station=st,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[gi].sum())))
   if (k+1)%12==0:
    status=dict(status='RUNNING',completed_cycles=k+1,total_cycles=144,wall_seconds=time.time()-started,retries=retries);dump(d/'进度.json',status)
    print(name,json.dumps(status),flush=True)
  assert phash=={key:shaarr(getattr(m,key)) for key in phash}
  pd.DataFrame(rows).to_csv(d/'半小时水位流量.csv',index=False)
  status=dict(status='COMPLETED',completed_cycles=144,total_cycles=144,days=3,finite_state=all(np.isfinite(getattr(m,key)).all() for key in STATE_KEYS),parameters_unchanged=True,wall_seconds=time.time()-started,retries=retries,max_head_residual_m=float(maxh),max_gate_flow_residual_m3s=float(maxq),numba_actual=jit_evidence(),net_input_delta_m3=config['input_net_delta_m3'])
  dump(d/'进度.json',status);finished.append(status)
 except BaseException as ex:
  dump(d/'失败.json',dict(error=repr(ex),traceback=traceback.format_exc(),completed_cycles=len(rows)//59));pd.DataFrame(rows).to_csv(d/'失败前半小时水位流量.csv',index=False);raise
print(json.dumps(finished,ensure_ascii=False,indent=2),flush=True)
