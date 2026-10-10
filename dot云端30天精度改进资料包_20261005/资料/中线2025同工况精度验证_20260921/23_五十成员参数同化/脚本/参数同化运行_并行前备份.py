"""50成员局地参数EnKF：仅改明渠糙率/闸系数，30min观测更新；不改水位流量状态。"""
from pathlib import Path
import argparse,sys,json,time,copy,hashlib,csv
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
from 试验公共 import Forcing,meshes,weights_at
sys.path.insert(0,str(R/'04_脚本'))
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import ParameterEnKF,STATE_KEYS,restore,volume
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--days',type=float,default=30);p.add_argument('--members',type=int,default=50);p.add_argument('--no-analysis',action='store_true');p.add_argument('--resume-from',type=Path);a=p.parse_args();O=R/'23_五十成员参数同化'/a.name;O.mkdir(exist_ok=False);started=time.time();seed=20260923
model=R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json';e,s,c=build_engine_physics(model,gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()));m=e.model
with np.load(R/'06_结果/A03R_C07物理加初态自适应30天/初始化后状态_0s.npz') as state:restore(m,{k:state[k].copy() for k in STATE_KEYS})
m.t=0.;m.iter_count=0;f=Forcing(e,s,c);ens=ParameterEnKF(e,c,n=a.members,seed=seed,radius=20000.)
mesh=meshes(s,c);mapping=json.loads((R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/水位测点位置审核.json').read_text());obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');valid=obs[obs.valid].set_index(['station_code','metric','time_s']);ops=[]
for z in mapping['station_side_summary']:
 if not z['candidate_interpolation']:continue
 st=z['stationcode'];met='forebay_m' if z['metric']=='front' else 'afterbay_m';vv=obs[(obs.station_code==st)&(obs.metric==met)]
 if not bool(vv.pdf_comparable.iloc[0]):continue
 row=next(q for q in mapping['sensor_rows'] if q['stationcode']==st and q['metric']==z['metric'] and q['status']=='可作位置插值候选');ch=row['model_brackets'][0]['channel'];ids,w=weights_at(mesh[ch],row['sensor_chainage']);ops.append(dict(station=st,metric=met,state='h_Ik',indices=ids.tolist(),weights=w.tolist(),offset=float(m._z_inv_Ik[ids]@w),sigma=.03,assimilated=int(st[3:])%5!=0,chainage=row['sensor_chainage']))
(O/'观测算子与留出.json').write_text(json.dumps(ops,ensure_ascii=False,indent=2));groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1)) for st,g in c['orifices'].assign(station=c['orifices'].name.str.split('-').str[0]).groupby('station')};total_cycles=int(round(a.days*48));retries=0;max_head=0.;max_gate=0.;ledger=[];parameter_log=[]
config=dict(runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),filter_sha256=hashlib.sha256((Path(__file__).parent/'局地参数EnKF.py').read_bytes()).hexdigest(),seed=seed,members=a.members,days=a.days,no_analysis=a.no_analysis,update_s=1800,observation_sigma_m=.03,localization_radius_m=20000,parameter_initial_log_std=.05,parameter_spread_floor=.01,max_parameter_update_fraction=.01,state_process_noise=False,parameter_count=len(ens.parameters),parameter_bounds=ens.parameters,solver_identity=e.solver_identity,input_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),initial_snapshot='A03R初始化后状态_0s.npz',sensor_ops=len(ops),assimilated_ops=sum(x['assimilated'] for x in ops),heldout_ops=sum(not x['assimilated'] for x in ops),note='历史已知开度及边界回放，非实时未来预报；水位算子候选的历史仪器身份尚未完全闭合')
(O/'运行配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));print('CONFIG',json.dumps(config,ensure_ascii=False),flush=True)

start_cycle=0
if a.resume_from:
 source=a.resume_from.resolve()
 previous=json.loads((source/'运行配置.json').read_text())
 for key in config:
  if key not in {'runner_sha256','days'} and config[key]!=previous[key]:
   raise ValueError(f'恢复配置不一致: {key}')
 stamp=json.loads((source/'最新集合时间与随机状态.json').read_text())
 start_cycle=int(stamp['cycle']);resume_time=float(stamp['time_s'])
 assert resume_time==start_cycle*1800 and 0<start_cycle<total_cycles
 with np.load(source/'最新完整集合.npz') as saved:
  theta=saved['parameter_log_ratios'].copy()
  assert theta.shape==ens.theta.shape and np.isfinite(theta).all()
  for i,member in enumerate(ens.members):
   restored={k:saved[f'm{i}_{k}'].copy() for k in STATE_KEYS}
   assert all(v.shape==getattr(member,k).shape and np.isfinite(v).all() for k,v in restored.items())
   restore(member,restored);member.t=resume_time;member.iter_count=0
 ens.theta=theta;ens.apply()
 ens.rng.bit_generator.state=stamp['parameter_rng'];ens.analysis_rng.bit_generator.state=stamp['analysis_rng']
 def prefix(filename):
  path=source/filename
  if path.stat().st_size<2:return []
  frame=pd.read_csv(path);return frame[frame.time_s<=resume_time].to_dict('records')
 ledger=prefix('同化等效水量.csv');parameter_log=prefix('参数轨迹.csv');ens.log=prefix('逐观测更新记录.csv')
 assert len(ledger)==start_cycle and len(parameter_log)==start_cycle*len(ens.parameters)
 with (source/'预报与分析逐点.csv').open(newline='') as src,(O/'预报与分析逐点.csv').open('w',newline='') as dst:
  reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames);writer.writeheader()
  for row in reader:
   if float(row['time_s'])<=resume_time:writer.writerow(row)
 evidence=dict(source=str(source),cycle=start_cycle,time_s=resume_time,original_config=previous,
  checkpoint_sha256=hashlib.sha256((source/'最新完整集合.npz').read_bytes()).hexdigest(),
  rng_sha256=hashlib.sha256((source/'最新集合时间与随机状态.json').read_bytes()).hexdigest(),
  note='原目录不修改；仅复制检查点及之前记录，检查点之后重新计算，不重复计分。恢复段重试及耗时单独计数。')
 (O/'恢复记录.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
 print('RESUME',start_cycle,resume_time,flush=True)

def forecast_member(member,t0,t1):
 global retries,max_head,max_gate
 assert abs(member.t-t0)<1e-8
 for ts in np.arange(t0,t1,15.):
  pending=[(ts,15.)]
  while pending:
   tt,dt=pending.pop(0);kw=f.at(tt);kw.update(dt=dt,num_iter=20,strict=True)
   try:member.step(**kw)
   except RuntimeError as err:
    if 'did not converge' not in str(err) or dt<=1.875:raise
    retries+=1;pending[:0]=[(tt,dt/2),(tt+dt/2,dt/2)];continue
   max_head=max(max_head,member.last_head_residual);max_gate=max(max_gate,member.last_gate_flow_residual)
   if not all(np.isfinite(getattr(member,k)).all() for k in STATE_KEYS):raise ValueError('nonfinite ensemble')

def write_scores(writer,t,phase):
 mean=np.mean([mm.H_j for mm in ens.members],axis=0)
 for st,(ju,jd) in groups.items():
  for met,j in [('forebay_m',ju),('afterbay_m',jd)]:
   key=(st,met,t)
   if key in valid.index:
    v=valid.loc[key];writer.writerow([t,phase,'闸端全线',st,met,float(mean[j]),float(v.value),bool(v.pdf_comparable),''])
 for op in ops:
  key=(op['station'],op['metric'],t)
  if key in valid.index:writer.writerow([t,phase,'同测点位置',op['station'],op['metric'],float(ens.values(op).mean()),float(valid.loc[key,'value']),True,'同化点' if op['assimilated'] else '空间留出点'])

try:
 with (O/'预报与分析逐点.csv').open('a' if start_cycle else 'w',newline='') as handle:
  writer=csv.writer(handle)
  if not start_cycle:writer.writerow(['time_s','phase','operator','station','metric','sim','observed','pdf_comparable','group'])
  for cycle in range(start_cycle,total_cycles):
   t0=cycle*1800.;now=t0+1800
   hydraulic_before=[{k:getattr(mm,k).copy() for k in STATE_KEYS} for mm in ens.members]
   rejuvenated=ens.rejuvenate()
   assert all(np.array_equal(getattr(mm,k),old[k]) for mm,old in zip(ens.members,hydraulic_before) for k in STATE_KEYS)
   for member in ens.members:forecast_member(member,t0,now)
   write_scores(writer,now,'forecast_before_update');predvol=float(np.mean([volume(mm) for mm in ens.members]));update=[]
   for op in ops:
    key=(op['station'],op['metric'],now)
    if op['assimilated'] and key in valid.index:update.append({**op,'value':float(valid.loc[key,'value'])})
   hydraulic_before=[{k:getattr(mm,k).copy() for k in STATE_KEYS} for mm in ens.members]
   traces=[] if a.no_analysis else ens.analysis(update,now)
   assert all(np.array_equal(getattr(mm,k),old[k]) for mm,old in zip(ens.members,hydraulic_before) for k in STATE_KEYS)
   av=float(np.mean([volume(mm) for mm in ens.members]));write_scores(writer,now,'after_parameter_update_same_state');assert abs(av-predvol)<1e-8;ledger.append(dict(time_s=now,analysis_delta_volume_mean_m3=av-predvol,rejuvenated_parameters=rejuvenated,state_unchanged=True,accepted=sum(q['status']=='used_for_parameters' for q in traces),rejected=sum(q['status']!='used_for_parameters' for q in traces)))
   for j,par in enumerate(ens.parameters):
    ratio=np.exp(ens.theta[j]);parameter_log.append(dict(time_s=now,parameter=par['name'],mean_ratio=float(ratio.mean()),min_ratio=float(ratio.min()),max_ratio=float(ratio.max()),std_log=float(ens.theta[j].std(ddof=1)),at_lower=int((ens.theta[j]<=ens.lower[j]+1e-10).sum()),at_upper=int((ens.theta[j]>=ens.upper[j]-1e-10).sum())))
   if (cycle+1)%48==0 or cycle+1==total_cycles:
    checkpoint={f'm{i}_{k}':getattr(mm,k) for i,mm in enumerate(ens.members) for k in STATE_KEYS}
    checkpoint['parameter_log_ratios']=ens.theta.copy()
    np.savez_compressed(O/'最新完整集合.npz',**checkpoint)
    (O/'最新集合时间与随机状态.json').write_text(json.dumps(dict(cycle=cycle+1,time_s=now,parameter_rng=ens.rng.bit_generator.state,analysis_rng=ens.analysis_rng.bit_generator.state),ensure_ascii=False,indent=2))
   if (cycle+1)%12==0 or cycle+1==total_cycles:
    handle.flush();pd.DataFrame(parameter_log).to_csv(O/'参数轨迹.csv',index=False);pd.DataFrame(ledger).to_csv(O/'同化等效水量.csv',index=False);pd.DataFrame(ens.log).to_csv(O/'逐观测更新记录.csv',index=False)
    from numba import config as numba_config
    from pipedream_solver import nsuperlink
    from pipedream_solver.gate_hydraulics import rectangular_gate_terms
    jit=dict(backend=e.solver_identity,model_class=type(m).__module__+'.'+type(m).__name__,jit_disabled=bool(numba_config.DISABLE_JIT),momentum_compiled_signatures=len(nsuperlink.numba_b_ik.signatures),gate_compiled_signatures=len(rectangular_gate_terms.signatures))
    assert jit['model_class']=='pipedream_solver.nsuperlink.nSuperLink' and not jit['jit_disabled'] and jit['momentum_compiled_signatures']>0 and jit['gate_compiled_signatures']>0
    (O/'Numba实际执行证据.json').write_text(json.dumps(jit,ensure_ascii=False,indent=2))
    elapsed=time.time()-started;status=dict(status='RUNNING',completed_cycles=cycle+1,total_cycles=total_cycles,simulated_days=(cycle+1)/48,wall_seconds=elapsed,estimated_total_seconds=elapsed/(cycle+1-start_cycle)*(total_cycles-start_cycle),retries=retries,max_head_residual=max_head,max_gate_residual=max_gate)
    (O/'进度.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));print(json.dumps(status,ensure_ascii=False),flush=True)
 status['status']='COMPLETED';(O/'进度.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
 checkpoint={f'm{i}_{k}':getattr(mm,k) for i,mm in enumerate(ens.members) for k in STATE_KEYS};checkpoint['parameter_log_ratios']=ens.theta.copy();np.savez_compressed(O/'末态集合.npz',**checkpoint)
except BaseException as ex:
 (O/'失败.json').write_text(json.dumps(dict(error=repr(ex),cycle=locals().get('cycle'),wall_seconds=time.time()-started),ensure_ascii=False,indent=2));raise
