"""20成员局地化状态EnSRF，30min水位更新；独立留出站及同集合无更新对照。"""
from pathlib import Path
import argparse,sys,json,time,copy,hashlib,csv
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
from 试验公共 import Forcing,meshes,weights_at
sys.path.insert(0,str(R/'04_脚本'))
from 并联结构物理修复 import build_engine_physics
from 连续EnKF核心 import LocalEnKF,STATE_KEYS,restore,volume
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--days',type=float,default=30);p.add_argument('--members',type=int,default=20);p.add_argument('--no-analysis',action='store_true');a=p.parse_args();O=R/'20_连续状态同化'/a.name;O.mkdir(exist_ok=False);started=time.time();seed=20260923
model=R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json';e,s,c=build_engine_physics(model,gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()));m=e.model
with np.load(R/'06_结果/A03R_C07物理加初态自适应30天/初始化后状态_0s.npz') as state:restore(m,{k:state[k].copy() for k in STATE_KEYS})
m.t=0.;m.iter_count=0;f=Forcing(e,s,c);ens=LocalEnKF(e,n=a.members,seed=seed,radius=20000.)
mesh=meshes(s,c);mapping=json.loads((R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/水位测点位置审核.json').read_text());obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');valid=obs[obs.valid].set_index(['station_code','metric','time_s']);ops=[]
for z in mapping['station_side_summary']:
 if not z['candidate_interpolation']:continue
 st=z['stationcode'];met='forebay_m' if z['metric']=='front' else 'afterbay_m';vv=obs[(obs.station_code==st)&(obs.metric==met)]
 if not bool(vv.pdf_comparable.iloc[0]):continue
 row=next(q for q in mapping['sensor_rows'] if q['stationcode']==st and q['metric']==z['metric'] and q['status']=='可作位置插值候选');ch=row['model_brackets'][0]['channel'];ids,w=weights_at(mesh[ch],row['sensor_chainage']);ops.append(dict(station=st,metric=met,state='h_Ik',indices=ids.tolist(),weights=w.tolist(),offset=float(m._z_inv_Ik[ids]@w),sigma=.03,assimilated=int(st[3:])%5!=0,chainage=row['sensor_chainage']))
(O/'观测算子与留出.json').write_text(json.dumps(ops,ensure_ascii=False,indent=2));groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1)) for st,g in c['orifices'].assign(station=c['orifices'].name.str.split('-').str[0]).groupby('station')};rng=np.random.default_rng(seed+1);knots=np.arange(np.nanmin(ens.x),np.nanmax(ens.x)+20000,20000.);total_cycles=int(round(a.days*48));retries=0;max_head=0.;max_gate=0.;ledger=[]
config=dict(runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),filter_sha256=hashlib.sha256((Path(__file__).parent/'连续EnKF核心.py').read_bytes()).hexdigest(),seed=seed,members=a.members,days=a.days,no_analysis=a.no_analysis,update_s=1800,observation_sigma_m=.03,localization_radius_m=20000,process_head_std_m=.005,process_flow_std_m3s=.1,input_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),initial_snapshot='A03R初始化后状态_0s.npz',sensor_ops=len(ops),assimilated_ops=sum(x['assimilated'] for x in ops),heldout_ops=sum(not x['assimilated'] for x in ops),note='历史已知开度及边界回放，非实时未来预报；水位算子候选的历史仪器身份尚未完全闭合')
(O/'运行配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2));print('CONFIG',json.dumps(config,ensure_ascii=False),flush=True)

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
 with (O/'预报与分析逐点.csv').open('w',newline='') as handle:
  writer=csv.writer(handle);writer.writerow(['time_s','phase','operator','station','metric','sim','observed','pdf_comparable','group'])
  for cycle in range(total_cycles):
   t0=cycle*1800.;now=t0+1800
   # Shared zero-mean smooth process perturbations, identical seed for the paired control.
   z=rng.normal(size=(len(knots),a.members));z-=z.mean(axis=1,keepdims=True);noise=np.stack([np.interp(ens.x,knots,z[:,i]) for i in range(a.members)],axis=1);scale=np.where(ens.heads,.005,.1);delta=noise*scale[:,None]*ens.mask[:,None];pv=[]
   for i,member in enumerate(ens.members):
    before=volume(member);ens.unpack(member,ens.pack(member)+delta[:,i]);pv.append(volume(member)-before);forecast_member(member,t0,now)
   write_scores(writer,now,'forecast_before_update');predvol=float(np.mean([volume(mm) for mm in ens.members]));update=[]
   for op in ops:
    key=(op['station'],op['metric'],now)
    if op['assimilated'] and key in valid.index:update.append({**op,'value':float(valid.loc[key,'value'])})
   traces=[] if a.no_analysis else ens.analysis(update,now)
   av=float(np.mean([volume(mm) for mm in ens.members]));write_scores(writer,now,'analysis_after_update');ledger.append(dict(time_s=now,process_delta_volume_mean_m3=float(np.mean(pv)),analysis_delta_volume_mean_m3=av-predvol,accepted=sum(q['status']=='applied' for q in traces),rejected=sum(q['status']!='applied' for q in traces)))
   if (cycle+1)%12==0 or cycle+1==total_cycles:
    handle.flush();pd.DataFrame(ledger).to_csv(O/'同化等效水量.csv',index=False);pd.DataFrame(ens.log).to_csv(O/'逐观测更新记录.csv',index=False)
    elapsed=time.time()-started;status=dict(status='RUNNING',completed_cycles=cycle+1,total_cycles=total_cycles,simulated_days=(cycle+1)/48,wall_seconds=elapsed,estimated_total_seconds=elapsed/(cycle+1)*total_cycles,retries=retries,max_head_residual=max_head,max_gate_residual=max_gate)
    (O/'进度.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));print(json.dumps(status,ensure_ascii=False),flush=True)
 status['status']='COMPLETED';(O/'进度.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
 checkpoint={f'm{i}_{k}':getattr(mm,k) for i,mm in enumerate(ens.members) for k in STATE_KEYS};np.savez_compressed(O/'末态集合.npz',**checkpoint)
except BaseException as ex:
 (O/'失败.json').write_text(json.dumps(dict(error=repr(ex),cycle=locals().get('cycle'),wall_seconds=time.time()-started),ensure_ascii=False,indent=2));raise
