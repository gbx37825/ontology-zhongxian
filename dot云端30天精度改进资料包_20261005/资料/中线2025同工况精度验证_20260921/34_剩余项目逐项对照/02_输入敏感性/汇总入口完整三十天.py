from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');obs=obs[obs.valid&obs.metric.isin(['forebay_m','afterbay_m'])&(obs.time_s>0)].rename(columns={'station_code':'station','value':'observed'})
names=['M00_原输入完整30天','M01_陶岔增加完整30天','M02_陶岔减少完整30天'];datasets={};checks=[];configs={}
for name in names:
 d=O/name;cfg=json.loads((d/'运行配置.json').read_text());configs[name]=cfg;st=json.loads((d/'进度.json').read_text());assert st['status']=='COMPLETED' and st['completed_cycles']==1440 and not(d/'失败.json').exists()
 raw=pd.read_csv(d/'半小时水位流量.csv');assert len(raw)==59*1440 and raw.station.nunique()==59 and raw.time_s.nunique()==1440 and raw.time_s.min()==1800 and raw.time_s.max()==2592000 and not raw[['station','time_s']].duplicated().any()
 old=pd.read_csv(O/cfg['prior_3day']/'半小时水位流量.csv').sort_values(['station','time_s']).reset_index(drop=True);prefix=raw[raw.time_s<=259200].sort_values(['station','time_s']).reset_index(drop=True);assert np.array_equal(old.to_numpy(),prefix.to_numpy());pre=dict(exact_equal_after_same_csv_parser=True,rows=len(old),max_abs_numeric_difference=float(abs(old[['forebay_m','afterbay_m','gate_flow_m3s']].to_numpy()-prefix[['forebay_m','afterbay_m','gate_flow_m3s']].to_numpy()).max()))
 with np.load(d/'末态.npz') as z:finite=all(np.isfinite(z[k]).all() for k in ('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o'));assert finite
 a=raw.melt(id_vars=['station','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs[['station','time_s','metric','observed','pdf_comparable']],on=['station','time_s','metric'],validate='one_to_one');assert len(a)==1440*118;a['error']=a.sim-a.observed;datasets[name]=a
 checks.append(dict(name=name,progress=st,config=cfg,prefix_match=pre,final_state_finite=finite,input_hash_unchanged=hashlib.sha256((B/'共享输入/C07.json').read_bytes()).hexdigest()==cfg['input_sha256'],initial_hash_unchanged=hashlib.sha256((B/'共享输入/初始化后状态_0s.npz').read_bytes()).hexdigest()==cfg['initial_sha256']))
assert all(c['input_hash_unchanged'] and c['initial_hash_unchanged'] for c in checks)
assert all(c['config']['initial_state_sha256']==checks[0]['config']['initial_state_sha256'] and c['config']['parameter_sha256']==checks[0]['config']['parameter_sha256'] for c in checks)
periods=[('首3天诊断段',0,259200),('后27天保留段',259200,2592000),('完整30天',0,2592000),('末3天',2332800,2592000)];total=[];point=[];influence=[]
for name,a in datasets.items():
 paired=a.merge(datasets[names[0]][['time_s','station','metric','sim']],on=['time_s','station','metric'],suffixes=('','_baseline'),validate='one_to_one');paired['water_delta']=paired.sim-paired.sim_baseline
 for period,lo,hi in periods:
  data=paired[paired.time_s.between(lo,hi,inclusive='right')]
  for scope,q in [('全线118个水位测点',data),('作者可比88个水位测点',data[data.pdf_comparable])]:total.append(dict(run=name,period=period,scope=scope,n=len(q),MAE=float(q.error.abs().mean()),RMSE=float(np.sqrt((q.error**2).mean())),bias=float(q.error.mean()),max_abs=float(q.error.abs().max()),net_input_difference_m3=configs[name]['inlet_delta_m3s']*(hi-lo)))
  for (station,metric),q in data.groupby(['station','metric']):
   point.append(dict(run=name,period=period,station=station,metric=metric,n=len(q),MAE=float(q.error.abs().mean()),RMSE=float(np.sqrt((q.error**2).mean())),bias=float(q.error.mean()),max_abs=float(q.error.abs().max())))
   peak=q.loc[q.water_delta.abs().idxmax()];influence.append(dict(run=name,period=period,station=station,metric=metric,mean_change_m=float(q.water_delta.mean()),max_abs_change_m=float(q.water_delta.abs().max()),peak_time_s=float(peak.time_s),peak_signed_change_m=float(peak.water_delta)))
S=pd.DataFrame(total);P=pd.DataFrame(point);D=pd.DataFrame(influence);S.to_csv(O/'入口30天分段总体精度.csv',index=False);P.to_csv(O/'入口30天逐测点精度.csv',index=False);D.to_csv(O/'入口30天逐测点水位影响.csv',index=False)
judgements=[]
for period,lo,hi in periods:
 base=P[(P.run==names[0])&(P.period==period)]
 for name in names[1:]:
  z=P[(P.run==name)&(P.period==period)].merge(base[['station','metric','MAE']],on=['station','metric'],suffixes=('','_base'));v=z.MAE-z.MAE_base;judgements.append(dict(run=name,period=period,improved_more_1mm=int((v<-.001).sum()),worsened_more_1mm=int((v>.001).sum()),within_1mm=int((v.abs()<=.001).sum())))
dump=dict(completed=True,three_runs_complete=True,cycles_per_run=1440,scored_samples_per_run=1440*118,initial_and_parameters_equal=True,first_three_days_match_exactly=True,frozen_files_unchanged=True,cases=checks,point_improvement_counts=judgements)
(O/'入口30天完成核验.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2));pd.DataFrame(judgements).to_csv(O/'入口30天测点改善与恶化数量.csv',index=False)
print(S.to_string(index=False));print(pd.DataFrame(judgements).to_string(index=False));print(D[D.period=='完整30天'].groupby('run').max_abs_change_m.max())
