from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');obs=obs[obs.valid&(obs.metric.isin(['forebay_m','afterbay_m']))].rename(columns={'station_code':'station','value':'observed'})
expected=['I00_原输入三天','I01_陶岔增加三天','I02_陶岔减少三天','I04_袁庄最早差异窗原插值三天','I05_袁庄最早差异窗前值保持三天'];datasets={};checks=[];configs={}
for name in expected:
 d=O/name;cfg=json.loads((d/'运行配置.json').read_text());configs[name]=cfg;status=json.loads((d/'进度.json').read_text());assert status['status']=='COMPLETED' and status['completed_cycles']==144 and not(d/'失败.json').exists();a=pd.read_csv(d/'半小时水位流量.csv');lo=cfg.get('start_s',0);hi=cfg.get('end_s',259200);assert len(a)==144*59 and a.station.nunique()==59 and a.time_s.nunique()==144 and a.time_s.max()==hi and a.time_s.min()==lo+1800;assert not a[['station','time_s']].duplicated().any()
 a=a.melt(id_vars=['station','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs[['station','time_s','metric','observed','pdf_comparable']],on=['station','time_s','metric'],validate='one_to_one');assert len(a)==144*118;a['error']=a.sim-a.observed;datasets[name]=a
 checks.append(dict(name=name,source_sha_ok=hashlib.sha256((B/'共享输入/C07.json').read_bytes()).hexdigest()==cfg['input_sha256'],initial_sha_ok=hashlib.sha256((B/'共享输入/初始化后状态_0s.npz').read_bytes()).hexdigest()==cfg['initial_sha256'],state_sha=cfg['initial_state_sha256'],parameter_sha=cfg['parameter_sha256'],config=cfg,status=status))
assert all(x['source_sha_ok'] and x['initial_sha_ok'] for x in checks)
for group in [checks[:3],checks[3:]]:assert all(x['state_sha']==group[0]['state_sha'] and x['parameter_sha']==group[0]['parameter_sha'] for x in group)
assert all(x['parameter_sha']==checks[0]['parameter_sha'] for x in checks)
windows=[('窗口首6小时',0,21600),('窗口第6至24小时',21600,86400),('窗口第2天',86400,172800),('窗口第3天',172800,259200),('完整3天窗口',0,259200)]
summary=[];point=[];influence=[]
for name,a in datasets.items():
 start=configs[name].get('start_s',0)
 for window,lo,hi in windows:
  cut=a[a.time_s.between(start+lo,start+hi,inclusive='right')]
  for scope,q in [('全线118测点',cut),('作者可比88测点',cut[cut.pdf_comparable])]:summary.append(dict(run=name,window=window,scope=scope,n=len(q),MAE=float(q.error.abs().mean()),RMSE=float(np.sqrt((q.error**2).mean())),bias=float(q.error.mean()),max_abs=float(q.error.abs().max())))
  for (st,metric),q in cut.groupby(['station','metric']):point.append(dict(run=name,window=window,station=st,metric=metric,n=len(q),MAE=float(q.error.abs().mean()),bias=float(q.error.mean()),max_abs=float(q.error.abs().max())))
 baserun=expected[0] if name in expected[:3] else expected[3];base=datasets[baserun];paired=a.merge(base[['station','time_s','metric','sim']],on=['station','time_s','metric'],suffixes=('','_baseline'),validate='one_to_one');paired['delta']=paired.sim-paired.sim_baseline
 for (st,metric),q in paired.groupby(['station','metric']):
  at=q.loc[q.delta.abs().idxmax()];influence.append(dict(run=name,reference=baserun,station=st,metric=metric,mean_water_change_m=float(q.delta.mean()),max_abs_water_change_m=float(q.delta.abs().max()),time_of_max_s=float(at.time_s),water_change_at_max_m=float(at.delta)))
S=pd.DataFrame(summary);P=pd.DataFrame(point);D=pd.DataFrame(influence);S.to_csv(O/'分时段总体精度.csv',index=False);P.to_csv(O/'逐站分时段精度.csv',index=False);D.to_csv(O/'逐站水位影响.csv',index=False)
near=P[P.station.isin(['JZZ2','JZZ3','JZZ4','JZZ32','JZZ33','JZZ34','JZZ35'])];near.to_csv(O/'入口及袁庄邻近闸站精度.csv',index=False)
result=dict(completed=True,all_five_runs_144_cycles=True,within_each_comparison_initial_states_exactly_equal=True,parameters_exactly_equal_and_unchanged=True,frozen_files_unchanged=True,case_checks=checks,no_difference_first_3day_yuan=json.loads((O/'I03_袁庄补值前值保持三天/无差异预检.json').read_text()),yuan_history=json.loads((O/'袁庄共同历史推进/进度.json').read_text()))
(O/'完成与单因素核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(S[S.window=='完整3天窗口'].to_string(index=False));print(D.groupby('run').max_abs_water_change_m.max().to_string());print(near[(near.window=='完整3天窗口')&near.station.isin(['JZZ2','JZZ33','JZZ34'])].to_string(index=False))
