from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'25_JZZ13参数隔离';source=R/'24_参数同化根因核查/F01_拓扑分组修复六小时';rows=[];checks=[]
for name in ['G03_只冻结沙河闸流系数','G04_只冻结第12渠池糙率']:
 d=O/name;status=json.loads((d/'进度.json').read_text());assert status['status']=='COMPLETED';cfg=json.loads((d/'运行配置.json').read_text());par=json.loads((d/'隔离分支.json').read_text())['frozen'];j=next(i for i,p in enumerate(cfg['parameter_bounds']) if p['name']==par)
 with np.load(source/'末态集合.npz') as a,np.load(d/'末态集合.npz') as b:
  assert len(b.files)==301 and all(np.isfinite(b[k]).all() for k in b.files);assert np.array_equal(a['parameter_log_ratios'][j],b['parameter_log_ratios'][j])
 track=pd.read_csv(d/'参数轨迹.csv');track=track[(track.time_s>=21600)&(track.parameter==par)];assert track.mean_ratio.max()-track.mean_ratio.min()<1e-12
 v=pd.read_csv(d/'同化等效水量.csv');assert v.state_unchanged.all() and (v.analysis_delta_volume_mean_m3==0).all();checks.append(dict(run=name,cycles=48,branch_cycles=36,parameter=par,frozen_members_exact=True,water_state_unchanged=True,instantaneous_volume_adjustment=0))
for label,d in [('正常同化',R/'24_参数同化根因核查/F03_拓扑分组修复30天'),('只冻结沙河闸流系数',O/'G03_只冻结沙河闸流系数'),('只冻结第12渠池糙率',O/'G04_只冻结第12渠池糙率')]:
 a=pd.read_csv(d/'预报与分析逐点.csv');a=a[(a.phase=='forecast_before_update')&(a.time_s>21600)&(a.time_s<=86400)];a['err']=a.sim-a.observed
 for scope,g in [('沙河闸前',a[(a.operator=='闸端全线')&(a.station=='JZZ13')&(a.metric=='forebay_m')]),('沙河闸后测点',a[(a.operator=='同测点位置')&(a.station=='JZZ13')&(a.metric=='afterbay_m')]),('全线水位测点',a[a.operator=='闸端全线']),('检验水位测点',a[a.group=='空间留出点'])]:rows.append(dict(run=label,scope=scope,n=len(g),mae=abs(g.err).mean(),bias=g.err.mean()))
x=pd.DataFrame(rows);x.to_csv(O/'单参数冻结误差对照.csv',index=False);(O/'单参数冻结核验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print(x.to_string(index=False))
