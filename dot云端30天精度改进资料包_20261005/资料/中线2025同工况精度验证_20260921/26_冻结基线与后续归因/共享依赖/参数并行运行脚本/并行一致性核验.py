from pathlib import Path
import json,numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1]
serial=P/'T01B_1进程六小时验证';parallel=P/'T04B_4进程六小时验证'
report={'comparison':'相同第348周期检查点与随机数状态，计算12周期至第360周期','state_arrays':{},'tables':{}}
for d in (serial,parallel):
 status=json.loads((d/'进度.json').read_text());assert status['status']=='COMPLETED' and status['completed_cycles']==360
 ledger=pd.read_csv(d/'同化等效水量.csv');assert ledger.state_unchanged.all() and (ledger.analysis_delta_volume_mean_m3==0).all()
with np.load(serial/'末态集合.npz') as a,np.load(parallel/'末态集合.npz') as b:
 assert set(a.files)==set(b.files) and len(a.files)==301
 for k in a.files:
  difference=float(np.max(np.abs(a[k]-b[k])));report['state_arrays'][k]=difference
  assert np.array_equal(a[k],b[k]),(k,difference)
for name in ['预报与分析逐点.csv','参数轨迹.csv','同化等效水量.csv','逐观测更新记录.csv']:
 a=pd.read_csv(serial/name);b=pd.read_csv(parallel/name);a=a[a.time_s>626400].reset_index(drop=True);b=b[b.time_s>626400].reset_index(drop=True)
 pd.testing.assert_frame_equal(a,b,check_exact=True);report['tables'][name]={'rows':len(a),'exact_match':True}
assert json.loads((serial/'最新集合时间与随机状态.json').read_text())==json.loads((parallel/'最新集合时间与随机状态.json').read_text())
report['random_states_exact_match']=True;report['timing']={}
for label,d in [('serial',serial),('parallel',parallel)]:
 t=json.loads((d/'水力推进计时.json').read_text());v=[r['wall_seconds'] for r in t[1:]]
 report['timing'][label]={'steady_cycles':len(v),'mean_hydraulic_seconds':float(np.mean(v)),'median_hydraulic_seconds':float(np.median(v)),'total_seconds':json.loads((d/'进度.json').read_text())['wall_seconds'],'process_startup':json.loads((d/'计算进程配置.json').read_text())}
report['speedup']=report['timing']['serial']['mean_hydraulic_seconds']/report['timing']['parallel']['mean_hydraulic_seconds'];report['status']='PASS'
(P/'四进程并行一致性与性能验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='state_arrays'},ensure_ascii=False,indent=2))
