"""已有30分钟输出逐孔筛查；不是全部内部步或稳定性验收。"""
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'11_Pipedream振荡根因';records=[];summary=[]
for run,file in [('B00_实际开度同工况30天','C00_实际开度负值修复_2025同工况30天.json'),('B01_目标开度同工况30天','C01_目标开度独立假设_2025同工况30天.json')]:
 s=next(iter(json.loads((R/'02_输入'/file).read_text()).values()));data=pd.read_csv(R/'06_结果'/run/'逐闸水位流量.csv');links=s['timeSeries']['seriesLink'];series={v[1]:np.asarray(s['timeSeries']['seriesData'][k]) for k,v in links.items() if v[0]=='gates'}
 for name,g in s['baseData']['gates'].items():
  st=name.split('-')[0];df=data[(data.station_code==st)&(data.time_s>0)].sort_values('time_s');ts=df.time_s.to_numpy();a=np.interp(ts-15,series[name][:,0],series[name][:,1]);effective_zb=max(g['zb'],s['baseData']['nodes'][g['node1']]['zb']);hu=df.forebay_m.to_numpy()-effective_zb;hd=df.afterbay_m.to_numpy()-effective_zb;hi=np.maximum(hu,hd);lo=np.minimum(hu,hd)
  c=(a>=hi)&(hi>0)&(a>0);wet=c&(lo>0);ratio=np.divide(a,hi,out=np.full_like(a,np.nan),where=hi>0)
  records.append(dict(run=run,station=st,gate=name,output_points=len(df),max_opening=float(a.max()),max_opening_depth_ratio=float(np.nanmax(ratio)),max_hi_depth=float(hi.max()),min_hi_depth=float(hi.min()),branch_c_points=int(c.sum()),wet_c_points=int(wet.sum()),negative_station_points=int((df.flow_m3s<0).sum()),first_c_time_s=float(ts[c][0]) if c.any() else None))
 sub=pd.DataFrame([d for d in records if d['run']==run]);summary.append(dict(run=run,gates=len(sub),stations=sub.station.nunique(),no_sampled_c_gates=int((sub.branch_c_points==0).sum()),no_sampled_c_stations=int((sub.groupby('station').branch_c_points.sum()==0).sum()),c_stations=sub.loc[sub.branch_c_points>0,'station'].unique().tolist(),negative_stations=sub.loc[sub.negative_station_points>0,'station'].unique().tolist()))
pd.DataFrame(records).to_csv(O/'全线闸孔流态触发筛查.csv',index=False);(O/'全线闸孔流态触发筛查.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
df=pd.DataFrame(records);print(df[df.branch_c_points>0].to_string(index=False));print(df[(df.run.str.startswith('B00'))&df.station.isin(['JZZ1','JZZ20','JZZ26','JZZ41','JZZ43'])][['gate','max_opening','max_opening_depth_ratio']].to_string(index=False))
