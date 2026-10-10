from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');summaries=[];per=[]
for out in sorted((R/'06_结果').iterdir()):
 if out.name.startswith('V00') or not (out/'逐闸水位流量.csv').exists() or not (out/'运行摘要.json').exists():continue
 status=json.loads((out/'运行摘要.json').read_text())
 if status['status']!='COMPLETED':continue
 d=pd.read_csv(out/'逐闸水位流量.csv').melt(id_vars=['station_code','time_s'],var_name='metric',value_name='sim').merge(obs,on=['station_code','time_s','metric'],validate='one_to_one');d['raw_error']=d.sim-d.raw_value;d['author_error']=d.sim-d.value
 d.to_csv(out/'逐时刻双口径对照.csv',index=False)
 for period,lo,hi in [('全30天',0,2592000),('前6天',0,518400),('前三天',0,259200),('后27天',259200,2592000),('首6小时',0,21600),('第2至6天',86400,518400),('第4至6天',259200,518400)]:
  if d.time_s.max()<hi:continue
  v=d[d.valid&(d.metric!='flow_m3s')&(d.time_s>lo)&(d.time_s<=hi)]
  if v.empty:continue
  for area,g in [('全线118侧',v),('作者可读88侧',v[v.pdf_comparable]),('作者未展示30侧',v[~v.pdf_comparable]),('JZZ56至60',v[v.station_code.isin([f'JZZ{i}' for i in range(56,61)])])]:
   for basis,col in [('原始观测','raw_error'),('作者已核实修正','author_error')]:
    e=g[col];means=g.groupby(['station_code','metric'])[col].apply(lambda x:x.abs().mean());summaries.append(dict(run=out.name,period=period,actual_end_s=float(g.time_s.max()),area=area,basis=basis,n=len(e),mae=e.abs().mean(),bias=e.mean(),rmse=np.sqrt((e*e).mean()),p95=e.abs().quantile(.95),max_error=e.abs().max(),series=len(means),pass_mae_0_1=int((means<=.1).sum()),fraction_over_0_1=float((e.abs()>.1).mean())))
  for (station,metric),g in v.groupby(['station_code','metric']):
   for basis,col in [('原始观测','raw_error'),('作者已核实修正','author_error')]:per.append(dict(run=out.name,period=period,station=station,sheet=g.sheet.iloc[0],metric=metric,basis=basis,n=len(g),mae=g[col].abs().mean(),bias=g[col].mean(),max_error=g[col].abs().max(),pdf_comparable=bool(g.pdf_comparable.iloc[0])))
pd.DataFrame(summaries).to_csv(R/'07_对比分析/2025同工况精度汇总.csv',index=False);pd.DataFrame(per).to_csv(R/'07_对比分析/2025逐测点精度.csv',index=False)
print(pd.DataFrame(summaries).query("period=='全30天' and area=='作者可读88侧' and basis=='作者已核实修正'")[['run','mae','bias','max_error','pass_mae_0_1']].to_string(index=False))
