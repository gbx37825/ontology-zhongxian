from pathlib import Path
import pandas as pd,numpy as np,json
R=Path('outputs/中线2025同工况精度验证_20260921');P=R/'23_五十成员参数同化';O=R/'24_参数同化根因核查'
a=pd.read_csv(P/'P01M4_50成员30天参数同化/预报与分析逐点.csv');b=pd.read_csv(P/'P02M4_50成员30天无同化/预报与分析逐点.csv');a=a[a.phase=='forecast_before_update'];b=b[b.phase=='forecast_before_update'];keys=['time_s','operator','station','metric'];x=a.merge(b,on=keys,suffixes=('_a','_b'));x['ea']=x.sim_a-x.observed_a;x['eb']=x.sim_b-x.observed_b
rows=[]
for (st,met),g in x[x.group_a=='空间留出点'].groupby(['station','metric']):
 rows.append(dict(station=st,metric=met,mae_a=abs(g.ea).mean(),mae_b=abs(g.eb).mean(),delta=abs(g.ea).mean()-abs(g.eb).mean(),bias_a=g.ea.mean(),bias_b=g.eb.mean(),last_day_bias_a=g[g.time_s>29*86400].ea.mean(),last_day_bias_b=g[g.time_s>29*86400].eb.mean()))
r=pd.DataFrame(rows).sort_values('delta',ascending=False);r.to_csv(O/'九侧留出误差分解.csv',index=False);print(r.to_string(index=False))
tr=pd.read_csv(P/'30天参数轨迹与触界汇总.csv');print(tr[(tr.run.str.startswith('P01'))&tr.parameter.isin(['POOL44_n','POOL45_n','POOL46_n','JZZ44_C','JZZ45_C','JZZ46_C','POOL3_n','POOL13_n'])][['parameter','first_mean','final_mean','final_std_log']].to_string(index=False))
ops=pd.read_csv(O/'同化测点图距离.csv');print('nearJZZ45',ops[(ops.graph>860000)&(ops.graph<940000)].to_string(index=False))
