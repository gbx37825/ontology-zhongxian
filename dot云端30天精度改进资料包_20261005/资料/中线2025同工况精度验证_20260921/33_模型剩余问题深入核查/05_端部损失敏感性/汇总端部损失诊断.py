from pathlib import Path
import json,numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];obs=pd.read_csv(R/'26_冻结基线与后续归因/共享输入/作者已核实修正观测层.csv');obs=obs[obs.valid];rows=[];frames={}
for name in ['L00_保持默认','L01_默认端损减20','L02_默认端损加20']:
 d=O/name;status=json.loads((d/'结果.json').read_text());assert status['status']=='COMPLETED' and status['periods']==132
 f=pd.read_csv(d/'水位流量.csv').melt(id_vars=['time_s','station'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim').merge(obs[['station_code','metric','time_s','value','pdf_comparable']],left_on=['station','metric','time_s'],right_on=['station_code','metric','time_s'],validate='one_to_one');assert len(f)==132*118;f['error']=f.sim-f.value;frames[name]=f
 for label,g in [('全线118测点',f),('作者88测点',f[f.pdf_comparable]),('JZZ21闸前',f[(f.station=='JZZ21')&(f.metric=='forebay_m')]),('JZZ15至16',f[f.station.isin(['JZZ15','JZZ16'])]),('JZZ30附近',f[f.station.isin(['JZZ30','JZZ31'])])]:rows.append(dict(case=name,scope=label,n=len(g),mae=abs(g.error).mean(),max_abs=abs(g.error).max(),bias=g.error.mean()))
base=frames['L00_保持默认'];deltas=[]
for name in ['L01_默认端损减20','L02_默认端损加20']:
 f=frames[name].merge(base[['station','metric','time_s','sim']],on=['station','metric','time_s'],suffixes=('','_base'))
 for label,g in [('全线118测点',f),('作者88测点',f[f.pdf_comparable])]:deltas.append(dict(case=name,scope=label,mean_absolute_head_change=abs(g.sim-g.sim_base).mean(),max_absolute_head_change=abs(g.sim-g.sim_base).max()))
a=pd.DataFrame(rows);a.to_csv(O/'端部损失精度对照.csv',index=False);pd.DataFrame(deltas).to_csv(O/'端部损失水位变化.csv',index=False);print(a.to_string(index=False));print(deltas)
