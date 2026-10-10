from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'19_水位精度根因';obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');runs=['S02_C07输入及闸流修复严格30天','A01_C07观测初态30天','A02R_C07物理转换自适应30天','A03R_C07物理加初态自适应30天','A04_C07物理初态原JSON闸系数30天'];rows=[];sides=[];checks=[]
for run in runs:
 p=R/'06_结果'/run;status=json.loads((p/'运行摘要.json').read_text());assert status['status']=='COMPLETED';raw=pd.read_csv(p/'逐闸水位流量.csv');d=raw.melt(id_vars=['station_code','time_s'],var_name='metric',value_name='sim').merge(obs,on=['station_code','metric','time_s'],validate='one_to_one');d=d[d.valid&(d.metric!='flow_m3s')&(d.time_s>0)];d['raw_error']=d.sim-d.raw_value;d['author_error']=d.sim-d.value
 for period,lo,hi in [('全30天',0,2592000),('首6小时',0,21600),('前3天',0,259200),('后27天',259200,2592000),('最后3天',2332800,2592000)]:
  v=d[(d.time_s>lo)&(d.time_s<=hi)]
  for area,g in [('全线118侧',v),('作者88侧',v[v.pdf_comparable]),('作者未展示30侧',v[~v.pdf_comparable])]:
   for basis,col in [('原始观测','raw_error'),('作者修正','author_error')]:
    means=g.groupby(['station_code','metric'])[col].apply(lambda z:abs(z).mean());rows.append(dict(run=run,period=period,area=area,basis=basis,n=len(g),mae=float(abs(g[col]).mean()),bias=float(g[col].mean()),max_error=float(abs(g[col]).max()),pass_mae_0p1=int((means<=.1).sum())))
 for (st,metric),g in d.groupby(['station_code','metric']):sides.append(dict(run=run,station=st,metric=metric,mae=float(abs(g.author_error).mean()),bias=float(g.author_error.mean()),pdf_comparable=bool(g.pdf_comparable.iloc[0])))
 step=pd.read_csv(p/'逐步收敛核验.csv');dt=step.dt_s if 'dt_s' in step else pd.Series(15.,index=step.index);assert abs(dt.sum()-2592000)<1e-8;assert np.allclose(np.diff(np.r_[0,step.time_s]),dt);assert step.head_residual_m.max()<.0015 and step.gate_residual_m3s.max()<.01
 retry=p/'失败步减半记录.csv';retries=len(pd.read_csv(retry)) if retry.exists() and retry.stat().st_size>1 else 0
 init=p/'初始化增量.csv';updates=pd.read_csv(init) if init.stat().st_size>1 else pd.DataFrame();checks.append(dict(run=run,steps=len(step),seconds=float(dt.sum()),retries=retries,min_dt=float(dt.min()),negative_hole_steps=int((step.min_hole_flow<0).sum()),max_head_residual=float(step.head_residual_m.max()),max_gate_residual=float(step.gate_residual_m3s.max()),init_changes=updates.to_dict('records')))
pd.DataFrame(rows).to_csv(O/'五组单因素精度汇总.csv',index=False);pd.DataFrame(sides).to_csv(O/'五组逐测点误差.csv',index=False);(O/'五组时间步与初态核验.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));s=pd.DataFrame(rows);print(s[(s.area=='作者88侧')&(s.basis=='作者修正')].to_string(index=False));print(json.dumps(checks,ensure_ascii=False,indent=2))
