from pathlib import Path
import sys,json
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parents[1]/'water/pipedream-runner'))
# import the active validated scalar law; no hydraulic state changes.
sys.path.insert(0,str(Path('/Users/gaobingxiang/code/HydroOS-hangzhou/water/pipedream-runner')))
from pipedream_solver.gate_hydraulics import rectangular_gate_discharge
O=R/'19_水位精度根因';raw=pd.read_csv(R/'12_全量实测质量核查/五个重点站30天开度换算米.csv.gz');obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');model=next(iter(json.loads((R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json').read_text()).values()));cs=json.loads((R/'01_原始资料/Pipedream闸流系数.json').read_text());series={name:np.asarray(model['timeSeries']['seriesData'][k],float) for k,(typ,name) in model['timeSeries']['seriesLink'].items() if typ=='gates'};cv=pd.read_csv(R/'06_结果/A03R_C07物理加初态自适应30天/测流至闸门水量换算.csv');rows=[]
for st,g in cv.groupby('station_code'):
 x=obs[obs.station_code==st].pivot(index='time_s',columns='metric',values='value').sort_index();x=x.join(g.set_index('time_s')[['direction','offtake_mean','storage_rate']]);qobs=x.flow_m3s-x.direction*(x.offtake_mean+x.storage_rate);pred=np.zeros(len(x));unit=np.zeros(len(x));openingmaxchange=np.zeros(len(x))
 for name,gate in model['baseData']['gates'].items():
  if not name.startswith(st+'-'):continue
  a=series[name][:,1];coef=cs.get(name,gate.get('c1',.9));zb=max(gate['zb'],model['baseData']['nodes'][gate['node1']]['zb']);q1=np.array([rectangular_gate_discharge(hu-zb,hd-zb,aa,gate['b'],1.) for hu,hd,aa in zip(x.forebay_m,x.afterbay_m,a)]);unit+=q1;pred+=coef*q1;openingmaxchange=np.maximum(openingmaxchange,np.r_[0,abs(np.diff(a))])
 qvalid=(qobs>0)&(unit>0)&(x.forebay_m>x.afterbay_m)&(x.index>0)&(openingmaxchange<.03)&(x.forebay_m.diff().abs()<.02)&(x.afterbay_m.diff().abs()<.02)
 ratio=qobs[qvalid]/pred[qvalid];ce=qobs[qvalid]/unit[qvalid];rows.append(dict(station=st,n=int(qvalid.sum()),equivalent_C_median=float(ce.median()),equivalent_C_p10=float(ce.quantile(.1)),equivalent_C_p90=float(ce.quantile(.9)),observed_to_legacy_formula_ratio=float(ratio.median()),legacy_formula_flow_bias=float((pred[qvalid]-qobs[qvalid]).mean())))
d=pd.DataFrame(rows);d.to_csv(O/'实测水位开度反算等效闸系数_仅诊断.csv',index=False);print(d.to_string(index=False));print('median ratio',d.observed_to_legacy_formula_ratio.median())
