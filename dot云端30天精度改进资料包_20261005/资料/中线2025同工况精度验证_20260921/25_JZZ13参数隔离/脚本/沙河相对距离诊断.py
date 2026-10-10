from pathlib import Path
import sys,json,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'25_JZZ13参数隔离';sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本')]
from 试验公共 import meshes,weights_at
from 并联结构物理修复 import build_engine_physics
e,s,c=build_engine_physics(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()));mesh=meshes(s,c);gate=c['orifices'][c['orifices'].name.str.startswith('JZZ13-')].iloc[0];rows=[];obs=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv')
for name,t in [('F01_拓扑分组修复六小时',21600),('F03_拓扑分组修复30天',2592000)]:
 with np.load(R/'24_参数同化根因核查'/name/'末态集合.npz') as z:
  H=np.mean([z[f'm{i}_H_j'] for i in range(50)],axis=0);h=np.mean([z[f'm{i}_h_Ik'] for i in range(50)],axis=0)+e.model._z_inv_Ik
 for metric,channel,position,node in [('forebay_m','POOL12-2-DHX14JJZ13',241930-115,int(gate.sj_0)),('afterbay_m','POOL13-1-DHX15DHX16',241930+105,int(gate.sj_1))]:
  ids,w=weights_at(mesh[channel],position);relative=float(h[ids]@w);measured=float(obs[(obs.station_code=='JZZ13')&(obs.metric==metric)&(obs.time_s==t)].value.iloc[0]);old=float(H[node])
  if metric=='afterbay_m':
   ids,w=weights_at(mesh[channel],242823);old=float(h[ids]@w)
  rows.append(dict(run=name,time_s=t,metric=metric,observed=measured,current_scoring_level=old,relative_distance_candidate_level=relative,candidate_minus_current=relative-old,candidate_error=relative-measured,note='只诊断相对闸距假设，不能据此改观测桩号或模型'))
a=pd.DataFrame(rows);a.to_csv(O/'沙河相对闸距水位量级.csv',index=False);print(a.to_string(index=False))
