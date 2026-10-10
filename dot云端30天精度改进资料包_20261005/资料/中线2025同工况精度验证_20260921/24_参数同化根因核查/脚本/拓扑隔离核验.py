from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[2];P=R/'23_五十成员参数同化';O=R/'24_参数同化根因核查'
sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(P/'脚本')]
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import ParameterEnKF as Old
from 拓扑参数EnKF import ParameterEnKF as New
e,s,c=build_engine_physics(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()))
ops=json.loads((P/'P01M4_50成员30天参数同化/观测算子与留出.json').read_text());op=next(x for x in ops if x['station']=='JZZ10' and x['metric']=='forebay_m');report=[]
for cls in (Old,New):
 ens=cls(e,c);j=next(i for i,p in enumerate(ens.parameters) if p['name']=='POOL3_n');p=ens.parameters[j]
 # 人工构造一致的参数-水位相关性，仅验证距离屏蔽，不是实测精度试验。
 for i,m in enumerate(ens.members):m.h_Ik[op['indices']]=4+ens.theta[j,i]
 before=ens.theta.copy();states=[m.h_Ik.copy() for m in ens.members];obs=ens.values(op).mean()+.02
 ens.analysis([{**op,'value':obs}],1800.)
 delta=float(abs(ens.theta[j]-before[j]).max());unchanged=all(np.array_equal(m.h_Ik,h) for m,h in zip(ens.members,states));assert unchanged
 report.append(dict(implementation=cls.__module__,pool3_span_m=[p['xmin'],p['xmax']],remote_pool3_update_max=delta,state_unchanged=unchanged))
assert report[0]['remote_pool3_update_max']>0 and report[1]['remote_pool3_update_max']==0
(O/'拓扑隔离核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report)
