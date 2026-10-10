from pathlib import Path
import sys,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'17_求解器修复/运行脚本'));from 试验公共 import Forcing,hydraulic_volume
sys.path.insert(0,str(R/'04_脚本'));from 并联结构物理修复 import build_engine_physics
O=R/'20_连续状态同化';e,s,c=build_engine_physics(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()));m=e.model;forcing=Forcing(e,s,c);tail=e.node_index['NODE-main-119'];rows=[]
p=R/'06_结果/A03R_C07物理加初态自适应30天'
with np.load(p/'内部断面状态.npz') as sn:
 times=sn['time_s'];snap={k:sn[k].copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']}
with np.load(p/'初始化后状态_0s.npz') as ini:
 for k in snap:snap[k][0]=ini[k]
for j,t in enumerate(times):
 if j%48==0 or j==len(times)-1:
  m.load_state({k:v[j] for k,v in snap.items()});v=hydraulic_volume(m);rows.append(dict(time_s=t,day=t/86400,volume_m3=v))
qout=snap['Q_dk'][:,m._J_dk==tail].sum(axis=1)-snap['Q_uk'][:,m._J_uk==tail].sum(axis=1)+snap['Q_o'][:,m._J_do==tail].sum(axis=1)-snap['Q_o'][:,m._J_uo==tail].sum(axis=1)
net=np.array([forcing.at(t)['Q_in'].sum()+forcing.at(t)['Q_0Ik'].sum() for t in times]);balance=np.trapezoid(net-qout,times);delta=rows[-1]['volume_m3']-rows[0]['volume_m3'];result=dict(initial_volume=rows[0]['volume_m3'],final_volume=rows[-1]['volume_m3'],volume_change_m3=delta,mean_geometric_storage_rate=delta/2592000,mean_boundary_net_before_tail=float(np.trapezoid(net,times)/2592000),mean_tail_outflow=float(np.trapezoid(qout,times)/2592000),sampled_flux_net_m3=float(balance),geometry_vs_sampled_flux_difference_m3=float(delta-balance),note='几何蓄量与半小时采样通量积分的诊断，不是每15秒离散守恒精确审计；初态使用观测初始化后快照')
pd.DataFrame(rows).to_csv(O/'模型逐日蓄量.csv',index=False);(O/'模型蓄量漂移摘要.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
