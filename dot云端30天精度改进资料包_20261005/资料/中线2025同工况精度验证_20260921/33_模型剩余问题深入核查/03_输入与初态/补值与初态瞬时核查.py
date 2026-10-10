from pathlib import Path
import json,sys
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因'
sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
from 试验公共 import Forcing
sys.path.insert(0,str(R/'04_脚本'))
from 并联结构物理修复 import build_engine_physics
s=json.loads((B/'共享输入/C07.json').read_text());s=s if 'baseData'in s else next(iter(s.values()))
x=json.loads((R/'02_输入/C00_实际开度负值修复_2025同工况30天.json').read_text());x=x if 'baseData'in x else next(iter(x.values()))
changes=[]
for key,link in s['timeSeries']['seriesLink'].items():
 a=np.asarray(x['timeSeries']['seriesData'][key],float);b=np.asarray(s['timeSeries']['seriesData'][key],float);assert np.array_equal(a[:,0],b[:,0]);d=b[:,1]-a[:,1]
 if (abs(d)>1e-12).any():changes.append(dict(type=link[0],name=link[1],count=int((abs(d)>1e-12).sum()),max_abs_change=float(abs(d).max()),first_s=float(b[(abs(d)>1e-12),0].min()),last_s=float(b[(abs(d)>1e-12),0].max())))
pd.DataFrame(changes).to_csv(O/'C07相对初版输入实际改动.csv',index=False)
fix=pd.read_csv(R/'03_运行参数/2025输入补值逐点记录.csv'); fs=[]
for (kind,name),g in fix.groupby(['kind','name']):fs.append(dict(kind=kind,name=name,n=len(g),min_replacement=float(g.replacement.min()),max_replacement=float(g.replacement.max()),reason='；'.join(g.reason.unique())))
pd.DataFrame(fs).to_csv(O/'初版输入补值按站汇总.csv',index=False)
# 对负分水真实输入的净量级，不据此清零。
negative=[]
for key,(typ,name) in s['timeSeries']['seriesLink'].items():
 if typ!='bleeders':continue
 v=np.array(s['timeSeries']['seriesData'][key]);n=v[:,1]<0
 if n.any():negative.append(dict(name=name,count=int(n.sum()),min=float(v[:,1].min()),net_added_volume_if_compare_zero_m3=float(-np.trapezoid(np.minimum(v[:,1],0),v[:,0]))))
# 实测来源西黑山/文村北不同断面数值差只用于量级，不替代已冻结别名。
raw=pd.read_csv(R/'02_输入/2025分退水原始窗口.csv',parse_dates=['时间']);raw=raw[raw['时间'].between('2025-07-22 12:00','2025-08-21 12:00')];series=raw[raw['闸站名称'].isin(['西黑山进口闸','文村北调节池'])].pivot(index='时间',columns='闸站名称',values='总瞬时流量');diff=series['西黑山进口闸']-series['文村北调节池'];tm=(series.index-series.index[0]).total_seconds();alias=dict(names=list(series.columns),mean_difference_m3s=float(np.trapezoid(diff,tm)/tm[-1]),max_abs_m3s=float(abs(diff).max()),volume_difference_m3=float(np.trapezoid(diff,tm)),note='支线入口与下游不同断面；差额可含分水和蓄变，不是仪表偏移，不得重复扣除或任选替换')
# 只推1个15秒步，判断已存起点是否有重新调整；无需创建长过程输出。
e,ss,c=build_engine_physics(B/'共享输入/C07.json',gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()));m=e.model;f=Forcing(e,ss,c);n=np.load(B/'共享输入/初始化后状态_0s.npz');state={k:n[k].copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']};m.load_state(state,compute_hydraulic_geometries=True);m.compute_storage_volumes();m.t=0;m.iter_count=0
kw=f.at(0);kw.update(num_iter=20,strict=True,dt=15.);m.step(**kw)
dH=m.H_j-state['H_j'];dQ=m.Q_o-state['Q_o'];j=int(abs(dH).argmax());i=int(abs(dQ).argmax());instant=dict(seconds=15,head_max_abs_change_m=float(abs(dH).max()),head_max_node=e.node_names[j],gate_hole_max_abs_change_m3s=float(abs(dQ).max()),gate_hole_name=c['orifices'].iloc[i]['name'],max_internal_depth_change_m=float(abs(m.h_Ik-state['h_Ik']).max()),max_internal_flow_change_m3s=float(abs(m.Q_ik-state['Q_ik']).max()),head_residual_m=float(m.last_head_residual),gate_residual_m3s=float(m.last_gate_flow_residual),note='共同基础初态与基础系数的单步核验，不是50成员平均；未启用同化，没有改观测或冻结数据，不外推此瞬态解释整月误差')
result=dict(changes_vs_C00=changes,initial_fill=fs,small_negative_offtake=negative,west_black_alias_pair=alias,one_step=instant)
(O/'补值与初态瞬时核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(result,ensure_ascii=False,indent=2))
