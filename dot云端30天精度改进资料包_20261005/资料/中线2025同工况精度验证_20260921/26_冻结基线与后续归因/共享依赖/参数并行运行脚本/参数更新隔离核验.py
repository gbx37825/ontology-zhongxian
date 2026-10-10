from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'17_求解器修复/运行脚本'));from 试验公共 import *
sys.path.insert(0,str(R/'04_脚本'));from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import ParameterEnKF,STATE_KEYS,restore,volume
e,s,c=build_engine_physics(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text()))
with np.load(R/'06_结果/A03R_C07物理加初态自适应30天/初始化后状态_0s.npz') as z:restore(e.model,{k:z[k] for k in STATE_KEYS})
f=ParameterEnKF(e,c,n=50);before=[{k:getattr(m,k).copy() for k in STATE_KEYS} for m in f.members];v0=[volume(m) for m in f.members];t0=[m.t for m in f.members];old=f.theta.copy();node=e.node_index['NODE-main-0'];op=dict(station='合成',metric='水位',state='H_j',indices=[node],weights=[1.],sigma=.03,value=10.05)
f.values=lambda obs:10+.3*f.theta[0].copy()
trace=f.analysis([op],0)
assert f.theta[0].mean()>old[0].mean();assert abs(f.theta-old).max()<=np.log(1.01)+1e-12
far=np.array([p['xmin']>f.xparts['H_j'][node]+20000 for p in f.parameters]);assert np.array_equal(old[far],f.theta[far]);assert np.all(f.theta>=f.lower-1e-12) and np.all(f.theta<=f.upper+1e-12)
assert all(np.array_equal(getattr(m,k),z[k]) for m,z in zip(f.members,before) for k in STATE_KEYS);assert all(m.t==t for m,t in zip(f.members,t0));assert all(volume(m)==v for m,v in zip(f.members,v0))
x=f.theta.copy();row=f.analysis([{**op,'value':1000}],1)[0];assert row['status']=='rejected_innovation';assert np.array_equal(x,f.theta)
# Roughness changes must reach internal and endpoint arrays; siphon n stays fixed.
for m in f.members:
 assert np.array_equal(m._n_uk,m._n_ik[m._i_1k]);assert np.array_equal(m._n_dk,m._n_ik[m._i_nk]);assert np.array_equal(m._n_ik[m._parallel_i],f.base_n[m._parallel_i])
result=dict(parameters=len(f.parameters),members=50,parameter_changes_toward_synthetic_observation=True,max_cycle_change_bounded=True,localization_cutoff_preserved=True,parameter_bounds=True,all_hydraulic_states_time_and_volume_unchanged=True,bad_innovation_rejected=True,roughness_endpoint_arrays_synchronized=True,siphon_roughness_unchanged=True)
(R/'23_五十成员参数同化/参数更新隔离核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result)
