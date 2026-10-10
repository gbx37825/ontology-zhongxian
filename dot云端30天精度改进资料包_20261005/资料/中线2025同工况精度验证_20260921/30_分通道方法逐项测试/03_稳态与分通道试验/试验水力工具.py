from pathlib import Path
import sys,json,time,copy,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];B=R/'26_冻结基线与后续归因'
sys.path[:0]=[str(R/'17_求解器修复/运行脚本'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本'),str(R/'24_参数同化根因核查/脚本')]
from 试验公共 import Forcing,meshes,weights_at,flow_operator
from 并联结构物理修复 import build_engine_physics
from 局地参数EnKF import STATE_KEYS,restore
from 常驻水力进程 import advance
from wxq_pipedream.warmup import adaptive_warmup,WarmupConfig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def build():
 e,s,c=build_engine_physics(B/'共享输入/C07.json',gate_max_normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text()))
 with np.load(B/'共享输入/初始化后状态_0s.npz') as z:restore(e.model,{k:z[k].copy() for k in STATE_KEYS})
 e.model.t=0.;e.model.iter_count=0;f=Forcing(e,s,c)
 rr=np.r_[e.model._J_uk,e.model._J_dk];cc=np.r_[e.model._J_dk,e.model._J_uk];_,lab=connected_components(csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(e.model.M,e.model.M)),directed=False)
 params=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json').read_text())['parameter_bounds']
 return e,s,c,f,params

def save(m,path):np.savez_compressed(path,**{k:getattr(m,k) for k in STATE_KEYS})
def fixed_steady(e,f,t,folder,label,max_seconds=180):
 folder=Path(folder);folder.mkdir(exist_ok=True,parents=True);m=e.model;kw=f.at(t);original=m.step
 def step(**kwargs):
  kwargs.update(Q_0Ik=kw['Q_0Ik'],strict=True,num_iter=20)
  try:return original(**kwargs)
  except RuntimeError as ex:
   if 'did not converge' not in str(ex):raise
   raise FloatingPointError(str(ex)) from ex
 m.step=step
 cfg=WarmupConfig(initial_dt=15.,min_dt=1.875,max_dt=300.,num_iter=20,max_steps=50000,max_wall_seconds=max_seconds,verification_steps=240)
 try:result=adaptive_warmup(e,cfg,boundary_heads=kw['H_bc'],node_inflows=kw['Q_in'],gate_openings=kw['u_o'])
 finally:m.step=original
 (folder/(label+'_稳态诊断.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));save(m,folder/(label+'_状态.npz'))
 print(json.dumps(dict(label=label,converged=result['converged'],status=result['status'],wall=result.get('wall_seconds')),ensure_ascii=False),flush=True)
 return result

def change(m,params,name,factor):
 p=next(x for x in params if x['name']==name);arr=m._n_ik if p['kind']=='n' else m._Co;arr[p['indices']]*=factor;m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk]

def operators(e,s,c,stations):
 source=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text());ops=[x for x in source if x['station'] in stations]
 for op in ops:op['name']=op['station']+'_'+op['metric']
 pos=pd.read_csv(R/'02_输入/优先级确认后测流映射.csv').set_index('stationcode')
 for st in stations:
  if bool(pos.loc[st,'precise_position_usable']):
   q=flow_operator(e,s,c,st,float(pos.loc[st,'usable_meter_chainage']))
   if 'indices' in q:ops.append({**q,'name':st+'_flow_m3s','metric':'flow_m3s','offset':0})
 return ops

def values(m,ops):return {p['name']:float(getattr(m,p['state'])[p['indices']]@np.array(p['weights'])+p.get('offset',0)) for p in ops}
