from pathlib import Path
import json
import numpy as np,pandas as pd
from 试验公共 import R,build_engine,Forcing
O=R/'17_求解器修复';norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());results=[]
for st,start,duration in [('JZZ9',1546200,7200),('JZZ21',2338200,10800)]:
 states={}
 with np.load(R/'06_结果/S01_C06闸流修复严格30天/内部断面状态.npz') as d:
  j=int(np.flatnonzero(d['time_s']==start)[0]);initial={k:d[k][j].copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']}
 for dt in [15.,7.5]:
  e,s,c=build_engine(R/'16_两站分别清洗复跑/C06_济河金水河合并清洗_2025同工况30天.json',gate_max_normalization=norm);m=e.model;f=Forcing(e,s,c);m.load_state({**initial,'t':float(start)});ids=np.flatnonzero(c['orifices'].name.str.startswith(st+'-'));up=m._J_uo[ids[0]];dn=m._J_do[ids[0]];rows=[];maxmass=0
  for t in np.arange(start,start+duration,dt):
   kw=f.at(t);kw.update(dt=dt,num_iter=20,strict=True);H=m.H_j.copy();h=m.h_Ik.copy();m.step(**kw);M=m.M;net=np.r_[kw['Q_in'],kw['Q_0Ik']].copy()
   for name,u,v in [('Q_ik',m._Ik+M,m._Ip1k+M),('Q_uk',m._J_uk,m._I_1k+M),('Q_dk',m._I_Np1k+M,m._J_dk),('Q_o',m._J_uo,m._J_do)]:
    q=getattr(m,name);np.add.at(net,u,-q);np.add.at(net,v,q)
   res=net-np.r_[m._A_sj*(m.H_j-H)/dt,m._E_Ik*(m.h_Ik-h)];dynamic=np.r_[~m.bc,np.ones(m.NIk,bool)];mass=float(abs(res[dynamic]).max());maxmass=max(maxmass,mass);rows.append(dict(time_s=float(t+dt),hu=m.H_j[up],hd=m.H_j[dn],q=m.Q_o[ids].sum(),max_node_continuity_residual=mass))
  d=pd.DataFrame(rows);d.to_csv(O/f'{st}_修复后dt{dt:g}.csv',index=False);states[dt]=d;results.append(dict(station=st,dt=dt,steps=len(d),max_node_continuity_residual=maxmass,min_q=float(d.q.min()),max_q=float(d.q.max())))
 join=states[15.].merge(states[7.5],on='time_s',suffixes=('_15','_7p5'));results.append(dict(station=st,dt_compare=[15,7.5],max_differences={k:float(abs(join[k+'_15']-join[k+'_7p5']).max()) for k in ['hu','hd','q']}))
(O/'修复后水量与步长核验.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False,indent=2))
