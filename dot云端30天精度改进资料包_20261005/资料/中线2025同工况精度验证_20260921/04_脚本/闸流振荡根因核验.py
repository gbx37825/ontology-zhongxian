"""冻结输入/4h初态的闸流消融；诊断补丁不进入正式求解器。"""
import json,types,argparse,time
import numpy as np
import pandas as pd
from 试验公共 import R,Forcing
from 并联结构物理修复 import build_engine_physics
from pipedream_solver.nsuperlink import numba_solve_orifice_flows
OUT=R/'11_Pipedream振荡根因';OUT.mkdir(exist_ok=True)
NORM=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text())
def original_q(m,H,u):
 return numba_solve_orifice_flows(H,u,m._z_inv_j,m._z_o,m._tau_o,m._y_max_o,m._Co,m._Ao,m._unidir_o,m._J_uo,m._J_do)
def integrated_q(m,H,u):
 # 矩形竖向开口静水压差积分，仅作连续闸流关系的因果消融，不视为率定后的工程闸流公式。
 z=m._z_inv_j[m._J_uo]+m._z_o
 hu=H[m._J_uo]-z;hd=H[m._J_do]-z;hi=np.maximum(hu,hd);lo=np.maximum(np.minimum(hu,hd),0)
 a=np.minimum(np.maximum(hi,0),m._y_max_o*u);wet=np.minimum(a,lo);delta=np.maximum(hi-lo,0)
 term=wet*np.sqrt(delta)+(2/3)*(np.maximum(hi-wet,0)**1.5-np.maximum(hi-a,0)**1.5)
 return np.sign(hu-hd)*m._Co*m._real_width*np.sqrt(2*9.81)*term

def coeff(m,u=None):
 m._diag_class.orifice_flow_coefficients(m,u=u)
 saved=[x.copy() for x in (m._alpha_o,m._beta_o,m._chi_o)]
 fun=integrated_q if m._law=='integral' else original_q
 q=fun(m,m.H_j,u)
 if m._linear=='tangent':
  eps=1e-5;up=m.H_j.copy();up[m._J_uo]+=eps;dn=m.H_j.copy();dn[m._J_do]+=eps
  # 并联闸孔共享节点，但每个向量中的节点索引赋值只加一次。
  alpha=(fun(m,up,u)-q)/eps;beta=(fun(m,dn,u)-q)/eps
  m._alpha_o[:]=alpha;m._beta_o[:]=beta;m._chi_o[:]=q-alpha*m.H_j[m._J_uo]-beta*m.H_j[m._J_do]
 else:
  # 对照：保留旧1/|Q_previous|割线形式，只让非线性流量关系连续。
  dh=m.H_j[m._J_uo]-m.H_j[m._J_do]
  gamma=np.divide(q*q,np.abs(m.Q_o)*np.abs(dh),out=np.zeros_like(q),where=(np.abs(m.Q_o)*np.abs(dh)>1e-16))
  m._alpha_o[:]=gamma;m._beta_o[:]=-gamma;m._chi_o[:]=0

 if hasattr(m,'_only'):
  for dst,src in zip((m._alpha_o,m._beta_o,m._chi_o),saved):dst[~m._only]=src[~m._only]

def solve(m,dt,u=None):
 if m._law=='integral':
  q=integrated_q(m,m.H_j,u);cap=np.where(m.H_j[m._J_uo]>m.H_j[m._J_do],m._V_sj[m._J_uo],m._V_sj[m._J_do])/dt
  new=np.sign(q)*np.minimum(np.abs(q),cap)
  if hasattr(m,'_only'):
   m._diag_class.solve_orifice_flows(m,dt,u=u);new[~m._only]=m.Q_o[~m._only]
  m._Qo=new
 else:m._diag_class.solve_orifice_flows(m,dt,u=u)

def track(m,*args,**kwargs):
 m._diag_class._solve_step(m,*args,**kwargs);m._heads.append(m.H_j.copy())

def build(case,base,start=14400,model=None,physics='physical'):
 e,s,c=build_engine_physics(**({'model':model} if model else {}),gate_max_normalization=NORM,mode=physics);m=e.model;f=Forcing(e,s,c)
 with np.load(R/'06_结果'/base/'内部断面状态.npz') as d:
  j=int(np.flatnonzero(d['time_s']==start)[0]);initial={k:d[k][j].copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']}
 m.load_state({**initial,'t':float(start)});m._diag_class=type(m);m._real_width=np.array([s['baseData']['gates'][n]['b'] for n in c['orifices'].name])
 if case!='original' and 'only' not in case:m._g1_o[:]=m._y_max_o;m._g2_o[:]=m._real_width
 if 'only42' in case:m._only=c['orifices'].name.str.startswith('JZZ42-').to_numpy() & c['orifices'].name.str.endswith('-3').to_numpy()
 assert not set(m._J_uo)&set(m._J_do), 'vector derivative assumes disjoint gate end nodes'
 m._law='integral' if case.startswith('integral') else 'original';m._linear='tangent' if case.endswith('tangent') else 'secant'
 if m._law=='integral' or m._linear=='tangent':
  m.orifice_flow_coefficients=types.MethodType(coeff,m);m.solve_orifice_flows=types.MethodType(solve,m)
 m._solve_step=types.MethodType(track,m)
 return e,s,c,f

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',default='original');p.add_argument('--iterations',type=int,default=1);p.add_argument('--dt',type=float,default=15);p.add_argument('--sweep',action='store_true');p.add_argument('--start',type=float,default=14400);p.add_argument('--duration',type=float,default=7200);p.add_argument('--base',default='P00_并联物理修复稳态6小时');p.add_argument('--model');p.add_argument('--station',default='JZZ42');p.add_argument('--only-gate');p.add_argument('--physics',default='physical',choices=['physical','legacy']);a=p.parse_args()
 e,s,c,f=build(a.case,a.base,a.start,a.model,a.physics);m=e.model
 if a.only_gate:m._only=c['orifices'].name.eq(a.only_gate).to_numpy();assert m._only.sum()==1
 ids=np.flatnonzero(c['orifices'].name.str.startswith(a.station+'-'));up=m._J_uo[ids[0]];dn=m._J_do[ids[0]]
 if a.sweep:
  hole=ids[-1]
  depth=m.H_j[up]-m._z_inv_j[up]-m._z_o[hole];rows=[]
  for opening in [0,1,4,depth-1e-5,depth,depth+1e-5,5.551,7.787]:
   u=f.at(a.start)['u_o'].copy();u[hole]=opening/m._y_max_o[hole];m.orifice_hydraulic_geometry(u=u)
   rows.append(dict(opening=opening,hu=float(m.H_j[up]),hd=float(m.H_j[dn]),depth=float(depth),area=float(m._Ao[hole]),q_original=float(original_q(m,m.H_j,u)[hole]),q_integral=float(integrated_q(m,m.H_j,u)[hole]),g1=float(m._g1_o[hole]),g2=float(m._g2_o[hole]),ymax=float(m._y_max_o[hole]),width=float(m._real_width[hole])))
  pd.DataFrame(rows).to_csv(OUT/f'定水位开度扫描_{a.case}.csv',index=False);print(json.dumps(rows,ensure_ascii=False));return
 rows=[];started=time.time()
 for i in range(round(a.duration/a.dt)):
  t=a.start+i*a.dt;kw=f.at(t);kw.update(dt=a.dt,num_iter=a.iterations,head_tol=.0015);before=m.H_j.copy();qbefore=m.Q_o.copy();m._heads=[];m.step(**kw)
  last=m._heads[-2] if len(m._heads)>1 else before
  linear=m._alpha_o*m.H_j[m._J_uo]+m._beta_o*m.H_j[m._J_do]+m._chi_o
  j=ids[-1];opening=kw['u_o'][j]*m._y_max_o[j]
  row=dict(time_s=t+a.dt,forebay_m=m.H_j[up],afterbay_m=m.H_j[dn],gate_flow=m.Q_o[ids].sum(),tracked_hole_q=m.Q_o[j],tracked_opening=opening,head_residual=float(abs(m.H_j-last).max()),iterations=m.iter_elapsed,gate_balance_mismatch=float(abs(linear-m.Q_o).max()),station_mismatch=float(abs(linear[ids]-m.Q_o[ids]).max()),pre_hu=float(before[up]),pre_hd=float(before[dn]),pre_tracked_hole_q=float(qbefore[j]),tracked_alpha=float(m._alpha_o[j]),linear_tracked_hole_q=float(linear[j]),tracked_area=float(m._Ao[j]),max_step_dH=float(abs(m.H_j-before).max()),min_all_gate_flow=float(m.Q_o.min()))
  rows.append(row)
  if not np.isfinite(m.H_j).all():raise ValueError('nonfinite')
 df=pd.DataFrame(rows);name=f'{a.case}_iter{a.iterations}_dt{a.dt:g}'+('' if a.station=='JZZ42' else '_'+a.station)+('' if a.physics=='physical' else '_legacy');df.to_csv(OUT/(name+'.csv'),index=False)
 summary=dict(case=a.case,physics=a.physics,station=a.station,tracked_gate=str(c['orifices'].iloc[ids[-1]]['name']),only_gate=a.only_gate,start_s=a.start,duration_s=a.duration,base=a.base,iterations=a.iterations,dt=a.dt,min_q=float(df.gate_flow.min()),max_q=float(df.gate_flow.max()),min_H=float(df.forebay_m.min()),max_H=float(df.forebay_m.max()),max_residual=float(df.head_residual.max()),unconverged=int((df.head_residual>=.0015).sum()),max_gate_balance_mismatch=float(df.gate_balance_mismatch.max()),max_station_mismatch=float(df.station_mismatch.max()),max_step_dH=float(df.max_step_dH.max()),wall_s=time.time()-started)
 (OUT/(name+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
