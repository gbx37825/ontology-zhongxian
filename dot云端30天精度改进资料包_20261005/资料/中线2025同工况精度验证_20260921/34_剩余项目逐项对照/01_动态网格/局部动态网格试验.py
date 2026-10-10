"""用真实 Pipedream step 推进两个原断面之间的受控时变试验；不改全线或基线。"""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent;R=O.parents[1]
B=R/'26_冻结基线与后续归因';B2=B/'B02_渠池归属修复后的参数同化诊断基线'
sys.path.insert(0,str(R/'30_分通道方法逐项测试/03_稳态与分通道试验'))
from 试验水力工具 import build
from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine
from pipedream_solver import nsuperlink
from numba import config

DT=1.;DURATION=21600.;T0=21600.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
_,source,converted,_,_=build()
checkpoint=B2/'六小时诊断起点/最新完整集合.npz'
with np.load(checkpoint) as z:
 h0=np.mean([z[f'm{i}_h_Ik'] for i in range(50)],axis=0)
 q0=np.mean([z[f'm{i}_Q_ik'] for i in range(50)],axis=0)
obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv')
cases=[]
for lid,station in [(1067,'JZZ22'),(1445,'JZZ58')]:
 row=converted['links'].loc[lid]
 name=converted['superlinks'].set_index('id').loc[row.k,'name']
 meta=converted['channel_discretization'][name]
 endpoints=[next(p for p in meta['sections'] if p['junction_id']==j) for j in [int(row.j_0),int(row.j_1)]]
 hh=[h0[int(row.j_0)],h0[int(row.j_1)]];H=[hh[j]+endpoints[j]['z_inv'] for j in range(2)]
 sign=float(np.sign(q0[lid]));in_node=0 if sign>0 else 1;out_node=1-in_node
 vv=obs[(obs.station_code==station)&obs.time_s.between(T0,T0+DURATION)&obs.valid]
 qq=vv[vv.metric=='flow_m3s'].sort_values('time_s');zz=vv[vv.metric=='afterbay_m'].sort_values('time_s')
 assert len(qq)==len(zz)==13 and np.array_equal(qq.time_s,zz.time_s)
 times=qq.time_s.to_numpy(float)-T0
 qin=abs(float(q0[lid]))*qq.value.to_numpy(float)/float(qq.value.iloc[0])
 zout=H[out_node]+zz.value.to_numpy(float)-float(zz.value.iloc[0])
 cases.append(dict(link_id=lid,channel=name,station=station,length=float(row.dx),n=float(row.n),endpoints=endpoints,initial_H=H,signed_initial_Q=float(q0[lid]),flow_sign=sign,in_node=in_node,out_node=out_node,times=times.tolist(),qin=qin.tolist(),zout=zout.tolist()))

def make(case,segments,*,reset_initial=True):
 p0,p1=case['endpoints'];in_node=case['in_node'];out_node=case['out_node'];frac=np.linspace(0,1,segments+1)
 g={key:(1-frac)*p0[key]+frac*p1[key] for key in ['z_inv','height','bottom_width','side_slope']}
 initial_H=(1-frac)*case['initial_H'][0]+frac*case['initial_H'][1]
 sj=pd.DataFrame([dict(id=j,name=f'cut{j}',z_inv=p['z_inv'],h_0=case['initial_H'][j]-p['z_inv'],bc=j==out_node,storage='functional',a=0.,b=0.,c=1.,max_depth=np.inf) for j,p in enumerate([p0,p1])])
 # Cut boundaries are not physical fittings; negligible identical end terms avoid grid-dependent surrogate losses.
 sl=pd.DataFrame([dict(id=0,name='selected_original_interval',sj_0=0,sj_1=1,in_offset=0.,out_offset=0.,dx=case['length'],n=case['n'],shape='trapezoidal',g1=float(g['height'].mean()),g2=float(g['bottom_width'].mean()),g3=float(g['side_slope'].mean()),g4=0.,Q_0=case['signed_initial_Q'],h_0=float((initial_H-g['z_inv']).mean()),ctrl=False,A_s=0.,A_c=0.,C=1.,C_uk=1e9,C_dk=1e9,j_0=0,j_1=segments)])
 jj=pd.DataFrame([dict(id=j,k=0,h_0=float(initial_H[j]-g['z_inv'][j]),A_s=0.,z_inv=float(g['z_inv'][j])) for j in range(segments+1)])
 ll=[]
 for j in range(segments):
  mid={key:float((a[j]+a[j+1])/2) for key,a in g.items()}
  ll.append(dict(id=j,k=0,j_0=j,j_1=j+1,dx=case['length']/segments,n=case['n'],shape='rect_open' if mid['side_slope']==0 else 'trapezoidal',g1=mid['height'],g2=mid['bottom_width'],g3=mid['side_slope'],g4=0.,Q_0=case['signed_initial_Q'],ctrl=False,A_c=0.,C=1.,friction_method='cm'))
 e=PipedreamHydraulicEngine(dict(superjunctions=sj,superlinks=sl,links=pd.DataFrame(ll),junctions=jj,transects={}),dt=DT,backend='numba');m=e.model;m._method='lsq'
 e.constructor_probe=dict(t=float(m.t),max_abs_Q_before_restore=float(np.max(abs(m.Q_ik))),max_initial_H_difference=float(abs(m.H_j-np.asarray(case['initial_H'])).max()))
 if reset_initial:
  # SuperLink.__init__ calls a micro-step before backend adaptation; restore the intended common initial state.
  initial=dict(H_j=np.asarray(case['initial_H'],float),h_Ik=initial_H-g['z_inv'],Q_ik=np.full(segments,case['signed_initial_Q']),Q_uk=np.asarray([case['signed_initial_Q']]),Q_dk=np.asarray([case['signed_initial_Q']]),Q_o=np.empty(0),t=0.)
  m.load_state(initial);m.compute_storage_volumes();m.iter_count=0;m.save_state()
  assert all(np.array_equal(getattr(m,k),v) for k,v in initial.items() if k!='t')
 assert m._J_uk[0]==0 and m._J_dk[0]==1
 assert abs(m._dx_ik.sum()-case['length'])<1e-12
 return e,g

def geometry_volume(m,case):
 m.link_hydraulic_geometry();m.compute_storage_volumes()
 return float(m._A_ik@m._dx_ik +m.V_j[case['in_node']])

def step(m,case,q,z,dt):
 qin=np.zeros(2);qin[case['in_node']]=q;hbc=np.full(2,np.nan);hbc[case['out_node']]=z
 m.step(Q_in=qin,H_bc=hbc,dt=dt,num_iter=30,head_tol=1e-9,flow_tol=1e-9,strict=True)
 assert all(np.isfinite(getattr(m,k)).all() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o'])

def extract(m,case):
 qout=float(m.Q_dk[0]) if case['flow_sign']>0 else -float(m.Q_uk[0])
 return float(m.H_j[case['in_node']]),qout

results=[];all_times=[];started=time.time()
for case in cases:
 for nseg in [1,2,4,8]:
  e,g=make(case,nseg);m=e.model;start=time.time();nsteps=0;min_margin=np.inf;min_depth=np.inf;stable=0
  warm_q=case['qin'][0];warm_z=case['zout'][0]
  try:
   for w in range(int(3600/DT)):
    hp=np.r_[m.H_j,m.h_Ik].copy();qp=np.r_[m.Q_ik,m.Q_uk,m.Q_dk].copy()
    step(m,case,warm_q,warm_z,DT);nsteps+=1
    dh=float(abs(np.r_[m.H_j,m.h_Ik]-hp).max());dq=float(abs(np.r_[m.Q_ik,m.Q_uk,m.Q_dk]-qp).max())
    stable=stable+1 if dh<1e-9 and dq<1e-7 else 0
    if stable>=12:break
   if stable<12:raise RuntimeError(f'Warmup not steady: {case["link_id"]}/{nseg}: {dh},{dq}')
  except RuntimeError as ex:
   failed=dict(link_id=case['link_id'],segments=nseg,status='FAILED_WARMUP',error=str(ex),head_residual_m=float(m.last_head_residual),real_step_calls=nsteps,dynamic_steps=0,dt=DT,wall_s=time.time()-start)
   results.append(failed);print(json.dumps(failed,ensure_ascii=False),flush=True);continue
  warm=dict(steps=w+1,seconds=(w+1)*DT,max_head_step_m=dh,max_flow_step_m3s=dq,upstream_H=extract(m,case)[0])
  m.t=0.;m.iter_count=0
  V0=geometry_volume(m,case);Vp=V0;net=0.;disc_storage=0.;geom_step_residual=0.;disc_step_residual=0.;rows=[]
  for it in range(int(DURATION/DT)):
   t=(it+1)*DT;q=float(np.interp(t,case['times'],case['qin']));z=float(np.interp(t,case['times'],case['zout']))
   hp=m.h_Ik.copy();Hsuper=m.H_j[case['in_node']]
   step(m,case,q,z,DT);nsteps+=1
   # E contains dV/dh / dt. The nonfixed superjunction has constant 1m2 storage.
   dVdisc=float(m._E_Ik@(m.h_Ik-hp))*DT+float(m.H_j[case['in_node']]-Hsuper)
   Hup,qout=extract(m,case);net_step=(q-qout)*DT;net+=net_step;disc_storage+=dVdisc
   V=geometry_volume(m,case);geom_step_residual=max(geom_step_residual,abs((net_step-(V-Vp))/DT));disc_step_residual=max(disc_step_residual,abs((net_step-dVdisc)/DT));Vp=V
   min_margin=min(min_margin,float(np.min(g['height']-m.h_Ik)));min_depth=min(min_depth,float(m.h_Ik.min()))
   rows.append(dict(link_id=case['link_id'],segments=nseg,time_s=t,upstream_H_m=Hup,outlet_Q_m3s=qout,inlet_Q_m3s=q,outlet_H_m=z,volume_m3=V,net_inflow_m3=net))
  assert min_margin>0 and min_depth>0,(case['link_id'],nseg,min_margin,min_depth)
  result=dict(link_id=case['link_id'],segments=nseg,status='COMPLETED',real_step_calls=nsteps,dynamic_steps=int(DURATION/DT),dt=DT,duration_s=DURATION,wall_s=time.time()-start,warmup=warm,min_contour_margin_m=min_margin,min_depth_m=min_depth,net_inflow_m3=net,geometric_storage_change_m3=V-V0,geometric_mass_residual_m3=net-(V-V0),discrete_storage_change_m3=disc_storage,discrete_mass_residual_m3=net-disc_storage,max_step_geometric_residual_m3s=geom_step_residual,max_step_discrete_residual_m3s=disc_step_residual,solver_identity=e.solver_identity,numba_momentum_signature_count=len(nsuperlink.numba_b_ik.signatures),end_loss_coefficient=m._C_uk.tolist(),constructor_probe=e.constructor_probe,initial_state_explicitly_restored=True)
  results.append(result);all_times.extend(rows)
  print(json.dumps(result,ensure_ascii=False),flush=True)

frame=pd.DataFrame(all_times)
for row in results:
 if row['status']!='COMPLETED':continue
 ref=int(frame[frame.link_id==row['link_id']].segments.max())
 a=frame[(frame.link_id==row['link_id'])&(frame.segments==row['segments'])].set_index('time_s');b=frame[(frame.link_id==row['link_id'])&(frame.segments==ref)].set_index('time_s')
 row['reference_segments']=ref
 for key in ['upstream_H_m','outlet_Q_m3s','volume_m3']:
  diff=a[key]-b[key];row[key+'_difference_vs_reference']=dict(mae=float(abs(diff).mean()),max_abs=float(abs(diff).max()),end=float(diff.iloc[-1]))
out=dict(status='COMPLETED' if all(x['status']=='COMPLETED' for x in results) else 'PARTIAL_NUMERICAL_FAILURE',elapsed_s=time.time()-started,scope='受控局部时变离散验证，附近观测仅提供变化波形，以冻结状态为锚；不是局地实测边界率定，不是全线MAE验证。',cases=cases,results=results,source_sha256={'script':sha(Path(__file__)), 'checkpoint':sha(checkpoint),'C07':sha(B/'共享输入/C07.json'),'observations':sha(B/'共享输入/作者已核实修正观测层.csv')},numba_disable_jit=bool(config.DISABLE_JIT),boundary_note='外部切口不另设实体构筑物损失；各网格使用同一近零端头系数1e-18。仅一段SuperLink，每次细分只改变内部网格。')
for path in [Path(nsuperlink.__file__),Path(nsuperlink.__file__).with_name('superlink.py'),Path(nsuperlink.__file__).with_name('ngeometry.py'),R.parents[1]/'water/pipedream-runner/wxq_pipedream/backends.py',R.parents[1]/'water/pipedream-runner/wxq_pipedream/hydraulic_engine.py']:
 out['source_sha256'][str(path)]=sha(path)
(O/'动态网格结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False));frame[(frame.time_s%60==0)|(frame.time_s==DT)].to_csv(O/'动态过程抽样.csv',index=False)
