"""三种有实际桩号切口的局部模型；复用safe转换与Forcing，不做在线同化。"""
from pathlib import Path
from functools import lru_cache
import copy,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'26_冻结基线与后续归因'
sys.path.insert(0,str(ROOT/'17_求解器修复/运行脚本'))
from 试验公共 import Forcing,meshes,weights_at
from wxq_pipedream.wxq_to_superlink import parse_wxq_json
from wxq_pipedream.section_geometry import classify_section,interpolate_section_geometry
from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine
from wxq_pipedream.warmup import adaptive_warmup,WarmupConfig

STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
CASES={
 'JZZ6_short':dict(station='JZZ6',up=96769.,down=97119.,tail_station='JZZ6',tail_metric='afterbay_m',channels=['POOL5-1-JJZ5JJZ6','POOL6-1-JJZ6DHX1'],observations=[('JZZ6','forebay_m',96949.)]),
 'JZZ16_short':dict(station='JZZ16',up=300091.,down=300721.,tail_station='JZZ16',tail_metric='afterbay_m',channels=['POOL15-2-DHX20JJZ16','POOL16-1-JJZ15DHX21'],observations=[('JZZ16','forebay_m',300551.)]),
 'JZZ16_pool':dict(station='JZZ16',up=300091.,down=327100.,tail_station='JZZ17',tail_metric='forebay_m',channels=['POOL15-2-DHX20JJZ16','POOL16-1-JJZ15DHX21'],observations=[('JZZ16','forebay_m',300551.),('JZZ16','afterbay_m',300721.)]),
}

@lru_cache(None)
def data():
 raw=json.loads((FROZEN/'共享输入/C07.json').read_text())['中线全线']
 obs=pd.read_csv(FROZEN/'共享输入/作者已核实修正观测层.csv')
 norm=json.loads((FROZEN/'共享输入/2025开度归一化上限.json').read_text())
 coeff=json.loads((FROZEN/'共享输入/Pipedream闸流系数.json').read_text())
 return raw,obs,norm,coeff

def observed(station,metric):
 _,obs,_,_=data();rows=obs[(obs.station_code==station)&(obs.metric==metric)].sort_values('time_s')
 assert len(rows)==1441 and rows.valid.all() and np.isfinite(rows.value).all()
 return rows.time_s.to_numpy(float),rows.value.to_numpy(float)

@lru_cache(None)
def parameter_bounds():
 cfg=json.loads((FROZEN/'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json').read_text())
 return {p['name']:(p['lower'],p['upper']) for p in cfg['parameter_bounds']}

def _cut_profile(bd,channel,lo,hi):
 names=bd['channels'][channel]['sec_names'];points=[];source=[bd['sections'][name] for name in names]
 for a,b in zip(source[:-1],source[1:]):
  ga,gb=classify_section(a['yz']),classify_section(b['yz']);count=int(a.get('num_interp',0))+1
  for i in range(count):
   frac=i/count;points.append((float(a['location']+(b['location']-a['location'])*frac),interpolate_section_geometry(ga,gb,frac)))
 points.append((float(source[-1]['location']),classify_section(source[-1]['yz'])))
 points.sort(key=lambda p:p[0]);xs=np.array([p[0] for p in points]);assert lo>=xs[0] and hi<=xs[-1]
 def at(x):
  same=np.flatnonzero(abs(xs-x)<1e-9)
  if same.size:return points[int(same[0])][1]
  j=int(np.searchsorted(xs,x)-1);return interpolate_section_geometry(points[j][1],points[j+1][1],(x-xs[j])/(xs[j+1]-xs[j]))
 return [(lo,at(lo))]+[p for p in points if lo<p[0]<hi]+[(hi,at(hi))]

def build(case_name,*,t0=0.,c_factor=1.,initial_offset=0.):
 spec=CASES[case_name];raw,obs,norm,coeff=data();bd=raw['baseData'];station=spec['station']
 lower,upper=parameter_bounds()[station+'_C']
 if not lower<=c_factor<=upper:raise ValueError(f'C倍率超冻结B02范围: {c_factor} outside {lower,upper}')
 gates={k:copy.deepcopy(v) for k,v in bd['gates'].items() if k.startswith(station+'-')};assert len(gates)==2
 gate_nodes=list(dict.fromkeys(n for g in gates.values() for n in [g['node1'],g['node2']]))
 nodes={name:copy.deepcopy(bd['nodes'][name]) for name in gate_nodes};channels={};sections={};profile_info=[]
 for index,name in enumerate(spec['channels']):
  orig=bd['channels'][name];allx=[float(bd['sections'][sn]['location']) for sn in orig['sec_names']]
  lo=max(spec['up'],min(allx));hi=min(spec['down'],max(allx));profile=_cut_profile(bd,name,lo,hi)
  ch=copy.deepcopy(orig);ch['sec_names']=[]
  for j,(x,g) in enumerate(profile):
   sn=f'local_{index}_{j}';sections[sn]=dict(name=sn,location=x,yz=[list(z) for z in g.raw_yz],num_interp=0);ch['sec_names'].append(sn)
  if index==0:
   ch['node1']='LOCAL_Q';nodes['LOCAL_Q']=dict(x=bd['nodes'][orig['node1']]['x'],y=bd['nodes'][orig['node1']]['y'],zb=profile[0][1].z_inv,Amin=0.,nodeType=2,max_depth=float('inf'))
  else:
   ch['node2']='LOCAL_Z';nodes['LOCAL_Z']=dict(x=bd['nodes'][orig['node2']]['x'],y=bd['nodes'][orig['node2']]['y'],zb=profile[-1][1].z_inv,Amin=0.,nodeType=1,max_depth=float('inf'))
  channels[name]=ch;profile_info.append(dict(channel=name,start_m=lo,end_m=hi,retained_calculation_sections=len(profile),n=ch['nc']))
 bleeds={k:copy.deepcopy(v) for k,v in bd['bleeders'].items() if v.get('channel') and v['channel'][0] in channels and spec['up']<float(v['channel'][1])<spec['down']}
 mini=copy.deepcopy(raw);mini['baseData']={**{k:{} for k in bd},'nodes':nodes,'channels':channels,'sections':sections,'gates':gates,'bleeders':bleeds,'siphons':{}}
 times,q=observed(station,'flow_m3s');ht,z=observed(spec['tail_station'],spec['tail_metric']);assert np.array_equal(times,ht)
 mini['initialData']={**copy.deepcopy(raw['initialData']),'channels':{'global_Z':float(z[0]),'global_H':1.},'boundaries':{'LOCAL_Q':float(q[0]),'LOCAL_Z':float(z[0])},'linear_boundary':{},'gates':{k:raw['initialData']['gates'][k] for k in gates},'bleeders':{k:raw['initialData']['bleeders'].get(k,0.) for k in bleeds},'pumps':{},'valves':{},'turbines':{}}
 ts=dict(seriesLink={},seriesData={});ix=0
 for sid,(kind,name) in raw['timeSeries']['seriesLink'].items():
  if (kind=='gates' and name in gates) or (kind=='bleeders' and name in bleeds):ts['seriesLink'][str(ix)]=[kind,name];ts['seriesData'][str(ix)]=copy.deepcopy(raw['timeSeries']['seriesData'][sid]);ix+=1
 for name,values in [('LOCAL_Q',q),('LOCAL_Z',z)]:ts['seriesLink'][str(ix)]=['nodes',name];ts['seriesData'][str(ix)]=np.column_stack([times,values]).tolist();ix+=1
 mini['timeSeries']=ts
 converted=parse_wxq_json(mini,gate_max_openings=norm,section_geometry_mode='safe',minimum_initial_depth_m=.01)
 cj=converted['superjunctions'];sl=converted['superlinks'];ids=cj.set_index('name').id.to_dict()
 # Artificial mathematical cuts have no storage and no added fitting loss. Original gate-adjacent ends remain unchanged.
 cj.loc[cj.name.isin(['LOCAL_Q','LOCAL_Z']),['a','b','c','A_s']]=0.
 sl['C_uk']=1/.67**2;sl['C_dk']=1/.67**2
 for cut in ['LOCAL_Q','LOCAL_Z']:
  sl.loc[sl.sj_0==ids[cut],'C_uk']=1e9;sl.loc[sl.sj_1==ids[cut],'C_dk']=1e9
 converted['orifices']['C']=converted['orifices'].name.map(coeff)*c_factor
 assert converted['orifices'].C.notna().all()
 e=PipedreamHydraulicEngine(converted,dt=15.,backend='numba');m=e.model;m._method='lsq';force=Forcing(e,mini,converted)
 H=float(np.interp(t0,times,z))+initial_offset;Q=float(np.interp(t0,times,q));initial_head=np.maximum(H,m._z_inv_j+.05);initial_head[ids['LOCAL_Z']]=float(np.interp(t0,times,z))
 depth=np.maximum(H-m._z_inv_Ik,.05);signs=np.array([converted['channel_discretization'][name]['flow_sign'] for name in e.channel_names])
 initial=dict(H_j=initial_head,h_Ik=depth,Q_ik=Q*signs[m._ki],Q_uk=Q*signs,Q_dk=Q*signs,Q_o=np.full(m.n_o,Q/m.n_o))
 constructor_q=float(np.max(abs(m.Q_ik)));m.load_state(initial);m.compute_storage_volumes();m.t=t0;m.iter_count=0;m.save_state()
 assert all(np.array_equal(getattr(m,k),v) for k,v in initial.items())
 mesh=meshes(mini,converted);ops=[]
 for st,metric,x in spec['observations']:
  hit=[(name,a) for name,a in mesh.items() if a['x'][0]<=x<=a['x'][-1]];assert len(hit)==1
  name,a=hit[0];jj,w=weights_at(a,x);ops.append(dict(name=st+'_'+metric,station=st,metric=metric,chainage=x,channel=name,state='h_Ik',indices=jj.tolist(),weights=w.tolist(),offset=float(m._z_inv_Ik[jj]@w)))
 audit=dict(case=case_name,spec=spec,profiles=profile_info,bleeders=force.routing,cut_flow_node=ids['LOCAL_Q'],cut_head_node=ids['LOCAL_Z'],gate_names=e.gate_names,gate_geometry=converted['orifices'][['name','g1','g2','z_o','y_max','C']].to_dict('records'),original_gate_input=gates,constructor_max_abs_Q_before_reset=constructor_q,initial_state_explicitly_restored=True,artificial_cut_storage_zero=True,artificial_cut_loss_factor=1e-18,original_adjacent_end_loss_factor=.67**4,operators=ops,within_raw_contour_initial=bool(np.all(m.h_Ik<=np.array([p['height'] for a in converted['channel_discretization'].values() for p in a['sections']]))),solver_identity=e.solver_identity)
 heights=np.zeros(m.NIk)
 for a in converted['channel_discretization'].values():
  for p in a['sections']:heights[p['junction_id']]=p['height']
 actual_ends=[]
 for k in range(m.NK):
  for side,node,factor in [('up',int(m._J_uk[k]),float(m._C_uk[k])),('down',int(m._J_dk[k]),float(m._C_dk[k]))]:
   artificial=node in [ids['LOCAL_Q'],ids['LOCAL_Z']]
   assert abs(factor-(1e-18 if artificial else .67**4))<1e-15
   actual_ends.append(dict(superlink=k,side=side,node=e.node_names[node],artificial_cut=artificial,runtime_loss_factor=factor))
 assert m._A_sj[ids['LOCAL_Q']]==0 and m._A_sj[ids['LOCAL_Z']]==0
 audit['actual_cut_storage_areas_m2']={name:float(m._A_sj[ids[name]]) for name in ['LOCAL_Q','LOCAL_Z']}
 audit['actual_endpoint_losses']=actual_ends
 audit['coefficient_multiplier_bounds']=[lower,upper];audit['cut_node_GPS_note']='桩号由原断面插值确定；切口节点x/y仅保留端点占位坐标，不作测量定位依据'
 runtime=dict(checked_hydraulic_attempts=0,maximum_depth_excess_m=float((m.h_Ik-heights).max()),exceedance_attempts=0,exceedance_point_attempts=0,note='含预热尝试与正式推进每一步；若预热回退，已检查尝试仍留痕；超轮廓立即抛异常')
 return dict(engine=e,model=m,source=mini,converted=converted,forcing=force,operators=ops,audit=audit,spec=spec,raw_heights=heights,runtime_audit=runtime)

def forcing(local,t):return local['forcing'].at(t)

def check_range(local):
 m=local['model']
 if not all(np.isfinite(getattr(m,k)).all() for k in STATE_KEYS):raise FloatingPointError('局部水力状态非有限')
 excess=m.h_Ik-local['raw_heights'];bad=excess>1e-7;record=local['runtime_audit']
 record['checked_hydraulic_attempts']+=1;record['maximum_depth_excess_m']=max(record['maximum_depth_excess_m'],float(excess.max()))
 if bad.any():
  record['exceedance_attempts']+=1;record['exceedance_point_attempts']+=int(bad.sum());record['last_exceedance']=dict(time_s=float(m.t),junction_ids=np.flatnonzero(bad).tolist(),excess_m=excess[bad].tolist())
  raise FloatingPointError('局部水位超原轮廓，拒绝以截高状态进行率定')

def sample(local):
 m=local['model'];d={o['name']:float(getattr(m,o['state'])[o['indices']]@np.array(o['weights'])+o.get('offset',0.)) for o in local['operators']};a=local['audit'];u=a['cut_flow_node'];v=a['cut_head_node']
 d['boundary_tailwater_m']=float(m.H_j[v]);d['boundary_inlet_water_m']=float(m.H_j[u]);d['gate_flow_m3s']=float(m.Q_o.sum())
 d['cut_inlet_flow_m3s']=float(m.Q_uk[m._J_uk==u].sum()-m.Q_dk[m._J_dk==u].sum());d['cut_outlet_flow_m3s']=float(m.Q_dk[m._J_dk==v].sum()-m.Q_uk[m._J_uk==v].sum())
 return d

def mean_forcing(local,lo,hi):
 ts=np.r_[lo,local['forcing'].times[(local['forcing'].times>lo)&(local['forcing'].times<hi)],hi];kw={}
 for key in ['H_bc','Q_in','Q_0Ik','u_o']:
  a=np.stack([forcing(local,t)[key] for t in ts]);kw[key]=np.trapezoid(a,ts,axis=0)/(hi-lo)
 kw.update(dt=15.,num_iter=20,strict=True);return kw

def steady(local,*,t=None,kw=None,max_seconds=60.):
 m=local['model'];e=local['engine'];hold=forcing(local,t) if kw is None else kw;old=m.step;clock=m.t
 def step(**args):
  args.update(Q_0Ik=hold['Q_0Ik'],num_iter=20,strict=True)
  try:
   result=old(**args);check_range(local);return result
  except RuntimeError as ex:
   if 'did not converge' not in str(ex):raise
   raise FloatingPointError(str(ex)) from ex
 m.step=step
 try:r=adaptive_warmup(e,WarmupConfig(initial_dt=15.,min_dt=1.875,max_dt=300.,num_iter=20,max_steps=30000,max_wall_seconds=max_seconds,verification_steps=240),boundary_heads=hold['H_bc'],node_inflows=hold['Q_in'],gate_openings=hold['u_o'])
 finally:m.step=old
 m.t=clock;m.iter_count=0
 return r

def advance(local,t0,t1,*,output_interval=1800.):
 m=local['model'];assert abs(m.t-t0)<1e-8;rows=[];retries=0
 for ts in np.arange(t0,t1,15.):
  pending=[(float(ts),min(15.,t1-ts))]
  while pending:
   tt,dt=pending.pop(0);kw=forcing(local,tt);kw.update(dt=dt,num_iter=20,strict=True)
   try:m.step(**kw)
   except RuntimeError as ex:
    if 'did not converge' not in str(ex) or dt<=1.875:raise
    retries+=1;pending[:0]=[(tt,dt/2),(tt+dt/2,dt/2)];continue
   check_range(local)
  if abs((m.t-t0)%output_interval)<1e-8 or m.t==t1:rows.append(dict(time_s=float(m.t),**sample(local)))
 return rows,retries
