"""A/B测点假设的离线局部试验接口；求解器仅绑定26冻结依赖，不加载Corelib。"""
from pathlib import Path
from functools import lru_cache
import ast,copy,hashlib,json,os,re,sys,types
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'26_冻结基线与后续归因'
DEPS=FROZEN/'共享依赖'
sys.dont_write_bytecode=True
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/pipedream_frozen_38_numba')
for name,mod in list(sys.modules.items()):
 if name.startswith(('wxq_pipedream','pipedream_solver')) and getattr(mod,'__file__',None):
  if not Path(mod.__file__).resolve().is_relative_to(DEPS.resolve()):raise RuntimeError('请在新进程加载38接口，已存在非冻结水力模块: '+name)
sys.path.insert(0,str(DEPS))
from wxq_pipedream.wxq_to_superlink import parse_wxq_json
from wxq_pipedream.section_geometry import classify_section,interpolate_section_geometry
from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine
from wxq_pipedream.warmup import adaptive_warmup,WarmupConfig

STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
UP_CHANNEL='POOL15-2-DHX20JJZ16'
POOL_CHANNEL='POOL16-1-JJZ15DHX21'
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
_definition_bindings=[]

def _bind_definitions(path,names,bindings,label):
 """原样加载指定函数/类定义，排除历史文件顶层旧water路径注入和旧入口导入。"""
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('冻结定义缺失')
 mod=types.ModuleType(label);mod.__file__=str(path);mod.__dict__.update(bindings);sys.modules[label]=mod
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),mod.__dict__)
 _definition_bindings.append(dict(module=label,file=str(path),sha256=SHA(path),definitions=names,top_level_imports_executed=False))
 return mod

_common=_bind_definitions(DEPS/'水力运行脚本/试验公共.py',['meshes','weights_at','Forcing'],dict(np=np,pd=pd,re=re,Path=Path,R=ROOT),'_38_frozen_forcing_definitions')
meshes=_common.meshes
weights_at=_common.weights_at
Forcing=_common.Forcing
_cut=_bind_definitions(ROOT/'35_局部率定前提复核/03_局部模型/局部闸段.py',['_cut_profile'],dict(np=np,classify_section=classify_section,interpolate_section_geometry=interpolate_section_geometry),'_38_reused_cut_definition')

@lru_cache(None)
def data():
 raw=json.loads((FROZEN/'共享输入/C07.json').read_text())['中线全线']
 obs=pd.read_csv(FROZEN/'共享输入/作者已核实修正观测层.csv')
 norms=json.loads((FROZEN/'共享输入/2025开度归一化上限.json').read_text())
 coeff=json.loads((FROZEN/'共享输入/Pipedream闸流系数.json').read_text())
 return raw,obs,norms,coeff

def observed(station,metric):
 rows=data()[1];rows=rows[(rows.station_code==station)&(rows.metric==metric)].sort_values('time_s')
 if len(rows)!=1441 or not rows.valid.all() or not np.isfinite(rows.value).all():raise ValueError('观测缺失或不在既定有效口径内')
 return rows.time_s.to_numpy(float),rows.value.to_numpy(float)

@lru_cache(None)
def parameter_bounds():
 cfg=json.loads((FROZEN/'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json').read_text())
 return {p['name']:(p['lower'],p['upper']) for p in cfg['parameter_bounds']}

def _local_legacy_build(model,gate_max_normalization=None,**unused):
 source=copy.deepcopy(model) if isinstance(model,dict) else next(iter(json.loads(Path(model).read_text()).values()))
 c=parse_wxq_json(source,gate_max_openings=gate_max_normalization,section_geometry_mode='safe',minimum_initial_depth_m=.01)
 c['orifices']['C']=c['orifices'].name.map(data()[3]);assert c['orifices'].C.notna().all()
 e=PipedreamHydraulicEngine(c,dt=15.,backend='numba');e.model._method='lsq'
 return e,source,c

_physics=_bind_definitions(DEPS/'并联结构物理修复.py',['_correct_section','_link_geometry','_up_geometry','_down_geometry','_link_coeffs','attach_parallel_physics','build_engine_physics'],dict(np=np,types=types,copy=copy,hashlib=hashlib,Path=Path,R=ROOT,_legacy_build=_local_legacy_build,PipedreamHydraulicEngine=PipedreamHydraulicEngine),'_38_frozen_parallel_physics')

def runtime_identity():
 modules=[];bad=[];core=[]
 for name,mod in sorted(sys.modules.items()):
  path=getattr(mod,'__file__',None)
  if 'corelib' in name.lower() or (path and 'corelib' in path.lower()):core.append(dict(name=name,file=path))
  if name.startswith(('wxq_pipedream','pipedream_solver','_38_')) and path:
   row=dict(name=name,file=str(Path(path).resolve()),sha256=SHA(path));modules.append(row)
   if name.startswith(('wxq_pipedream','pipedream_solver')) and not Path(path).resolve().is_relative_to(DEPS.resolve()):bad.append(row)
 if bad or core:raise RuntimeError('运行身份不符合冻结水力包/不加载Corelib约定')
 from numba import config
 from pipedream_solver import nsuperlink
 from pipedream_solver.gate_hydraulics import rectangular_gate_terms
 return dict(modules=modules,definition_bindings=_definition_bindings,old_water_hydraulic_modules=bad,corelib_modules=core,numba_jit_disabled=bool(config.DISABLE_JIT),momentum_compiled_signatures=len(nsuperlink.numba_b_ik.signatures),gate_compiled_signatures=len(rectangular_gate_terms.signatures),interpreter=sys.executable,numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'),formal_chain_reproduction=False)

def _source(hypothesis):
 if hypothesis not in ['A','B']:raise ValueError('hypothesis须为A或B')
 raw,obs,norms,coeff=data();bd=raw['baseData'];gates={k:copy.deepcopy(v) for k,v in bd['gates'].items() if k.startswith('JZZ16-')};assert len(gates)==2
 nodes={n:copy.deepcopy(bd['nodes'][n]) for g in gates.values() for n in [g['node1'],g['node2']]}
 channels={};sections={};end_x=327100. if hypothesis=='A' else max(bd['sections'][s]['location'] for s in bd['channels'][POOL_CHANNEL]['sec_names'])
 for index,name in enumerate([UP_CHANNEL,POOL_CHANNEL]):
  original=bd['channels'][name];xs=[bd['sections'][s]['location'] for s in original['sec_names']];lo=max(300091.,min(xs));hi=min(end_x,max(xs));profile=_cut._cut_profile(bd,name,lo,hi)
  ch=copy.deepcopy(original);ch['sec_names']=[]
  for j,(x,g) in enumerate(profile):
   sn=f'hyp_{index}_{j}';sections[sn]=dict(name=sn,location=x,yz=[list(v) for v in g.raw_yz],num_interp=0);ch['sec_names'].append(sn)
  if index==0:
   ch['node1']='LOCAL_Q';nodes['LOCAL_Q']=dict(x=0.,y=0.,zb=profile[0][1].z_inv,Amin=0.,nodeType=2)
  if index==1 and hypothesis=='A':
   ch['node2']='LOCAL_Z';nodes['LOCAL_Z']=dict(x=0.,y=0.,zb=profile[-1][1].z_inv,Amin=0.,nodeType=1)
  channels[name]=ch
 siphons={}
 if hypothesis=='B':
  upstream17={g['node1'] for name,g in bd['gates'].items() if name.startswith('JZZ17-')};assert len(upstream17)==1;tail=next(iter(upstream17))
  siphons={k:copy.deepcopy(s) for k,s in bd['siphons'].items() if s['node1']==channels[POOL_CHANNEL]['node2'] and s['node2']==tail};assert len(siphons)==1
  for s in siphons.values():
   for n in [s['node1'],s['node2']]:nodes[n]=copy.deepcopy(bd['nodes'][n])
  nodes[tail]['nodeType']=1
 else:tail='LOCAL_Z'
 bleeds={k:copy.deepcopy(v) for k,v in bd['bleeders'].items() if v.get('channel') and v['channel'][0] in channels and 300091.<v['channel'][1]<end_x}
 assert set(bleeds)=={'FSK28-2-pool15','FSK29-1-pool16','FSK30-2-pool16','FSK31-3-pool16'}
 mini=copy.deepcopy(raw);mini['baseData']={**{k:{} for k in bd},'nodes':nodes,'channels':channels,'sections':sections,'gates':gates,'siphons':siphons,'bleeders':bleeds}
 times,q=observed('JZZ16','flow_m3s');tz,z=observed('JZZ17','forebay_m');assert np.array_equal(times,tz)
 mini['initialData']={**copy.deepcopy(raw['initialData']),'channels':{'global_Z':0.,'global_H':7.},'boundaries':{'LOCAL_Q':float(q[0]),tail:float(z[0])},'linear_boundary':{},'gates':{k:raw['initialData']['gates'][k] for k in gates},'bleeders':{k:raw['initialData']['bleeders'].get(k,0.) for k in bleeds},'pumps':{},'valves':{},'turbines':{}}
 ts=dict(seriesLink={},seriesData={});index=0
 for sid,(kind,name) in raw['timeSeries']['seriesLink'].items():
  if (kind=='gates' and name in gates) or (kind=='bleeders' and name in bleeds):ts['seriesLink'][str(index)]=[kind,name];ts['seriesData'][str(index)]=copy.deepcopy(raw['timeSeries']['seriesData'][sid]);index+=1
 for name,values in [('LOCAL_Q',q),(tail,z)]:ts['seriesLink'][str(index)]=['nodes',name];ts['seriesData'][str(index)]=np.column_stack([times,values]).tolist();index+=1
 mini['timeSeries']=ts
 return mini,tail,end_x

def change(local,parameter=None,factor=None,*,n_factor=None,c_factor=None):
 if parameter is not None:
  if parameter in ['n','n16','POOL16_n']:n_factor=factor
  elif parameter in ['C','C16','JZZ16_C']:c_factor=factor
  else:raise ValueError(parameter)
 factors=dict(local['factors'])
 if n_factor is not None:factors['n']=float(n_factor)
 if c_factor is not None:factors['C']=float(c_factor)
 for k,p in [('n','POOL16_n'),('C','JZZ16_C')]:
  lo,hi=parameter_bounds()[p]
  if not lo<=factors[k]<=hi:raise ValueError(f'{p}倍率超冻结界: {factors[k]} outside {lo,hi}')
 m=local['model'];before={k:getattr(m,k).copy() for k in STATE_KEYS}
 m._n_ik[:]=local['base_n'];m._n_ik[local['n_indices']]*=factors['n'];m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m._Co[:]=local['base_C']*factors['C']
 assert all(np.array_equal(getattr(m,k),v) for k,v in before.items());local['factors']=factors

def build(hypothesis,*,t0=0.,n_factor=1.,c_factor=1.,initial_offset=0.):
 source,tail,end_x=_source(hypothesis);norms=data()[2];previous,s,c=_physics.build_engine_physics(source,gate_max_normalization=norms)
 c=copy.deepcopy(c);sj=c['superjunctions'];sl=c['superlinks'];ids=sj.set_index('name').id.to_dict();artificial=['LOCAL_Q']+(['LOCAL_Z'] if hypothesis=='A' else [])
 sj.loc[sj.name.isin(artificial),['a','b','c','A_s']]=0.;sj.loc[:,'max_depth']=np.inf
 sl['C_uk']=1/.67**2;sl['C_dk']=1/.67**2
 for name in artificial:sl.loc[sl.sj_0==ids[name],'C_uk']=1e9;sl.loc[sl.sj_1==ids[name],'C_dk']=1e9
 e=PipedreamHydraulicEngine(c,dt=15.,backend='numba');m=e.model;m._method='lsq';_physics.attach_parallel_physics(e,c);e.parallel_physics_manifest=previous.parallel_physics_manifest
 f=Forcing(e,s,c);times,q=observed('JZZ16','flow_m3s');_,z=observed('JZZ17','forebay_m');H=float(np.interp(t0,times,z));Q=float(np.interp(t0,times,q))
 raw_heights=np.full(m.NIk,np.inf);node_tops=np.full(m.M,np.inf)
 for name,meta in c['channel_discretization'].items():
  if name not in s['baseData']['channels']:continue
  for p in meta['sections']:raw_heights[p['junction_id']]=p['height']
  k=meta['superlink_id'];a,b=meta['sections'][0],meta['sections'][-1]
  node_tops[m._J_uk[k]]=min(node_tops[m._J_uk[k]],a['z_inv']+a['height']);node_tops[m._J_dk[k]]=min(node_tops[m._J_dk[k]],b['z_inv']+b['height'])
 # Both starts are guesses inside the original open contour. This only prepares t0, never clamps a simulated step.
 Hj=np.maximum(H+initial_offset,m._z_inv_j+.05);Hj=np.minimum(Hj,node_tops-.05);Hj[ids[tail]]=H
 h=np.maximum(H+initial_offset-m._z_inv_Ik,.05);h=np.minimum(h,raw_heights-.05)
 signs=np.array([c['channel_discretization'][name]['flow_sign'] for name in e.channel_names])
 initial=dict(H_j=Hj,h_Ik=h,Q_ik=Q*signs[m._ki],Q_uk=Q*signs,Q_dk=Q*signs,Q_o=np.full(m.n_o,Q/m.n_o))
 ctor_q=float(np.max(abs(m.Q_ik)));m.load_state(initial);m.compute_storage_volumes();m.t=t0;m.iter_count=0;m.save_state();assert all(np.array_equal(getattr(m,k),v) for k,v in initial.items())
 mesh=meshes(s,c);ops=[]
 for name,x in [('JZZ16_forebay_m',300551.),('JZZ16_afterbay_m',300721.)]:
  candidates=[(channel,a) for channel,a in mesh.items() if a['x'][0]<=x<=a['x'][-1]];assert len(candidates)==1
  channel,a=candidates[0];jj,w=weights_at(a,x);ops.append(dict(name=name,state='h_Ik',indices=jj.tolist(),weights=w.tolist(),offset=float(m._z_inv_Ik[jj]@w),chainage_m=x,channel=channel,role='internal_target'))
 k_pool=int(sl.loc[sl.name==POOL_CHANNEL,'id'].iloc[0]);n_ids=np.flatnonzero(m._ki==k_pool)
 local=dict(hypothesis=hypothesis,engine=e,model=m,source=s,converted=c,forcing=f,operators=ops,raw_heights=raw_heights,base_n=m._n_ik.copy(),base_C=m._Co.copy(),n_indices=n_ids,factors={'n':1.,'C':1.},runtime_audit={'checked_states':0,'exceedance_attempts':0,'exceedance_point_attempts':0,'maximum_depth_excess_m':float((m.h_Ik-raw_heights).max())})
 change(local,n_factor=n_factor,c_factor=c_factor)
 assert m.n_o==2 and all(n.startswith('JZZ16-') for n in e.gate_names)
 assert not any(n.startswith('JZZ17-') for n in e.gate_names)
 accident={k:v for k,v in data()[0]['baseData']['sections'].items() if k in ['q16s13','q16s14','q16s15','q16s16']}
 preserved=[]
 for key,v in accident.items():
  hits=[x for x in s['baseData']['sections'].values() if abs(x['location']-v['location'])<1e-8];assert len(hits)==1
  ga=classify_section(v['yz']);gb=classify_section(hits[0]['yz']);assert ga.analytic_geometry()==gb.analytic_geometry();preserved.append(dict(original_key=key,chainage_m=v['location'],geometry_unchanged=True))
 ends=[]
 for k in range(m.NK):
  for side,node,C in [('up',int(m._J_uk[k]),float(m._C_uk[k])),('down',int(m._J_dk[k]),float(m._C_dk[k]))]:ends.append(dict(superlink=e.channel_names[k],side=side,node=e.node_names[node],runtime_loss_factor=C))
 audit=dict(hypothesis=hypothesis,inflow=dict(station='JZZ16',metric='flow_m3s',chainage_m=300091.,node='LOCAL_Q'),tailwater=dict(station='JZZ17',metric='forebay_m',node=tail,chainage_m=327100. if hypothesis=='A' else None,meaning='沿用绝对桩号' if hypothesis=='A' else '出口闸前水头近似；非0.5m精确映射',scored=False),open_domain_end_chainage_m=end_x,gate_names=e.gate_names,gate_rows=c['orifices'][['name','sj_0','sj_1','g1','g2','z_o','y_max','C']].to_dict('records'),siphons=e.parallel_physics_manifest,bleeders=f.routing,n_parameter_channels=[POOL_CHANNEL],n_parameter_indices=n_ids.tolist(),operator_rows=ops,cut_storage_m2={name:float(m._A_sj[ids[name]]) for name in artificial},endpoint_terms=ends,raw_open_point_count=int(np.isfinite(raw_heights).sum()),closed_pressure_points_excluded_from_open_contour_guard=int(np.isinf(raw_heights).sum()),accident_gate=preserved,accident_gate_assumption='保留C07禹州事故闸几何，无额外可变节流假设；不认定现场全程全开',constructor_max_Q_before_reset=ctor_q,initial_state_restored=True,initial_guess_note='原轮廓内水平初猜，超范围位置在起算前压至原顶下0.05m；正式计算从不夹内部水位',initial_guess_limited_open_points=int(((H+initial_offset-m._z_inv_Ik)>raw_heights-.05).sum()),factor_bounds={'n':parameter_bounds()['POOL16_n'],'C':parameter_bounds()['JZZ16_C']},runtime_identity=runtime_identity())
 local['audit']=audit;local['inlet_node']=ids['LOCAL_Q'];local['tail_node']=ids[tail]
 check_range(local)
 return local

def forcing(local,t):return local['forcing'].at(t)

def mean_forcing(local,lo,hi):
 times=local['forcing'].times;ts=np.r_[lo,times[(times>lo)&(times<hi)],hi]
 result={k:np.trapezoid(np.stack([forcing(local,t)[k] for t in ts]),ts,axis=0)/(hi-lo) for k in ['H_bc','Q_in','Q_0Ik','u_o']}
 result.update(dt=15.,num_iter=20,strict=True);return result

def sample(local):
 m=local['model'];values={o['name']:float(m.h_Ik[o['indices']]@np.array(o['weights'])+o['offset']) for o in local['operators']}
 u=local['inlet_node'];d=local['tail_node'];values.update(boundary_tailwater_m=float(m.H_j[d]),boundary_inlet_water_m=float(m.H_j[u]),gate_flow_m3s=float(m.Q_o.sum()),cut_inlet_flow_m3s=float(m.Q_uk[m._J_uk==u].sum()-m.Q_dk[m._J_dk==u].sum()),cut_outlet_flow_m3s=float(m.Q_dk[m._J_dk==d].sum()-m.Q_uk[m._J_uk==d].sum()))
 return values

def check_range(local):
 m=local['model'];record=local['runtime_audit'];excess=m.h_Ik-local['raw_heights'];bad=excess>1e-7
 if not all(np.isfinite(getattr(m,k)).all() for k in STATE_KEYS):raise FloatingPointError('nonfinite hydraulic states')
 record['checked_states']+=1;record['maximum_depth_excess_m']=max(record['maximum_depth_excess_m'],float(excess.max()))
 if bad.any():
  record['exceedance_attempts']+=1;record['exceedance_point_attempts']+=int(bad.sum());record['last_exceedance']=dict(time_s=float(m.t),junction_ids=np.flatnonzero(bad).tolist(),excess_m=excess[bad].tolist())
  raise FloatingPointError('局部开敞水位超原轮廓；闭管压力水头不受此项约束')

def steady(local,*,t=None,kw=None,max_seconds=60.):
 m=local['model'];hold=forcing(local,t) if kw is None else kw;original=m.step;clock=m.t
 def step(**args):
  args.update(Q_0Ik=hold['Q_0Ik'],strict=True,num_iter=20)
  try:result=original(**args);check_range(local);return result
  except RuntimeError as ex:
   if 'did not converge' not in str(ex):raise
   raise FloatingPointError(str(ex)) from ex
 m.step=step
 try:r=adaptive_warmup(local['engine'],WarmupConfig(initial_dt=15.,min_dt=1.875,max_dt=300.,num_iter=20,max_steps=30000,max_wall_seconds=max_seconds,verification_steps=240),boundary_heads=hold['H_bc'],node_inflows=hold['Q_in'],gate_openings=hold['u_o'])
 finally:m.step=original
 m.t=clock;m.iter_count=0;excess=m.h_Ik-local['raw_heights'];r['accepted_final_max_depth_excess_m']=float(excess.max());r['accepted_final_exceeded_points']=int((excess>1e-7).sum());r['runtime_attempt_audit']=copy.deepcopy(local['runtime_audit'])
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
