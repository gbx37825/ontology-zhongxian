"""两个合格工况的n4/C4/C5数值可辨识性；不率定、不改真实工程参数。"""
from pathlib import Path
from collections import deque
import copy,hashlib,json,sys,time
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'35_局部率定前提复核';B=R/'26_冻结基线与后续归因';W=O.parent/'03_联合辨识条件'
sys.path[:0]=[str(P/'03_局部模型'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
import 局部闸段 as api
from 并联结构物理修复 import build_engine_physics,attach_parallel_physics
from 常驻水力进程 import jit_evidence

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
raw,obs,norm,coeff=api.data();bd=raw['baseData']
windows=pd.read_csv(W/'联合诊断不重叠通过窗口.csv');windows=windows[(windows.window_h==6)&windows.passed]
assert len(windows)>=2 and windows.end_s.max()<=1296000
assert all(float(a.end_s)<float(b.start_s) for a,b in zip(windows.itertuples(),list(windows.itertuples())[1:]))
cuts=(48495.37,74731.);targets=[('JZZ4_forebay_m',48695.37),('JZZ4_afterbay_m',48795.37)]
first=[n for n,c in bd['channels'].items() if min(bd['sections'][s]['location'] for s in c['sec_names'])<cuts[0]<max(bd['sections'][s]['location'] for s in c['sec_names'])]
last=[n for n,c in bd['channels'].items() if min(bd['sections'][s]['location'] for s in c['sec_names'])<cuts[1]<max(bd['sections'][s]['location'] for s in c['sec_names'])]
assert len(first)==len(last)==1
start=bd['channels'][first[0]]['node2'];end=bd['channels'][last[0]]['node1']
edges={}
for kind in ['channels','siphons','gates']:
 for name,x in bd[kind].items():
  key=(kind,name.split('-')[0] if kind=='gates' else name,x['node1'],x['node2'])
  edges.setdefault(key,[]).append(name)
adj={}
for edge in edges:adj.setdefault(edge[2],[]).append(edge)
queue=deque([(start,[])]);seen=set();paths=[]
while queue:
 node,path=queue.popleft()
 if node==end:paths.append(path);continue
 if node in seen:continue
 seen.add(node)
 for edge in adj.get(node,[]):queue.append((edge[3],path+[edge]))
assert len(paths)==1;path=paths[0]
selected={k:[] for k in ['channels','siphons','gates']}
for edge in path:selected[edge[0]].extend(edges[edge])
middle_channels=selected['channels'].copy()
selected['channels']=[first[0]]+middle_channels+[last[0]]
assert len(middle_channels)==2 and len(selected['siphons'])==2 and len(selected['gates'])==6
nodes={n:copy.deepcopy(bd['nodes'][n]) for edge in path for n in [edge[2],edge[3]]};channels={};sections={}
for ci,name in enumerate(selected['channels']):
 ch=copy.deepcopy(bd['channels'][name]);xs=[bd['sections'][s]['location'] for s in ch['sec_names']];lo=max(min(xs),cuts[0]);hi=min(max(xs),cuts[1]);points=api._cut_profile(bd,name,lo,hi);ch['sec_names']=[]
 for i,(x,g) in enumerate(points):
  key=f'joint_{ci}_{i}';sections[key]=dict(name=key,location=x,yz=[list(p) for p in g.raw_yz],num_interp=0);ch['sec_names'].append(key)
 if ci==0:
  ch['node1']='LOCAL_Q';nodes['LOCAL_Q']=dict(x=0.,y=0.,zb=points[0][1].z_inv,Amin=0.,nodeType=2)
 if ci==len(selected['channels'])-1:
  ch['node2']='LOCAL_Z';nodes['LOCAL_Z']=dict(x=0.,y=0.,zb=points[-1][1].z_inv,Amin=0.,nodeType=1)
 channels[name]=ch
gates={k:copy.deepcopy(bd['gates'][k]) for k in selected['gates']};siphons={k:copy.deepcopy(bd['siphons'][k]) for k in selected['siphons']}
bleeds={k:copy.deepcopy(v) for k,v in bd['bleeders'].items() if v.get('channel') and v['channel'][0] in channels and cuts[0]<v['channel'][1]<cuts[1]}
assert set(bleeds)=={'FSK6-2-pool3','FSK7-1-pool4'}
times,q=api.observed('JZZ4','flow_m3s');times2,z=api.observed('JZZ5','afterbay_m');assert np.array_equal(times,times2)
mini=copy.deepcopy(raw);mini['baseData']={**{k:{} for k in bd},'nodes':nodes,'channels':channels,'sections':sections,'gates':gates,'siphons':siphons,'bleeders':bleeds}
mini['initialData']={**copy.deepcopy(raw['initialData']),'channels':{'global_Z':0.,'global_H':7.},'boundaries':{'LOCAL_Q':float(q[0]),'LOCAL_Z':float(z[0])},'linear_boundary':{},'gates':{k:raw['initialData']['gates'][k] for k in gates},'bleeders':{k:raw['initialData']['bleeders'].get(k,0.) for k in bleeds},'pumps':{},'valves':{},'turbines':{}}
ts=dict(seriesLink={},seriesData={});i=0
for sid,(kind,name) in raw['timeSeries']['seriesLink'].items():
 if (kind=='gates' and name in gates) or (kind=='bleeders' and name in bleeds):ts['seriesLink'][str(i)]=[kind,name];ts['seriesData'][str(i)]=copy.deepcopy(raw['timeSeries']['seriesData'][sid]);i+=1
for name,y in [('LOCAL_Q',q),('LOCAL_Z',z)]:ts['seriesLink'][str(i)]=['nodes',name];ts['seriesData'][str(i)]=np.column_stack([times,y]).tolist();i+=1
mini['timeSeries']=ts;model_file=O/'联合段候选输入.json';dump(model_file,{'严陵河淇河条件性联合段':mini})
source_files=[Path(__file__),Path(api.__file__),model_file,R/'04_脚本/并联结构物理修复.py',W/'联合诊断选窗结果.json',W/'联合诊断不重叠通过窗口.csv',B/'共享输入/C07.json',B/'共享输入/Pipedream闸流系数.json']
configuration=dict(scope='数值多工况灵敏度与可辨识性，不求工程参数；C5显式作为第三参数',cuts_m=cuts,targets=targets,selected=selected,topology_path=[dict(kind=e[0],names=edges[e],from_node=e[2],to_node=e[3]) for e in path],middle_roughness_channels=middle_channels,bleeders=bleeds,windows=windows[['start_s','end_s']].to_dict('records'),perturbation=.05,observation_scale_m=.03,observation_scale_note='沿用B02的0.03m仅作数值尺度，不证明噪声独立或已实测估计',fixed_uncertainties=['西赵河/淇河旧几何差异未裁决','两座倒虹吸n/K保留原先验','端部损失原值保留','安装位置/高程/仪器身份仍是候选'],source_sha256={str(p):sha(p) for p in source_files})
dump(O/'预定试验约定.json',configuration)

def build(t,ratios):
 old,s,c=build_engine_physics(model_file,gate_max_normalization=norm)
 c=copy.deepcopy(c);sj=c['superjunctions'];sl=c['superlinks'];nodeids=sj.set_index('name').id.to_dict()
 sj.loc[sj.name.isin(['LOCAL_Q','LOCAL_Z']),['a','b','c','A_s']]=0.
 sl['C_uk']=1/.67**2;sl['C_dk']=1/.67**2
 for cut in ['LOCAL_Q','LOCAL_Z']:
  sl.loc[sl.sj_0==nodeids[cut],'C_uk']=1e9;sl.loc[sl.sj_1==nodeids[cut],'C_dk']=1e9
 ks=sl.loc[sl.name.isin(middle_channels),'id'].to_numpy(int);mask=c['links'].k.isin(ks)
 c['links'].loc[mask,'n']*=ratios['n4'];sl.loc[sl.id.isin(ks),'n']*=ratios['n4']
 for name in ['C4','C5']:
  gate=c['orifices'].name.str.startswith('JZZ'+name[1]+'-');c['orifices'].loc[gate,'C']*=ratios[name]
 e=api.PipedreamHydraulicEngine(c,dt=15.,backend='numba');e.model._method='lsq';attach_parallel_physics(e,c);e.parallel_physics_manifest=old.parallel_physics_manifest
 m=e.model;force=api.Forcing(e,s,c);H=float(np.interp(t,times,z));Q=float(np.interp(t,times,q));heads=np.maximum(H,m._z_inv_j+.05);heads[nodeids['LOCAL_Z']]=H
 signs=np.asarray([c['channel_discretization'][name]['flow_sign'] for name in e.channel_names])
 initial=dict(H_j=heads,h_Ik=np.maximum(H-m._z_inv_Ik,.05),Q_ik=Q*signs[m._ki],Q_uk=Q*signs,Q_dk=Q*signs,Q_o=np.asarray([Q/(2 if name.startswith('JZZ4-') else 4) for name in e.gate_names]))
 m.load_state(initial);m.compute_storage_volumes();m.t=t;m.iter_count=0;m.save_state();assert all(np.array_equal(getattr(m,k),v) for k,v in initial.items())
 raw_heights=np.full(m.NIk,np.inf)
 for name,meta in c['channel_discretization'].items():
  if name not in s['baseData']['channels']:continue
  for p in meta['sections']:raw_heights[p['junction_id']]=p['height']
 mesh=api.meshes(s,c);ops=[]
 for name,x in targets:
  hit=[(cn,a) for cn,a in mesh.items() if a['x'][0]<=x<=a['x'][-1]];assert len(hit)==1
  cn,a=hit[0];jj,w=api.weights_at(a,x);ops.append(dict(name=name,state='h_Ik',indices=jj.tolist(),weights=w.tolist(),offset=float(m._z_inv_Ik[jj]@w),chainage=x,channel=cn))
 assert m._A_sj[nodeids['LOCAL_Q']]==m._A_sj[nodeids['LOCAL_Z']]==0
 physical=[]
 for row in c['siphons_info']:
  ids=np.flatnonzero(m._ki==row['superlink_id']);assert abs(m._dx_ik[ids].sum()-row['physical_length'])<1e-8
  assert np.all(m._n_ik[ids]==siphons[row['name']]['nc'])
  physical.append({k:row[k] for k in ['name','physical_length','n_orifice','width','height','loss_coeff','equivalent_length_added']})
 audit=dict(ratios=ratios,operators=ops,cut_flow_node=nodeids['LOCAL_Q'],cut_head_node=nodeids['LOCAL_Z'],physical_siphons=physical,initial_state_restored=True,solver_identity=e.solver_identity,gate_coefficients=m._Co.tolist(),gate_names=e.gate_names)
 local=dict(engine=e,model=m,source=s,converted=c,forcing=force,operators=ops,audit=audit,raw_heights=raw_heights,runtime_audit=dict(checked_hydraulic_attempts=0,maximum_depth_excess_m=float((m.h_Ik-raw_heights).max()),exceedance_attempts=0,exceedance_point_attempts=0))
 api.check_range(local)
 return local

def sample(local):
 m=local['model'];values=np.array([m.h_Ik[o['indices']]@np.asarray(o['weights'])+o['offset'] for o in local['operators']])
 return values

trials=[];stack=[];started=time.time()
try:
 for wi,w in enumerate(windows.itertuples()):
  states={}
  for parameter,factor in [('baseline',1.),('n4',.95),('n4',1.05),('C4',.95),('C4',1.05),('C5',.95),('C5',1.05)]:
   ratios=dict(n4=1.,C4=1.,C5=1.)
   if parameter!='baseline':ratios[parameter]=factor
   for k,source_name in [('n4','POOL4_n'),('C4','JZZ4_C'),('C5','JZZ5_C')]:lo,hi=api.parameter_bounds()[source_name];assert lo<=ratios[k]<=hi
   local=build(float(w.end_s),ratios);hold=api.mean_forcing(local,float(w.start_s),float(w.end_s));result=api.steady(local,kw=hold,max_seconds=60.)
   if not result['converged'] or local['runtime_audit']['exceedance_attempts']:raise RuntimeError('稳态或原轮廓门槛未通过')
   value=sample(local);states[(parameter,factor)]=value
   row=dict(window_index=wi,start_s=int(w.start_s),end_s=int(w.end_s),parameter=parameter,factor=factor,water_levels_m=value.tolist(),audit=local['audit'],runtime=local['runtime_audit'],steady=result,numba=jit_evidence())
   trials.append(row);dump(O/'逐次水力灵敏度试验.json',trials);print(json.dumps(dict(window=wi,parameter=parameter,factor=factor,water_levels=value.tolist()),ensure_ascii=False),flush=True)
  J=np.column_stack([(states[(p,1.05)]-states[(p,.95)])/.1 for p in ['n4','C4','C5']]);stack.append(J)
 matrix=np.vstack(stack);G=matrix*.05/.03
 U,svals,Vt=np.linalg.svd(G,full_matrices=False)
 correlation=G.T@G/np.outer(np.linalg.norm(G,axis=0),np.linalg.norm(G,axis=0))
 result=dict(status='COMPLETED',conditioned_on_old_geometry_and_siphon_parameters=True,parameters=['n4','C4','C5'],targets=[p[0] for p in targets],windows=configuration['windows'],raw_jacobian_m_per_multiplier=matrix.tolist(),scaled_J_for_5percent_over_003m=G.tolist(),singular_values=svals.tolist(),numerical_rank=int(np.linalg.matrix_rank(G)),condition_number=float(svals[0]/svals[-1]) if svals[-1]>0 else None,column_cosine_similarity=correlation.tolist(),weakest_parameter_direction=Vt[-1].tolist(),smallest_response_to_unit_5percent_parameter_combination=float(svals[-1]),unit_5percent_note='参数坐标以5%为1单位，水位以0.03m为1单位；最小奇异值不能单独作为工程验收',single_window_numerical_ranks=[int(np.linalg.matrix_rank(x)) for x in stack],trials=len(trials),source_hashes_unchanged=all(sha(p)==h for p,h in configuration['source_sha256'].items()),wall_seconds=time.time()-started,parameters_identified=False)
 assert result['source_hashes_unchanged']
 dump(O/'两工况可辨识性结果.json',result)
 pd.DataFrame(matrix,index=[f'window{i}_{target}' for i in range(len(stack)) for target,_ in targets],columns=['n4','C4','C5']).to_csv(O/'水位灵敏度矩阵.csv')
 print(json.dumps(result,ensure_ascii=False),flush=True)
except BaseException as ex:
 extra=dict(local.get('runtime_audit',{})) if 'local' in globals() else {}
 for k,v in list(extra.items()):
  if isinstance(v,float) and not np.isfinite(v):extra[k]=None
 dump(O/'失败.json',dict(error=repr(ex),completed_trials=len(trials),runtime=extra,wall_seconds=time.time()-started));raise
