"""固定水动力源码上的输入、实际桩号取水与断面观测算子。"""
from pathlib import Path
import json,sys,re
import numpy as np
import pandas as pd
R=Path('/Users/gaobingxiang/code/HydroOS-hangzhou/outputs/中线2025同工况精度验证_20260921')
sys.path.insert(0,'/Users/gaobingxiang/code/HydroOS-hangzhou/water/pipedream-runner')
from scripts.run_simulation import load_wxq
from wxq_pipedream.wxq_to_superlink import parse_wxq_json
from wxq_pipedream.hydraulic_engine import PipedreamHydraulicEngine
from wxq_pipedream.warmup import adaptive_warmup
from wxq_pipedream.warmup_io import resolve_config

DEFAULT_MODEL=R/'02_输入/C00_实际开度负值修复_2025同工况30天.json'

def build_engine(model=DEFAULT_MODEL,restart=None,roughness_factor=1.,roughness_pools=None,gate_max_normalization=None,use_calibrated_coefficients=True):
    source=load_wxq(Path(model))
    gate_max={k:float(v.get('b',1) or 1) for k,v in source['baseData']['gates'].items()}
    if gate_max_normalization:gate_max.update(gate_max_normalization)
    c=parse_wxq_json(source,gate_max_openings=gate_max,section_geometry_mode='safe',minimum_initial_depth_m=7.)
    c['superjunctions'].loc[:,'h_0']=7.
    c['superjunctions'].loc[:,'max_depth']=np.maximum(c['superjunctions'].max_depth.to_numpy(),7.)
    co=json.loads((R/'01_原始资料/Pipedream闸流系数.json').read_text()) if use_calibrated_coefficients else {}
    c['orifices'].loc[:,'C']=c['orifices'].name.map(co).fillna(c['orifices'].C)
    if roughness_factor!=1:
        names=c['superlinks'].name
        mask=names.str.upper().str.startswith('POOL')
        if roughness_pools is not None:mask &= names.str.upper().str.extract(r'^POOL(\d+)')[0].astype(float).isin(roughness_pools)
        ks=set(c['superlinks'].loc[mask,'id']);c['superlinks'].loc[mask,'n']*=roughness_factor;c['links'].loc[c['links'].k.isin(ks),'n']*=roughness_factor
    e=PipedreamHydraulicEngine(c,dt=15.,backend='numba');e.model._method='lsq'
    if restart is not None:
        state=json.loads(Path(restart).read_text()) if isinstance(restart,(str,Path)) else restart
        for key in ['max_depth','max_depth_k']:
            if key in state.get('arrays',{}):state['arrays'][key]['values']=[float('inf') if x is None else x for x in state['arrays'][key]['values']]
        e.load_restart_state(state)
    return e,source,c

def meshes(source,c):
    sections=source['baseData']['sections'];result={}
    for name,meta in c['channel_discretization'].items():
        points=[]
        for p in meta['sections']:
            a,b=p['source_interval']
            if a not in sections or b not in sections:continue
            x=(1-p['fraction'])*sections[a]['location']+p['fraction']*sections[b]['location']
            points.append((float(x),int(p['junction_id'])))
        if len(points)>=2:
            points=sorted(points);result[name]=dict(x=np.array([v[0] for v in points]),ids=np.array([v[1] for v in points]),k=meta['superlink_id'],sign=meta['flow_sign'])
    return result

def weights_at(mesh,x):
    xs=mesh['x'];i=int(np.clip(np.searchsorted(xs,x,side='right')-1,0,len(xs)-2));f=float((x-xs[i])/(xs[i+1]-xs[i]))
    if f < -1e-8 or f>1+1e-8:raise ValueError(f'桩号{x}超出断面范围{xs[0]}..{xs[-1]}')
    return mesh['ids'][i:i+2],np.array([1-f,f])

class Forcing:
    def __init__(self,e,source,c,positioned=True):
        self.engine=e;self.source=source;self.converted=c;self.mesh=meshes(source,c);self.positioned=positioned
        series=list(c['time_series'].values());self.times=np.array(series[0]['times_s'],float)
        if not all(np.array_equal(s['times_s'],self.times) for s in series):raise ValueError('本试验要求共同等距输入时刻')
        nt=len(self.times);self.h=np.full((nt,e.model.M),np.nan);self.q=np.zeros((nt,e.model.M));self.q0=np.zeros((nt,e.model.NIk));self.u=np.zeros((nt,e.model.n_o));self.bleeder_values={};self.routing=[]
        types={k:v.get('nodeType',0) for k,v in source['baseData']['nodes'].items()}
        flow_nodes={s['target_name'] for s in series if s['target_type']=='nodes'}
        for name,value in c.get('boundaries',{}).items():
            if name not in e.node_index:continue
            i=e.node_index[name]
            if types.get(name)==1:self.h[:,i]=value
            elif name not in flow_nodes:self.q[:,i]+=value
        for name,i in e.gate_index.items():self.u[:,i]=float(c['initial_data']['gates'].get(name,0))/float(c['orifices'].iloc[i].y_max)
        for s in series:
            v=np.array(s['values'],float);name=s['target_name'];typ=s['target_type']
            if typ=='nodes':
                i=e.node_index[name]
                if types.get(name)==1:self.h[:,i]=v
                else:self.q[:,i]+=v
            elif typ=='gates':self.u[:,e.gate_index[name]]=v/float(c['orifices'].iloc[e.gate_index[name]].y_max)
            elif typ=='bleeders':self.bleeder_values[name]=v
            elif typ=='channels':self.q[:,e.node_index[source['baseData']['channels'][name]['node1']]]+=v
        for row in c['bleeders']:
            name=row['name'];v=self.bleeder_values.get(name,np.full(nt,float(c['initial_data']['bleeders'].get(name,0))))
            b=source['baseData']['bleeders'][name];channel=b.get('channel');target=None;reason='原节点'
            if positioned and channel and channel[0] in self.mesh:
                mesh=self.mesh[channel[0]];x=float(channel[1])
                if mesh['x'][0]-1e-6<=x<=mesh['x'][-1]+1e-6:
                    ids,weights=weights_at(mesh,x);target=(ids,weights);self.q0[:,ids]-=v[:,None]*weights;reason='按模型真实桩号向相邻内部节点守恒分配'
                else:reason='桩号不在所属渠道断面内，保留原节点并标记'
            if target is None:
                if row['node'] not in e.node_index:raise ValueError(f'分水未接入:{name}')
                self.q[:,e.node_index[row['node']]]-=v
            self.routing.append(dict(target=name,channel=channel,mode=reason,junction_ids=[] if target is None else target[0].tolist(),weights=[] if target is None else target[1].tolist()))
        self.u=np.clip(self.u,0,1)
        assert np.isfinite(self.q).all() and np.isfinite(self.q0).all() and np.isfinite(self.h[:,e.model.bc]).all()
    def at(self,t):
        i=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2));f=float(np.clip((t-self.times[i])/(self.times[i+1]-self.times[i]),0,1))
        return dict(H_bc=self.h[i]*(1-f)+self.h[i+1]*f,Q_in=self.q[i]*(1-f)+self.q[i+1]*f,Q_0Ik=self.q0[i]*(1-f)+self.q0[i+1]*f,u_o=self.u[i]*(1-f)+self.u[i+1]*f,dt=15.,num_iter=1)
    inputs_at=at

def warmup(e,source,forcing):
    kw=forcing.at(0);fixed_q0=kw['Q_0Ik'];original=e.model.step
    def step(**kwargs):kwargs.setdefault('Q_0Ik',fixed_q0);return original(**kwargs)
    e.model.step=step
    cfg=resolve_config(source.get('initialData',{}).get('options',{}),15.,R/'01_原始资料/自适应预热.json',{'internal_method':'lsq','num_iter':1})
    try:result=adaptive_warmup(e,cfg,boundary_heads=kw['H_bc'],node_inflows=kw['Q_in'],gate_openings=kw['u_o'])
    finally:e.model.step=original
    if not result['converged']:raise RuntimeError(f'预热未收敛:{result.get("status")}')
    e.model.t=0.;e.model.iter_count=0
    return result

def flow_operator(e,source,c,station,x):
    mesh=meshes(source,c);no=int(station.removeprefix('JZZ'));options=[]
    for channel,g in mesh.items():
        p=re.match(r'POOL(\d+)',channel,re.I)
        if not p or int(p[1]) not in [no-1,no]:continue
        if g['x'][0]<=x<=g['x'][-1]:options.append((channel,g))
    if len(options)!=1:return dict(station=station,status='无唯一明渠断面对照',candidate_channels=[k for k,g in options],chainage=x)
    name,g=options[0];links=c['links'][c['links'].k==g['k']];positions=[];ids=[]
    byid=dict(zip(g['ids'],g['x']))
    for row in links.itertuples():positions.append((byid[row.j_0]+byid[row.j_1])/2);ids.append(row.id)
    order=np.argsort(positions);positions=np.array(positions)[order];ids=np.array(ids)[order]
    # 流量取包围测点的内部连接中心线性插值；端部取最近连接，不跨闸。
    if x<=positions[0]:idx=[int(ids[0])];w=[float(g['sign'])]
    elif x>=positions[-1]:idx=[int(ids[-1])];w=[float(g['sign'])]
    else:
        j=np.searchsorted(positions,x)-1;f=(x-positions[j])/(positions[j+1]-positions[j]);idx=ids[j:j+2].tolist();w=((1-f)*g['sign'],f*g['sign'])
    return dict(station=station,state='Q_ik',indices=idx,weights=list(w),channel=name,chainage=x,status='按实际测流桩号取渠内流量')

def flow_operators(e,source,c):
    ops=[];audit=[]
    for row in pd.read_csv(R/'02_输入/测点位置映射.csv').itertuples():
        if row.status!='已核对' or not np.isfinite(row.meter_chainage):audit.append(dict(station=row.stationcode,status='位置冲突或缺失，不启用流量同化'));continue
        op=flow_operator(e,source,c,row.stationcode,row.meter_chainage);audit.append(op)
        if 'indices'in op:ops.append(op)
    return ops,audit

def observe(model,op):return float(np.dot(getattr(model,op['state'])[op['indices']],op['weights']))

def control_volumes(e,ops):
    """由测流断面和闸孔两条切口解节点权重，保留离散连续方程的同一控制体。"""
    from collections import deque
    m=e.model;M=m.M;edges=[]
    for typ,up,down in [('Q_ik',m._Ik+M,m._Ip1k+M),('Q_uk',m._J_uk,m._I_1k+M),('Q_dk',m._I_Np1k+M,m._J_dk),('Q_o',m._J_uo,m._J_do)]:
        edges.extend((int(a),int(b),typ,i) for i,(a,b) in enumerate(zip(up,down)))
    edge_index={(typ,i):j for j,(a,b,typ,i) in enumerate(edges)};adj=[[] for _ in range(M+m.NIk)]
    for j,(a,b,typ,i) in enumerate(edges):adj[a].append((b,j,1));adj[b].append((a,j,-1))
    positions=pd.read_csv(R/'02_输入/测点位置映射.csv').set_index('stationcode');records=[];vectors=[]
    for op in ops:
        station=op['station'];direction=1. if positions.loc[station,'upstream_downstream']=='upstream' else -1.
        d=np.zeros(len(edges));holes=[i for name,i in e.gate_index.items() if name.startswith(station+'-')]
        for i,weight in zip(op['indices'],op['weights']):d[edge_index[('Q_ik',i)]]+=direction*weight
        for i in holes:d[edge_index[('Q_o',i)]]-=direction
        w=np.full(len(adj),np.nan);w[e.node_index['NODE-main-0']]=0.;queue=deque([e.node_index['NODE-main-0']]);error=0.
        while queue:
            a=queue.popleft()
            for b,j,sign in adj[a]:
                v=w[a]+sign*d[j]
                if np.isnan(w[b]):w[b]=v;queue.append(b)
                else:error=max(error,abs(w[b]-v))
        ok=np.isfinite(w).all() and error<1e-9 and w.min()>-1e-8 and w.max()<1+1e-8 and np.max(np.abs(w[:M][m.bc]))<1e-8
        records.append(dict(station=station,direction=direction,gate_indices=holes,graph_consistency_error=error,valid=bool(ok),weight_min=float(np.nanmin(w)),weight_max=float(np.nanmax(w))))
        if not ok:raise ValueError(records[-1])
        vectors.append(w)
    return np.asarray(vectors),records

def hydraulic_volume(model):
    model.link_hydraulic_geometry();model.compute_storage_volumes()
    return float(np.sum(model._A_ik*model._dx_ik)+np.sum(model.V_j[~model.bc]))
