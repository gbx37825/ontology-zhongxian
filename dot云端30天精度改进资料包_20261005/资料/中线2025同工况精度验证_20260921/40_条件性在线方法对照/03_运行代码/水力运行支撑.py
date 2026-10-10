"""40实验运行支撑：只从26冻结水力包构模，子进程显式接收分析后的状态。"""
from pathlib import Path
import copy,hashlib,importlib.util,json,multiprocessing as mp,re,time,traceback
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,dijkstra

HERE=Path(__file__).resolve().parent
R40=HERE.parent
R=R40.parent
B=R/'26_冻结基线与后续归因'
spec=importlib.util.spec_from_file_location('_40_full38',R/'38_测点假设下的联合辨识/06_全线联合参数转移/全线固定参数转移.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full);api=full.api
STATE_KEYS=api.STATE_KEYS
# step.save_state uses these geometry caches as previous-step data. Preserve them when mirroring a worker.
SNAPSHOT_KEYS=(*STATE_KEYS,'A_ik','A_uk','A_dk','A_sj','V_j','x_Ik')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

parameter_defs=api._bind_definitions(B/'B02_渠池归属修复后的参数同化诊断基线/运行源码/拓扑参数EnKF.py',
    ['ParameterEnKF','restore','volume'],dict(copy=copy,re=re,np=np,csr_matrix=csr_matrix,dijkstra=dijkstra,connected_components=connected_components,STATE_KEYS=STATE_KEYS),'_40_frozen_parameter')
state_defs=api._bind_definitions(B/'B01_允许修改水位的状态同化参考基线/运行源码/连续EnKF核心.py',
    ['LocalEnKF','snapshot','restore','volume'],dict(copy=copy,json=json,np=np,csr_matrix=csr_matrix,dijkstra=dijkstra,STATE_KEYS=STATE_KEYS),'_40_frozen_state')
advance=api._bind_definitions(api.DEPS/'参数并行运行脚本/常驻水力进程.py',['advance'],dict(np=np,STATE_KEYS=STATE_KEYS),'_40_frozen_advance').advance

def dump(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)

def make_full():
    old=api._physics._legacy_build
    try:
        api._physics._legacy_build=full.frozen_full_build
        e,s,c=api._physics.build_engine_physics(gate_max_normalization=api.data()[2])
    finally:api._physics._legacy_build=old
    return e,s,c,api.Forcing(e,s,c)

def snapshot(m):
    return dict(arrays={k:getattr(m,k).copy() for k in SNAPSHOT_KEYS},time_s=float(m.t),iter_count=int(m.iter_count))

def restore_exact(m,state):
    m.load_state(state['arrays'],compute_hydraulic_geometries=False)
    m.t=float(state['time_s']);m.iter_count=int(state['iter_count'])

def save_snapshot(path,state):
    np.savez_compressed(path,**state['arrays'],time_s=np.array(state['time_s']),iter_count=np.array(state['iter_count']))

def read_snapshot(path):
    with np.load(path) as z:return dict(arrays={k:z[k].copy() for k in SNAPSHOT_KEYS},time_s=float(z['time_s']),iter_count=int(z['iter_count']))

def score_operators(c):
    mapped=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text())
    assert len(mapped)==43 and sum(o['assimilated'] for o in mapped)==34
    all_ops=[]
    for station,g in c['orifices'].assign(station=c['orifices'].name.str.split('-').str[0]).groupby('station'):
        for metric,j in [('forebay_m',int(g.iloc[0].sj_0)),('afterbay_m',int(g.iloc[0].sj_1))]:
            all_ops.append(dict(scope='闸节点118',station=station,metric=metric,state='H_j',indices=[j],weights=[1.],offset=0.,group='节点表示'))
    all_ops += [{**o,'scope':'安装位置43','group':'更新34' if o['assimilated'] else '留出9'} for o in mapped]
    audit=json.loads((R/'30_分通道方法逐项测试/02_输入与测点核查/输入与算子核验.json').read_text())
    q=next(x['flow_operator'] for x in audit['stations'] if x['station']=='JZZ16')
    assert q['chainage']==300091 and audit['frozen_B02_input_sha256']==sha(B/'共享输入/C07.json')
    all_ops.append({**q,'metric':'flow_m3s','scope':'兰河测流切口','group':'不参与更新','offset':0.})
    return mapped,all_ops

def scores(m,ops):
    return np.array([getattr(m,o['state'])[o['indices']]@np.asarray(o['weights'])+o.get('offset',0.) for o in ops],float)

def raw_heights(m,s,c):
    heights=np.full(m.NIk,np.inf)
    for name,meta in c['channel_discretization'].items():
        if name in s['baseData']['channels']:
            for p in meta['sections']:heights[p['junction_id']]=p['height']
    return heights

def geometry_record(m,heights):
    excess=m.h_Ik-heights
    return dict(open_points_above_input_contour=int((excess>1e-7).sum()),maximum_excess_m=float(excess.max()),minimum_internal_depth_m=float(m.h_Ik.min()))

class ConstantForcing:
    def __init__(self,f,t):self.kw=f.at(t)
    def at(self,t):return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.kw.items()}

def worker(conn):
    try:
        e,s,c,f=make_full();_,ops=score_operators(c);heights=raw_heights(e.model,s,c);members=[];ids=[]
        conn.send(dict(ready=True,runtime=api.runtime_identity()))
        while True:
            command=conn.recv();kind=command['kind']
            if kind=='stop':break
            if kind=='init':
                ids=command['ids'];members=[copy.deepcopy(e.model) for _ in ids]
                for m,n,co in zip(members,command['n'],command['C']):
                    restore_exact(m,command['state']);m._n_ik[:]=n;m._Co[:]=co;m._n_uk[:]=n[m._i_1k];m._n_dk[:]=n[m._i_nk]
                conn.send(dict(initialized=True,ids=ids));continue
            if kind not in ['advance','forecast']:raise ValueError(kind)
            reset=command.get('reset_states')
            for j,(m,n,co) in enumerate(zip(members,command['n'],command['C'])):
                if reset is not None:restore_exact(m,reset[j])
                m._n_ik[:]=n;m._Co[:]=co;m._n_uk[:]=n[m._i_1k];m._n_dk[:]=n[m._i_nk]
                assert m.t==command['start_s']
            if kind=='advance':
                rows=[]
                for m in members:
                    stats=advance(m,f,command['start_s'],command['end_s'])
                    rows.append(dict(state=snapshot(m),stats=stats,geometry=geometry_record(m,heights)))
                conn.send(dict(rows=rows,runtime=api.runtime_identity()));continue
            # Both forecasting tracks start from the same cloned ensemble; original adaptive members stay unchanged.
            start=command['start_s'];leads=command['leads_s'];before=[snapshot(m) for m in members];tracks={}
            for track in ['known_historical_inputs','frozen_at_issue']:
                forcing=f if track=='known_historical_inputs' else ConstantForcing(f,start)
                clones=[copy.deepcopy(m) for m in members];lead_rows=[];cursor=start
                fixed=[(m._n_ik.copy(),m._Co.copy()) for m in clones]
                for lead in leads:
                    end=start+lead;rows=[]
                    for j,m in enumerate(clones):
                        stats=advance(m,forcing,cursor,end)
                        assert np.array_equal(m._n_ik,fixed[j][0]) and np.array_equal(m._Co,fixed[j][1])
                        rows.append(dict(values=scores(m,ops),stats=stats,geometry=geometry_record(m,heights)))
                    lead_rows.append(dict(lead_s=lead,rows=rows));cursor=end
                tracks[track]=lead_rows
            assert all(m.t==a['time_s'] and all(np.array_equal(getattr(m,k),a['arrays'][k]) for k in SNAPSHOT_KEYS) for m,a in zip(members,before))
            conn.send(dict(tracks=tracks,adaptive_members_unchanged=True,runtime=api.runtime_identity()))
    except BaseException:
        try:conn.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:conn.close()

class WorkerPool:
    def __init__(self,n=50,workers=4):
        self.groups=[list(map(int,x)) for x in np.array_split(np.arange(n),workers) if len(x)];self.children=[];ctx=mp.get_context('spawn')
        try:
            for ids in self.groups:
                parent,child=ctx.Pipe();process=ctx.Process(target=worker,args=(child,));process.start();child.close();self.children.append((parent,process))
            self.ready=[self.receive(conn,p) for conn,p in self.children]
        except BaseException:self.close();raise
    @staticmethod
    def receive(conn,process):
        while not conn.poll(.2):
            if not process.is_alive():raise RuntimeError(f'水力子进程退出: {process.pid}, {process.exitcode}')
        result=conn.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def initialize(self,members,state):
        for ids,(conn,p) in zip(self.groups,self.children):conn.send(dict(kind='init',ids=ids,state=state,n=[members[i]._n_ik for i in ids],C=[members[i]._Co for i in ids]))
        for conn,p in self.children:assert self.receive(conn,p)['initialized']
    def run(self,members,start_s,*,end_s=None,leads_s=None,reset_states=False):
        kind='advance' if end_s is not None else 'forecast'
        for ids,(conn,p) in zip(self.groups,self.children):
            command=dict(kind=kind,start_s=start_s,n=[members[i]._n_ik for i in ids],C=[members[i]._Co for i in ids],reset_states=[snapshot(members[i]) for i in ids] if reset_states else None)
            if kind=='advance':command['end_s']=end_s
            else:command['leads_s']=leads_s
            conn.send(command)
        returned=[]
        for ids,(conn,p) in zip(self.groups,self.children):
            response=self.receive(conn,p)
            if kind=='advance':
                for i,row in zip(ids,response['rows']):restore_exact(members[i],row['state'])
            returned.append((ids,response))
        return returned
    def close(self):
        for conn,p in self.children:
            if p.is_alive():
                try:conn.send(dict(kind='stop'))
                except (BrokenPipeError,EOFError,OSError):pass
        for conn,p in self.children:
            p.join(timeout=3)
            if p.is_alive():p.terminate();p.join(timeout=3)
            conn.close()
