"""常驻成员分组；仅并行水力推进，随机数、同化与文件写入留在主进程。"""
import copy,json,multiprocessing as mp,os,time,traceback
from pathlib import Path
import numpy as np
from 局地参数EnKF import STATE_KEYS,restore,volume

def advance(member,forcing,t0,t1):
    retries=0;head=gate=0.
    assert abs(member.t-t0)<1e-8
    for ts in np.arange(t0,t1,15.):
        pending=[(ts,15.)]
        while pending:
            tt,dt=pending.pop(0);kw=forcing.at(tt);kw.update(dt=dt,num_iter=20,strict=True)
            try:member.step(**kw)
            except RuntimeError as err:
                if 'did not converge' not in str(err) or dt<=1.875:raise
                retries+=1;pending[:0]=[(tt,dt/2),(tt+dt/2,dt/2)];continue
            head=max(head,member.last_head_residual);gate=max(gate,member.last_gate_flow_residual)
            if not all(np.isfinite(getattr(member,k)).all() for k in STATE_KEYS):raise ValueError('nonfinite ensemble')
    return retries,head,gate

def jit_evidence():
    from numba import config
    from pipedream_solver import nsuperlink
    from pipedream_solver.gate_hydraulics import rectangular_gate_terms
    evidence=dict(pid=os.getpid(),jit_disabled=bool(config.DISABLE_JIT),momentum_compiled_signatures=len(nsuperlink.numba_b_ik.signatures),gate_compiled_signatures=len(rectangular_gate_terms.signatures))
    assert not evidence['jit_disabled'] and evidence['momentum_compiled_signatures'] and evidence['gate_compiled_signatures']
    return evidence

def worker(conn,root,initial):
    try:
        from 试验公共 import Forcing
        from 并联结构物理修复 import build_engine_physics
        started=time.perf_counter();root=Path(root)
        engine,s,c=build_engine_physics(root/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',gate_max_normalization=json.loads((root/'03_运行参数/2025开度归一化上限.json').read_text()))
        forcing=Forcing(engine,s,c);members=[]
        for state,t,n,co in initial:
            m=copy.deepcopy(engine.model);restore(m,state);m.t=t;m.iter_count=0
            m._n_ik[:]=n;m._Co[:]=co;m._n_uk[:]=n[m._i_1k];m._n_dk[:]=n[m._i_nk];members.append(m)
        # 用副本预热，不推进真实成员，不消耗任何同化随机数。
        warm=copy.deepcopy(members[0]);advance(warm,forcing,warm.t,warm.t+15.)
        conn.send(dict(ready=True,warmup_seconds=time.perf_counter()-started,**jit_evidence()))
        while True:
            command=conn.recv()
            if command is None:break
            t0,t1,parameters=command;started=time.perf_counter();results=[]
            for m,(n,co) in zip(members,parameters):
                m._n_ik[:]=n;m._Co[:]=co;m._n_uk[:]=n[m._i_1k];m._n_dk[:]=n[m._i_nk]
                stats=advance(m,forcing,t0,t1)
                results.append(({k:getattr(m,k).copy() for k in STATE_KEYS},float(m.t),volume(m),stats,{k:getattr(m,k).copy() for k in ('A_ik','A_uk','A_dk','V_j')}))
            conn.send(dict(results=results,seconds=time.perf_counter()-started))
    except BaseException:
        try:conn.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError,EOFError):pass
    finally:conn.close()

class HydraulicWorkers:
    def __init__(self,root,members,count):
        assert 1<count<=len(members)
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
        self.groups=[list(map(int,ids)) for ids in np.array_split(np.arange(len(members)),count)];self.children=[];self.ready=[]
        ctx=mp.get_context('spawn');started=time.perf_counter()
        try:
            for ids in self.groups:
                parent,child=ctx.Pipe();initial=[({k:getattr(members[i],k).copy() for k in STATE_KEYS},members[i].t,members[i]._n_ik.copy(),members[i]._Co.copy()) for i in ids]
                process=ctx.Process(target=worker,args=(child,str(root),initial));process.start();child.close();self.children.append((parent,process))
            self.ready=[self.receive(conn,p) for conn,p in self.children]
        except BaseException:self.close();raise
        self.startup_seconds=time.perf_counter()-started
    @staticmethod
    def receive(conn,process):
        while not conn.poll(.2):
            if not process.is_alive():raise RuntimeError(f'水力工作进程 {process.pid} 退出: {process.exitcode}')
        result=conn.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def advance(self,members,t0,t1):
        started=time.perf_counter()
        for ids,(conn,p) in zip(self.groups,self.children):conn.send((t0,t1,[(members[i]._n_ik.copy(),members[i]._Co.copy()) for i in ids]))
        retries=0;head=gate=0.;times=[]
        for ids,(conn,p) in zip(self.groups,self.children):
            message=self.receive(conn,p);times.append(message['seconds']);assert len(message['results'])==len(ids)
            for i,(state,t,v,stats,storage) in zip(ids,message['results']):
                assert t==t1
                restore(members[i],state);members[i].t=t
                # 保留求解器本步的蓄水面积/体积缓存，不能用重新计算的几何覆盖其统计口径。
                for key,value in storage.items():getattr(members[i],key)[:]=value
                assert abs(volume(members[i])-v)<1e-8
                retries+=stats[0];head=max(head,stats[1]);gate=max(gate,stats[2])
        return (retries,head,gate),dict(wall_seconds=time.perf_counter()-started,worker_seconds=times)
    def close(self):
        for conn,p in self.children:
            if p.is_alive():
                try:conn.send(None)
                except (BrokenPipeError,EOFError,OSError):pass
        for conn,p in self.children:
            p.join(timeout=3)
            if p.is_alive():p.terminate();p.join(timeout=3)
            conn.close()
