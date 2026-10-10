"""仅转移固定n16/C16；全线原输入、原初态，冻结模块，不加载旧water实现。"""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, os, time, traceback
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
R38=HERE.parent
R=R38.parent
B=R/'26_冻结基线与后续归因'
X00=R/'35_局部率定前提复核/05_系数全线转移/X00_原系数含测点30天'
API=R38/'01_局部模型/测点假设局部模型.py'
spec=importlib.util.spec_from_file_location('local38',API)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda v:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
PARAM_KEYS=('_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk')

def dump(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)

def frozen_full_build(*args,gate_max_normalization=None,**kwargs):
    # 原共享build_engine的默认全线构模步骤；仅把输入/系数文件绑定26冻结路径。
    source=api.data()[0]
    gate_max={k:float(v.get('b',1) or 1) for k,v in source['baseData']['gates'].items()}
    if gate_max_normalization:gate_max.update(gate_max_normalization)
    c=api.parse_wxq_json(source,gate_max_openings=gate_max,section_geometry_mode='safe',minimum_initial_depth_m=7.)
    c['superjunctions'].loc[:,'h_0']=7.
    c['superjunctions'].loc[:,'max_depth']=np.maximum(c['superjunctions'].max_depth.to_numpy(),7.)
    c['orifices'].loc[:,'C']=c['orifices'].name.map(api.data()[3]).fillna(c['orifices'].C)
    engine=api.PipedreamHydraulicEngine(c,dt=15.,backend='numba');engine.model._method='lsq'
    return engine,source,c

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['G00','G16']);args=parser.parse_args()
    out=HERE/({'G00':'G00_冻结原参数','G16':'G16_联合参数转移'}[args.case]);out.mkdir(exist_ok=False)
    started=time.time();completed=0
    status=dict(status='RUNNING',case=args.case,pid=os.getpid(),completed_cycles=0)
    try:
        fit_path=R38/'04_分通道联合辨识/A/辨识结果.json'
        fit=json.loads(fit_path.read_text());factors=fit['ratios'] if args.case=='G16' else {'n':1.,'C':1.}
        assert json.loads((R38/'05_固定参数复核/局部保留检验与完整性核验.json').read_text())['eligible_for_diagnostic_full_line_transfer']
        frozen=api.DEPS/'参数并行运行脚本'
        ops_path=B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json'
        params_path=B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json'
        sources={str(p):sha(p) for p in [Path(__file__),API,fit_path,HERE/'运行前冻结约定.json',ops_path,params_path,B/'共享输入/C07.json',B/'共享输入/初始化后状态_0s.npz',B/'共享输入/2025开度归一化上限.json',B/'共享输入/Pipedream闸流系数.json',B/'共享输入/作者已核实修正观测层.csv',frozen/'常驻水力进程.py',frozen/'局地参数EnKF.py']}
        sources.update({x['file']:x['sha256'] for x in fit['runtime_identity']['modules']})
        assert all(sha(p)==v for p,v in sources.items())
        advance=api._bind_definitions(frozen/'常驻水力进程.py',['advance'],dict(np=np,STATE_KEYS=api.STATE_KEYS),'_38_frozen_full_advance').advance
        restore=api._bind_definitions(frozen/'局地参数EnKF.py',['restore'],{},'_38_frozen_full_restore').restore
        api._physics._legacy_build=frozen_full_build
        engine,source,c=api._physics.build_engine_physics(gate_max_normalization=api.data()[2]);m=engine.model
        with np.load(B/'共享输入/初始化后状态_0s.npz') as z:restore(m,{k:z[k].copy() for k in api.STATE_KEYS})
        m.t=0.;m.iter_count=0
        initial={k:arrsha(getattr(m,k)) for k in api.STATE_KEYS}
        original={k:getattr(m,k).copy() for k in PARAM_KEYS}
        ref=json.loads((X00/'运行配置.json').read_text())
        assert initial==ref['initial_state_sha256']
        assert all(arrsha(original[k])==v for k,v in ref['parameter_sha256'].items())
        params=json.loads(params_path.read_text())['parameter_bounds']
        npar=next(p for p in params if p['name']=='POOL16_n');cpar=next(p for p in params if p['name']=='JZZ16_C')
        nids=np.asarray(npar['indices'],int);cids=np.asarray(cpar['indices'],int)
        pool=c['superlinks'][c['superlinks'].name==api.POOL_CHANNEL]
        assert len(pool)==1 and np.array_equal(nids,np.flatnonzero(m._ki==int(pool.iloc[0].id)))
        assert np.array_equal(cids,np.flatnonzero(c['orifices'].name.str.startswith('JZZ16-')))
        for p,factor in [(npar,factors['n']),(cpar,factors['C'])]:assert p['lower']<=factor<=p['upper']
        m._n_ik[nids]*=factors['n'];m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m._Co[cids]*=factors['C']
        assert initial=={k:arrsha(getattr(m,k)) for k in api.STATE_KEYS}
        assert np.array_equal(m._n_ik[np.setdiff1d(np.arange(len(m._n_ik)),nids)],original['_n_ik'][np.setdiff1d(np.arange(len(m._n_ik)),nids)])
        assert np.array_equal(m._Co[np.setdiff1d(np.arange(len(m._Co)),cids)],original['_Co'][np.setdiff1d(np.arange(len(m._Co)),cids)])
        assert all(np.array_equal(getattr(m,k),original[k]) for k in ['_C_uk','_C_dk'])
        fixed={k:arrsha(getattr(m,k)) for k in PARAM_KEYS};forcing=api.Forcing(engine,source,c)
        ops=json.loads(ops_path.read_text());assert len(ops)==43 and sum(op['assimilated'] for op in ops)==34
        groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')}
        heights=np.full(m.NIk,np.inf);pool_points=[]
        for name,meta in c['channel_discretization'].items():
            if name not in source['baseData']['channels']:continue
            for p in meta['sections']:
                heights[p['junction_id']]=p['height']
                if name==api.POOL_CHANNEL:pool_points.append(p['junction_id'])
        mesh=api.meshes(source,c)[api.POOL_CHANNEL];tail_ids,tail_weights=api.weights_at(mesh,327100.)
        cfg=dict(case=args.case,factors=factors,source_hashes=sources,initial_state_sha256=initial,parameter_sha256=fixed,
            applied_n=m._n_ik[nids].tolist(),applied_C=m._Co[cids].tolist(),n_indices=nids.tolist(),C_indices=cids.tolist(),
            n_channel=api.POOL_CHANNEL,n_scope_extra_307m=True,siphon_n_K_unchanged=True,internal_observation_state_updates=0,
            all_geometry_unchanged=True,boundaries_unchanged=True,runtime_identity=api.runtime_identity(),
            forcing_array_sha256={k:arrsha(getattr(forcing,k)) for k in ['times','h','q','q0','u']})
        dump(out/'运行配置.json',cfg)
        raw_rows=[];mapped_rows=[];range_rows=[];tail_rows=[];retries=0;head=gate=0.
        for cycle in range(1440):
            stats=advance(m,forcing,cycle*1800.,(cycle+1)*1800.);retries+=stats[0];head=max(head,stats[1]);gate=max(gate,stats[2]);t=(cycle+1)*1800.
            for station,(ju,jd,ids) in groups.items():raw_rows.append(dict(time_s=t,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[ids].sum())))
            for op in ops:mapped_rows.append(dict(time_s=t,station=op['station'],metric=op['metric'],mapped_sim=float(getattr(m,op['state'])[op['indices']]@np.asarray(op['weights'])+op.get('offset',0)),group='旧参与同化测点' if op['assimilated'] else '旧留出检验测点',chainage_m=op['chainage']))
            excess=m.h_Ik-heights
            range_rows.append(dict(time_s=t,all_open_exceeded_points=int((excess>1e-7).sum()),all_open_max_depth_excess_m=float(excess.max()),pool16_open_exceeded_points=int((excess[pool_points]>1e-7).sum()),pool16_open_max_depth_excess_m=float(excess[pool_points].max())))
            tail_rows.append(dict(time_s=t,A_tail_chainage_327100_head_m=float((m.h_Ik[tail_ids]+m._z_inv_Ik[tail_ids])@tail_weights),JZZ16_gate_flow_m3s=float(m.Q_o[cids].sum())))
            completed=cycle+1
            if completed%48==0:
                assert all(arrsha(getattr(m,k))==v for k,v in fixed.items())
                status.update(completed_cycles=completed,model_time_s=float(m.t),days=completed/48,retries=retries,max_head_residual_m=float(head),max_gate_flow_residual_m3s=float(gate),wall_seconds=time.time()-started)
                dump(out/'进度.json',status);print(json.dumps(status),flush=True)
                if args.case=='G00' and completed==48:
                    prior=pd.read_csv(X00/'半小时水位流量.csv');prior=prior[prior.time_s<=86400].reset_index(drop=True)
                    assert pd.read_csv(io.StringIO(pd.DataFrame(raw_rows).to_csv(index=False))).equals(prior),'冻结运行时第一天未逐值复现X00'
                    dump(out/'第一天同源复现.json',dict(exact_equal=True,rows=len(prior)))
        pd.DataFrame(raw_rows).to_csv(out/'半小时水位流量.csv',index=False)
        pd.DataFrame(mapped_rows).to_csv(out/'实际位置水位.csv',index=False)
        pd.DataFrame(range_rows).to_csv(out/'半小时开敞轮廓范围.csv',index=False)
        pd.DataFrame(tail_rows).to_csv(out/'局部切口同位置全线响应.csv',index=False)
        np.savez_compressed(out/'末态.npz',**{k:getattr(m,k) for k in api.STATE_KEYS})
        if args.case=='G00':
            assert pd.read_csv(out/'半小时水位流量.csv').equals(pd.read_csv(X00/'半小时水位流量.csv'))
            with np.load(X00/'末态.npz') as z:assert all(np.array_equal(getattr(m,k),z[k]) for k in api.STATE_KEYS)
            # CSV十进制读取可差一个末位，节点全轨迹及六个二进制末态必须完全一致。
            dump(out/'全30天同源复现.json',dict(exact_node_trajectory_equal=True,exact_six_final_states_equal=True,reference=str(X00)))
        assert all(sha(p)==v for p,v in sources.items())
        status.update(status='COMPLETED',parameters_fixed=True,source_hashes_unchanged=True,runtime_identity=api.runtime_identity(),wall_seconds=time.time()-started)
    except BaseException as exc:
        status.update(status='FAILED',completed_cycles=completed,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        raise
    finally:dump(out/'进度.json',status)

if __name__=='__main__':main()
