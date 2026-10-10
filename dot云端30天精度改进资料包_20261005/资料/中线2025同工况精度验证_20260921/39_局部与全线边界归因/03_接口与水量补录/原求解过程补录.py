"""原全线计算只加读数：补齐逐成功子步的测流切口及离散连续方程水量账。"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
R39=HERE.parent
R=R39.parent
R38=R/'38_测点假设下的联合辨识'
BASE=R38/'06_全线联合参数转移'
spec=importlib.util.spec_from_file_location('full38',BASE/'全线固定参数转移.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
api=full.api
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=full.arrsha
dump=full.dump

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['G00','G16']);args=parser.parse_args()
    label={'G00':'G00_冻结原参数','G16':'G16_联合参数转移'}[args.case]
    out=HERE/label;out.mkdir(parents=True,exist_ok=False)
    ref=BASE/label;old=json.loads((ref/'运行配置.json').read_text())
    assert json.loads((ref/'进度.json').read_text())['status']=='COMPLETED'
    started=time.time();status=dict(status='RUNNING',case=args.case,completed_cycles=0)
    try:
        T08=R/'30_分通道方法逐项测试/02_输入与测点核查'
        qfile=T08/'输入与算子核验.json';wfile=T08/'兰河测流到闸门控制体权重.npy'
        source_hashes={**old['source_hashes'],str(Path(__file__)):sha(Path(__file__)),str(R39/'预定诊断约定.json'):sha(R39/'预定诊断约定.json'),str(qfile):sha(qfile),str(wfile):sha(wfile)}
        assert all(sha(p)==h for p,h in source_hashes.items())
        b=api.DEPS/'参数并行运行脚本'
        advance=api._bind_definitions(b/'常驻水力进程.py',['advance'],dict(np=np,STATE_KEYS=api.STATE_KEYS),'_38_trace_advance').advance
        restore=api._bind_definitions(b/'局地参数EnKF.py',['restore'],{},'_38_trace_restore').restore
        api._physics._legacy_build=full.frozen_full_build
        engine,source,c=api._physics.build_engine_physics(gate_max_normalization=api.data()[2]);m=engine.model
        with np.load(full.B/'共享输入/初始化后状态_0s.npz') as z:restore(m,{k:z[k].copy() for k in api.STATE_KEYS})
        m.t=0.;m.iter_count=0
        assert {k:arrsha(getattr(m,k)) for k in api.STATE_KEYS}==old['initial_state_sha256']
        m._n_ik[old['n_indices']]*=old['factors']['n'];m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m._Co[old['C_indices']]*=old['factors']['C']
        assert {k:arrsha(getattr(m,k)) for k in full.PARAM_KEYS}==old['parameter_sha256']
        forcing=api.Forcing(engine,source,c)
        assert {k:arrsha(getattr(forcing,k)) for k in ['times','h','q','q0','u']}==old['forcing_array_sha256']
        audit=json.loads(qfile.read_text());qop=next(x['flow_operator'] for x in audit['stations'] if x['station']=='JZZ16')
        assert audit['frozen_B02_input_sha256']==sha(full.B/'共享输入/C07.json')
        flow_fn=api._bind_definitions(api.DEPS/'水力运行脚本/试验公共.py',['flow_operator'],dict(np=np,re=api.re,meshes=api.meshes),'_38_trace_flow_operator').flow_operator
        rebuilt=flow_fn(engine,source,c,'JZZ16',300091.)
        assert rebuilt==qop
        qids=np.asarray(qop['indices']);qw=np.asarray(qop['weights']);gids=np.asarray(old['C_indices'])
        W=np.load(wfile);assert W.shape==(m.M+m.NIk,)
        # The weighted continuity equation has exactly two flux cuts, without crossing a model boundary.
        for field,up,down in [('Q_ik',m._Ik+m.M,m._Ip1k+m.M),('Q_uk',m._J_uk,m._I_1k+m.M),('Q_dk',m._I_Np1k+m.M,m._J_dk),('Q_o',m._J_uo,m._J_do)]:
            expected=np.zeros(len(up))
            if field=='Q_ik':expected[qids]=qw
            if field=='Q_o':expected[gids]=-1.
            assert np.max(abs((W[down]-W[up])-expected))<1e-12
        assert np.max(abs(W[:m.M][m.bc]))<1e-12
        jids=np.flatnonzero(W[:m.M]);iids=np.flatnonzero(W[m.M:]);wj=W[jids];wi=W[m.M+iids]
        ops=json.loads((full.B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json').read_text())
        hops={key:next(op for op in ops if op['station']==station and op['metric']==metric) for key,station,metric in [('H16_fore_m','JZZ16','forebay_m'),('H16_after_m','JZZ16','afterbay_m'),('H17_cut_m','JZZ17','forebay_m')]}
        def heads():return {key:float(getattr(m,op['state'])[op['indices']]@np.asarray(op['weights'])+op.get('offset',0)) for key,op in hops.items()}
        def reads():return dict(Q16_cut_m3s=float(m.Q_ik[qids]@qw),Q16_gate_m3s=float(m.Q_o[gids].sum()),**heads())
        rows=[];initial=dict(time_s=0.,**reads());original_step=m.step
        def record_step(**kw):
            t0=float(m.t);dt=float(kw['dt']);before_H=m.H_j[jids].copy();before_h=m.h_Ik[iids].copy()
            result=original_step(**kw) # Failed steps roll back in the original solver and are not ledger entries.
            dv=float(wj@(m._A_sj[jids]*(m.H_j[jids]-before_H))+wi@(m._E_Ik[iids]*(m.h_Ik[iids]-before_h)*dt))
            take=-float(wj@kw['Q_in'][jids]+wi@kw['Q_0Ik'][iids])*dt
            now=reads();residual=now['Q16_cut_m3s']-now['Q16_gate_m3s']-take/dt-dv/dt
            rows.append(dict(interval_start_s=t0,time_s=float(m.t),dt_s=dt,**now,offtake_volume_m3=take,native_storage_change_m3=dv,continuity_residual_m3s=residual))
            return result
        m.step=record_step
        groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')}
        config=dict(case=args.case,source_hashes=source_hashes,reference=str(ref),flow_operator=qop,control_volume_weights=str(wfile),support=dict(superjunction_ids=jids.tolist(),internal_junction_ids=iids.tolist(),superjunction_weights=wj.tolist(),internal_weights=wi.tolist()),initial_readings=initial,initial_state_identical=True,parameter_arrays_identical=True,forcing_arrays_identical=True,read_only_observer=True,storage_semantics='原离散连续方程系数下的模型蓄变；非独立实测体积',time_semantics='Q/H为成功子步末态；输入外部forcing仍取该子步起点；记录真实减步')
        dump(out/'补录配置.json',config)
        node_rows=[];retries=0
        for cycle in range(1440):
            retry,_,_=advance(m,forcing,cycle*1800.,(cycle+1)*1800.);retries+=retry;t=(cycle+1)*1800
            for station,(ju,jd,ids) in groups.items():node_rows.append(dict(time_s=t,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[ids].sum())))
            if (cycle+1)%48==0:
                status.update(completed_cycles=cycle+1,model_time_s=float(m.t),days=(cycle+1)/48,successful_substeps=len(rows),retries=retries,wall_seconds=time.time()-started)
                dump(out/'进度.json',status);print(json.dumps(status),flush=True)
        m.step=original_step
        frame=pd.DataFrame(rows)
        assert frame.time_s.iloc[-1]==2592000 and frame.interval_start_s.iloc[0]==0
        assert np.array_equal(frame.interval_start_s.to_numpy()[1:],frame.time_s.to_numpy()[:-1])
        assert np.array_equal(frame.time_s-frame.interval_start_s,frame.dt_s)
        pd.DataFrame(node_rows).to_csv(out/'半小时水位流量.csv',index=False)
        assert pd.read_csv(out/'半小时水位流量.csv').equals(pd.read_csv(ref/'半小时水位流量.csv'))
        with np.load(ref/'末态.npz') as z:assert all(np.array_equal(getattr(m,k),z[k]) for k in api.STATE_KEYS)
        np.savez_compressed(out/'逐成功子步接口及水量.npz',**{key:frame[key].to_numpy() for key in frame})
        np.savez_compressed(out/'末态.npz',**{key:getattr(m,key) for key in api.STATE_KEYS})
        frame['half_hour_end_s']=(np.ceil(frame.time_s/1800)*1800).astype(int)
        ledger=[]
        for ts,g in frame.groupby('half_hour_end_s'):
            duration=float(g.dt_s.sum());assert duration==1800.
            qm=float((g.Q16_cut_m3s*g.dt_s).sum()/duration);qg=float((g.Q16_gate_m3s*g.dt_s).sum()/duration);take=float(g.offtake_volume_m3.sum()/duration);dv=float(g.native_storage_change_m3.sum()/duration)
            ledger.append(dict(time_s=int(ts),duration_s=duration,model_meter_mean_m3s=qm,model_gate_mean_m3s=qg,offtake_mean_m3s=take,native_storage_rate_m3s=dv,continuity_residual_m3s=qm-qg-take-dv))
        pd.DataFrame(ledger).to_csv(out/'半小时测流至闸水量账.csv',index=False)
        snap=frame[frame.time_s.mod(1800)==0].drop(columns='half_hour_end_s');assert len(snap)==1440
        snap.to_csv(out/'半小时同位置接口读数.csv',index=False)
        assert {k:arrsha(getattr(m,k)) for k in full.PARAM_KEYS}==old['parameter_sha256']
        assert all(sha(p)==h for p,h in source_hashes.items())
        residual_max=float(frame.continuity_residual_m3s.abs().max())
        status.update(status='COMPLETED',exact_node_trajectory=True,exact_six_final_states=True,parameters_unchanged=True,source_hashes_unchanged=True,substep_continuity_max_abs_m3s=residual_max,control_volume_gate_pass=residual_max<=1e-6,half_hour_rows=1440,wall_seconds=time.time()-started,runtime_identity=api.runtime_identity())
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        raise
    finally:dump(out/'进度.json',status)

if __name__=='__main__':main()
