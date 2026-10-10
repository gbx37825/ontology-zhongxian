"""独立接口诊断：原水量门槛未通过的状态保留，本试验不调参或用于替换。"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;R39=HERE.parent;R=R39.parent
R38=R/'38_测点假设下的联合辨识'
spec=importlib.util.spec_from_file_location('full38',R38/'06_全线联合参数转移/全线固定参数转移.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full);api=full.api
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['G00','G16']);case=p.parse_args().case
    out=HERE/(case+'_DFF');out.mkdir(parents=True,exist_ok=False);started=time.time()
    status=dict(status='RUNNING',case=case,completed_substeps=0)
    try:
        label={'G00':'G00_冻结原参数','G16':'G16_联合参数转移'}[case]
        trace_dir=R39/'03_接口与水量补录'/label
        tr_status=json.loads((trace_dir/'进度.json').read_text())
        assert tr_status['status']=='COMPLETED' and tr_status['exact_node_trajectory'] and tr_status['exact_six_final_states']
        cfg=json.loads((R38/'06_全线联合参数转移'/label/'运行配置.json').read_text())
        protocol=R39/'接口独立诊断分支约定.json';contract=json.loads(protocol.read_text())
        trace_path=trace_dir/'逐成功子步接口及水量.npz'
        sources={**cfg['source_hashes'],str(protocol):sha(protocol),str(trace_path):sha(trace_path),str(Path(__file__)):sha(Path(__file__))}
        assert all(sha(path)==h for path,h in sources.items())
        with np.load(trace_path) as z:trace={key:z[key].copy() for key in z.files}
        assert trace['interval_start_s'][0]==0 and trace['time_s'][-1]==2592000
        assert np.array_equal(trace['interval_start_s'][1:],trace['time_s'][:-1])
        # Restore the original callback even if full-line construction fails.
        saved_callback=api._physics._legacy_build
        try:
            api._physics._legacy_build=full.frozen_full_build
            ge,gs,gc=api._physics.build_engine_physics(gate_max_normalization=api.data()[2])
        finally:api._physics._legacy_build=saved_callback
        gm=ge.model
        with np.load(full.B/'共享输入/初始化后状态_0s.npz') as z:
            gm.load_state({k:z[k].copy() for k in api.STATE_KEYS},compute_hydraulic_geometries=True)
        gm.compute_storage_volumes();gm.t=0.;gm.iter_count=0
        assert {k:full.arrsha(getattr(gm,k)) for k in api.STATE_KEYS}==cfg['initial_state_sha256']
        assert api._physics._legacy_build is api._local_legacy_build
        local=api.build('A',n_factor=cfg['factors']['n'],c_factor=cfg['factors']['C'])
        lm=local['model'];lc=local['converted'];ls=local['source']
        assert lm.M==4 and lm.n_o==2 and lm.NK==2 and len(lm.Q_ik)==71
        gmesh=api.meshes(gs,gc);lmesh=api.meshes(ls,lc)
        centers={}
        for name,lgrid in lmesh.items():
            ggrid=gmesh[name];assert ggrid['sign']==lgrid['sign']==1
            gx=dict(zip(ggrid['ids'],ggrid['x']));lx=dict(zip(lgrid['ids'],lgrid['x']))
            glinks=gc['links'][gc['links'].k==ggrid['k']]
            llinks=lc['links'][lc['links'].k==lgrid['k']]
            gp=np.asarray([(gx[row.j_0]+gx[row.j_1])/2 for row in glinks.itertuples()]);order=np.argsort(gp)
            centers[name]=(gp[order],glinks.id.to_numpy(int)[order],llinks,lx)
        def full_head(name,x):
            grid=gmesh[name];ids=grid['ids'];return np.interp(x,grid['x'],gm.h_Ik[ids]+gm._z_inv_Ik[ids])
        def full_native_q(name,x):
            xx,ids,_,_=centers[name];return np.interp(x,xx,gm.Q_ik[ids])
        state={k:np.full_like(getattr(lm,k),np.nan) for k in api.STATE_KEYS}
        for name,idx in local['engine'].node_index.items():
            if name=='LOCAL_Q':value=full_head(api.UP_CHANNEL,300091.)
            elif name=='LOCAL_Z':value=full_head(api.POOL_CHANNEL,327100.)
            else:value=gm.H_j[ge.node_index[name]]
            state['H_j'][idx]=value
        for name,grid in lmesh.items():
            ids=grid['ids'];state['h_Ik'][ids]=full_head(name,grid['x'])-lm._z_inv_Ik[ids]
            _,_,links,lx=centers[name]
            for row in links.itertuples():state['Q_ik'][row.id]=full_native_q(name,(lx[row.j_0]+lx[row.j_1])/2)
            k=int(grid['k']);gk=int(gmesh[name]['k'])
            state['Q_uk'][k]=full_native_q(name,300091.) if name==api.UP_CHANNEL else gm.Q_uk[gk]
            state['Q_dk'][k]=full_native_q(name,327100.) if name==api.POOL_CHANNEL else gm.Q_dk[gk]
        for name,idx in local['engine'].gate_index.items():state['Q_o'][idx]=gm.Q_o[ge.gate_index[name]]
        assert all(np.isfinite(v).all() for v in state.values())
        lm.load_state(state,compute_hydraulic_geometries=True);lm.compute_storage_volumes();lm.t=0.;lm.iter_count=0;lm.save_state();api.check_range(local)
        np.savez_compressed(out/'映射起算状态.npz',**state)
        initial=api.sample(local)
        full_initial=json.loads((trace_dir/'补录配置.json').read_text())['initial_readings']
        initial_differences={key:initial[key]-full_initial[ref] for key,ref in [('JZZ16_forebay_m','H16_fore_m'),('JZZ16_afterbay_m','H16_after_m')]}
        assert max(abs(x) for x in initial_differences.values())<1e-9
        fixed={key:full.arrsha(getattr(lm,key)) for key in ['_n_ik','_n_uk','_n_dk','_Co']}
        full.dump(out/'运行配置.json',dict(case=case,source_hashes=sources,factors=cfg['factors'],local_identity=dict(nodes=lm.M,links=len(lm.Q_ik),gates=lm.n_o),callback_restored=True,
            initial_projection='全线t0模型状态投影，无内部实测、无再次预热',initial_target_differences_m=initial_differences,
            reported_discharge_conservation_gate_from_phase1_pass=tr_status['control_volume_gate_pass'],not_override_failed_original_water_gate=True,
            protocol=contract,runtime_identity=api.runtime_identity()))
        nsteps=len(trace['time_s']);differences=np.empty((nsteps,3));actual=np.empty((nsteps,3));boundary_max_H=boundary_max_Q=0.;rows=[]
        for i in range(nsteps):
            t0=float(trace['interval_start_s'][i]);t1=float(trace['time_s'][i]);dt=float(trace['dt_s'][i]);assert lm.t==t0
            kw=api.forcing(local,t0);kw['H_bc']=kw['H_bc'].copy();kw['Q_in']=kw['Q_in'].copy()
            Q=float(trace['Q16_cut_m3s'][i]);H=float(trace['H17_cut_m'][i])
            kw['Q_in'][local['inlet_node']]=Q;kw['H_bc'][local['tail_node']]=H
            kw.update(dt=dt,num_iter=20,strict=True)
            lm.step(**kw);api.check_range(local);assert lm.t==t1
            values=api.sample(local)
            boundary_max_H=max(boundary_max_H,abs(values['boundary_tailwater_m']-H))
            boundary_max_Q=max(boundary_max_Q,abs(values['cut_inlet_flow_m3s']-Q))
            actual[i]=[values['JZZ16_forebay_m'],values['JZZ16_afterbay_m'],values['gate_flow_m3s']]
            differences[i]=actual[i]-np.asarray([trace['H16_fore_m'][i],trace['H16_after_m'][i],trace['Q16_gate_m3s'][i]])
            status.update(completed_substeps=i+1,model_time_s=t1)
            if t1%1800==0:rows.append(dict(time_s=t1,**values,full_H16_fore_m=trace['H16_fore_m'][i],full_H16_after_m=trace['H16_after_m'][i],full_gate_Q_m3s=trace['Q16_gate_m3s'][i]))
            if t1%86400==0:
                status['wall_seconds']=time.time()-started;full.dump(out/'进度.json',status);print(case,int(t1/86400),flush=True)
        assert len(rows)==1440 and all(full.arrsha(getattr(lm,key))==h for key,h in fixed.items())
        weights=trace['dt_s'];assert weights.sum()==2592000
        stats=[]
        for j,(name,unit,mae_limit,max_limit) in enumerate([('H16_fore','m',.002,.01),('H16_after','m',.002,.01),('gate_Q','m3/s',.05,.5)]):
            d=differences[:,j];mae=float(np.average(abs(d),weights=weights));maximum=float(abs(d).max())
            stats.append(dict(quantity=name,unit=unit,MAE=mae,RMSE=float(np.sqrt(np.average(d*d,weights=weights))),bias=float(np.average(d,weights=weights)),max_abs=maximum,MAE_limit=mae_limit,max_limit=max_limit,passed=mae<=mae_limit and maximum<=max_limit))
        boundaries=dict(max_actual_head_difference_m=boundary_max_H,max_actual_cut_Q_difference_m3s=boundary_max_Q,passed=boundary_max_H<=1e-10 and boundary_max_Q<=1e-8)
        pd.DataFrame(rows).to_csv(out/'半小时局部全线接口对照.csv',index=False)
        np.savez_compressed(out/'逐步接口差异.npz',time_s=trace['time_s'],dt_s=weights,local=actual,differences=differences)
        np.savez_compressed(out/'末态.npz',**{key:getattr(lm,key) for key in api.STATE_KEYS})
        assert all(sha(path)==h for path,h in sources.items())
        status.update(status='COMPLETED',interface_pass=all(x['passed'] for x in stats) and boundaries['passed'],metrics=stats,boundary_checks=boundaries,parameters_unchanged=True,source_hashes_unchanged=True,runtime_audit=local['runtime_audit'],runtime_identity=api.runtime_identity(),wall_seconds=time.time()-started,not_override_original_water_gate=True,baseline_replaced=False)
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        raise
    finally:full.dump(out/'结果.json',status)
if __name__=='__main__':main()
