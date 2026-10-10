"""预定共同起点上的50成员方法对照；未来观测只在预测返回后用于计分。"""
from 水力运行支撑 import *
from 在线窗口与观测 import choose_window,observations_at,common_noise
import 条件性同化机制 as mechanism
import argparse,csv,os

METHODS=['F','P','W','S','H']
SEED=20260923
LEADS=[1800,3600,10800,21600]

def score_rows(matrix,ops,valid,issue,at,phase,track='',lead=0):
    matrix=np.asarray(matrix);assert matrix.shape==(50,len(ops)) and np.isfinite(matrix).all()
    mean=matrix.mean(axis=0);spread=matrix.std(axis=0,ddof=1);rows=[]
    for j,op in enumerate(ops):
        value=valid.loc[(op['station'],op['metric'],at)];assert bool(value['valid'])
        rows.append(dict(issue_s=issue,time_s=at,lead_s=lead,phase=phase,track=track,scope=op['scope'],group=op['group'],station=op['station'],metric=op['metric'],sim_mean=float(mean[j]),ensemble_std=float(spread[j]),observed=float(value['value']),pdf_comparable=bool(value['pdf_comparable'])))
    return rows

def parameter_rows(ens,t):
    rows=[]
    for j,p in enumerate(ens.parameters):
        ratios=np.exp(ens.theta[j]);center=ens.center_ratios[j]
        rows.append(dict(time_s=t,parameter=p['name'],center_ratio=float(center),mean_ratio=float(ratios.mean()),minimum_ratio=float(ratios.min()),maximum_ratio=float(ratios.max()),std_log=float(ens.theta[j].std(ddof=1)),at_lower=int((ens.theta[j]<=ens.lower[j]+1e-12).sum()),at_upper=int((ens.theta[j]>=ens.upper[j]-1e-12).sum()),effective_lower=float(np.exp(ens.lower[j,0])),effective_upper=float(np.exp(ens.upper[j,0]))))
    return rows

def run_case(pool,method,origin_id,entry,valid,protocol,sources):
    issue=float(entry['issue_s']);start=float(entry['adaptation_start_s'])
    out=R40/'06_配对运行'/f'起点_{int(issue)}s'/method
    if out.exists():
        old=json.loads((out/'结果.json').read_text()) if (out/'结果.json').exists() else {}
        if old.get('status')=='COMPLETED':
            previous=json.loads((out/'运行配置.json').read_text())
            assert previous['source_hashes']==sources,'源码改变，禁止混用旧试验成绩'
            return dict(method=method,issue_s=issue,status='COMPLETED',reused_completed=True)
        raise RuntimeError(f'存在未完成记录，须先核对真实进程/终态，不能自动重启: {out}')
    out.mkdir(parents=True);started=time.time();status=dict(status='RUNNING',method=method,issue_s=issue,adaptation_start_s=start,completed_adaptation_cycles=0,process_id=os.getpid())
    dump(out/'进度.json',status)
    try:
        common=read_snapshot(entry['path']);assert common['time_s']==start and sha(entry['path'])==entry['sha256']
        e,s,c,f=make_full();restore_exact(e.model,common)
        # The base arrays remain the original parameter reference; the candidate is a log-ratio centre.
        ens=parameter_defs.ParameterEnKF(e,c,n=50,seed=SEED,radius=20000.)
        frozen_parameters=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json').read_text())['parameter_bounds']
        assert ens.parameters==frozen_parameters
        view=state_defs.LocalEnKF(e,n=50,seed=SEED,radius=20000.)
        fit=json.loads((R/'38_测点假设下的联合辨识/04_分通道联合辨识/A/辨识结果.json').read_text())
        prior=mechanism.initialize(ens,view,{'POOL16_n':fit['ratios']['n'],'JZZ16_C':fit['ratios']['C']},SEED)
        assert all(np.array_equal(getattr(m,k),common['arrays'][k]) for m in ens.members for k in SNAPSHOT_KEYS)
        assert all(np.array_equal(ens.xparts[k],view.xparts[k]) for k in ['H_j','h_Ik'])
        mapped,ops=score_operators(c);update_ops=[o for o in mapped if o['assimilated']]
        y_max=c['orifices'].y_max.to_numpy(float)
        initial_theta=ens.theta.copy();theta_trace=[initial_theta.copy()];parameter_log=parameter_rows(ens,start)
        config=dict(method=method,origin_id=origin_id,issue_s=issue,adaptation_start_s=start,common_state_path=entry['path'],common_state_sha256=entry['sha256'],members=50,seed=SEED,prior=prior,prior_parameter_statistics=parameter_log,initial_theta_sha256=arrsha(initial_theta),initial_states_sha256=entry['state_sha256'],score_operators=ops,update_operators=update_ops,discarded_B01_constructor_state_perturbations=True,state_process_noise=False,source_hashes=sources,protocol=protocol,runtime_identity=api.runtime_identity())
        dump(out/'运行配置.json',config);np.savez_compressed(out/'初始参数集合.npz',theta=initial_theta)
        pool.initialize(ens.members,common)
        adaptation=[];ledger=[];traces=[];window_records=[];water_geometry=[];used_samples=set();theta_window=None;Y_buffer=[];y_buffer=[];time_buffer=[];window=None;dirty=False;retry_total=0
        for cycle in range(12):
            t0=start+cycle*1800;now=t0+1800
            rejuvenation=None
            if method in ['P','H'] or (method=='W' and window is None):
                rejuvenation=mechanism.rejuvenate(ens,common_noise(SEED,origin_id,cycle,0,ens.theta.shape))
            if method=='W' and window is None:
                window=choose_window(f,t0,y_max,12-cycle);window_records.append(window);theta_window=ens.theta.copy();Y_buffer=[];y_buffer=[];time_buffer=[]
            if method=='W':assert np.array_equal(ens.theta,theta_window),'窗口内部参数发生变化'
            replies=pool.run(ens.members,t0,end_s=now,reset_states=dirty);dirty=False
            for ids,response in replies:
                assert response['runtime']['momentum_compiled_signatures']>0 and response['runtime']['gate_compiled_signatures']>0
                retry_total+=sum(row['stats'][0] for row in response['rows'])
                water_geometry += [dict(time_s=now,phase='forecast_before_update',member=i,**row['geometry']) for i,row in zip(ids,response['rows'])]
            prior_scores=np.stack([scores(m,ops) for m in ens.members])
            adaptation += score_rows(prior_scores,ops,valid,issue,now,'forecast_before_update')
            audit=None;source_times=[]
            if method!='F':
                obs=observations_at(valid,mapped,now,now)
                if method=='W':
                    Y_buffer.append(np.stack([ens.values(o) for o in obs]));y_buffer.append(np.array([o['value'] for o in obs]));time_buffer.append(now)
                    if now==window['window_end_s']:
                        source_times=list(time_buffer);mean_y=np.mean(y_buffer,axis=0)
                        averaged=[{**o,'value':float(mean_y[j]),'time_s':now,'window_start_s':window['window_start_s'],'window_sample_times_s':source_times} for j,o in enumerate(obs)]
                        audit=mechanism.analysis(ens,view,averaged,common_noise(SEED,origin_id,cycle,1,(34,50)),True,False,Y_override=np.mean(Y_buffer,axis=0),elapsed_cycles=len(time_buffer))
                        window=None;theta_window=None
                else:
                    source_times=[now]
                    audit=mechanism.analysis(ens,view,obs,common_noise(SEED,origin_id,cycle,1,(34,50)),method in ['P','H'],method in ['S','H'])
                if audit is not None:
                    sample_keys={(o['station'],o['metric'],t) for o in update_ops for t in source_times}
                    assert not (sample_keys & used_samples),'重复消费窗口观测'
                    assert max(source_times)<=now;used_samples.update(sample_keys)
                    dirty=audit['state_dirty']
                    traces += [{**row,'batch_end_s':now,'source_times_s':source_times,'method':method} for row in audit['trace']]
            post_scores=np.stack([scores(m,ops) for m in ens.members])
            if method in ['F','P','W']:assert np.array_equal(prior_scores,post_scores),'仅参数路线即时水位/流量被更改'
            if method in ['F','S']:assert np.array_equal(ens.theta,initial_theta),'固定参数或仅状态路线修改了参数'
            adaptation += score_rows(post_scores,ops,valid,issue,now,'analysis_after_update')
            heights=raw_heights(e.model,s,c)
            water_geometry += [dict(time_s=now,phase='analysis_after_update',member=i,**geometry_record(m,heights)) for i,m in enumerate(ens.members)]
            ledger.append(dict(time_s=now,source_observation_times_s=source_times,rejuvenation=rejuvenation,analysis=audit))
            parameter_log+=parameter_rows(ens,now);theta_trace.append(ens.theta.copy())
            status.update(completed_adaptation_cycles=cycle+1,model_time_s=now,phase='adaptation',retries=retry_total,wall_seconds=time.time()-started)
            dump(out/'进度.json',status);dump(out/'更新水量与参数审计.json',ledger);dump(out/'窗口记录.json',window_records)
            print(json.dumps(status),flush=True)
        assert all(m.t==issue for m in ens.members) and (method!='W' or window is None)
        assert len(used_samples)==(0 if method=='F' else 34*12)
        np.savez_compressed(out/'参数集合轨迹.npz',theta=np.stack(theta_trace),time_s=np.arange(start,issue+1,1800))
        pd.DataFrame(parameter_log).to_csv(out/'参数轨迹.csv',index=False)
        pd.DataFrame(adaptation).to_csv(out/'适应期先验与分析.csv',index=False)
        dump(out/'逐观测更新记录.json',traces)
        checkpoint={f'm{i}_{k}':getattr(m,k) for i,m in enumerate(ens.members) for k in SNAPSHOT_KEYS}
        checkpoint.update(theta=ens.theta.copy(),time_s=np.array(issue),iter_counts=np.array([m.iter_count for m in ens.members]))
        np.savez_compressed(out/'发行时集合检查点.npz',**checkpoint)
        status.update(phase='frozen_update_forecasts',wall_seconds=time.time()-started);dump(out/'进度.json',status)
        replies=pool.run(ens.members,issue,leads_s=LEADS,reset_states=dirty)
        forecasts=np.empty((2,4,50,len(ops)));tracks=['known_historical_inputs','frozen_at_issue'];forecasts_stats=[]
        for ids,response in replies:
            assert response['adaptive_members_unchanged']
            for it,track in enumerate(tracks):
                for j,lead_record in enumerate(response['tracks'][track]):
                    assert lead_record['lead_s']==LEADS[j]
                    forecasts[it,j,ids,:]=np.stack([z['values'] for z in lead_record['rows']])
                    forecasts_stats += [dict(track=track,lead_s=LEADS[j],member=i,retries=row['stats'][0],max_head_residual_m=row['stats'][1],max_gate_residual_m3s=row['stats'][2],**row['geometry']) for i,row in zip(ids,lead_record['rows'])]
        assert np.isfinite(forecasts).all() and all(m.t==issue for m in ens.members)
        assert np.array_equal(ens.theta,checkpoint['theta'])
        future_rows=[]
        for it,track in enumerate(tracks):
            for j,lead in enumerate(LEADS):future_rows+=score_rows(forecasts[it,j],ops,valid,issue,issue+lead,'no_updates_after_issue',track,lead)
        pd.DataFrame(future_rows).to_csv(out/'冻结更新预测评分.csv',index=False)
        np.savez_compressed(out/'各成员未来预测.npz',predictions=forecasts,lead_s=np.asarray(LEADS))
        pd.DataFrame(forecasts_stats).to_csv(out/'预测数值与几何审计.csv',index=False)
        pd.DataFrame(water_geometry).to_csv(out/'适应期输入轮廓检查.csv',index=False)
        assert all(sha(p)==h for p,h in sources.items())
        status.update(status='COMPLETED',phase='complete',forecast_tracks=tracks,forecast_leads_s=LEADS,forecast_rows=len(future_rows),consumed_update_samples=len(used_samples),future_observations_used_by_algorithm=0,future_parameter_updates=0,future_state_updates=0,source_hashes_unchanged=True,wall_seconds=time.time()-started,baseline_replaced=False,engineering_acceptance=False)
        dump(out/'结果.json',status);dump(out/'进度.json',status);print(json.dumps(status),flush=True)
        return status
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        dump(out/'结果.json',status);dump(out/'进度.json',status)
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    protocol_path=R40/'预定在线对照约定.json';protocol=json.loads(protocol_path.read_text())
    assert args.workers==protocol['runtime']['workers']
    assert json.loads((R40/'05_先行核验/并行状态同步自检结果.json').read_text())['status']=='PASSED'
    common_path=R40/'04_起点与公共初态/公共初态清单.json';common=json.loads(common_path.read_text());assert common['status']=='COMPLETED' and common['exact_reference_node_trajectory']
    selection_path=R40/'04_起点与公共初态/预定起点.json';selection=json.loads(selection_path.read_text())
    entries=sorted(common['states'],key=lambda x:x['issue_s']);assert [x['issue_s'] for x in entries]==selection['cut_times_s']
    sources={str(p):sha(p) for p in [protocol_path,common_path,selection_path,Path(__file__),HERE/'水力运行支撑.py',HERE/'条件性同化机制.py',HERE/'在线窗口与观测.py',B/'共享输入/作者已核实修正观测层.csv']}
    sources.update(common['source_hashes']);sources.update(protocol['source_sha256'])
    assert all(sha(p)==h for p,h in sources.items())
    obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');valid=obs.set_index(['station_code','metric','time_s']);assert not valid.index.duplicated().any()
    pool=WorkerPool(n=50,workers=args.workers);all_status=[]
    try:
        for origin_id,entry in enumerate(entries):
            if args.pilot and entry['issue_s']!=16*86400:continue
            for method in METHODS:
                try:result=run_case(pool,method,origin_id,entry,valid,protocol,sources)
                except (KeyboardInterrupt,SystemExit):raise
                except Exception as exc:
                    result=dict(status='FAILED',method=method,issue_s=entry['issue_s'],error=repr(exc));pool.close();pool=WorkerPool(n=50,workers=args.workers)
                all_status.append(result)
                dump(R40/('先行试验批次进度.json' if args.pilot else '全部配对批次进度.json'),dict(status='RUNNING',pilot=args.pilot,cases=all_status))
        dump(R40/('先行试验批次进度.json' if args.pilot else '全部配对批次进度.json'),dict(status='COMPLETED_WITH_FAILURES' if any(x['status']!='COMPLETED' for x in all_status) else 'COMPLETED',pilot=args.pilot,cases=all_status))
    finally:pool.close()
if __name__=='__main__':main()
