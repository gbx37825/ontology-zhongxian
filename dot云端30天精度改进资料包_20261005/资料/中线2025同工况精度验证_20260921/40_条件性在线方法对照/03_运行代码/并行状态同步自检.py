"""真实冻结求解器的短步自检：串并行、分析状态同步及预测副本隔离。"""
from 水力运行支撑 import *

def equal(a,b):
    return a['time_s']==b['time_s'] and all(np.array_equal(a['arrays'][k],b['arrays'][k]) for k in SNAPSHOT_KEYS)

def main():
    out=R40/'05_先行核验';out.mkdir(exist_ok=True)
    manifest=json.loads((R40/'04_起点与公共初态/公共初态清单.json').read_text());assert manifest['status']=='COMPLETED'
    entry=next(x for x in manifest['states'] if x['issue_s']==16*86400)
    state=read_snapshot(entry['path']);e,s,c,f=make_full();m=e.model;restore_exact(m,state)
    fit=json.loads((R/'38_测点假设下的联合辨识/04_分通道联合辨识/A/辨识结果.json').read_text())
    cfg=json.loads((R/'38_测点假设下的联合辨识/06_全线联合参数转移/G16_联合参数转移/运行配置.json').read_text())
    m._n_ik[cfg['n_indices']]*=fit['ratios']['n'];m._Co[cfg['C_indices']]*=fit['ratios']['C'];m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk]
    serial=[copy.deepcopy(m) for _ in range(2)];mirrors=[copy.deepcopy(m) for _ in range(2)];pool=WorkerPool(n=2,workers=2)
    checks={}
    try:
        pool.initialize(mirrors,state);t=state['time_s']
        for mm in serial:advance(mm,f,t,t+1800)
        pool.run(mirrors,t,end_s=t+1800)
        checks['parameter_only_native_continuation_exact']=all(equal(snapshot(a),snapshot(b)) for a,b in zip(serial,mirrors));assert checks['parameter_only_native_continuation_exact']
        t+=1800
        # Explicit known analysis increment at an interior sensor; no boundary or parameter changes.
        ops,allops=score_operators(c);op=next(x for x in ops if x['station']=='JZZ16' and x['metric']=='forebay_m')
        for members in [serial,mirrors]:
            for j,mm in enumerate(members):
                values={k:getattr(mm,k).copy() for k in STATE_KEYS};values['h_Ik'][op['indices']]+=(j+1)*.001
                parameter_defs.restore(mm,values)
        for mm in serial:advance(mm,f,t,t+1800)
        pool.run(mirrors,t,end_s=t+1800,reset_states=True)
        checks['state_analysis_sent_to_workers_exact']=all(equal(snapshot(a),snapshot(b)) for a,b in zip(serial,mirrors));assert checks['state_analysis_sent_to_workers_exact']
        t+=1800;before=[snapshot(mm) for mm in mirrors]
        reply=pool.run(mirrors,t,leads_s=[1800])
        expected={}
        for track in ['known_historical_inputs','frozen_at_issue']:
            clones=[copy.deepcopy(mm) for mm in serial];forcing=f if track=='known_historical_inputs' else ConstantForcing(f,t)
            for mm in clones:advance(mm,forcing,t,t+1800)
            expected[track]=np.stack([scores(mm,allops) for mm in clones])
        for ids,response in reply:
            assert response['adaptive_members_unchanged']
            for track in expected:
                actual=np.stack([row['values'] for row in response['tracks'][track][0]['rows']]);assert np.array_equal(actual,expected[track][ids])
        checks['both_forecast_tracks_exact']=True
        checks['parent_members_unchanged_by_forecast']=all(equal(snapshot(m),b) for m,b in zip(mirrors,before));assert checks['parent_members_unchanged_by_forecast']
        for mm in serial:advance(mm,f,t,t+1800)
        pool.run(mirrors,t,end_s=t+1800)
        checks['worker_members_unchanged_by_forecast']=all(equal(snapshot(a),snapshot(b)) for a,b in zip(serial,mirrors));assert checks['worker_members_unchanged_by_forecast']
        dump(out/'并行状态同步自检结果.json',dict(status='PASSED',members=2,scope='仅真实求解器短步协议核验，不用于50成员方法评分',checks=checks,source_sha256={str(Path(__file__)):sha(Path(__file__)),str(HERE/'水力运行支撑.py'):sha(HERE/'水力运行支撑.py')},worker_runtime=pool.ready))
        print(json.dumps(checks),flush=True)
    finally:pool.close()
if __name__=='__main__':main()
