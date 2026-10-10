"""以固定候选沿已知过去输入推进；只保存各短期比较的共同起算状态。"""
from 水力运行支撑 import *

def main():
    out=R40/'04_起点与公共初态';selection_path=out/'预定起点.json'
    selection=json.loads(selection_path.read_text());cuts=selection['cut_times_s'];starts=sorted({float(t-21600) for t in cuts})
    assert min(starts)>=1296000 and max(cuts)+21600<=2592000
    state_dir=out/'公共模型状态';state_dir.mkdir(exist_ok=False)
    reference=R/'38_测点假设下的联合辨识/06_全线联合参数转移/G16_联合参数转移'
    cfg=json.loads((reference/'运行配置.json').read_text());source_hashes={**cfg['source_hashes'],str(selection_path):sha(selection_path),str(Path(__file__)):sha(Path(__file__)),str(HERE/'水力运行支撑.py'):sha(HERE/'水力运行支撑.py')}
    assert all(sha(p)==h for p,h in source_hashes.items())
    e,s,c,f=make_full();m=e.model
    with np.load(B/'共享输入/初始化后状态_0s.npz') as z:parameter_defs.restore(m,{k:z[k].copy() for k in STATE_KEYS})
    m.t=0.;m.iter_count=0
    assert {k:arrsha(getattr(m,k)) for k in STATE_KEYS}==cfg['initial_state_sha256']
    m._n_ik[cfg['n_indices']]*=cfg['factors']['n'];m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk];m._Co[cfg['C_indices']]*=cfg['factors']['C']
    assert {k:arrsha(getattr(m,k)) for k in full.PARAM_KEYS}==cfg['parameter_sha256']
    groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')}
    node_rows=[];records=[];started=time.time();retries=0
    for t0 in np.arange(0,max(starts),1800.):
        now=t0+1800;rt,_,_=advance(m,f,t0,now);retries+=rt
        for st,(ju,jd,ids) in groups.items():node_rows.append(dict(time_s=int(now),station=st,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[ids].sum())))
        if now in starts:
            path=state_dir/f'起点_{int(now)}s.npz';save_snapshot(path,snapshot(m))
            records.append(dict(adaptation_start_s=now,issue_s=now+21600,path=str(path.resolve()),sha256=sha(path),state_sha256={k:arrsha(getattr(m,k)) for k in STATE_KEYS}))
        if now%86400==0:
            status=dict(status='RUNNING',time_s=now,days=now/86400,states_saved=len(records),retries=retries,wall_seconds=time.time()-started)
            dump(out/'公共初态进度.json',status);print(json.dumps(status),flush=True)
    pd.DataFrame(node_rows).to_csv(out/'公共推进节点核对.csv',index=False)
    new=pd.read_csv(out/'公共推进节点核对.csv');old=pd.read_csv(reference/'半小时水位流量.csv');old=old[old.time_s<=max(starts)].reset_index(drop=True)
    assert new.equals(old);assert len(records)==len(starts)
    assert all(sha(p)==h for p,h in source_hashes.items())
    dump(out/'公共初态清单.json',dict(status='COMPLETED',states=records,exact_reference_node_trajectory=True,reference=str(reference),time_s=float(m.t),retries=retries,source_hashes=source_hashes,runtime_identity=api.runtime_identity(),note='使用前15天拟合的固定候选作离线历史初始化；在线比较只在第15天以后发行，不把训练期推进称因果预测'))
    dump(out/'公共初态进度.json',dict(status='COMPLETED',states_saved=len(records),time_s=float(m.t),wall_seconds=time.time()-started))
if __name__=='__main__':main()
