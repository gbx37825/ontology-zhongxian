"""真实水动力回放、糙率与初始化对照；不覆盖旧基线。"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from 试验公共 import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model',default=str(DEFAULT_MODEL));p.add_argument('--days',type=float,default=30);p.add_argument('--positioned',action='store_true');p.add_argument('--roughness',type=float,default=1);p.add_argument('--pools',default='');p.add_argument('--initialize',choices=['none','local','all','profile'],default='none');p.add_argument('--init-hour',type=float,default=24);p.add_argument('--init-cap',type=float,default=.35);p.add_argument('--reinit',action='store_true');p.add_argument('--forecast-boundary',action='store_true');p.add_argument('--observations',default=str(R/'02_输入/实测评分长表.csv'));p.add_argument('--gate-max-normalization',default=str(R/'03_运行参数/2025开度归一化上限.json'));p.add_argument('--coefficients',choices=['legacy','json'],default='legacy');p.add_argument('--output-step',type=int,default=1800);p.add_argument('--init-lookback-hours',type=float,default=6);p.add_argument('--reinit-interval-hours',type=float,default=24);a=p.parse_args()
    assert a.output_step>0 and a.output_step%15==0
    output_steps=a.output_step//15
    out=R/'06_结果'/a.name
    if out.exists():raise FileExistsError(out)
    out.mkdir();started=time.time();(R/'03_运行参数'/(a.name+'.json')).write_text(json.dumps({**vars(a),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shared_sha256':hashlib.sha256((R/'04_脚本/试验公共.py').read_bytes()).hexdigest(),'execution_cwd':str(Path.cwd())},ensure_ascii=False,indent=2))
    e,source,c=build_engine(a.model,roughness_factor=a.roughness,roughness_pools=[int(v) for v in a.pools.split(',') if v] or None,gate_max_normalization=json.loads(Path(a.gate_max_normalization).read_text()) if a.gate_max_normalization else None,use_calibrated_coefficients=a.coefficients=='legacy');f=Forcing(e,source,c,positioned=a.positioned);m=e.model
    warm=warmup(e,source,f);(out/'预热核验.json').write_text(json.dumps(warm,ensure_ascii=False,indent=2,default=str));print(a.name,'WARMUP',warm['converged'],flush=True)
    ops,audit=flow_operators(e,source,c);W,cv=control_volumes(e,ops);W=csr_matrix(W);(out/'测流控制体.json').write_text(json.dumps(cv,ensure_ascii=False,indent=2));(out/'分水位置投影.json').write_text(json.dumps(f.routing,ensure_ascii=False,indent=2))
    holes=c['orifices'].copy();holes['station']=holes.name.str.extract(r'^(JZZ\d+)-')[0];groups={s:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int)) for s,g in holes.groupby('station')}
    measured=pd.read_csv(a.observations);obs=measured[measured.metric!='flow_m3s'].set_index(['station_code','metric','time_s'])
    sn=[];raw=[];ledger=[];updates=[];history=[];history_times=[];snapshots={k:[] for k in ['time_s','H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']};interval_storage=np.zeros(len(ops));interval_offtake=np.zeros(len(ops));interval_qmeter=np.zeros(len(ops));interval_gate=np.zeros(len(ops));last_output=0.;initial_n=m.NIk
    def output(t):
        for station,(up,down,gi) in groups.items():raw.append(dict(station_code=station,time_s=t,forebay_m=float(m.H_j[up]),afterbay_m=float(m.H_j[down]),flow_m3s=float(m.Q_o[gi].sum())))
        for op in ops:sn.append(dict(station_code=op['station'],time_s=t,meter_flow=observe(m,op)))
        for k in snapshots:snapshots[k].append(float(t) if k=='time_s' else getattr(m,k).copy())
        history_times.append(float(t))
        history.append({(s,k):float(m.H_j[up if k=='forebay_m' else down]) for s,(up,down,gi) in groups.items() for k in ['forebay_m','afterbay_m']})
    def initialize(t):
        targets=[s for s in groups if a.initialize in ['all','profile'] or int(s[3:])>=56];delta=np.zeros(m.M);count=np.zeros(m.M)
        for station in targets:
            up,down,gi=groups[station]
            for metric,node in [('forebay_m',up),('afterbay_m',down)]:
                vals=[]
                for k,tt in enumerate(history_times):
                    if not t-a.init_lookback_hours*3600<tt<=t:continue
                    key=(station,metric,tt)
                    if key in obs.index and bool(obs.loc[key,'valid']):vals.append(float(obs.loc[key,'value'])-history[k][(station,metric)])
                if vals:delta[node]+=float(np.clip(np.mean(vals),-a.init_cap,a.init_cap));count[node]+=1
        delta=np.divide(delta,count,out=np.zeros_like(delta),where=count>0);delta[m.bc]=0
        if a.initialize=='profile':
            from scipy.sparse.csgraph import dijkstra
            edges={}
            for aa,bb,ll in zip(m._J_uk,m._J_dk,m._dx_k):edges[(int(aa),int(bb))]=max(float(ll),1.)
            for aa,bb in zip(m._J_uo,m._J_do):edges[(int(aa),int(bb))]=1.
            rr,cc=zip(*edges);graph=csr_matrix((list(edges.values()),(rr,cc)),shape=(m.M,m.M));xj=dijkstra(graph,directed=False,indices=e.node_index['NODE-main-0'])
            known=(count>0)|m.bc;known[e.node_index['NODE-main-0']]=True
            order=np.argsort(xj[known]);delta=np.interp(xj,xj[known][order],delta[known][order]);delta[m.bc]=0
        # 沿每条连接渠道在两端修正量之间线性分配，保留完整水深/流量状态；不逐点锯齿改水位。
        dh=np.zeros(m.NIk)
        for k in range(m.NK):
            ids=np.where(m._kI==k)[0];x=m._x_Ik[ids];ratio=(x-x.min())/max(x.max()-x.min(),1e-9);dh[ids]=(1-ratio)*delta[m._J_uk[k]]+ratio*delta[m._J_dk[k]]
        v0=hydraulic_volume(m);state=dict(H_j=m.H_j+delta,h_Ik=np.maximum(m.h_Ik+dh,.01),Q_ik=m.Q_ik.copy(),Q_uk=m.Q_uk.copy(),Q_dk=m.Q_dk.copy(),Q_o=m.Q_o.copy(),t=m.t)
        m.load_state(state);v1=hydraulic_volume(m);updates.append(dict(time_s=t,max_head_increment=float(np.abs(delta).max()),delta_volume_m3=v1-v0,mode=a.initialize,cap=a.init_cap));print(a.name,'INIT',updates[-1],flush=True)
        m.save_state();np.savez_compressed(out/f'初始化后状态_{int(t)}s.npz',**m.states)
    output(0)
    if a.initialize!='none' and a.init_hour==0:initialize(0)
    steps=int(round(a.days*86400/15));failure=None;step_checks=[]
    try:
        for i in range(steps):
            t=i*15.;kw=f.at(t)
            # 无未来输入试验：72h后边界、分水和开度均保持起报值，不读取未来观测作输入。
            if a.forecast_boundary and t>=259200:
                kw=f.at(259200)
            kw.update(num_iter=20, strict=True)
            H=m.H_j.copy();h=m.h_Ik.copy();m.step(**kw)
            step_checks.append((float(t+15), m.iter_elapsed, m.last_head_residual, m.last_gate_flow_residual, float(m.Q_o.min()),float(m.Q_o.max())))
            if not np.isfinite(m.H_j).all() or not np.isfinite(m.Q_ik).all():raise FloatingPointError(f'nonfinite at {t+15}')
            storage=np.r_[m._A_sj*(m.H_j-H),m._E_Ik*(m.h_Ik-h)*15]
            interval_storage+=W@storage;interval_offtake-=W@np.r_[kw['Q_in'],kw['Q_0Ik']]*15
            interval_qmeter+=np.array([observe(m,op) for op in ops])*15
            interval_gate+=np.array([m.Q_o[r['gate_indices']].sum() for r in cv])*15
            now=t+15
            if (i+1)%output_steps==0:
                output(now);duration=now-last_output
                for j,record in enumerate(cv):
                    direction=record['direction'];balance=direction*(interval_qmeter[j]-interval_gate[j])-interval_offtake[j]-interval_storage[j]
                    ledger.append(dict(station_code=record['station'],time_s=now,duration_s=duration,direction=direction,sim_meter_mean=interval_qmeter[j]/duration,sim_gate_mean=interval_gate[j]/duration,offtake_mean=interval_offtake[j]/duration,storage_rate=interval_storage[j]/duration,continuity_residual=balance/duration))
                interval_storage[:]=0;interval_offtake[:]=0;interval_qmeter[:]=0;interval_gate[:]=0;last_output=now
                if a.initialize!='none' and now==a.init_hour*3600:initialize(now)
                elif a.initialize!='none' and a.reinit and now>a.init_hour*3600 and now%(a.reinit_interval_hours*3600)==0:
                    selected=[station for station in groups if a.initialize in ['all','profile'] or int(station[3:])>=56]
                    errors=[abs(history[-1][(station,metric)]-float(obs.loc[(station,metric,now),'value'])) for station in selected for metric in ['forebay_m','afterbay_m'] if (station,metric,now) in obs.index and bool(obs.loc[(station,metric,now),'valid'])]
                    if errors and max(errors)>.1:initialize(now)
                if now in [21600,86400,108000,172800,259200,345600,432000]:
                    m.save_state();np.savez_compressed(out/f'状态_{int(now/3600)}h.npz',**m.states)
                if (i+1)%5760==0:print(a.name,'DAY',int(now/86400),flush=True)
    except Exception as ex:failure=str(ex);raise
    finally:
        pd.DataFrame(step_checks,columns=['time_s','iterations','head_residual_m','gate_residual_m3s','min_hole_flow','max_hole_flow']).to_csv(out/'逐步收敛核验.csv',index=False);pd.DataFrame(raw).to_csv(out/'逐闸水位流量.csv',index=False);pd.DataFrame(sn).to_csv(out/'真实测流断面流量.csv',index=False);pd.DataFrame(ledger).to_csv(out/'测流至闸门水量换算.csv',index=False);pd.DataFrame(updates).to_csv(out/'初始化增量.csv',index=False)
        np.savez_compressed(out/'内部断面状态.npz',**{k:np.asarray(v) for k,v in snapshots.items()})
        summary=dict(name=a.name,status='FAILED' if failure else 'COMPLETED',failure=failure,completed_steps=i+1,days=a.days,positioned=a.positioned,roughness=a.roughness,initialize=a.initialize,wall_seconds=time.time()-started,model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),output_dt_s=a.output_step,observation_step_s=1800,warmup_converged=warm['converged']);(out/'运行摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
