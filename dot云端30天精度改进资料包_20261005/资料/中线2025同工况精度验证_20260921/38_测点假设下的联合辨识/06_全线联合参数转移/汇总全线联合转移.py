"""仅汇总两组已完成的全线转移；不加载求解器、不启动或修改仿真。"""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent
P=O.parent
R=P.parent
B=R/'26_冻结基线与后续归因'
RUNS=['G00_冻结原参数','G16_联合参数转移']
REFERENCE_X=R/'35_局部率定前提复核/05_系数全线转移/X00_原系数含测点30天'
REFERENCE_M=R/'34_剩余项目逐项对照/02_输入敏感性/M00_原输入完整30天'
STATE_KEYS=['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']
PERIODS=[('前15天',0,1296000),('后15天',1296000,2592000),('完整30天',0,2592000)]
KEYS=['station','metric','time_s']
@lru_cache(None)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x): (O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def finite_number(value): return None if not np.isfinite(value) else float(value)
def stats(e):
    e=np.asarray(e,dtype=float)
    assert np.isfinite(e).all() and len(e)>0
    return dict(samples=len(e),MAE_m=float(abs(e).mean()),RMSE_m=float(np.sqrt(np.mean(e**2))),bias_m=float(e.mean()),max_abs_m=float(abs(e).max()))
def assert_timeline(frame,points):
    assert len(frame)==points*1440
    assert np.array_equal(np.sort(frame.time_s.unique()),np.arange(1800,2592001,1800))
    assert frame.groupby('time_s').size().eq(points).all()

def main():
    # Do not inspect partial CSV/npz: both final progress records must pass first.
    progress={name:json.loads((O/name/'进度.json').read_text()) for name in RUNS}
    if not all(x['status']=='COMPLETED' and x['completed_cycles']==1440 for x in progress.values()):
        raise SystemExit('两组尚未完成，未读取任何结果CSV或npz；等待主任务监测。')
    protocol_path=O/'运行前冻结约定.json'
    protocol=json.loads(protocol_path.read_text())
    fit_path=P/'04_分通道联合辨识/A/辨识结果.json'
    fit=json.loads(fit_path.read_text())
    observations_path=B/'共享输入/作者已核实修正观测层.csv'
    obs=pd.read_csv(observations_path).rename(columns={'station_code':'station','value':'observed'})
    obs=obs[(obs.time_s>0)&obs.metric.isin(['forebay_m','afterbay_m'])]
    assert not obs.duplicated(KEYS).any()
    obs=obs[KEYS+['observed','valid','pdf_comparable']]
    op_path=B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json'
    ops=json.loads(op_path.read_text())
    assert len(ops)==43 and sum(o['assimilated'] for o in ops)==34
    op_keys={(o['station'],o['metric']) for o in ops}
    cfgs={name:json.loads((O/name/'运行配置.json').read_text()) for name in RUNS}
    c0,c1=[cfgs[name] for name in RUNS]
    assert c0['initial_state_sha256']==c1['initial_state_sha256']
    assert c0['forcing_array_sha256']==c1['forcing_array_sha256']
    xcfg=json.loads((REFERENCE_X/'运行配置.json').read_text())
    assert c0['initial_state_sha256']==xcfg['initial_state_sha256']
    assert c0['parameter_sha256']==xcfg['parameter_sha256']
    assert c1['factors']==fit['ratios'] and c0['factors']=={'n':1.,'C':1.}
    bounds=json.loads((B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/运行配置.json').read_text())['parameter_bounds']
    npar=next(p for p in bounds if p['name']=='POOL16_n')
    Cpar=next(p for p in bounds if p['name']=='JZZ16_C')
    assert c1['n_indices']==c0['n_indices']==npar['indices']
    assert c1['C_indices']==c0['C_indices']==Cpar['indices']
    assert np.array_equal(np.asarray(c1['applied_n']),np.asarray(c0['applied_n'])*fit['ratios']['n'])
    assert np.array_equal(np.asarray(c1['applied_C']),np.asarray(c0['applied_C'])*fit['ratios']['C'])
    assert all(c0['parameter_sha256'][k]==c1['parameter_sha256'][k] for k in ['_C_uk','_C_dk'])
    source_hashes={str(p):sha(p) for p in [protocol_path,fit_path,op_path,observations_path,Path(__file__)]}
    datasets={}; raw_nodes={}; states={}; run_integrity=[]; reconstruction=[]
    for name in RUNS:
        d=O/name; cfg=cfgs[name]; p=progress[name]
        assert all(sha(k)==v for k,v in cfg['source_hashes'].items())
        assert p['parameters_fixed'] and p['source_hashes_unchanged']
        assert cfg['internal_observation_state_updates']==0 and cfg['siphon_n_K_unchanged']
        assert cfg['boundaries_unchanged'] and cfg['all_geometry_unchanged']
        ident=p['runtime_identity']
        assert not ident['old_water_hydraulic_modules'] and not ident['corelib_modules']
        assert not ident['numba_jit_disabled'] and ident['momentum_compiled_signatures']>0 and ident['gate_compiled_signatures']>0
        raw=pd.read_csv(d/'半小时水位流量.csv')
        assert_timeline(raw,59)
        assert raw.station.nunique()==59 and not raw.duplicated(['station','time_s']).any()
        assert np.isfinite(raw[['forebay_m','afterbay_m','gate_flow_m3s']].to_numpy()).all()
        raw_nodes[name]=raw
        node=raw.melt(id_vars=['station','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim')
        node=node.merge(obs,on=KEYS,validate='one_to_one',how='left',indicator=True)
        assert node['_merge'].eq('both').all() and node.valid.all()
        assert_timeline(node,118)
        node['error']=node.sim-node.observed
        mapped=pd.read_csv(d/'实际位置水位.csv')
        assert_timeline(mapped,43)
        assert not mapped.duplicated(KEYS).any()
        assert set(zip(mapped.station,mapped.metric))==op_keys
        # Fresh one-to-one association, not copying observed columns from old run outputs.
        mapped=mapped.merge(obs,on=KEYS,validate='one_to_one',how='left',indicator=True)
        assert mapped['_merge'].eq('both').all() and mapped.valid.all()
        assert np.isfinite(mapped[['mapped_sim','observed']].to_numpy()).all()
        expected_group={(o['station'],o['metric']):'旧参与同化测点' if o['assimilated'] else '旧留出检验测点' for o in ops}
        assert all(row.group==expected_group[(row.station,row.metric)] for row in mapped.itertuples())
        mapped['error']=mapped.mapped_sim-mapped.observed
        datasets[name]=(node,mapped)
        with np.load(d/'末态.npz') as z:
            assert set(z.files)==set(STATE_KEYS)
            states[name]={k:z[k].copy() for k in STATE_KEYS}
        assert all(np.isfinite(v).all() for v in states[name].values())
        end=mapped[mapped.time_s==2592000].set_index(['station','metric'])
        differences=[]
        for op in ops:
            v=float(states[name][op['state']][op['indices']]@np.asarray(op['weights'])+op.get('offset',0))
            differences.append(abs(v-end.loc[(op['station'],op['metric']),'mapped_sim']))
        assert max(differences)<1e-12
        reconstruction.append(dict(run=name,points=43,max_abs_difference_m=max(differences)))
        run_integrity.append(dict(run=name,cycles=1440,model_time_s=p['model_time_s'],node_score_samples=len(node),mapped_score_samples=len(mapped),holdout_per_target=720,retries=p['retries'],finite_final_states=True,parameters_fixed=True,source_hashes_match=True,numba_actual=True,solver_runtime_frozen=True))
        for fn in ['运行配置.json','进度.json','半小时水位流量.csv','实际位置水位.csv','末态.npz','局部切口同位置全线响应.csv','半小时开敞轮廓范围.csv']:
            source_hashes[str(d/fn)]=sha(d/fn)
    # Close the independent reference chain G00 -> X00 -> M00.
    references=[]
    for label,folder in [('35_X00',REFERENCE_X),('34_M00',REFERENCE_M)]:
        raw=pd.read_csv(folder/'半小时水位流量.csv')
        assert raw_nodes[RUNS[0]].equals(raw)
        with np.load(folder/'末态.npz') as z:
            eq={k:bool(np.array_equal(states[RUNS[0]][k],z[k])) for k in STATE_KEYS}
        assert all(eq.values())
        references.append(dict(reference=label,exact_node_trajectory=True,rows=59*1440,exact_six_states=eq))
    old_mapped=pd.read_csv(REFERENCE_X/'实际位置水位.csv')
    pair=datasets[RUNS[0]][1].merge(old_mapped[KEYS+['mapped_sim']],on=KEYS,suffixes=('','_old'),validate='one_to_one')
    mapped_reference_diff=float(abs(pair.mapped_sim-pair.mapped_sim_old).max())
    assert mapped_reference_diff<1e-12
    totals=[]; points=[]; range_summary=[]; range_data={}
    for name,(node,mapped) in datasets.items():
        ranges=pd.read_csv(O/name/'半小时开敞轮廓范围.csv')
        assert len(ranges)==1440 and np.array_equal(ranges.time_s,np.arange(1800,2592001,1800))
        range_data[name]=ranges
        for period,lo,hi in PERIODS:
            n=node[node.time_s.between(lo,hi,inclusive='right')]
            m=mapped[mapped.time_s.between(lo,hi,inclusive='right')]
            per_target=1440 if period=='完整30天' else 720
            assert len(n)==118*per_target and len(m)==43*per_target
            groups=[('闸节点118测点',n),('闸节点作者可比88测点',n[n.pdf_comparable]),('安装位置43测点',m),('安装位置旧参与34测点',m[m.group=='旧参与同化测点']),('安装位置旧留出9测点',m[m.group=='旧留出检验测点'])]
            for scope,g in groups:
                totals.append(dict(run=name,period=period,scope=scope,**stats(g.error)))
            for location,data in [('闸节点',n),('安装位置候选',m)]:
                for (station,metric),g in data.groupby(['station','metric']):
                    points.append(dict(run=name,period=period,location=location,station=station,metric=metric,**stats(g.error)))
            r=ranges[ranges.time_s.between(lo,hi,inclusive='right')]
            for scope,ncol,vcol in [('全线开敞计算点','all_open_exceeded_points','all_open_max_depth_excess_m'),('POOL16开敞计算点','pool16_open_exceeded_points','pool16_open_max_depth_excess_m')]:
                range_summary.append(dict(run=name,period=period,scope=scope,half_hour_samples=len(r),samples_with_exceedance=int((r[ncol]>0).sum()),minimum_exceeded_points=int(r[ncol].min()),maximum_exceeded_points=int(r[ncol].max()),mean_exceeded_points=float(r[ncol].mean()),point_times=int(r[ncol].sum()),max_excess_m=float(r[vcol].max()),min_of_sample_max_excess_m=float(r[vcol].min()),note='仅半小时记录，输入轮廓并非已核定堤顶；point_times不是独立测点数'))
    S=pd.DataFrame(totals); T=pd.DataFrame(points)
    S.to_csv(O/'全线联合转移分组精度.csv',index=False)
    T.to_csv(O/'全线联合转移逐点精度.csv',index=False)
    pd.DataFrame(range_summary).to_csv(O/'半小时开敞轮廓超限比较.csv',index=False)
    b=T[T.run==RUNS[0]]; c=T[T.run==RUNS[1]]
    point_compare=c.merge(b[['period','location','station','metric','MAE_m','RMSE_m','bias_m','max_abs_m']],on=['period','location','station','metric'],suffixes=('_candidate','_base'),validate='one_to_one')
    point_compare['MAE_change_m']=point_compare.MAE_m_candidate-point_compare.MAE_m_base
    point_compare['RMSE_change_m']=point_compare.RMSE_m_candidate-point_compare.RMSE_m_base
    point_compare.to_csv(O/'全线联合转移逐点变化.csv',index=False)
    worsened=point_compare[(point_compare.period=='后15天') & (point_compare.MAE_change_m>.01+1e-12)].sort_values('MAE_change_m',ascending=False)
    worsened.to_csv(O/'后15天MAE恶化超过1厘米清单.csv',index=False)
    nearby=['JZZ14','JZZ15','JZZ16','JZZ17','JZZ18']
    near=point_compare[point_compare.station.isin(nearby)]
    near.to_csv(O/'兰河及邻站逐点比较.csv',index=False)
    count_rows=[]
    for period,_,_ in PERIODS:
        for location in ['闸节点','安装位置候选']:
            g=point_compare[(point_compare.period==period)&(point_compare.location==location)]
            count_rows.append(dict(period=period,location=location,points=len(g),improved_over_1cm=int((g.MAE_change_m<-.01-1e-12).sum()),worsened_over_1cm=int((g.MAE_change_m>.01+1e-12).sum()),within_1cm=int((g.MAE_change_m.abs()<=.01+1e-12).sum())))
    pd.DataFrame(count_rows).to_csv(O/'全线逐点改善恶化计数.csv',index=False)
    # Apply the written thresholds without choosing a new tolerance from the results.
    target=T[(T.period=='后15天') & (T.location=='安装位置候选') & (T.station=='JZZ16')]
    tb=target[target.run==RUNS[0]].set_index('metric'); tc=target[target.run==RUNS[1]].set_index('metric')
    assert set(tb.index)=={'forebay_m','afterbay_m'} and set(tc.index)==set(tb.index)
    mean_base=float(tb.MAE_m.mean());mean_candidate=float(tc.MAE_m.mean());gain=mean_base-mean_candidate;minimum=max(.002,mean_base*.05)
    target_checks=dict(mean_MAE_base_m=mean_base,mean_MAE_candidate_m=mean_candidate,gain_m=gain,minimum_gain_m=minimum,mean_MAE_gain_pass=gain>=minimum,each_RMSE_no_worse=bool((tc.RMSE_m<=tb.RMSE_m+1e-12).all()))
    group_checks=[]
    for scope in ['闸节点118测点','安装位置43测点','安装位置旧留出9测点']:
        g=S[(S.period=='后15天')&(S.scope==scope)].set_index('run')
        row=dict(scope=scope,baseline=g.loc[RUNS[0],['MAE_m','RMSE_m','max_abs_m']].to_dict(),candidate=g.loc[RUNS[1],['MAE_m','RMSE_m','max_abs_m']].to_dict(),MAE_no_worse=bool(g.loc[RUNS[1],'MAE_m']<=g.loc[RUNS[0],'MAE_m']+1e-12),RMSE_no_worse=bool(g.loc[RUNS[1],'RMSE_m']<=g.loc[RUNS[0],'RMSE_m']+1e-12))
        group_checks.append(row)
    pass_criteria=target_checks['mean_MAE_gain_pass'] and target_checks['each_RMSE_no_worse'] and all(x['MAE_no_worse'] and x['RMSE_no_worse'] for x in group_checks) and len(worsened)==0
    # Compare available full-line cut records with actual local-A outputs. Do not rename gate Q as meter Q.
    cut_stats=[]; cut_series=[]; cutoff_final=[]
    old_qop_path=R/'30_分通道方法逐项测试/02_输入与测点核查/输入与算子核验.json'
    qaudit=json.loads(old_qop_path.read_text())
    assert qaudit['frozen_B02_input_sha256']==sha(B/'共享输入/C07.json')
    qop=next(x['flow_operator'] for x in qaudit['stations'] if x['station']=='JZZ16')
    assert qop['chainage']==300091. and qop['state']=='Q_ik'
    given=pd.read_csv(P/'02_试验约定/共同30天输入与内部目标.csv').set_index('time_s')
    for run,local_name in zip(RUNS,['A_原参数','A_联合辨识参数']):
        cut=pd.read_csv(O/run/'局部切口同位置全线响应.csv')
        local=pd.read_csv(P/'05_固定参数复核'/local_name/'半小时复算结果.csv')
        assert len(cut)==len(local)==1440 and np.array_equal(cut.time_s,local.time_s)
        cutoff=cut.merge(local[['time_s','boundary_tailwater_m','cut_inlet_flow_m3s','gate_flow_m3s','cut_outlet_flow_m3s']],on='time_s',validate='one_to_one')
        actual_h17=datasets[run][1]
        actual_h17=actual_h17[(actual_h17.station=='JZZ17')&(actual_h17.metric=='forebay_m')]
        assert np.array_equal(actual_h17.time_s.to_numpy(),cut.time_s.to_numpy())
        assert np.max(abs(actual_h17.mapped_sim.to_numpy()-cut.A_tail_chainage_327100_head_m.to_numpy()))<1e-12
        cutoff['run']=run
        cutoff['H17_observation_sample_m']=given.loc[cutoff.time_s,'H17_forebay_m'].to_numpy()
        cutoff['Q16_observation_sample_m3s']=given.loc[cutoff.time_s,'Q16_m3s'].to_numpy()
        cutoff['full_minus_local_applied_tailwater_m']=cutoff.A_tail_chainage_327100_head_m-cutoff.boundary_tailwater_m
        cutoff['full_minus_H17_sample_m']=cutoff.A_tail_chainage_327100_head_m-cutoff.H17_observation_sample_m
        cutoff['full_minus_local_gate_flow_m3s']=cutoff.JZZ16_gate_flow_m3s-cutoff.gate_flow_m3s
        cutoff['local_inlet_is_boundary_not_independent_observation']=True
        cut_series.append(cutoff)
        for period,lo,hi in PERIODS:
            g=cutoff[cutoff.time_s.between(lo,hi,inclusive='right')]
            for label,col,unit in [('327100m水位：全线减局部实际施加尾水','full_minus_local_applied_tailwater_m','m'),('327100m水位：全线减同时间实测样本','full_minus_H17_sample_m','m'),('JZZ16过闸流量：全线减局部过闸流量','full_minus_local_gate_flow_m3s','m³/s')]:
                vstats=stats(g[col])
                generic=dict(samples=vstats['samples'],MAE=vstats['MAE_m'],RMSE=vstats['RMSE_m'],bias=vstats['bias_m'],max_abs=vstats['max_abs_m'])
                cut_stats.append(dict(run=run,period=period,quantity=label,unit=unit,**generic))
        # Only the end state can be reconstructed for full-line meter Q; no full time series was saved.
        flow=float(states[run]['Q_ik'][qop['indices']]@np.asarray(qop['weights']))
        cutoff_final.append(dict(run=run,time_s=2592000,full_Q_at_300091_m3s=flow,local_applied_cut_inlet_m3s=float(cutoff.cut_inlet_flow_m3s.iloc[-1]),observation_Q16_sample_m3s=float(cutoff.Q16_observation_sample_m3s.iloc[-1]),full_JZZ16_gate_flow_m3s=float(cutoff.JZZ16_gate_flow_m3s.iloc[-1]),note='仅末态按已存同位置算子重算；不得冒充30天切口Q时序或忽略测流至闸间蓄变'))
    pd.concat(cut_series,ignore_index=True).to_csv(O/'局部与全线同位置尾水及过闸流量比较.csv',index=False)
    pd.DataFrame(cut_stats).to_csv(O/'局部与全线切口比较分时段统计.csv',index=False)
    pd.DataFrame(cutoff_final).to_csv(O/'末时刻测流切口流量核验.csv',index=False)
    # Algebraic decomposition of the two internal targets; it does not identify a physical cause by itself.
    algebra=[]
    for run,(_,mapped) in datasets.items():
        target=mapped[mapped.station=='JZZ16']
        front=target[target.metric=='forebay_m'][['time_s','mapped_sim','observed']]
        back=target[target.metric=='afterbay_m'][['time_s','mapped_sim','observed']]
        a=front.merge(back,on='time_s',suffixes=('_front','_back'),validate='one_to_one')
        a['front_error']=a.mapped_sim_front-a.observed_front
        a['back_error']=a.mapped_sim_back-a.observed_back
        a['point_level_difference_error']=(a.mapped_sim_front-a.mapped_sim_back)-(a.observed_front-a.observed_back)
        residual=a.front_error-a.back_error-a.point_level_difference_error
        assert residual.abs().max()<1e-12
        for period,lo,hi in PERIODS:
            g=a[a.time_s.between(lo,hi,inclusive='right')]
            algebra.append(dict(run=run,period=period,front_bias_m=float(g.front_error.mean()),back_bias_m=float(g.back_error.mean()),point_level_difference_bias_m=float(g.point_level_difference_error.mean()),point_level_difference_MAE_m=float(g.point_level_difference_error.abs().mean()),identity_max_abs_residual_m=float(residual.abs().max())))
    pd.DataFrame(algebra).to_csv(O/'兰河两测点有符号误差分解.csv',index=False)
    report=dict(status='COMPLETED',assessment='PASS' if pass_criteria else 'NOT_PASS_PREDECLARED_FULL_LINE_TRANSFER_CRITERIA',no_simulation_started_by_this_summary=True,no_inputs_or_runners_modified=True,protocol=protocol,protocol_sha256=sha(protocol_path),run_integrity=run_integrity,reference_chain=references,mapped_reference_max_difference_m=mapped_reference_diff,final43point_reconstruction=reconstruction,initial_state_hashes_equal=True,forcing_array_hashes_equal=True,only_n16_C16_changed_according_to_recorded_indices_and_runner_guards=True,parameter_application=dict(n_indices=c1['n_indices'],C_indices=c1['C_indices'],n_factor=c1['factors']['n'],C_factor=c1['factors']['C'],n_values=sorted(set(c1['applied_n'])),C_values=c1['applied_C'],extrapolation_beyond_A_m=307,siphon_n_K_fixed=True,other_loss_coefficient_hashes_equal=True),heldout_criteria=dict(target_checks=target_checks,global_group_checks=group_checks,any_point_MAE_worsening_above_001=len(worsened)>0,worsened_measurement_representations=len(worsened),passed=pass_criteria,numerical_equality_tolerance=1e-12),point_change_counts=count_rows,worst_point_changes=worsened.head(12).to_dict('records'),group_metrics=S.to_dict('records'),contour_metrics=range_summary,cut_differences=cut_stats,terminal_meter_flow=cutoff_final,algebraic_decomposition=algebra,boundary_evidence_limits=dict(H17_full_time_series_same_position=True,Q16_full_upstream_cut_series_available=False,Q16_terminal_only_available=True,stored_Q_is_gate_discharge_not_upstream_meter_Q=True,flux_operator_source=str(old_qop_path),flux_operator_source_sha256=sha(old_qop_path),flux_operator=qop,interpretation='H差是两种边界/模型情境的同位置代数差；过闸Q差也不能独自裁决原因。上游Q和尾水、初态、附加307m及下游结构耦合均不同；不能只归尾水，也不能以闸流充当测流切口Q。'),limitations=['安装位置沿用冻结候选算子，JZZ17真实仪器位置仍为假设','POOL16及全线超限是超过输入轮廓，不是实测漫顶或已确认堤顶失效','半小时检查不等于每个15秒均未越界','局部保留检验通过不能替代本轮全线门槛','后15天没有用于重新选择n/C或位置解释；历史已知边界回放不是真实未知未来预报','同化/留出34/9仅为历史分组，本轮两组均无状态同化'],engineering_acceptance=False,baseline_replaced=False,T11_started=False,source_hashes=source_hashes)
    dump('全线联合参数转移汇总报告.json',report)
    print(S[S.period=='后15天'].to_string(index=False))
    print(json.dumps(dict(assessment=report['assessment'],target_checks=target_checks,global_checks=group_checks,point_changes=[x for x in count_rows if x['period']=='后15天'],worst_points=worsened[['location','station','metric','MAE_change_m']].head(5).to_dict('records')),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
