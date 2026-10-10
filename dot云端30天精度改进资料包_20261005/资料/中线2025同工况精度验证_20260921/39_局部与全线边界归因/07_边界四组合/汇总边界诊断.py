"""固定四组合事后诊断；保留原验收结论，不重新选参。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;R39=HERE.parent;R=R39.parent
folders={'FF':R39/'06_独立接口复算/G16_DFF',**{k:HERE/k for k in ['HF','FH','HH']}}
labels={'FF':'全线流量与全线尾水','HF':'历史流量与全线尾水','FH':'全线流量与历史尾水','HH':'历史流量与历史尾水'}
periods=[('完整30天',0,2592000),('前15天',0,1296000),('后15天',1296000,2592000)]
observed=pd.read_csv(R/'26_冻结基线与后续归因/共享输入/作者已核实修正观测层.csv')
data={};checks=[];scores=[];native={};initials=[];sources={}
for code,folder in folders.items():
    result=json.loads((folder/'结果.json').read_text());cfg=json.loads((folder/'运行配置.json').read_text())
    assert result['status']=='COMPLETED' and result['model_time_s']==2592000 and result['parameters_unchanged'] and result['source_hashes_unchanged']
    assert result['boundary_checks']['passed'] and result['runtime_audit']['exceedance_attempts']==0
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['source_hashes'].items())
    with np.load(folder/'映射起算状态.npz') as z:initials.append({key:z[key].copy() for key in z.files})
    with np.load(folder/'逐步接口差异.npz') as z:native[code]={key:z[key].copy() for key in z.files}
    assert native[code]['dt_s'].sum()==2592000
    a=pd.read_csv(folder/'半小时局部全线接口对照.csv');assert len(a)==1440 and np.array_equal(a.time_s,np.arange(1800,2592001,1800))
    data[code]=a
    for target,metric in [('JZZ16_forebay_m','forebay_m'),('JZZ16_afterbay_m','afterbay_m')]:
        y=observed[(observed.station_code=='JZZ16')&(observed.metric==metric)&(observed.time_s>0)].sort_values('time_s')
        assert y.valid.all() and np.array_equal(y.time_s,a.time_s)
        a[target+'_observed']=y.value.to_numpy()
        for label,lo,hi in periods:
            mask=a.time_s.between(lo,hi,inclusive='right');error=a.loc[mask,target].to_numpy()-y.loc[y.time_s.between(lo,hi,inclusive='right'),'value'].to_numpy()
            scores.append(dict(case=code,label=labels[code],target=target,period=label,samples=len(error),MAE_m=float(abs(error).mean()),RMSE_m=float(np.sqrt(np.mean(error**2))),bias_m=float(error.mean()),max_abs_m=float(abs(error).max())))
    checks.append(dict(case=code,substeps=result['completed_substeps'],half_hour_rows=len(a),boundary_checks=result['boundary_checks'],no_parameter_or_internal_state_update=True))
    for name in ['结果.json','运行配置.json','逐步接口差异.npz','半小时局部全线接口对照.csv','映射起算状态.npz']:
        path=folder/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
assert all(all(np.array_equal(x[key],initials[0][key]) for key in x) for x in initials)
assert all(np.array_equal(x['time_s'],native['FF']['time_s']) and np.array_equal(x['dt_s'],native['FF']['dt_s']) for x in native.values())
pd.DataFrame(scores).to_csv(HERE/'四组合水位描述性误差.csv',index=False)
time=native['FF']['time_s'];dt=native['FF']['dt_s']
response={name:native[code]['local'] for code,name in [('FF','FF'),('HF','HF'),('FH','FH'),('HH','HH')]}
terms={'只换入口流量_HF减FF':response['HF']-response['FF'],'只换尾水_FH减FF':response['FH']-response['FF'],'交互项_HH减HF减FH加FF':response['HH']-response['HF']-response['FH']+response['FF'],'两边界总变化_HH减FF':response['HH']-response['FF']}
identity=terms['只换入口流量_HF减FF']+terms['只换尾水_FH减FF']+terms['交互项_HH减HF减FH加FF']-terms['两边界总变化_HH减FF']
assert abs(identity).max()<1e-12
effects=[]
for period,lo,hi in periods:
    mask=(time>lo)&(time<=hi)
    for term,values in terms.items():
        for j,(quantity,unit) in enumerate([('兰河闸前水位','m'),('兰河闸后水位','m'),('兰河过闸流量','m3/s')]):
            v=values[mask,j];w=dt[mask]
            effects.append(dict(period=period,term=term,quantity=quantity,unit=unit,time_weighted_signed_mean=float(np.average(v,weights=w)),time_weighted_mean_abs=float(np.average(abs(v),weights=w)),max_abs=float(abs(v).max())))
pd.DataFrame(effects).to_csv(HERE/'边界置换有符号响应分解.csv',index=False)
half=np.flatnonzero(time%1800==0)
frame=pd.DataFrame({'time_s':time[half]})
for name,values in terms.items():
    for j,q in enumerate(['forebay_m','afterbay_m','gate_m3s']):frame[name+'_'+q]=values[half,j]
frame.to_csv(HERE/'半小时边界响应分解.csv',index=False)
old=pd.read_csv(R/'38_测点假设下的联合辨识/05_固定参数复核/A_联合辨识参数/半小时复算结果.csv')
same_history=[]
for target in ['JZZ16_forebay_m','JZZ16_afterbay_m']:
    assert np.array_equal(old.time_s,data['HH'].time_s)
    for period,lo,hi in periods:
        mask=old.time_s.between(lo,hi,inclusive='right');delta=data['HH'].loc[mask,target]-old.loc[mask,target]
        same_history.append(dict(target=target,period=period,mean_abs_difference_m=float(delta.abs().mean()),max_abs_difference_m=float(delta.abs().max())))
pd.DataFrame(same_history).to_csv(HERE/'双历史边界对38局部轨迹复核.csv',index=False)
clock=[]
for metric in ['flow_m3s','forebay_m']:
    station='JZZ16' if metric=='flow_m3s' else 'JZZ17';y=observed[(observed.station_code==station)&(observed.metric==metric)].sort_values('time_s')
    diff=np.interp(time,y.time_s,y.value)-np.interp(time-dt,y.time_s,y.value)
    clock.append(dict(station=station,metric=metric,quantity='历史信号步末减步初，不当作模型响应差',weighted_mean=float(np.average(diff,weights=dt)),max_abs=float(abs(diff).max())))
summary=dict(status='COMPLETED',cases=checks,identical_projected_initial_states=True,identical_actual_step_grid=True,only_two_boundary_sources_varied=True,fixed_parameters=True,
    post15_scores=[s for s in scores if s['period']=='后15天'],post15_response=[e for e in effects if e['period']=='后15天'],factorial_identity_max_abs=float(abs(identity).max()),
    same_historical_boundary_check=same_history,historical_input_time_staggering=clock,
    original_reported_flow_1e6_gate_pass=False,original_38_full_line_transfer_pass=False,new_independent_holdout=False,baseline_replaced=False,T11_started=False,
    limitations=['四组合描述的是所保留模型和测点假设下的边界替换响应，不是现场因果识别','全线切口采用隐式步末值，历史边界按原Forcing步初值；记录该时标差而不暗中校时','有符号模型响应分解可相加，MAE差不可相加','原报告口径水量门槛失败及38全线失败保留','全局仍有输入轮廓适用范围及仪器位置未证问题'],source_hashes=sources)
(HERE/'边界四组合诊断结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(pd.DataFrame(scores).query('period=="后15天"').to_string(index=False))
print(pd.DataFrame(effects).query('period=="后15天" and unit=="m"').to_string(index=False))
print(pd.DataFrame(same_history).query('period=="后15天"').to_string(index=False))
