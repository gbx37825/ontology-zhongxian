"""只使用窗口起点以前的输入选择批次长度；计分观测不回流到算法。"""
import numpy as np

def input_window_features(forcing,end_s,span_s,y_max):
    lo=end_s-span_s
    if lo<forcing.times[0]:return dict(complete=False,span_s=span_s)
    times=np.r_[lo,forcing.times[(forcing.times>lo)&(forcing.times<end_s)],end_s]
    sampled=[forcing.at(float(t)) for t in times]
    heads=np.stack([x['H_bc'][forcing.engine.model.bc] for x in sampled])
    openings=np.stack([x['u_o']*y_max for x in sampled])
    inlet=forcing.engine.node_index['NODE-main-0']
    q=np.array([[x['Q_in'][inlet]] for x in sampled])
    takes=np.stack([np.interp(times,forcing.times,v) for v in forcing.bleeder_values.values()],axis=1)
    def relative(values):
        if values.shape[1]==0:return 0.
        mean=np.trapezoid(values,times,axis=0)/span_s
        return float(np.max(np.ptp(values,axis=0)/np.maximum(abs(mean),1.)))
    features=dict(complete=True,start_s=lo,end_s=end_s,span_s=span_s,samples=len(times),
        max_opening_range_m=float(np.ptp(openings,axis=0).max()),max_boundary_head_range_m=float(np.ptp(heads,axis=0).max()),
        max_inflow_relative_range=relative(q),max_offtake_relative_range=relative(takes))
    features['input_quiet']=features['max_opening_range_m']<=.02 and features['max_boundary_head_range_m']<=.03 and features['max_inflow_relative_range']<=.02 and features['max_offtake_relative_range']<=.02
    return features

def choose_window(forcing,start_s,y_max,remaining_cycles):
    rows=[input_window_features(forcing,start_s,span,y_max) for span in [21600.,10800.,3600.]]
    chosen=next((x for x in rows[:2] if x['complete'] and x['input_quiet']),rows[-1])
    planned=int(chosen['span_s']/1800);actual=min(planned,remaining_cycles)
    return dict(window_start_s=start_s,planned_cycles=planned,actual_cycles=actual,window_end_s=start_s+actual*1800,
        input_quiet=bool(chosen.get('input_quiet',False)),truncated_at_issue=actual!=planned,
        classification_uses_only_input_times_at_or_before=start_s,features=rows)

def observations_at(valid,ops,t,now):
    if t>now:raise ValueError('算法不得请求未来观测')
    result=[]
    for op in ops:
        if not op['assimilated']:continue
        value=valid.loc[(op['station'],op['metric'],t)]
        if not bool(value['valid']):raise ValueError('预定更新点存在无效值，不可静默换点')
        result.append({**op,'value':float(value['value']),'time_s':float(t)})
    assert len(result)==34
    return result

def common_noise(seed,origin_id,cycle,stream,shape):
    return np.random.default_rng(np.random.SeedSequence([seed,origin_id,cycle,stream])).normal(size=shape)
