"""按原门槛穷举另一候选联合段的前15天窗口；不构模、不率定。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因'
T=np.arange(1441)*1800
thfile=R/'30_分通道方法逐项测试/01_窗口筛选/筛选阈值.json'
th=json.loads(thfile.read_text());assert th['窗口小时']==[6,24]
s=json.loads((B/'共享输入/C07.json').read_text())['中线全线']
series=[];included=[]
for key,(kind,name) in s['timeSeries']['seriesLink'].items():
    take=kind=='bleeders' and s['baseData']['bleeders'][name].get('channel') and 96769<float(s['baseData']['bleeders'][name]['channel'][1])<=116647
    gate=kind=='gates' and name.split('-')[0] in ['JZZ6','JZZ7']
    if not(take or gate):continue
    v=np.asarray(s['timeSeries']['seriesData'][key],float);assert np.array_equal(v[:,0],T)
    label='gate' if gate else 'flow'
    series.append((name,v[:,1],np.isfinite(v[:,1]),label));included.append(dict(name=name,type=kind))
obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv')
for station,metric in [('JZZ6','flow_m3s'),('JZZ7','flow_m3s'),('JZZ6','forebay_m'),('JZZ6','afterbay_m'),('JZZ7','afterbay_m')]:
    g=obs[(obs.station_code==station)&(obs.metric==metric)].set_index('time_s').reindex(T)
    series.append((station+':'+metric,g.value.to_numpy(float),g.valid.fillna(False).to_numpy(bool),'flow' if metric=='flow_m3s'else'level'))
rows=[]
for hours in th['窗口小时']:
    span=hours*2
    for i in range(721-span):
        r=dict(window_h=hours,start_s=int(T[i]),end_s=int(T[i+span]));details=[]
        for name,v,valid,kind in series:
            z=v[i:i+span+1]
            den=max(abs(z.mean()),th['流量相对极差分母下限m3s']) if kind=='flow' else 1.
            limit={'flow':th['流量相对极差上限'],'level':th['水位极差上限m'],'gate':th['开度极差上限m']}[kind]
            ratio=float(np.ptp(z)/den/limit) if valid[i:i+span+1].all() else float('inf')
            r[name+'_ratio']=ratio;details.append((ratio,name))
        worst=max(details);r.update(passed=worst[0]<=1+1e-10,max_threshold_ratio=worst[0],worst=worst[1]);rows.append(r)
df=pd.DataFrame(rows);df.to_csv(O/'十二里河联合段全部窗口.csv',index=False)
summary=[]
for h,g in df.groupby('window_h'):
    q=g[g.passed];r=g.sort_values(['max_threshold_ratio','end_s']).iloc[0].to_dict()
    summary.append(dict(window_h=int(h),checked=len(g),passed=len(q),best=r))
result=dict(status='COMPLETED_READ_ONLY',upstream_Q_chainage_m=96769.,tailwater_chainage_m=116647.,targets=['JZZ6闸前','JZZ6闸后'],tailwater_not_scored=True,Q7_diagnostic_only=True,all_internal_inputs=included,summary=summary,original_threshold_sha256=hashlib.sha256(thfile.read_bytes()).hexdigest(),thresholds_unchanged=True,no_simulation=True,note='按相同原阈值筛候选；有时间合格窗也不自动证明位置、水量和几何前提通过。')
(O/'十二里河联合段选窗结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
