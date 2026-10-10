from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因'
sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
from 试验公共 import Forcing,flow_operators,observe
sys.path.insert(0,str(R/'04_脚本'))
from 并联结构物理修复 import build_engine_physics
from pipedream_solver.gate_hydraulics import rectangular_gate_discharge
model=B/'共享输入/C07.json';norm=json.loads((B/'共享输入/2025开度归一化上限.json').read_text())
e,s,c=build_engine_physics(model,gate_max_normalization=norm);m=e.model;f=Forcing(e,s,c);ts=f.times
series=list(c['time_series'].values());bleed={x['target_name']:np.array(x['values']) for x in series if x['target_type']=='bleeders'}
rows=[]
for d,row in zip(f.routing,c['bleeders']):
 name=d['target'];v=bleed[name]
 rows.append({**d,'runtime_node':row['node'],'mean_m3s':float(np.trapezoid(v,ts)/(ts[-1]-ts[0])),'min_m3s':float(v.min()),'max_m3s':float(v.max()),'nonzero_n':int(np.count_nonzero(v)),'negative_n':int((v<0).sum())})
pd.DataFrame(rows).to_csv(O/'全部分退水投影与流量.csv',index=False)
raw=pd.read_csv(R/'02_输入/2025节制闸原始窗口.csv',parse_dates=['时间']);start=pd.Timestamp(s['timeControl']['calculateStartData']);end=start+pd.Timedelta(seconds=ts[-1]);raw=raw[raw['时间'].between(start,end)]
bounds=[]
for node,station,col in [('NODE-main-0','陶岔渠首引水闸','瞬时流量'),('NODE-main-119','北拒马河暗渠进口节制闸','闸前水位')]:
 source=raw[raw['站点名称']==station].sort_values('时间');x=next(a for a in series if a['target_type']=='nodes' and a['target_name']==node);v=np.array(x['values']);expected=source[col].to_numpy()
 bounds.append(dict(node=node,source=station,column=col,n=len(v),max_source_difference=float(abs(v-expected).max()),mean=float(np.trapezoid(v,ts)/ts[-1]),min=float(v.min()),max=float(v.max()),negative_n=int((v<0).sum()),source_timestamps_exact=bool(np.array_equal((source['时间']-start).dt.total_seconds().to_numpy(),ts)),start=float(v[0]),end=float(v[-1])))
qin=np.array(next(x for x in series if x['target_name']=='NODE-main-0')['values']);outflow=np.sum(list(bleed.values()),axis=0)
check=dict(time_start=s['timeControl']['calculateStartData'],time_points=len(ts),all_time_axes_equal=all(np.array_equal(x['times_s'],ts) for x in series),intervals_s=np.unique(np.diff(ts)).tolist(),series_by_type=pd.Series([a['target_type'] for a in series]).value_counts().to_dict(),unique_targets=len({(a['target_type'],a['target_name']) for a in series}),boundaries=bounds,bleeders=len(rows),routing_modes=pd.Series([a['mode'] for a in rows]).value_counts().to_dict(),offtake_mean_m3s=float(np.trapezoid(outflow,ts)/ts[-1]),upstream_net_minus_offtakes_mean_m3s=float(np.trapezoid(qin-outflow,ts)/ts[-1]),mass_routing_max_abs_error_m3s=float(np.max(abs(f.q.sum(axis=1)+f.q0.sum(axis=1)-(qin-outflow)))),negative_offtakes=[x for x in rows if x['negative_n']],left_sample_15s_vs_linear_integral_net_m3=float(-7.5*((qin-outflow)[-1]-(qin-outflow)[0])))
openings=[]
for x in series:
 if x['target_type']!='gates':continue
 i=e.gate_index[x['target_name']];v=np.array(x['values']);actual=f.u[:,i]*float(c['orifices'].iloc[i].y_max)
 openings.append(dict(name=x['target_name'],max_diff=float(abs(v-actual).max()),n_clipped=int((abs(v-actual)>1e-12).sum()),min=float(v.min()),max=float(v.max())))
check['opening_clipped_n']=sum(x['n_clipped'] for x in openings);check['opening_max_restore_difference_m']=max(x['max_diff'] for x in openings)
npz=np.load(B/'共享输入/初始化后状态_0s.npz');state={k:npz[k].copy() for k in ('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')};m.load_state(state,compute_hydraulic_geometries=True);m.compute_storage_volumes()
kw=f.at(0);gates=c['orifices'].copy();gates['station']=gates.name.str.extract(r'^(JZZ\d+)-')[0];ob=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');ob=ob[ob.time_s==0].set_index(['station_code','metric']);g=[]
for station,a in gates.groupby('station'):
 ju=int(a.iloc[0].sj_0);jd=int(a.iloc[0].sj_1);qq=0
 for z in a.itertuples():
  sill=m._z_inv_j[ju]+z.z_o;qq+=rectangular_gate_discharge(m.H_j[ju]-sill,m.H_j[jd]-sill,kw['u_o'][z.id]*z.y_max,z.g2,z.C)
 stored=float(m.Q_o[a.id].sum());g.append(dict(station=station,initial_forebay=float(m.H_j[ju]),observed_forebay=float(ob.loc[(station,'forebay_m'),'value']),initial_afterbay=float(m.H_j[jd]),observed_afterbay=float(ob.loc[(station,'afterbay_m'),'value']),stored_flow_m3s=stored,flow_required_by_initial_heads_m3s=float(qq),flow_difference_m3s=float(qq-stored)))
pd.DataFrame(g).to_csv(O/'初始水面与闸流兼容性.csv',index=False)
ops,audit=flow_operators(e,s,c);flows=[]
for op in ops:
 key=(op['station'],'flow_m3s')
 if key not in ob.index or not ob.loc[key,'valid']:continue
 measured=float(ob.loc[key,'value']);calc=observe(m,op);flows.append(dict(station=op['station'],channel=op['channel'],model_flow=calc,measured_flow=measured,error_m3s=calc-measured))
pd.DataFrame(flows).to_csv(O/'初始同位置测流比较.csv',index=False)
check['initial_gate_flux_consistency']=dict(gates=len(g),mean_abs_m3s=float(np.mean([abs(z['flow_difference_m3s']) for z in g])),max_abs_m3s=float(max(abs(z['flow_difference_m3s']) for z in g)),noncompatible_gt_1m3s=sum(abs(z['flow_difference_m3s'])>1 for z in g),largest=sorted(g,key=lambda z:abs(z['flow_difference_m3s']),reverse=True)[:5])
check['initial_meter_flow']=dict(n=len(flows),MAE_m3s=float(np.mean([abs(v['error_m3s']) for v in flows])),bias_m3s=float(np.mean([v['error_m3s'] for v in flows])))
control=pd.read_csv(B/'B02_渠池归属修复后的参数同化诊断基线/匹配无同化对照/预报与分析逐点.csv');control=control[(control.operator=='闸端全线')&(control.phase=='forecast_before_update')];control['error']=control.sim-control.observed;stats=[]
for name,lo,hi in [('前半小时',0,1800),('首6小时',0,21600),('6至24小时',21600,86400),('第2天',86400,172800),('第3天',172800,259200),('末3天',27*86400,30*86400),('完整30天',0,30*86400)]:
 a=control[(control.time_s>lo)&(control.time_s<=hi)]
 for scope,d in [('全线118水位测点',a),('作者88水位测点',a[a.pdf_comparable])]:
  stats.append(dict(window=name,scope=scope,n=len(d),MAE=float(d.error.abs().mean()),bias=float(d.error.mean()),max_abs=float(d.error.abs().max())))
pd.DataFrame(stats).to_csv(O/'无同化对照误差随时间变化.csv',index=False)
check['files_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,B/'共享输入/初始化后状态_0s.npz',R/'17_求解器修复/运行脚本/试验公共.py']}
(O/'输入与初态复核.json').write_text(json.dumps(check,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(check,ensure_ascii=False,indent=2));print(pd.DataFrame(rows).query('mode != "按模型真实桩号向相邻内部节点守恒分配"').to_string(index=False));print(pd.DataFrame(stats).to_string(index=False))
