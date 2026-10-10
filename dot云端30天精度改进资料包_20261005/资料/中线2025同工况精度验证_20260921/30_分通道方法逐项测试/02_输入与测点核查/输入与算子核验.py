"""只读核查冻结 B02 与当前离线入口的输入、参数应用、观测位置。"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent
R=O.parents[1]
B=R/'26_冻结基线与后续归因'
B2=B/'B02_渠池归属修复后的参数同化诊断基线'
sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
from 试验公共 import Forcing,meshes,weights_at,flow_operator,control_volumes
sys.path.insert(0,str(R/'04_脚本'))
from 并联结构物理修复 import build_engine_physics
sys.path.insert(0,str(B2/'运行源码'))
from 拓扑参数EnKF import ParameterEnKF,STATE_KEYS,restore
from pipedream_solver.gate_hydraulics import rectangular_gate_discharge

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
model=R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json'
pairs=[(model,B/'共享输入/C07.json'),
 (R/'03_运行参数/2025开度归一化上限.json',B/'共享输入/2025开度归一化上限.json'),
 (R/'01_原始资料/Pipedream闸流系数.json',B/'共享输入/Pipedream闸流系数.json'),
 (R/'02_输入/作者已核实修正观测层.csv',B/'共享输入/作者已核实修正观测层.csv'),
 (R.parent/'中线测点修正与状态同化试验_20260916/07_对比分析/水位测点位置审核.json',B/'共享输入/水位测点位置审核.json'),
 (R/'17_求解器修复/运行脚本/试验公共.py',B/'共享依赖/水力运行脚本/试验公共.py'),
 (R/'04_脚本/并联结构物理修复.py',B/'共享依赖/并联结构物理修复.py'),
 (R/'23_五十成员参数同化/脚本/常驻水力进程.py',B/'共享依赖/参数并行运行脚本/常驻水力进程.py')]
project=R.parents[1]
for d in ['pipedream_solver','wxq_pipedream']:
 for f in (B/'共享依赖'/d).glob('*.py'):
  pairs.append((project/'water/pipedream-runner'/d/f.name,f))
hashes=[dict(active=str(a),frozen=str(b),active_sha256=sha(a),frozen_sha256=sha(b),equal=sha(a)==sha(b)) for a,b in pairs]
normalization=json.loads((B/'共享输入/2025开度归一化上限.json').read_text())
e,s,c=build_engine_physics(model,gate_max_normalization=normalization)
m=e.model;f=Forcing(e,s,c)
ens=ParameterEnKF(e,c,n=2)
config=json.loads((B2/'同化运行/运行配置.json').read_text())
assert ens.parameters==config['parameter_bounds']
assert sha(model)==config['input_sha256']
assert np.array_equal(ens.base_n,m._n_ik) and np.array_equal(ens.base_C,m._Co)
before=[{k:getattr(mm,k).copy() for k in STATE_KEYS} for mm in ens.members]
ens.theta[:]=0
for j,p in enumerate(ens.parameters):ens.theta[j]=np.log([1.02,.98])
ens.apply()
parameter_check=[]
for i,mm in enumerate(ens.members):
 checks=[]
 for j,p in enumerate(ens.parameters):
  ids=p['indices'];base=ens.base_n if p['kind']=='n' else ens.base_C;actual=mm._n_ik if p['kind']=='n' else mm._Co
  checks.append(np.array_equal(actual[ids],base[ids]*np.exp(ens.theta[j,i])))
 parameter_check.append(dict(member=i,every_parameter_applied=all(checks),endpoint_n_matches=bool(np.array_equal(mm._n_uk,mm._n_ik[mm._i_1k]) and np.array_equal(mm._n_dk,mm._n_ik[mm._i_nk])),state_unchanged=all(np.array_equal(getattr(mm,k),before[i][k]) for k in STATE_KEYS)))

interpolation=[]
for t in [0.,900.,1800.,21600.,21607.5,2592000.]:
 v=f.at(t);i=int(np.clip(np.searchsorted(f.times,t,side='right')-1,0,len(f.times)-2));a=float(np.clip((t-f.times[i])/(f.times[i+1]-f.times[i]),0,1))
 interpolation.append(dict(time_s=t,left_time_s=float(f.times[i]),right_time_s=float(f.times[i+1]),weight_right=a,all_arrays_equal=all(np.array_equal(v[k],x[i]*(1-a)+x[i+1]*a,equal_nan=True) for k,x in [('H_bc',f.h),('Q_in',f.q),('Q_0Ik',f.q0),('u_o',f.u)])))
obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv')
assert sorted(obs.time_s.unique())==f.times.tolist()
mapping=json.loads((B/'共享输入/水位测点位置审核.json').read_text())
ops=json.loads((B2/'同化运行/观测算子与留出.json').read_text())
positions=pd.read_csv(R/'02_输入/测点位置映射.csv').set_index('stationcode')
mesh=meshes(s,c)
saved=np.load(B2/'六小时诊断起点/最新完整集合.npz')
stages={k:np.mean([saved[f'm{i}_{k}'] for i in range(50)],axis=0) for k in STATE_KEYS}
restore(m,stages)
runtime_coeff=json.loads((B/'共享输入/Pipedream闸流系数.json').read_text())
old_cv=pd.read_csv(R/'19_水位精度根因/测流至闸门分水及蓄水量级.csv').set_index('station_code')
records=[]
for station in ['JZZ4','JZZ6','JZZ13','JZZ16']:
 gate=c['orifices'][c['orifices'].name.str.startswith(station+'-')]
 names=gate.name.tolist();ids=gate.id.to_numpy(int);ju=int(gate.iloc[0].sj_0);jd=int(gate.iloc[0].sj_1)
 pos=positions.loc[station];st_ops=[o for o in ops if o['station']==station]
 op_checks=[]
 for op in st_ops:
  metric='front' if op['metric']=='forebay_m' else 'back'
  original=next(x for x in mapping['sensor_rows'] if x['stationcode']==station and x['metric']==metric and x['status']=='可作位置插值候选')
  ch=original['model_brackets'][0]['channel'];ii,w=weights_at(mesh[ch],op['chainage']);z=float(m._z_inv_Ik[ii]@w)
  exact=bool(np.array_equal(ii,op['indices']) and np.array_equal(w,op['weights']) and z==op['offset'])
  head=stages['h_Ik'][ii]@w+z;j=ju if metric=='front' else jd
  op_checks.append(dict(metric=metric,channel=ch,indices=ii.tolist(),weights=w.tolist(),offset=z,operator_exactly_rebuilt=exact,
   chainage=op['chainage'],position_description=original['position_detail'],instrument_identity=original['historical_sensor_identity'],
   six_hour_mean_sensor_m=float(head),six_hour_mean_gate_node_m=float(stages['H_j'][j]),gate_minus_sensor_m=float(stages['H_j'][j]-head)))
 flow=None;cv=None
 if pos.status=='已核对' and np.isfinite(pos.meter_chainage):
  flow=flow_operator(e,s,c,station,float(pos.meter_chainage));v,notes=control_volumes(e,[flow]);cv=notes[0]
  z=f.at(21600);qoff=-float(v[0,:m.M]@z['Q_in']+v[0,m.M:]@z['Q_0Ik']);cv['six_hour_control_volume_offtake_m3s']=qoff
  cv['prior_30day_model_storage_rate_max_abs_m3s']=float(old_cv.loc[station,'max_abs_storage_rate'])
 openings=[]
 for name,i in zip(names,ids):
  row=gate.loc[i];g=s['baseData']['gates'][name];actual=f.u[:,i]*float(row.y_max)
  ser=next(ts for ts in c['time_series'].values() if ts['target_type']=='gates' and ts['target_name']==name)
  original=np.asarray(ser['values'],float)
  openings.append(dict(name=name,width_m=float(row.g2),normalization_height_m=float(row.y_max),effective_sill_m=float(m._z_inv_j[ju]+row.z_o),json_sill_m=g['zb'],json_down_sill_unused=g.get('down_zb'),json_C=g.get('c1'),runtime_C=float(m._Co[i]),override_C=runtime_coeff.get(name),opening_clipped_points=int(np.count_nonzero(abs(actual-original)>1e-12)),opening_recovery_max_abs_m=float(abs(actual-original).max()),opening_min_m=float(actual.min()),opening_max_m=float(actual.max())))
 # 同一公式固定水头/开度下对共同系数比例严格线性；只检查代数，不用模型流量冒充实测。
 qbase=qscaled=0.
 for row in gate.itertuples():
  sill=m._z_inv_j[ju]+row.z_o;hu=stages['H_j'][ju]-sill;hd=stages['H_j'][jd]-sill;opening=f.at(21600)['u_o'][row.id]*row.y_max
  qbase+=rectangular_gate_discharge(hu,hd,opening,row.g2,row.C)
  qscaled+=rectangular_gate_discharge(hu,hd,opening,row.g2,row.C*1.1)
 records.append(dict(station=station,name=pos['name'],flow_position={k:(None if isinstance(a,float) and not np.isfinite(a) else a) for k,a in pos.to_dict().items()},operators=op_checks,flow_operator=flow,control_volume=cv,gate_holes=openings,pressure_formula_scale_test_error=float(abs(qscaled-1.1*qbase))))

window=[]
for lo,hi in [(768600.,790200.),(59400.,81000.)]:
 station='JZZ16';flow=next(x['flow_operator'] for x in records if x['station']==station);w,cv=control_volumes(e,[flow]);w=w[0]
 np.save(O/'兰河测流到闸门控制体权重.npy',w)
 ts=f.times[(f.times>=lo)&(f.times<=hi)];sinks=np.array([-w@np.r_[f.at(t)['Q_in'],f.at(t)['Q_0Ik']] for t in ts])
 measured=obs[(obs.station_code==station)&obs.time_s.between(lo,hi)]
 stats={k:dict(valid_all=bool(v.valid.all()),samples=len(v),mean=float(v.value.mean()),min=float(v.value.min()),max=float(v.value.max()),net_change=float(v.value.iloc[-1]-v.value.iloc[0])) for k,v in measured.groupby('metric')}
 window.append(dict(station=station,start_s=lo,end_s=hi,offtake_mean_m3s=float(np.trapezoid(sinks,ts)/(hi-lo)),offtake_max_m3s=float(sinks.max()),offtake_min_m3s=float(sinks.min()),observations=stats,control_volume=cv[0],storage_status='未由独立实测确定；窗口水位小变不等于蓄变为零，须同时间段水力推进连续方程积分或带明确假设的断面体积估算'))

result=dict(input_hash_checks=hashes,hashes_all_equal=all(x['equal'] for x in hashes),frozen_B02_input_sha256=config['input_sha256'],
 source_start=s['timeControl']['calculateStartData'],time_step_s=1800,source_samples=len(f.times),end_s=float(f.times[-1]),
 forcing_sample_checks=interpolation,observation_times_exactly_match=True,parameter_application_checks=parameter_check,
 parameter_mapping_equals_B02=True,base_n_equals_converted_links=bool(np.array_equal(ens.base_n,c['links'].n.to_numpy())),base_C_equals_converted_orifices=bool(np.array_equal(ens.base_C,c['orifices'].C.to_numpy())),
 parallel_physics=e.parallel_physics_manifest,stations=records,JZZ16_windows=window,
 limitation='测点至闸节点水位差是冻结基线第6小时状态上的模型条件性估计，不是现场实测换算；控制体蓄量仍缺独立实测，不据此宣称物理率定完成。')
(O/'输入与算子核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps(dict(hashes_all_equal=result['hashes_all_equal'],hash_count=len(hashes),parameter_mapping_equals_B02=True,stations=[dict(station=x['station'],op_count=len(x['operators']),flow_operator=x['flow_operator'],head_offsets=[(v['metric'],v['gate_minus_sensor_m']) for v in x['operators']],cv=x['control_volume'],runtime_C=[v['runtime_C'] for v in x['gate_holes']]) for x in records]),ensure_ascii=False,indent=2))
