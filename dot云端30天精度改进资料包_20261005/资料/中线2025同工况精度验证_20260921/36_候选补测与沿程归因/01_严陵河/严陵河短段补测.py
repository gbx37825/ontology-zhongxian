"""显式注册JZZ4短段，复用35冻结API；前提通过后条件性辨识公共C，锁参验证。"""
from pathlib import Path
import copy,hashlib,json,os,sys,time,traceback
import numpy as np
import pandas as pd
from scipy.optimize import brentq

O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'35_局部率定前提复核';B=R/'26_冻结基线与后续归因'
sys.path[:0]=[str(P/'03_局部模型'),str(R/'23_五十成员参数同化/脚本')]
import 局部闸段 as api
from 常驻水力进程 import jit_evidence

CASE='JZZ4_short';STATION='JZZ4';LO=163800.;HI=185400.;END=2592000.
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
if (O/'运行约定.json').exists():raise RuntimeError('本目录已有试验约定，拒绝覆盖前次运行')
started=time.time();raw,observations,norm,coeff=api.data();bd=raw['baseData']
gates={k:v for k,v in bd['gates'].items() if k.startswith(STATION+'-')};assert len(gates)==2
up_nodes={g['node1'] for g in gates.values()};down_nodes={g['node2'] for g in gates.values()};assert len(up_nodes)==len(down_nodes)==1
def covering(name,x):
 xx=[bd['sections'][sn]['location'] for sn in bd['channels'][name]['sec_names']];return min(xx)<=x<=max(xx)
up=[k for k,v in bd['channels'].items() if v['node2'] in up_nodes and covering(k,48495.37)]
down=[k for k,v in bd['channels'].items() if v['node1'] in down_nodes and covering(k,48795.37)]
assert len(up)==len(down)==1
spec=dict(station=STATION,up=48495.37,down=48795.37,tail_station=STATION,tail_metric='afterbay_m',channels=up+down,observations=[(STATION,'forebay_m',48695.37)])
assert CASE not in api.CASES
api.CASES[CASE]=spec
bleeds={k:v for k,v in bd['bleeders'].items() if v.get('channel') and v['channel'][0] in spec['channels'] and spec['up']<v['channel'][1]<spec['down']}
assert list(bleeds)==['FSK6-2-pool3']
lower,upper=api.parameter_bounds()[STATION+'_C']
source_paths=[Path(__file__),Path(api.__file__),B/'共享输入/C07.json',B/'共享输入/作者已核实修正观测层.csv',B/'共享输入/2025开度归一化上限.json',B/'共享输入/Pipedream闸流系数.json',P/'02_短闸段窗口/固定选择训练窗口.csv',R/'17_求解器修复/运行脚本/试验公共.py']
runner=R.parents[1]/'water/pipedream-runner'
source_paths += [runner/p for p in ['wxq_pipedream/wxq_to_superlink.py','wxq_pipedream/section_geometry.py','wxq_pipedream/backends.py','wxq_pipedream/hydraulic_engine.py','wxq_pipedream/warmup.py','pipedream_solver/superlink.py','pipedream_solver/nsuperlink.py','pipedream_solver/ngeometry.py','pipedream_solver/gate_hydraulics.py']]
frozen=json.loads((P/'03_局部模型/接口冻结标识.json').read_text());assert sha(api.__file__)==frozen['sha256']
selected=pd.read_csv(P/'02_短闸段窗口/固定选择训练窗口.csv').set_index('station').loc[STATION];assert selected.passed and (selected.start_s,selected.end_s)==(LO,HI)
config=dict(case=CASE,pid=os.getpid(),registration='仅在本进程显式扩展api.CASES，不改35文件',registered_spec=spec,gate_input=gates,bleeders=bleeds,training_window=[LO,HI],C_factor_bounds=[lower,upper],head_difference_threshold_m=.03,gate_flow_relative_threshold=.02,independent_initial_offset_m=.2,training_target='前15天固定6h窗的时间加权平均闸前水位',evaluation_target='仅JZZ4闸前；闸后是边界不评分',roughness_fixed=True,no_assimilation=True,success_threshold='后15天MAE改善至少max(0.002m,原MAE的5%)且RMSE不恶化，全程不超原轮廓',source_hashes={str(p):sha(p) for p in source_paths},engineering_status='条件性有效系数试验；不称现场真系数，不开展全线转移')
dump(O/'运行约定.json',config)

def invariants(local):
 m=local['model'];a=local['audit'];assert all(v==0 for v in a['actual_cut_storage_areas_m2'].values())
 assert a['initial_state_explicitly_restored'] and a['within_raw_contour_initial']
 assert local['converted']['siphons_info']==[]
 original={n:np.asarray(raw['timeSeries']['seriesData'][sid],float) for sid,(kind,n) in raw['timeSeries']['seriesLink'].items()}
 holes=[]
 for row in local['converted']['orifices'].itertuples():
  delta=float(abs(local['forcing'].u[:,row.id]*row.y_max-original[row.name][:,1]).max())
  assert delta<1e-12 and row.g2==gates[row.name]['b'] and row.C==coeff[row.name]
  sill=float(m._z_inv_j[row.sj_0]+row.z_o);assert sill==gates[row.name]['zb']
  holes.append(dict(name=row.name,width_m=float(row.g2),normalization_height_not_mechanical_m=float(row.y_max),sill_m=sill,C=float(row.C),opening_max_difference_m=delta))
 for name,meta in local['converted']['channel_discretization'].items():assert np.all(m._n_ik[m._ki==meta['superlink_id']]==bd['channels'][name]['nc'])
 return dict(**a,holes_verified=holes,raw_n_unchanged=True)

def warm(local,hold):
 result=api.steady(local,kw=hold)
 if not result['converged']:raise RuntimeError('局部稳态未收敛: '+str(result['status']))
 assert local['runtime_audit']['exceedance_attempts']==0
 return result

def prerequisite():
 low=api.build(CASE,t0=HI);high=api.build(CASE,t0=HI,initial_offset=.2);end=api.build(CASE,t0=HI)
 audit=invariants(low);dump(O/'构模与原参数核验.json',audit)
 hold=api.mean_forcing(low,LO,HI)
 out=[]
 for label,loc,kw in [('窗均值低初值',low,hold),('窗均值高初值',high,hold),('末时固定',end,api.forcing(end,HI))]:
  result=warm(loc,kw);out.append(dict(label=label,steady=result,state=api.sample(loc),runtime_audit=loc['runtime_audit']))
 dual=max(abs(api.sample(low)[k]-api.sample(high)[k]) for k in ['JZZ4_forebay_m','boundary_inlet_water_m'])
 hist=api.build(CASE,t0=HI-86400.);init=warm(hist,api.forcing(hist,HI-86400.));rows,retries=api.advance(hist,HI-86400.,HI)
 frame=pd.DataFrame(rows);frame.to_csv(O/'提前24小时真实推进.csv',index=False);window=frame[frame.time_s>=LO];assert len(window)==13
 average={k:float(np.trapezoid(window[k],window.time_s)/(HI-LO)) for k in api.sample(hist)}
 checks=[]
 for name,x,y in [('末时固定与到达末态',api.sample(end),api.sample(hist)),('窗均值固定与真实窗均值',api.sample(low),average)]:
  hd=float(x['JZZ4_forebay_m']-y['JZZ4_forebay_m']);qd=abs(x['gate_flow_m3s']-y['gate_flow_m3s'])/max(abs(y['gate_flow_m3s']),1)
  checks.append(dict(mode=name,head_difference_m=hd,gate_flow_relative_difference=qd,passed=abs(hd)<=.03 and qd<=.02,reference=x,actual=y))
 result=dict(case=CASE,dual_initial_head_difference_m=dual,steady_runs=out,T04=checks,history_initialization=init,history_runtime_audit=hist['runtime_audit'],history_retries=retries,eligible_for_conditional_C_trial=dual<=1e-5 and all(v['passed'] for v in checks) and not hist['runtime_audit']['exceedance_attempts'])
 dump(O/'前提核验.json',result);return result,low,hold

trace=[]
try:
 pre,template,hold=prerequisite()
 if not pre['eligible_for_conditional_C_trial']:
  result=dict(status='PREREQUISITE_FAILED',prerequisite=pre,baseline_replaced=False)
 else:
  times,y=api.observed(STATION,'forebay_m');mask=(times>=LO)&(times<=HI);target=float(np.trapezoid(y[mask],times[mask])/(HI-LO))
  base_C=template['model']._Co.copy();base_n=template['model']._n_ik.copy();cache={}
  def evaluate(factor):
   factor=float(factor);assert lower<=factor<=upper
   if factor in cache:return cache[factor]
   test=copy.deepcopy(template);test['model']._Co[:]=base_C*factor;test['runtime_audit']=dict(checked_hydraulic_attempts=0,maximum_depth_excess_m=-float('inf'),exceedance_attempts=0,exceedance_point_attempts=0)
   try:
    state=warm(test,hold);res=float(api.sample(test)['JZZ4_forebay_m']-target)
    entry=dict(factor=factor,status='FEASIBLE',C=(base_C*factor).tolist(),residual_m=res,target_m=target,state=api.sample(test),runtime_audit=test['runtime_audit'],steady=state)
    cache[factor]=(res,test)
   except (RuntimeError,FloatingPointError,ValueError) as ex:
    entry=dict(factor=factor,status='INADMISSIBLE',error=repr(ex),runtime_audit=test['runtime_audit']);cache[factor]=(None,None)
   trace.append(entry);dump(O/'辨识逐次记录.json',trace);return cache[factor]
  f0,_=evaluate(1.);assert f0 is not None
  bracket=None;last=(1.,f0)
  # Search only within the frozen bounds, in the hydraulic direction that reduces the baseline residual.
  for cf in np.linspace(1.,upper if f0>0 else lower,9)[1:]:
   value,_=evaluate(float(cf))
   if value is None:continue
   if value*last[1]<=0:bracket=sorted([last[0],float(cf)]);break
   last=(float(cf),value)
  if abs(f0)<=1e-7:factor=1.;bracket=[1.,1.]
  elif bracket is not None:factor=float(brentq(lambda x:evaluate(x)[0],*bracket,xtol=1e-7,rtol=1e-7,maxiter=30))
  else:factor=None
  if factor is None:
   result=dict(status='NO_FEASIBLE_BRACKET_IN_SELECTED_DIRECTION',baseline_residual_m=f0,bounds=[lower,upper],target_m=target,trace=trace,bounds_relaxed=False,baseline_replaced=False)
  else:
   residual,fitted=evaluate(factor);assert lower+1e-5<factor<upper-1e-5 and fitted['runtime_audit']['exceedance_attempts']==0
   runs=[]
   for label,cf in [('原系数',1.),('辨识系数',factor)]:
    local=api.build(CASE,t0=0.,c_factor=cf);init=warm(local,api.forcing(local,0.));dump(O/(label+'_初态稳态.json'),init)
    rows=[];retries=0
    for day in range(30):
     part,n=api.advance(local,day*86400.,(day+1)*86400.);rows.extend(part);retries+=n
     progress=dict(status='RUNNING',branch=label,cycles=len(rows),total_cycles=1440,wall_seconds=time.time()-started);dump(O/'进度.json',progress);print(json.dumps(progress,ensure_ascii=False),flush=True)
    m=local['model'];assert m.t==END and np.array_equal(m._n_ik,base_n) and np.array_equal(m._Co,base_C*cf)
    table=pd.DataFrame(rows);assert np.array_equal(table.time_s,np.arange(1800,END+1,1800));table['observed_forebay_m']=np.interp(table.time_s,times,y);table['error_m']=table.JZZ4_forebay_m-table.observed_forebay_m
    table.to_csv(O/(label+'_30天水位.csv'),index=False);np.savez_compressed(O/(label+'_末态.npz'),**{k:getattr(m,k) for k in api.STATE_KEYS},C=m._Co,n=m._n_ik)
    metrics=[]
    for window,mask in [('辨识窗',table.time_s.between(LO,HI)),('前15天非辨识窗',(table.time_s<=1296000)&~table.time_s.between(LO,HI)),('后15天',table.time_s>1296000),('全30天',table.time_s>0)]:
     err=table.loc[mask,'error_m'];metrics.append(dict(window=window,count=len(err),mae_m=float(err.abs().mean()),rmse_m=float(np.sqrt((err**2).mean())),max_abs_m=float(err.abs().max()),bias_m=float(err.mean())))
    assert local['runtime_audit']['exceedance_attempts']==0
    runs.append(dict(branch=label,factor=cf,C=m._Co.tolist(),cycles=1440,end_s=float(m.t),retries=retries,runtime_audit=local['runtime_audit'],parameters_fixed=True,numba=jit_evidence(),metrics=metrics))
   orig=next(r for r in runs[0]['metrics'] if r['window']=='后15天');fit=next(r for r in runs[1]['metrics'] if r['window']=='后15天');threshold=max(.002,.05*orig['mae_m']);gain=orig['mae_m']-fit['mae_m']
   result=dict(status='COMPLETED',case=CASE,C_factor=factor,fitted_C=(base_C*factor).tolist(),bounds=[lower,upper],bracket=bracket,residual_at_calibration_mean_m=residual,runs=runs,holdout_MAE_gain_m=gain,predeclared_minimum_gain_m=threshold,useful_for_further_local_diagnostic=gain>=threshold and fit['rmse_m']<=orig['rmse_m'],baseline_replaced=False,full_line_transfer_done=False)
 assert all(sha(p)==h for p,h in config['source_hashes'].items())
 result.update(source_hashes_unchanged=True,wall_seconds=time.time()-started);dump(O/'结果.json',result);dump(O/'进度.json',dict(status=result['status'],wall_seconds=time.time()-started));print(json.dumps(result,ensure_ascii=False),flush=True)
except BaseException as ex:
 dump(O/'失败.json',dict(error=repr(ex),traceback=traceback.format_exc(),wall_seconds=time.time()-started));raise
