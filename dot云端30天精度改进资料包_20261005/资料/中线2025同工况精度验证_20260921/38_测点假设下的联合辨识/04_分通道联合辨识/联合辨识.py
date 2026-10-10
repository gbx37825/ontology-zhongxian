"""固定位置假设下的有界n/C分通道交替；只使用预定两训练窗。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

O=Path(__file__).resolve().parent;P=O.parent
sys.path.insert(0,str(P/'01_局部模型'))
from 测点假设局部模型 import build,change,mean_forcing,steady,sample,parameter_bounds,runtime_identity

p=argparse.ArgumentParser();p.add_argument('--hypothesis',choices=['A','B'],required=True);args=p.parse_args()
out=O/args.hypothesis;out.mkdir(parents=True,exist_ok=False);started=time.time()
protocol=json.loads((P/'预定试验约定.json').read_text())
pre_path=P/'01_局部模型'/f'{args.hypothesis}_前提总结果.json';pre=json.loads(pre_path.read_text())
assert pre['hypothesis']==args.hypothesis
if not pre['eligible_for_conditional_joint_trial']:
    (out/'辨识结果.json').write_text(json.dumps(dict(status='NOT_FITTED_PREREQUISITES_FAILED',hypothesis=args.hypothesis,source=str(pre_path),engineering_calibration=False),ensure_ascii=False,indent=2))
    raise SystemExit(0)
mean_file=P/'02_试验约定/两窗时间加权均值.csv';means=pd.read_csv(mean_file).sort_values('start_s')
assert means[['start_s','end_s']].to_numpy().tolist()==protocol['training']['windows']
assert (means.end_s<=protocol['training']['end_s']).all()
target=means[['H16_forebay_m','H16_afterbay_m']].to_numpy(float)
templates=[];holds=[]
for row in means.itertuples():
    local=build(args.hypothesis,t0=float(row.end_s))
    templates.append(local);holds.append(mean_forcing(local,float(row.start_s),float(row.end_s)))
bounds={'n':parameter_bounds()['POOL16_n'],'C':parameter_bounds()['JZZ16_C']}
source_paths=[Path(__file__),P/'预定试验约定.json',pre_path,mean_file,P/'01_局部模型/测点假设局部模型.py']
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
identity=runtime_identity();cfg=dict(hypothesis=args.hypothesis,pid=os.getpid(),method=protocol['calibration'],windows=means[['start_s','end_s']].to_dict('records'),target_time_weighted_means=target.tolist(),bounds=bounds,source_hashes={str(q):sha(q)for q in source_paths},runtime_identity=identity,no_future_target_used=True,engineering_calibration=False)
(out/'辨识约定.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
cache={};trials=[];cycles=[]

def evaluate(n_factor,c_factor):
    key=(round(float(n_factor),12),round(float(c_factor),12))
    if key in cache:return cache[key]
    trial=dict(trial=len(trials)+1,n_factor=float(n_factor),c_factor=float(c_factor),windows=[],valid=True)
    predictions=[]
    for i,(template,hold)in enumerate(zip(templates,holds)):
        local=copy.deepcopy(template);change(local,n_factor=float(n_factor),c_factor=float(c_factor))
        r=steady(local,kw=hold)
        good=bool(r['converged']and r['accepted_final_exceeded_points']==0)
        values=sample(local)
        entry=dict(window=i+1,converged=bool(r['converged']),status=r['status'],stop_reason=r.get('stop_reason'),accepted_final_exceeded_points=r['accepted_final_exceeded_points'],accepted_final_max_depth_excess_m=r['accepted_final_max_depth_excess_m'],verification_steps=r['verification_steps_run'],attempts=r['runtime_attempt_audit'],values=values)
        trial['windows'].append(entry)
        if not good:
            trial['valid']=False
            (out/f"试算{trial['trial']:03d}_窗{i+1}_未认证.json").write_text(json.dumps(r,ensure_ascii=False,indent=2))
            break
        predictions.append([values['JZZ16_forebay_m'],values['JZZ16_afterbay_m']])
    if trial['valid']:
        pred=np.asarray(predictions);residual=pred-target
        drop_residual=(pred[:,0]-pred[:,1])-(target[:,0]-target[:,1])
        trial.update(predictions=pred.tolist(),residual_m=residual.tolist(),drop_residual_m=drop_residual.tolist(),n_channel_loss=float(np.mean((residual[:,1]/.03)**2)),C_channel_loss=float(np.mean((drop_residual/.03)**2)),combined_target_MSE=float(np.mean(residual**2)))
    trials.append(trial);cache[key]=trial
    (out/'逐次试算记录.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2))
    (out/'进度.json').write_text(json.dumps(dict(status='RUNNING',stage='parameter_trials',evaluations=len(trials),cycles=len(cycles),wall_seconds=time.time()-started),ensure_ascii=False,indent=2))
    return trial

def loss(factors,parameter):
    t=evaluate(factors['n'],factors['C'])
    return t[parameter+'_channel_loss'] if t['valid'] else 1e8+(factors['n']-1)**2+(factors['C']-1)**2

try:
    current={'n':1.,'C':1.};initial=evaluate(1.,1.);assert initial['valid']
    converged=False
    for cycle in range(protocol['calibration']['max_alternating_cycles']):
        beginning=current.copy();steps=[]
        for parameter in ['C','n']:
            before_loss=loss(current,parameter)
            def objective(value):
                candidate={**current,parameter:float(value)}
                return loss(candidate,parameter)
            optimum=minimize_scalar(objective,bounds=bounds[parameter],method='bounded',options={'xatol':protocol['calibration']['xatol'],'maxiter':protocol['calibration']['max_scalar_iterations']})
            if not optimum.success:raise RuntimeError('有界一维求解未收敛: '+parameter)
            fraction=protocol['calibration']['damping'];accepted=False
            for backtrack in range(protocol['calibration']['max_damping_backtracks']+1):
                proposal={**current,parameter:float(current[parameter]+fraction*(optimum.x-current[parameter]))}
                trial=evaluate(proposal['n'],proposal['C'])
                if trial['valid']and trial[parameter+'_channel_loss']<=before_loss+1e-8:
                    accepted=True;break
                fraction*=.5
            if not accepted:raise RuntimeError('阻尼步未获得相应通道有效下降: '+parameter)
            steps.append(dict(parameter=parameter,before=current.copy(),bounded_optimum=float(optimum.x),objective_at_optimum=float(optimum.fun),damping=float(fraction),after=proposal,channel_loss_before=before_loss,channel_loss_after=trial[parameter+'_channel_loss']))
            current=proposal
        final=evaluate(current['n'],current['C']);delta=max(abs(current[k]-beginning[k])for k in current)
        cycles.append(dict(cycle=cycle+1,start=beginning,end=current.copy(),steps=steps,max_ratio_change=delta,training_residual_m=final['residual_m'],combined_target_MSE=final['combined_target_MSE']))
        (out/'交替迭代记录.json').write_text(json.dumps(cycles,ensure_ascii=False,indent=2))
        print(json.dumps(cycles[-1],ensure_ascii=False),flush=True)
        if delta<protocol['calibration']['ratio_change_stop']:
            converged=True;break
    final=evaluate(current['n'],current['C'])
    assert final['valid']and all(bounds[k][0]<=current[k]<=bounds[k][1]for k in current)
    assert all(sha(path)==v for path,v in cfg['source_hashes'].items())
    base=templates[0];n_values=base['base_n'][base['n_indices']]*current['n'];C_values=base['base_C']*current['C']
    result=dict(status='FITTED'if converged else'ITERATION_LIMIT_VALID_CANDIDATE',hypothesis=args.hypothesis,ratios=current,effective_n_min=float(n_values.min()),effective_n_max=float(n_values.max()),effective_C=C_values.tolist(),alternating_converged=converged,cycles=len(cycles),evaluations=len(trials),failed_steady_evaluations=sum(not t['valid']for t in trials),at_parameter_boundary={k:min(current[k]-bounds[k][0],bounds[k][1]-current[k])<1e-3 for k in current},training_predictions=final['predictions'],training_target=target.tolist(),training_residual_m=final['residual_m'],training_all_targets_within_003=bool(np.max(np.abs(final['residual_m']))<=.03),source_hashes_unchanged=True,wall_seconds=time.time()-started,runtime_identity=runtime_identity(),validation_done=False,engineering_calibration=False,baseline_replaced=False)
    (out/'辨识结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (out/'进度.json').write_text(json.dumps(dict(status=result['status'],ratios=current,wall_seconds=time.time()-started),ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items()if k!='runtime_identity'},ensure_ascii=False),flush=True)
except BaseException as exc:
    (out/'失败.json').write_text(json.dumps(dict(error=repr(exc),evaluations=len(trials),cycles=cycles,wall_seconds=time.time()-started),ensure_ascii=False,indent=2))
    raise
