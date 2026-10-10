"""两分支固定参数历史复算；评分只在复算结束后读取内部水位观测。"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, time, traceback
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
API = R/'01_局部模型/测点假设局部模型.py'
spec = importlib.util.spec_from_file_location('local38', API)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('branch', choices=['原参数', '联合辨识参数'])
    args = parser.parse_args()
    out = HERE/('A_'+args.branch)
    out.mkdir(parents=True, exist_ok=False)
    fit_path = R/'04_分通道联合辨识/A/辨识结果.json'
    fit = json.loads(fit_path.read_text())
    assert fit['status']=='FITTED' and fit['alternating_converged']
    factors = dict(n=1., C=1.) if args.branch=='原参数' else fit['ratios']
    sources = {str(p):sha(p) for p in [Path(__file__), API, fit_path, R/'预定试验约定.json']}
    sources.update({x['file']:x['sha256'] for x in fit['runtime_identity']['modules']})
    protocol = json.loads((R/'预定试验约定.json').read_text())
    sources.update({str(Path.cwd()/p):v for p,v in protocol['source_hashes'].items()})
    assert all(sha(p)==v for p,v in sources.items())
    started = time.time()
    result = dict(status='RUNNING', branch=args.branch, factors=factors, pid=os.getpid(),
        steps_completed=0, model_time_s=0., internal_observation_state_updates=0,
        initialization='各分支在t0同一Q/Z/开度/分水输入下，原自适应稳态认证；未用H16观测重置内部水位',
        boundary_data='历史已知边界，不代表未知未来预报', source_hashes=sources)
    save(out/'运行配置.json', result)
    local = None
    rows=[]
    try:
        local=api.build('A', n_factor=factors['n'], c_factor=factors['C'])
        m=local['model']
        warm=api.steady(local,t=0.)
        save(out/'起算稳态.json', warm)
        assert warm['converged'] and warm['accepted_final_exceeded_points']==0, '起算稳态未认证'
        save(out/'构模与初态核验.json', local['audit'])
        np.savez_compressed(out/'起算状态.npz', **{k:getattr(m,k) for k in api.STATE_KEYS})
        n_fixed=m._n_ik.copy(); C_fixed=m._Co.copy()
        result['numerical_step_retries']=0
        for day in range(30):
            new,retries=api.advance(local,day*86400.,(day+1)*86400.)
            rows.extend(new)
            assert np.array_equal(m._n_ik,n_fixed) and np.array_equal(m._Co,C_fixed)
            assert len(rows)==48*(day+1)
            result.update(steps_completed=len(rows),model_time_s=float(m.t),wall_seconds=time.time()-started)
            result['numerical_step_retries']+=retries
            save(out/'进度.json',result)
            print(json.dumps({k:result[k] for k in ['branch','steps_completed','model_time_s','wall_seconds']}),flush=True)
        frame=pd.DataFrame(rows)
        assert np.array_equal(frame.time_s.to_numpy(),np.arange(1800.,2592001.,1800.))
        frame.to_csv(out/'半小时复算结果.csv',index=False)
        # Internal observations are used here only, after every model step has finished.
        metrics=[]
        for target,metric in [('JZZ16_forebay_m','forebay_m'),('JZZ16_afterbay_m','afterbay_m')]:
            times,values=api.observed('JZZ16',metric)
            assert np.array_equal(times[1:],frame.time_s.to_numpy())
            frame[target+'_observed']=values[1:]
            frame[target+'_error']=frame[target]-values[1:]
            for label,mask in [('完整30天',frame.time_s>0),('前15天',frame.time_s<=1296000),('后15天保留检验',frame.time_s>1296000)]:
                errors=frame.loc[mask,target+'_error'].to_numpy()
                metrics.append(dict(branch=args.branch,target=target,period=label,samples=len(errors),
                    MAE_m=float(np.abs(errors).mean()),RMSE_m=float(np.sqrt(np.mean(errors**2))),
                    bias_m=float(errors.mean()),max_abs_m=float(np.abs(errors).max())))
        frame.to_csv(out/'内部水位检验.csv',index=False)
        pd.DataFrame(metrics).to_csv(out/'分时段误差指标.csv',index=False)
        np.savez_compressed(out/'末态.npz',**{k:getattr(m,k) for k in api.STATE_KEYS})
        result.update(status='COMPLETED',metrics=metrics,runtime_identity=api.runtime_identity(),
            runtime_attempt_audit=local['runtime_audit'],parameters_unchanged=True,
            source_hashes_unchanged=all(sha(p)==v for p,v in sources.items()),
            final_accepted_depth_excess_m=float((m.h_Ik-local['raw_heights']).max()))
        assert result['source_hashes_unchanged']
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        if rows: pd.DataFrame(rows).to_csv(out/'失败前已完成整日结果.csv',index=False)
        if local is not None:
            result.update(model_time_s=float(local['model'].t),runtime_attempt_audit=local['runtime_audit'],runtime_identity=api.runtime_identity())
        raise
    finally:
        result['wall_seconds']=time.time()-started
        save(out/'结果.json',result)

if __name__=='__main__':main()
