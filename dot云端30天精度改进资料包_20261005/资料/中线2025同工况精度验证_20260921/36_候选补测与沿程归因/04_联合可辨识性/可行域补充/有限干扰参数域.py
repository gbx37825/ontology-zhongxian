"""仅两个C5先验中心的有界可行域补充；不读H4实测残差、不辨识工程参数。"""
from pathlib import Path
import ast,copy,hashlib,json,sys,time
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent;D=O.parent;R=D.parents[1];P=R/'35_局部率定前提复核'
sys.path[:0]=[str(P/'03_局部模型'),str(R/'04_脚本'),str(R/'23_五十成员参数同化/脚本')]
import 局部闸段 as api
from 并联结构物理修复 import build_engine_physics,attach_parallel_physics
from 常驻水力进程 import jit_evidence
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
configuration=json.loads((D/'预定试验约定.json').read_text());model_file=D/'联合段候选输入.json';mini=next(iter(json.loads(model_file.read_text()).values()));siphons=mini['baseData']['siphons'];middle_channels=configuration['middle_roughness_channels'];targets=configuration['targets']
raw,obs,norm,coeff=api.data();times,q=api.observed('JZZ4','flow_m3s');_,z=api.observed('JZZ5','afterbay_m')
source=D/'联合段可辨识性.py';tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
assert all(sha(p)==h for p,h in configuration['source_sha256'].items())
rule=dict(C5_centers=[1.05,1.10],n4_center=1.,C4_center=1.,maximum_baseline_trials=4,windows=configuration['windows'],selection='两窗均认证且已接受状态不超轮廓；选两窗最终最小轮廓余量较大者，同值选较小倍率；不读取目标实测残差',perturbation='三个参数在选定同一中心各相对±5%；原B02倍率界内；任一失败停止，不变更幅度或再换中心',acceptance='采用原adaptive的最终converged与最终接受状态检查；拒绝尝试单独留痕',observation_scale_m=.03,engineering_parameters_identified=False,source_sha256={**configuration['source_sha256'],str(Path(__file__)):sha(Path(__file__))})
dump(O/'预定有限范围.json',rule)
started=time.time();records=[]

def evaluate(window_index,ratios,label):
 for key,name in [('n4','POOL4_n'),('C4','JZZ4_C'),('C5','JZZ5_C')]:lo,hi=api.parameter_bounds()[name];assert lo<=ratios[key]<=hi
 w=configuration['windows'][window_index];local=build(float(w['end_s']),ratios);hold=api.mean_forcing(local,float(w['start_s']),float(w['end_s']));r=api.steady(local,kw=hold,max_seconds=60.)
 m=local['model'];excess=m.h_Ik-local['raw_heights'];valid=bool(r['converged'] and np.isfinite(m.h_Ik).all() and np.all(excess<=1e-7))
 values=np.array([m.h_Ik[p['indices']]@np.asarray(p['weights'])+p['offset'] for p in local['operators']])
 row=dict(label=label,window_index=window_index,ratios=ratios,certified_and_in_domain=valid,accepted_final_max_depth_excess_m=float(excess.max()),accepted_final_exceeded_points=int((excess>1e-7).sum()),water_levels_m=values.tolist(),attempt_audit=local['runtime_audit'],steady=r,numba=jit_evidence(),physical_audit=local['audit'])
 records.append(row);dump(O/'逐次试验记录.json',records);print(json.dumps({k:v for k,v in row.items() if k not in ['steady','numba','physical_audit']},ensure_ascii=False),flush=True)
 return row

centers=[]
for center in rule['C5_centers']:
 pair=[evaluate(i,dict(n4=1.,C4=1.,C5=center),'center') for i in range(2)]
 centers.append(dict(C5=center,both_valid=all(r['certified_and_in_domain'] for r in pair),minimum_accepted_clearance_m=min(-r['accepted_final_max_depth_excess_m'] for r in pair),records=pair))
valid=[c for c in centers if c['both_valid']]
if not valid:
 result=dict(status='NO_CERTIFIED_CENTER_IN_PREDECLARED_SET',centers=centers,not_proof_of_global_infeasibility=True,parameters_identified=False)
else:
 selected=sorted(valid,key=lambda c:(-c['minimum_accepted_clearance_m'],c['C5']))[0];center=selected['C5'];dump(O/'按几何选定中心.json',dict(C5=center,minimum_clearance_m=selected['minimum_accepted_clearance_m'],selection_rule=rule['selection']))
 completed=[];failed=None;Jac=[];nonlinearity=[]
 for wi in range(2):
  states={}
  for parameter in ['n4','C4','C5']:
   for sign in [-1,1]:
    ratios=dict(n4=1.,C4=1.,C5=center);ratios[parameter]*=1+sign*.05
    row=evaluate(wi,ratios,parameter+('_minus' if sign<0 else '_plus'));completed.append(row)
    if not row['certified_and_in_domain']:failed=row;break
    states[parameter,sign]=np.array(row['water_levels_m'])
   if failed:break
  if failed:break
  J=np.column_stack([(states[p,1]-states[p,-1])/(.1*(center if p=='C5' else 1.)) for p in ['n4','C4','C5']]);Jac.append(J)
  baseline=np.array(selected['records'][wi]['water_levels_m']);nonlinearity.append({p:(states[p,1]+states[p,-1]-2*baseline).tolist() for p in ['n4','C4','C5']})
 if failed:
  result=dict(status='PERTURBATION_INCOMPLETE',C5_center=center,centers=centers,failed_perturbation=failed,completed_perturbation_trials=len(completed),complete_window_matrices=[J.tolist() for J in Jac],no_center_or_amplitude_change=True,parameters_identified=False)
 else:
  J=np.vstack(Jac);scale=np.array([.05,.05,.05*center]);G=J*scale/.03;U,S,Vt=np.linalg.svd(G,full_matrices=False);cos=G.T@G/np.outer(np.linalg.norm(G,axis=0),np.linalg.norm(G,axis=0))
  result=dict(status='COMPLETED_CONDITIONAL_INFORMATION_TEST',C5_center=center,centers=centers,parameters=['n4','C4','C5'],raw_jacobian_m_per_multiplier=J.tolist(),relative_five_percent_scaled_matrix=G.tolist(),singular_values=S.tolist(),numerical_rank=int(np.linalg.matrix_rank(G)),condition_number=float(S[0]/S[-1]) if S[-1]>0 else None,weakest_relative_parameter_combination=Vt[-1].tolist(),column_cosine_similarity=cos.tolist(),smallest_normalized_response=float(S[-1]),single_window_ranks=[int(np.linalg.matrix_rank(x)) for x in Jac],symmetric_nonlinearity_m=nonlinearity,parameters_identified=False,C5_is_unconfirmed_prior=True,not_restore_original_center_pass=True)
  pd.DataFrame(J,index=[f'window{w}_{t[0]}' for w in range(2) for t in targets],columns=['n4','C4','C5']).to_csv(O/'条件性灵敏度矩阵.csv')
assert all(sha(p)==h for p,h in rule['source_sha256'].items())
result.update(source_hashes_unchanged=True,total_trials=len(records),wall_seconds=time.time()-started,no_measurement_residual_used=True)
dump(O/'有限干扰域结果.json',result);print(json.dumps({k:v for k,v in result.items() if k!='centers'},ensure_ascii=False),flush=True)
