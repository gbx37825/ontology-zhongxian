"""只复现原第二窗基准；原build函数以AST原样加载，不运行原文件顶层14组循环。"""
from pathlib import Path
import ast,copy,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'35_局部率定前提复核'
sys.path[:0]=[str(P/'03_局部模型'),str(R/'04_脚本')]
import 局部闸段 as api
from 并联结构物理修复 import build_engine_physics,attach_parallel_physics
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
configuration=json.loads((O/'预定试验约定.json').read_text());model_file=O/'联合段候选输入.json'
mini=next(iter(json.loads(model_file.read_text()).values()));siphons=mini['baseData']['siphons']
middle_channels=configuration['middle_roughness_channels'];targets=configuration['targets']
raw,obs,norm,coeff=api.data();times,q=api.observed('JZZ4','flow_m3s');_,z=api.observed('JZZ5','afterbay_m')
source=O/'联合段可辨识性.py';tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
assert all(sha(p)==h for p,h in configuration['source_sha256'].items())
local=build(586800.,dict(n4=1.,C4=1.,C5=1.));hold=api.mean_forcing(local,565200.,586800.)
result=api.steady(local,kw=hold,max_seconds=60.)
m=local['model'];excess=m.h_Ik-local['raw_heights'];locations=[]
for name,meta in local['converted']['channel_discretization'].items():
 if name not in mini['baseData']['channels']:continue
 for p in meta['sections']:
  if p['junction_id']!=57:continue
  a,b=p['source_interval'];sa=mini['baseData']['sections'][a];sb=mini['baseData']['sections'][b];x=(1-p['fraction'])*sa['location']+p['fraction']*sb['location']
  original=[{'section_key':k,**v} for k,v in raw['baseData']['sections'].items() if abs(v['location']-x)<1e-6]
  locations.append(dict(channel=name,junction=57,chainage_m=x,section_bottom_m=p['z_inv'],original_depth_limit_m=p['height'],original_contour_top_m=p['z_inv']+p['height'],last_accepted_depth_m=float(m.h_Ik[57]),last_accepted_stage_m=float(m.h_Ik[57]+m._z_inv_Ik[57]),last_accepted_excess_m=float(excess[57]),original_sections=original))
out=dict(reproduction='仅第二窗基准同参一次证据复现；非新参数试验',start_s=565200.,end_s=586800.,ratios=dict(n4=1.,C4=1.,C5=1.),build_definition_source=str(source),build_source_sha256=sha(source),harness_sha256=sha(Path(__file__)),original_sources_unchanged=all(sha(p)==h for p,h in configuration['source_sha256'].items()),steady=result,runtime_attempt_audit=local['runtime_audit'],last_accepted_max_depth_excess_m=float(excess.max()),last_accepted_exceeded_points=int((excess>1e-7).sum()),locations=locations)
(O/'第二窗基准完整预热返回.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k!='steady'},ensure_ascii=False,indent=2));print(json.dumps({k:result[k] for k in ['status','converged','stop_reason','steps_run','accepted_steps','rejected_steps','verification_steps_run','numerical_errors','history']},ensure_ascii=False,indent=2))
