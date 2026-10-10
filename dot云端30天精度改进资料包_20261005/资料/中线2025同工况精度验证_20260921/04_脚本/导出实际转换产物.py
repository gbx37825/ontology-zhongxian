from pathlib import Path
import json,hashlib
import numpy as np
from 试验公共 import *
from 并联结构物理修复 import build_engine_physics
R=Path(__file__).resolve().parents[1]
source=R/'02_输入/C01_目标开度独立假设_2025同工况30天.json'
limits=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text())
for mode in ['legacy','physical']:
 out=R/'02_输入/转换检查'/mode;out.mkdir(parents=True,exist_ok=True)
 e,s,c=build_engine_physics(source,mode=mode,gate_max_normalization=limits)
 for name in ['superjunctions','superlinks','junctions','links','orifices']:c[name].to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
 f=Forcing(e,s,c,positioned=True)
 np.savez_compressed(out/'求解时序数组.npz',time_s=f.times,H_bc=f.h,Q_in=f.q,Q_0Ik=f.q0,u_o=f.u)
 manifest=dict(source_json=str(source.relative_to(R)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),conversion_function='wxq_pipedream.wxq_to_superlink.parse_wxq_json',converter_sha256=hashlib.sha256((R/'09_运行源码/wxq_pipedream/wxq_to_superlink.py').read_bytes()).hexdigest(),mode=mode,solver=e.solver_identity,table_rows={k:len(c[k]) for k in ['superjunctions','superlinks','junctions','links','orifices']},runtime_modifications='parallel Pe/R and K hook' if mode=='physical' else 'legacy',positioned_offtakes=True,observation_initialization=False,note='这些CSV由WXQ JSON转换并施加已记录运行配置后导出，不是手工建模输入；动态时序及运行时阻力hook不能由静态CSV单独替代')
 (out/'转换来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(mode,manifest['table_rows'])
