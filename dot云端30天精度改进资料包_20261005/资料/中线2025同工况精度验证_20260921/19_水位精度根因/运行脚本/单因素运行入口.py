from pathlib import Path
import sys,argparse,importlib.util
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'17_求解器修复/运行脚本'))
import 试验公共
import 运行水力对照 as runner
sys.path.insert(0,str(R/'04_脚本'))
from 并联结构物理修复 import build_engine_physics
p=argparse.ArgumentParser(add_help=False);p.add_argument('--physical',action='store_true');known,rest=p.parse_known_args()
if known.physical:
 def build(*args,**kwargs):
  e,s,c=build_engine_physics(*args,mode='physical',**kwargs)
  out=R/'06_结果'/rest[rest.index('--name')+1]
  import json
  (out/'物理转换清单.json').write_text(json.dumps(e.parallel_physics_manifest,ensure_ascii=False,indent=2))
  return e,s,c
 runner.build_engine=build
sys.argv=[sys.argv[0]]+rest
runner.main()
