"""只替换当前进程的构建入口，其余参数透传既有真实水力runner。"""
from pathlib import Path
import argparse,sys,json
import 运行水力对照 as runner
from 并联结构物理修复 import build_engine_physics
from 试验公共 import R
p=argparse.ArgumentParser(add_help=False);p.add_argument('--physics-mode',choices=['legacy','radius_only','physical'],default='physical');known,rest=p.parse_known_args()
if '--name' not in rest:raise ValueError('必须指定独立试验--name')
out=R/'06_结果'/rest[rest.index('--name')+1]
def build(*args,**kwargs):
    e,s,c=build_engine_physics(*args,mode=known.physics_mode,**kwargs)
    (out/'物理修复清单.json').write_text(json.dumps(e.parallel_physics_manifest,ensure_ascii=False,indent=2))
    print('PHYSICS_MODE',known.physics_mode,'N_STRUCTURES',len(c['siphons_info']),flush=True)
    return e,s,c
runner.build_engine=build
sys.argv=[sys.argv[0],*rest]
if __name__=='__main__':runner.main()
