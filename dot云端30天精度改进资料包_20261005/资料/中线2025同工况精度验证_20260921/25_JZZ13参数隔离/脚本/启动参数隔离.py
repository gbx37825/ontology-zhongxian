from pathlib import Path
import subprocess,os,json
root=Path.cwd();R=root/'outputs/中线2025同工况精度验证_20260921';O=R/'25_JZZ13参数隔离';records=[]
for name,kind in [('G01_仅更新糙率至第1天','n'),('G02_仅更新闸流系数至第1天','C')]:
 args=[str(root/'water/pipedream-runner/.venv/bin/python'),str(O/'脚本/参数隔离运行.py'),'--name',name,'--days','1','--workers','4','--resume-from',str(R/'24_参数同化根因核查/F01_拓扑分组修复六小时'),'--update-kind',kind]
 with (O/(name+'.log')).open('x') as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'NUMBA_CACHE_DIR':'/private/tmp/gate_fix_cache','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
 records.append(dict(name=name,pid=p.pid,args=args))
(O/'隔离试验进程.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(records)
