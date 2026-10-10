from pathlib import Path
import os,subprocess,json
root=Path(__file__).resolve().parents[4];o=Path(__file__).resolve().parents[1];records=[]
for name,source,flag in [('F03_拓扑分组修复30天','F01_拓扑分组修复六小时',[]),('F04_拓扑分组匹配无同化30天','F02_拓扑分组匹配无同化六小时',['--no-analysis'])]:
 args=[str(root/'water/pipedream-runner/.venv/bin/python'),str(o/'脚本/拓扑修复试验运行.py'),'--name',name,'--days','30','--workers','4','--resume-from',str(o/source)]+flag
 with (o/(name+'.log')).open('x') as f:p=subprocess.Popen(args,cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'NUMBA_CACHE_DIR':'/private/tmp/gate_fix_cache','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
 records.append(dict(name=name,pid=p.pid,args=args))
(o/'30天对照进程.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(records)
