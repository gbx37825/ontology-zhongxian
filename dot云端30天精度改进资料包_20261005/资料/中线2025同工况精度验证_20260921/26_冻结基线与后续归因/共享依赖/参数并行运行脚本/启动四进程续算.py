from pathlib import Path
import subprocess,os,json
root=Path.cwd();base=root/'outputs/中线2025同工况精度验证_20260921/23_五十成员参数同化'
records=[]
for code,old in [('P01','P01_50成员30天参数同化'),('P02','P02_50成员30天同集合无同化')]:
 name=code+'M4_50成员30天'+('参数同化' if code=='P01' else '无同化');log=base/(name+'.log')
 args=[str(root/'water/pipedream-runner/.venv/bin/python'),str(base/'脚本/参数同化运行.py'),'--name',name,'--days','30','--resume-from',str(base/(code+'R_50成员30天'+('参数同化续算' if code=='P01' else '无同化续算'))),'--workers','4']
 if code=='P02':args+=['--no-analysis']
 env={**os.environ,'NUMBA_CACHE_DIR':'/private/tmp/gate_fix_cache','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 with log.open('x') as h:p=subprocess.Popen(args,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env,cwd=root)
 records.append(dict(name=name,pid=p.pid,args=args))
(base/'四进程续算进程.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(records,ensure_ascii=False))
