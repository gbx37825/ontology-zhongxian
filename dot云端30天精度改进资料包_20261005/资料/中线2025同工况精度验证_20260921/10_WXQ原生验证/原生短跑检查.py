from pathlib import Path
import sys,json,time,traceback,hashlib,os,signal,csv
import numpy as np
R=Path('/out');sys.stdout=open(R/'原生运行日志.txt','w',buffering=1);sys.stderr=sys.stdout
started=time.time();report={'status':'RUNNING','phase':'import','completed_steps':0}
def save():
 (R/'原生短跑检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
def alarm(*args):raise TimeoutError('原生测试达到900秒时间上限')
signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
try:
 sys.path.insert(0,'/engine')
 import matplotlib;matplotlib.use('Agg')
 from src_hydro1d import Hydro1DimPreprocess as native
 report['engine_file']=native.__file__;report['engine_sha256']=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest();print('IMPORT_OK',native.__file__,flush=True)
 p=Path('/inputs/C00_实际开度负值修复_2025同工况30天.json');raw=json.loads(p.read_text());key,m=next(iter(raw.items())) if 'baseData'not in raw else ('model',raw)
 report['input_file']=p.name;report['input_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();report['model_container']=key
 # 文件外层是模型集合，选取其中一个模型；不更改数值或初始化设置。
 report['phase']='build';save();print('BUILD_START',flush=True)
 sim=native.processFdmSIMU_BASE(m);report['model_id']=str(sim.model_id);report['component_counts']={k:len(v) for k,v in sim.dict_AllComponent.items() if hasattr(v,'__len__')};print('BUILD_OK',report['component_counts'],flush=True)
 report['phase']='initialize';save();print('INIT_START',flush=True)
 native.processINITIALIZE(sim,m);print('INIT_OK',flush=True);report['initialize_seconds']=time.time()-started
 report['phase']='advance';save();station_rows=[];links=m['timeSeries']['seriesLink'];data=m['timeSeries']['seriesData'];arrays={k:np.asarray(v,float) for k,v in data.items()}
 for i in range(1440):
  elapsed=i*15
  for k,(typ,name) in links.items():
   a=arrays[k];v=float(np.interp(elapsed,a[:,0],a[:,1]));sim.dict_AllComponent[typ][name].updateBoundaryValue(v)
  sim.update1step(15);report['completed_steps']=i+1
  if (i+1)%120==0:
   rr=sim.getAllResult1Step();groups={}
   for name in m['baseData']['gates']:
    v=rr[name];station=name.split('-')[0];z=np.asarray(v['水位']).ravel();q=np.asarray(v['流量']).ravel();row=groups.setdefault(station,dict(station_code=station,time_s=(i+1)*15,forebay_m=float(z[0]),afterbay_m=float(z[-1]),flow_m3s=0.));row['flow_m3s']+=float(q[0])
   station_rows.extend(groups.values());save();print('STEP',i+1,flush=True)
 result=sim.getAllResult1Step()
 def safe(v):
  if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
  if isinstance(v,(tuple,list)):return [safe(x) for x in v]
  if isinstance(v,np.ndarray):return safe(v.tolist())
  if isinstance(v,np.generic):return v.item()
  return v
 values=safe(result);(R/'末步原生结果.json').write_text(json.dumps(values,ensure_ascii=False,indent=2,default=str))
 def numeric(v):
  if isinstance(v,dict):return [n for x in v.values() for n in numeric(x)]
  if isinstance(v,list):return [n for x in v for n in numeric(x)]
  return [float(v)] if isinstance(v,(int,float)) else []
 a=np.array(numeric(values));report['finite_result']=bool(np.isfinite(a).all());report['numeric_result_count']=len(a);report['simulated_seconds']=21600
 with (R/'原生6小时逐闸结果.csv').open('w',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=['station_code','time_s','forebay_m','afterbay_m','flow_m3s']);writer.writeheader();writer.writerows(station_rows)
 assert len(a)>0 and report['finite_result']
 report['status']='COMPLETED';report['phase']='done'
except BaseException as ex:
 report['status']='FAILED';report['error']=repr(ex);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 report['wall_seconds']=time.time()-started;save();print('REPORT',json.dumps(report,ensure_ascii=False,default=str),flush=True)
