"""将固定的局部C候选带回全线；不重新选参、不修改实测或冻结基线。"""
from pathlib import Path
import sys,json,time,hashlib,os,argparse,traceback,csv
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';M00=R/'34_剩余项目逐项对照/02_输入敏感性/M00_原输入完整30天';sys.path[:0]=[str(R/'30_分通道方法逐项测试/03_稳态与分通道试验')]
from 试验水力工具 import build,advance,STATE_KEYS
from 常驻水力进程 import jit_evidence

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shaar(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def dump(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
p=argparse.ArgumentParser();p.add_argument('--case',choices=['X00','X06','X16','X616'],required=True);a=p.parse_args();names={'X00':'X00_原系数含测点30天','X06':'X06_仅转移十二里河系数30天','X16':'X16_仅转移兰河系数30天','X616':'X616_同时转移两闸系数30天'};station_sets={'X00':[],'X06':['JZZ6'],'X16':['JZZ16'],'X616':['JZZ6','JZZ16']};d=O/names[a.case];d.mkdir(exist_ok=False);started=time.time();completed=0
try:
 results={};source_hash={}
 for st in ['JZZ6','JZZ16']:
  path=O.parent/'04_短闸段系数辨识'/f'{st}_short/结果.json';result=json.loads(path.read_text());assert result['status']=='COMPLETED';results[st]=result;source_hash[st]=sha(path)
 e,s,c,f,params=build();m=e.model;initial_hash={k:shaar(getattr(m,k))for k in STATE_KEYS};original_params={k:getattr(m,k).copy()for k in ['_n_ik','_n_uk','_n_dk','_Co','_C_uk','_C_dk']};old=json.loads((M00/'运行配置.json').read_text());assert initial_hash==old['initial_state_sha256'];assert all(shaar(original_params[k])==v for k,v in old['parameter_sha256'].items());changes=[];all_ids=[]
 for st in station_sets[a.case]:
  ids=c['orifices'].index[c['orifices'].name.str.startswith(st+'-')].to_numpy(int);factor=results[st]['factor'];m._Co[ids]*=factor;assert np.array_equal(m._Co[ids],np.array(results[st]['fitted_C']));assert (m._Co[ids]>0).all() and (m._Co[ids]<=1).all();all_ids+=ids.tolist()
  for i in ids:changes.append(dict(gate=c['orifices'].iloc[i]['name'],index=int(i),base_C=float(original_params['_Co'][i]),factor=factor,applied_C=float(m._Co[i])))
 assert initial_hash=={k:shaar(getattr(m,k))for k in STATE_KEYS};changed_ids=np.flatnonzero(m._Co!=original_params['_Co']).tolist();assert changed_ids==sorted(all_ids)
 phash={k:shaar(getattr(m,k))for k in original_params};assert all(np.array_equal(getattr(m,k),v)for k,v in original_params.items()if k!='_Co')
 opfile=B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/观测算子与留出.json';ops=json.loads(opfile.read_text());assert len(ops)==43 and sum(x['assimilated']for x in ops)==34;assert len({(x['station'],x['metric'])for x in ops})==43
 observed=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv').set_index(['station_code','metric','time_s']);groups={st:(int(g.iloc[0].sj_0),int(g.iloc[0].sj_1),g.id.to_numpy(int))for st,g in c['orifices'].assign(st=c['orifices'].name.str.split('-').str[0]).groupby('st')}
 cfg=dict(name=names[a.case],pid=os.getpid(),members=1,assimilation=False,days=30,total_cycles=1440,start_s=0,end_s=2592000,parameter_changes=changes,source_result_sha256=source_hash,source_result_paths={st:str(O.parent/'04_短闸段系数辨识'/f'{st}_short/结果.json')for st in results},input_sha256=sha(B/'共享输入/C07.json'),initial_sha256=sha(B/'共享输入/初始化后状态_0s.npz'),initial_state_sha256=initial_hash,parameter_sha256=phash,operator_sha256=sha(opfile),script_sha256=sha(Path(__file__)),strict=True,max_iterations=20,min_failed_step_s=1.875,dt=15.,sample_step=1800,sensor_groups={'旧参与同化测点':34,'旧留出检验测点':9},note='此轮全部无同化；43实际位置是冻结候选安装位置算子，不表示历史仪器身份已完全核实。后15天不再选择参数。')
 dump(d/'运行配置.json',cfg);dump(d/'进度.json',dict(status='RUNNING',pid=os.getpid(),completed_cycles=0,total_cycles=1440,wall_seconds=time.time()-started));retries=0;maxh=maxq=0.
 with (d/'半小时水位流量.csv').open('w',newline='')as hn,(d/'实际位置水位.csv').open('w',newline='')as hs:
  wn=csv.DictWriter(hn,fieldnames=['time_s','station','forebay_m','afterbay_m','gate_flow_m3s']);wn.writeheader();ws=csv.DictWriter(hs,fieldnames=['time_s','station','metric','mapped_sim','observed','valid','group','chainage_m']);ws.writeheader()
  for k in range(1440):
   st=advance(m,f,k*1800,(k+1)*1800);retries+=st[0];maxh=max(maxh,st[1]);maxq=max(maxq,st[2]);t=(k+1)*1800
   for station,(ju,jd,gi)in groups.items():wn.writerow(dict(time_s=t,station=station,forebay_m=float(m.H_j[ju]),afterbay_m=float(m.H_j[jd]),gate_flow_m3s=float(m.Q_o[gi].sum())))
   for op in ops:
    ob=observed.loc[(op['station'],op['metric'],t)];ws.writerow(dict(time_s=t,station=op['station'],metric=op['metric'],mapped_sim=float(np.dot(getattr(m,op['state'])[op['indices']],op['weights'])+op.get('offset',0)),observed=float(ob['value']),valid=bool(ob['valid']),group='旧参与同化测点'if op['assimilated']else'旧留出检验测点',chainage_m=op['chainage']))
   completed=k+1
   if completed%12==0:
    hn.flush();hs.flush();status=dict(status='RUNNING',pid=os.getpid(),completed_cycles=completed,total_cycles=1440,days=completed/48,retries=retries,wall_seconds=time.time()-started,max_head_residual_m=float(maxh),max_gate_flow_residual_m3s=float(maxq));dump(d/'进度.json',status);dump(d/'Numba实际执行证据.json',jit_evidence());print(json.dumps(status),flush=True)
   if a.case=='X00'and completed==144:
    new=pd.read_csv(d/'半小时水位流量.csv');oldrows=pd.read_csv(M00/'半小时水位流量.csv');oldrows=oldrows[oldrows.time_s<=259200].reset_index(drop=True);assert new.equals(oldrows);dump(d/'前三天对M00复现.json',dict(exact_equal=True,rows=len(new),max_abs_difference=0.))
 assert all(phash[key]==shaar(getattr(m,key))for key in phash);assert all(np.isfinite(getattr(m,key)).all()for key in STATE_KEYS);assert sha(B/'共享输入/C07.json')==cfg['input_sha256'];assert sha(B/'共享输入/初始化后状态_0s.npz')==cfg['initial_sha256'];np.savez_compressed(d/'末态.npz',**{key:getattr(m,key)for key in STATE_KEYS})
 if a.case=='X00':
  new=pd.read_csv(d/'半小时水位流量.csv');oldrows=pd.read_csv(M00/'半小时水位流量.csv');assert new.equals(oldrows);dump(d/'完整30天对M00复现.json',dict(exact_equal=True,rows=len(new),max_abs_difference=0.))
 status.update(status='COMPLETED',parameters_fixed=True,finite_state=True,input_unchanged=True,initial_unchanged=True,wall_seconds=time.time()-started);dump(d/'进度.json',status);print(json.dumps(status),flush=True)
except BaseException as ex:dump(d/'失败.json',dict(error=repr(ex),traceback=traceback.format_exc(),completed_cycles=completed,pid=os.getpid()));raise
