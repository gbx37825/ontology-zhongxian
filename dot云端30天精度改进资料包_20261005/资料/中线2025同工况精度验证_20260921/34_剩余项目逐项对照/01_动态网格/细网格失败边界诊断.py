"""保留1445八段局部反向坐标试验的有界失败诊断，不接受异常状态，不改正式求解器。"""
from pathlib import Path
import json
O=Path(__file__).resolve().parent;p=O/'局部动态网格试验.py'
scope={'__file__':str(p)}
exec(compile(p.read_text().split('results=[];all_times=[]')[0],str(p),'exec'),scope)
np=scope['np'];case=scope['cases'][1];rows=[]
for label,dt,loss in [('原dt1',1.,1e-18),('loss1e-8',1.,1e-8),('loss1e-6',1.,1e-6),('loss1e-4',1.,1e-4),('原默认loss',1.,.20151121),('dt0.2',.2,1e-18),('dt0.1',.1,1e-18),('dt0.05',.05,1e-18),('dt0.01',.01,1e-18),('dt0.001',.001,1e-18)]:
 e,g=scope['make'](case,8,reset_initial=False);m=e.model;m._C_uk[:]=loss;m._C_dk[:]=loss
 error=None;i=-1
 try:
  for i in range(10):scope['step'](m,case,case['qin'][0],case['zout'][0],dt)
 except RuntimeError as ex:error=str(ex)
 # Numerically finite output is not acceptance. Huge internal flow or contour violation rejects this probe.
 admissible=bool(np.isfinite(m.h_Ik).all() and np.isfinite(m.Q_ik).all() and np.all(m.h_Ik>0) and np.all(m.h_Ik<g['height']) and np.max(abs(m.Q_ik))<10*case['qin'][0])
 rows.append(dict(label=label,dt=dt,runtime_end_loss=loss,attempted_steps=i+1,error=error,physically_admissible_at_end=admissible,max_internal_abs_Q=float(np.max(abs(m.Q_ik))),status='REJECTED' if error or not admissible else 'SHORT_PROBE_ONLY'))
(O/'细网格失败诊断.json').write_text(json.dumps(dict(link_id=1445,segments=8,rows=rows,note='参数扫查只用于排查局部失败，不改变原实验、门槛或源K；无错误但内部流量异常的状态同样拒绝。'),ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False,indent=2))
