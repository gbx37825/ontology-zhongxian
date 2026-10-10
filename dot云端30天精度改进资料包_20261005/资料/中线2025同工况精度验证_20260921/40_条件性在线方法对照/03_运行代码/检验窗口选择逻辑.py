"""无水力求解的窗口逻辑补测，不把合成输入当作本月新增工况。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from 在线窗口与观测 import choose_window

class Inputs:
    def __init__(self,kind):
        self.times=np.arange(0.,43200.+1,1800.)
        self.q=np.full(len(self.times),100.);self.u=np.full(len(self.times),.5)
        if kind=='past_change':self.q[self.times>=10800]=103.
        if kind=='recent_change':self.u[self.times>=19800]=.6
        self.h=np.full(len(self.times),10.);self.bleeder_values={'take':np.ones(len(self.times))}
        self.engine=SimpleNamespace(model=SimpleNamespace(bc=np.array([False,True])),node_index={'NODE-main-0':0})
        self.queried=[]
    def at(self,t):
        self.queried.append(t)
        return dict(H_bc=np.array([np.nan,np.interp(t,self.times,self.h)]),Q_in=np.array([np.interp(t,self.times,self.q),0.]),u_o=np.array([np.interp(t,self.times,self.u)]))

records=[]
for kind,expected in [('constant',12),('past_change',6),('recent_change',2)]:
    forcing=Inputs(kind);x=choose_window(forcing,21600.,np.ones(1),12)
    assert x['planned_cycles']==expected and max(forcing.queried)<=21600.
    # Numeric future values must not influence a selection at the current start.
    future=forcing.times>21600;forcing.q[future]=1e6;forcing.u[future]=-1e6;forcing.h[future]=1e6;forcing.bleeder_values['take'][future]=1e6
    y=choose_window(forcing,21600.,np.ones(1),12);assert x==y
    records.append(dict(case=kind,planned_hours=expected*.5,future_mutation_does_not_change_choice=True,maximum_queried_time_s=max(forcing.queried)))
short=choose_window(Inputs('constant'),21600.,np.ones(1),1)
assert short['planned_cycles']==12 and short['actual_cycles']==1 and short['truncated_at_issue'] and short['window_end_s']==23400
out=Path(__file__).resolve().parents[1]/'05_先行核验/窗口选择逻辑自检.json'
source=Path(__file__).with_name('在线窗口与观测.py')
result=dict(status='PASSED',hydraulic_simulation_started=False,scope='仅选择逻辑及未来输入隔离；不代表本月实际出现3h/6h窗口或验证其精度',checks=records,final_window_truncation_passed=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
