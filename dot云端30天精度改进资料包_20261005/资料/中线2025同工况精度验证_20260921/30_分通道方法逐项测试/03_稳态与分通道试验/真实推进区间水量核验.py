from 试验水力工具 import *
O=Path(__file__).resolve().parent;end=790200.;start=end-86400;count_start=end-21600
e,s,c,f,params=build();m=e.model
with np.load(O/'提前24小时恒边界稳态_状态.npz') as z:restore(m,{k:z[k] for k in STATE_KEYS})
m.t=start;m.iter_count=0;W=np.load(O.parent/'02_输入与测点核查/兰河测流到闸门控制体权重.npy').reshape(-1);ids=c['orifices'].index[c['orifices'].name.str.startswith('JZZ16-')].to_numpy();op=next(x for x in operators(e,s,c,['JZZ16']) if x['metric']=='flow_m3s');tot=np.zeros(4);retries=0
for t in np.arange(start,end,15.):
 pending=[(t,15.)]
 while pending:
  ts,dt=pending.pop(0);kwargs=f.at(ts);kwargs.update(dt=dt,num_iter=20,strict=True);H=m.H_j.copy();h=m.h_Ik.copy()
  try:m.step(**kwargs)
  except RuntimeError as ex:
   if 'did not converge' not in str(ex) or dt<=1.875:raise
   retries+=1;pending[:0]=[(ts,dt/2),(ts+dt/2,dt/2)];continue
  if ts>=count_start:
   dv=W@np.r_[m._A_sj*(m.H_j-H),m._E_Ik*(m.h_Ik-h)*dt];take=-W@np.r_[kwargs['Q_in'],kwargs['Q_0Ik']]*dt;qm=float(m.Q_ik[op['indices']]@np.array(op['weights']));qg=float(m.Q_o[ids].sum());tot+=np.array([qm*dt,qg*dt,take,dv])
obs=pd.read_csv(B/'共享输入/作者已核实修正观测层.csv');q=obs[(obs.station_code=='JZZ16')&(obs.metric=='flow_m3s')&(obs.time_s>=count_start)&(obs.time_s<=end)].sort_values('time_s');assert len(q)==13 and q.valid.all();obsmean=float(np.trapezoid(q.value,q.time_s)/21600)
r=dict(window_start_s=count_start,window_end_s=end,model_meter_mean=tot[0]/21600,model_gate_mean=tot[1]/21600,model_offtake_mean=tot[2]/21600,model_storage_rate=tot[3]/21600,model_continuity_residual=(tot[0]-tot[1]-tot[2]-tot[3])/21600,observed_meter_time_mean=obsmean,observed_meter_minus_model_meter=obsmean-tot[0]/21600,conditional_observed_gate_mean=obsmean-(tot[2]+tot[3])/21600,retries=retries,note='条件性过闸流量使用模型蓄变，不是独立实测；模型内连续性不能证明现场数据准确。')
(O/'兰河真实推进水量核验.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False),flush=True)
