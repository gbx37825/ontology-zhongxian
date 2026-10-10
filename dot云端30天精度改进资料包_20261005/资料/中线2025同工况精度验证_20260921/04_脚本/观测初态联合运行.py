"""仅t0的水面线和测流匹配初态候选；连续方程剩余量明示为所需蓄量变化，不伪称稳态。"""
from pathlib import Path
import sys,json,time,hashlib,copy,argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from 试验公共 import *
STATE=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
def snap(m):return {k:getattr(m,k).copy() for k in STATE}
def restore(m,s):m.load_state(s,compute_hydraulic_geometries=True);m.compute_storage_volumes()
def graph_x(e,origin):
    m=e.model;edges={}
    for a,b,length in zip(m._J_uk,m._J_dk,m._dx_k):edges[int(a),int(b)]=max(float(length),1.)
    for a,b in zip(m._J_uo,m._J_do):edges[int(a),int(b)]=1.
    rr,cc=zip(*edges);return dijkstra(csr_matrix((list(edges.values()),(rr,cc)),shape=(m.M,m.M)),directed=False,indices=origin)
def head_profile(e,obs,cap=2.):
    m=e.model;delta=np.zeros(m.M);count=np.zeros(m.M)
    for row in e.converted_model['orifices'].itertuples():
        code=row.name.split('-')[0]
        # 同站多孔不能重复给权，以下仅在首次遇到节点时读取。
        for field,j in [('forebay_m',row.sj_0),('afterbay_m',row.sj_1)]:
            if count[j]:continue
            key=(code,field,0)
            if key in obs.index and bool(obs.loc[key,'valid']):delta[j]=np.clip(float(obs.loc[key,'value'])-m.H_j[j],-cap,cap);count[j]=1
    delta[m.bc]=0.;xj=graph_x(e,e.node_index['NODE-main-0']);known=(count>0)|m.bc;known[e.node_index['NODE-main-0']]=True;order=np.argsort(xj[known]);delta=np.interp(xj,xj[known][order],delta[known][order]);delta[m.bc]=0.
    dh=np.zeros(m.NIk)
    for k in range(m.NK):
        ids=np.where(m._kI==k)[0];x=m.x_Ik[ids];r=(x-x.min())/max(x.max()-x.min(),1e-9);dh[ids]=(1-r)*delta[m._J_uk[k]]+r*delta[m._J_dk[k]]
    s=snap(m);s['H_j']+=delta;s['h_Ik']=np.maximum(s['h_Ik']+dh,.01);restore(m,s)
    return dict(observed_sides=int(count.sum()),max_head_increment_m=float(abs(delta).max()))
def flux_residual(m,kw):
    result=np.r_[kw['Q_in'],kw['Q_0Ik']].copy();M=m.M
    for name,u,v in [('Q_ik',m._Ik+M,m._Ip1k+M),('Q_uk',m._J_uk,m._I_1k+M),('Q_dk',m._I_Np1k+M,m._J_dk),('Q_o',m._J_uo,m._J_do)]:
        q=getattr(m,name);np.add.at(result,u,-q);np.add.at(result,v,q)
    return result

def initialize_flow(e,forcing,ops,obs,out):
    m=e.model;kw=forcing.at(0);xj=graph_x(e,int(np.argmax(kw['Q_in'])));sign=np.sign(xj[m._J_dk]-xj[m._J_uk]);xh=xj[m._J_uk[m._kI]]+sign[m._kI]*m.x_Ik;xq=(xh[m._Ik]+xh[m._Ip1k])/2
    anchors=[]
    for op in ops:
        key=(op['station'],'flow_m3s',0)
        if key not in obs.index or not bool(obs.loc[key,'valid']):continue
        w=np.abs(op['weights']);xc=float(xq[op['indices']]@w/sum(w));anchors.append((xc,op,float(obs.loc[key,'value'])))
    anchors.sort(key=lambda z:z[0]);xx=np.array([v[0] for v in anchors]);target=np.array([v[2]-observe(m,v[1]) for v in anchors]);n=len(anchors);basis=np.zeros((len(xq),n))
    for j in range(n):
        v=np.zeros(n);v[j]=1.;basis[:,j]=np.interp(xq,np.r_[0,xx],np.r_[0,v])*sign[m._ki]
    A=np.array([basis[op['indices']].T@np.asarray(op['weights']) for _,op,_ in anchors]);coef,res,rank,sv=np.linalg.lstsq(A,target,rcond=None)
    if rank!=n:raise ValueError('流量插值约束秩不足')
    def correction(x):return np.interp(x,np.r_[0,xx],np.r_[0,coef])
    s=snap(m);s['Q_ik']+=basis@coef;s['Q_uk']+=correction(xj[m._J_uk])*sign;s['Q_dk']+=correction(xj[m._J_dk])*sign
    m.orifice_hydraulic_geometry(u=kw['u_o']);gates=e.converted_model['orifices'].copy();gates['station']=gates.name.str.extract(r'^(JZZ\d+)');gate_rows=[]
    for code,g in gates.groupby('station'):
        ids=g.id.to_numpy(int);up=int(g.iloc[0].sj_0);down=int(g.iloc[0].sj_1);direction=float(np.sign(xj[down]-xj[up]));qold=float(m.Q_o[ids].sum());qtotal=qold+direction*float(correction((xj[up]+xj[down])/2));area=m._Ao[ids].copy()
        if area.sum()>1e-12:s['Q_o'][ids]=qtotal*area/area.sum();status='实际过流面积分配'
        else:s['Q_o'][ids]=0.;status='全关闭无面积，置0并记录冲突'
        gate_rows.append(dict(station_code=code,old_gate_flow=qold,requested_gate_flow=qtotal,new_gate_flow=float(s['Q_o'][ids].sum()),status=status))
    restore(m,s);rows=[dict(station_code=op['station'],value=y,initial_meter_flow=observe(m,op),error=observe(m,op)-y,model_distance=x,channel=op['channel']) for x,op,y in anchors]
    pd.DataFrame(rows).to_csv(out/'流量锚点拟合.csv',index=False);pd.DataFrame(gate_rows).to_csv(out/'初始门流分配.csv',index=False)
    return dict(anchor_count=n,rank=int(rank),condition_number=float(sv[0]/sv[-1]),max_meter_error=float(max(abs(r['error']) for r in rows)),initial_profile_method='预热流量差沿模型连通距离插值，44个观测为约束；未闭合站仅插值通过，不使用本站Q作锚点')

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--days',type=float,default=.25);p.add_argument('--all-sides',action='store_true');a=p.parse_args();out=R/'06_结果'/a.name;out.mkdir(exist_ok=False);start=time.time()
    norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());e,source,c=build_engine(gate_max_normalization=norm);f=Forcing(e,source,c,positioned=True);m=e.model
    for name,value in norm.items():assert abs(float(c['orifices'].iloc[e.gate_index[name]].y_max)-value)<1e-12
    warm=warmup(e,source,f);(out/'预热核验.json').write_text(json.dumps(warm,ensure_ascii=False,default=str));print('WARMUP_OK',flush=True)
    # 原生装载后下一步等价核验，不推进本候选模型。
    b=copy.deepcopy(m);d=copy.deepcopy(m);restore(d,snap(d));b.step(**f.at(0));d.step(**f.at(0));restart_diff={k:float(abs(getattr(b,k)-getattr(d,k)).max()) for k in STATE};assert max(restart_diff.values())<1e-12
    ops,audit=flow_operators(e,source,c);print('OPERATORS',len(ops),flush=True)
    initfile=R/'02_输入'/('作者已核实修正观测层.csv' if a.all_sides else '仅作者88侧初始化观测.csv');measured=pd.read_csv(R/'02_输入/作者已核实修正观测层.csv');water=pd.read_csv(initfile).set_index(['station_code','metric','time_s']);obs=measured.set_index(['station_code','metric','time_s'])
    v0=hydraulic_volume(m);qwarm=snap(m);r0=flux_residual(m,f.at(0));hd=head_profile(e,water);fd=initialize_flow(e,f,ops,obs,out);v1=hydraulic_volume(m);r1=flux_residual(m,f.at(0));before=snap(m)
    pd.DataFrame(dict(kind=['superjunction']*m.M+['internal_junction']*m.NIk,index=list(range(m.M))+list(range(m.NIk)),warmup_storage_rate=r0,initial_required_storage_rate=r1)).to_csv(out/'连续方程隐含蓄量变化率.csv',index=False)
    config=dict(name=a.name,days=a.days,water_layer=str(initfile),time_used_s=0,online_assimilation=False,online_lateral_inflow_modification=False,gate_max_normalization_consumed=True,flow_observations=len(ops),head=hd,flow=fd,delta_volume_m3=v1-v0,initial_required_storage_abs_sum_m3s=float(abs(r1).sum()),initial_required_storage_abs_max_m3s=float(abs(r1).max()),warmup_storage_abs_sum_m3s=float(abs(r0).sum()),restart_next_step_max_differences=restart_diff)
    config['sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'04_脚本/试验公共.py',DEFAULT_MODEL,initfile,R/'02_输入/作者已核实修正观测层.csv',R/'03_运行参数/2025开度归一化上限.json']};(out/'初态构造清单.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print('INITIAL',config,flush=True)
    groups={}
    for row in c['orifices'].itertuples():groups.setdefault(row.name.split('-')[0],[row.sj_0,row.sj_1,[]])[2].append(row.id)
    raw=[];flows=[];shocks=[]
    def record(t):
        for code,(up,down,ids) in groups.items():raw.append(dict(station_code=code,time_s=t,forebay_m=m.H_j[up],afterbay_m=m.H_j[down],flow_m3s=m.Q_o[ids].sum()))
        for op in ops:flows.append(dict(station_code=op['station'],time_s=t,meter_flow=observe(m,op)))
    record(0);np.savez_compressed(out/'初态_0h.npz',**snap(m));steps=int(a.days*86400/15)
    for i in range(steps):
        t=i*15;prev=snap(m) if i<4 else None;m.step(**f.at(t));now=t+15
        if not all(np.isfinite(getattr(m,k)).all() for k in STATE):raise FloatingPointError(f'nonfinite at {now}')
        if i<4:
            shocks.append(dict(time_s=now,max_head_step_m=float(abs(m.H_j-prev['H_j']).max()),max_gate_step_m3s=float(abs(m.Q_o-prev['Q_o']).max()),max_internal_flow_step_m3s=float(abs(m.Q_ik-prev['Q_ik']).max()),head_max_departure_initial_m=float(abs(m.H_j-before['H_j']).max())))
            if i==0:pd.DataFrame([dict(station_code=code,initial_flow=float(before['Q_o'][ids].sum()),flow_after15s=float(m.Q_o[ids].sum()),head_drop_initial=float(before['H_j'][up]-before['H_j'][down])) for code,(up,down,ids) in groups.items()]).to_csv(out/'首步门流冲击.csv',index=False)
        if now%1800==0:record(now)
        if now in [21600,86400,259200,518400]:np.savez_compressed(out/f'状态_{int(now/3600)}h.npz',**snap(m));print('HOUR',now/3600,flush=True)
    pd.DataFrame(raw).to_csv(out/'逐闸水位流量.csv',index=False);pd.DataFrame(flows).to_csv(out/'真实测流断面流量.csv',index=False);pd.DataFrame(shocks).to_csv(out/'首分钟冲击.csv',index=False)
    long=pd.DataFrame(raw).melt(id_vars=['station_code','time_s'],value_vars=['forebay_m','afterbay_m'],var_name='metric',value_name='sim');paired=long.merge(measured,on=['station_code','time_s','metric'],validate='one_to_one');paired=paired[paired.valid].copy();paired['error']=paired.sim-paired.value;paired.to_csv(out/'逐点水位评分.csv',index=False)
    scores=[]
    for h in [0,6,24,72,144]:
        for scope in ['全118侧','作者88侧']:
            data=paired[(paired.time_s==h*3600)&((paired.pdf_comparable) if scope=='作者88侧' else True)]
            if len(data):scores.append(dict(hour=h,scope=scope,samples=len(data),MAE=float(data.error.abs().mean()),max_abs=float(data.error.abs().max()),bias=float(data.error.mean())))
    pd.DataFrame(scores).to_csv(out/'关键时刻误差.csv',index=False);summary=dict(status='COMPLETED',steps=steps,days=a.days,wall_seconds=time.time()-start);(out/'运行摘要.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
