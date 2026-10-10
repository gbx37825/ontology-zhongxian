"""解析阻力、并联流量、满顶切换和状态重载核验；无参数率定。"""
import json,copy
import numpy as np
import pandas as pd
from 试验公共 import R,PipedreamHydraulicEngine,Forcing
from 并联结构物理修复 import attach_parallel_physics,build_engine_physics

def toy(depth,N=2,separate=False,drop=.1,backend='numba',K=.1):
    L=100.;b=5.1;H=5.1;n=.015;nk=N if separate else 1;count=1 if separate else N
    sj=pd.DataFrame([dict(id=0,name='up',z_inv=0.,h_0=depth,bc=True,storage='functional',a=0.,b=0.,c=1.,max_depth=np.inf),dict(id=1,name='down',z_inv=-drop,h_0=depth,bc=True,storage='functional',a=0.,b=0.,c=1.,max_depth=np.inf)])
    sl=[];links=[];junctions=[];info=[]
    for k in range(nk):
        sl.append(dict(id=k,name=f'box{k}',sj_0=0,sj_1=1,in_offset=0.,out_offset=0.,dx=L,n=n,shape='rect_closed',g1=H,g2=count*b,g3=.001,g4=0.,Q_0=0.,h_0=depth,ctrl=False,A_s=0.,A_c=0.,C=1.,C_uk=1e9,C_dk=1e9,j_0=2*k,j_1=2*k+1))
        junctions.extend([dict(id=2*k,k=k,h_0=depth,A_s=0.,z_inv=0.),dict(id=2*k+1,k=k,h_0=depth,A_s=0.,z_inv=-drop)])
        links.append(dict(id=k,k=k,j_0=2*k,j_1=2*k+1,dx=L,n=n,shape='rect_closed',g1=H,g2=count*b,g3=.001,g4=0.,Q_0=0.,ctrl=False,A_c=0.,C=1.,friction_method='cm'))
        info.append(dict(name=f'box{k}',superlink_id=k,shape='rect_closed',n_orifice=count,width=b,height=H,loss_coeff=K,physical_length=L,length=L,equivalent_length_added=0.,nc=n))
    c=dict(superjunctions=sj,superlinks=pd.DataFrame(sl),links=pd.DataFrame(links),junctions=pd.DataFrame(junctions),siphons_info=info,transects={});e=PipedreamHydraulicEngine(c,dt=5.,backend=backend);e.model._method='lsq';attach_parallel_physics(e,c)
    A=N*b*min(depth,H);P=N*(b+2*depth) if depth<H else 2*N*(b+H);Rh=A/P
    expected=np.sign(drop)*np.sqrt(abs(drop)/(n*n*L/(A*A*Rh**(4/3))+K/(2*9.81*A*A))) if drop else 0.
    kw=dict(H_bc=np.array([depth,depth-drop]),Q_in=np.zeros(2),u_o=None,dt=5.,num_iter=1)
    for _ in range(1600):e.model.step(**kw)
    q=float(e.model.Q_ik.sum());err=abs(q-expected)/max(abs(expected),1.)
    return dict(depth=depth,N=N,separate=separate,drop=drop,backend=backend,K=K,expected_Q=expected,actual_Q=q,relative_error=err,A=float(e.model.A_ik.sum()),true_A=A,actual_R=e.model._R_ik.tolist(),true_R=Rh,physical_volume_m3=float(np.dot(e.model.A_ik,e.model._dx_ik))),e

results=[]
for depth,drop in [(2.,.1),(5.1-1e-6,.1),(5.1,.1),(5.1+1e-6,.1),(7.,.1),(7.,-.1),(7.,0.)]:
    row,e=toy(depth,drop=drop);results.append(row);print(row,flush=True)
    assert row['relative_error']<1e-5,row
    assert abs(row['A']-row['true_A'])<1e-7
    assert abs(row['physical_volume_m3']-row['true_A']*100)<1e-5
for separate in [True]:
    row,e=toy(7.,separate=separate);results.append(row);print(row,flush=True);assert row['relative_error']<1e-5,row
row,e=toy(7.,backend='standard');results.append(row);print(row,flush=True);assert row['relative_error']<1e-5,row
# 当前105结构物理长度与K项显式性，原n与外部闸C由builder断言不变。
norm=json.loads((R/'03_运行参数/2025开度归一化上限.json').read_text());e,s,c=build_engine_physics(gate_max_normalization=norm);f=Forcing(e,s,c,positioned=True);m=e.model
ids=m._parallel_i;m._Q_ik[ids]=10.;m.save_state();m._parallel_native_class.link_coeffs(m,_dt=15.);old=m._b_ik.copy();m.link_coeffs(_dt=15.);actual=m._b_ik-old;expected=np.zeros_like(actual);expected[ids]=m._parallel_Ki*10/(2*m._A_ik[ids]);assert np.allclose(actual,expected,atol=1e-12,rtol=1e-12)
ma=copy.deepcopy(m);mb=copy.deepcopy(m);mb.load_state({k:getattr(m,k).copy() for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']});ma.step(**f.at(0));mb.step(**f.at(0));diff=max(float(abs(getattr(ma,k)-getattr(mb,k)).max()) for k in ['H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o']);assert diff<1e-12,diff
out=dict(status='PASS',toy_cases=results,explicit_K_coefficient_max_error=float(abs(actual-expected).max()),restart_next_step_max_difference=diff,structures=len(c['siphons_info']),n_unchanged=True,gate_C_unchanged=True,full_manifest=e.parallel_physics_manifest)
(R/'07_对比分析/并联结构解析核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('ALL_ANALYTICAL_CHECKS_PASS',flush=True)
