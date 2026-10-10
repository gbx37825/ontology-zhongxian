"""独立进程物理修复：并联单孔湿周、真实管长、显式局部阻力。冻结09源码不变。"""
from pathlib import Path
import types,copy,hashlib
import numpy as np
from 试验公共 import build_engine as _legacy_build, PipedreamHydraulicEngine, R

def _correct_section(self,A,Pe,Rh,ids,N,b,H):
    area=A[ids];y=np.clip(area/(N*b),0,H)
    perimeter=N*(b+2*y)
    full=area>=N*b*H*(1-1e-12)
    perimeter[full]=2*N[full]*(b[full]+H[full])
    perimeter[area<=0]=0
    Pe[ids]=perimeter;Rh[ids]=np.divide(area,perimeter,out=np.zeros_like(area),where=perimeter>0)

def _link_geometry(self):
    self._parallel_native_class.link_hydraulic_geometry(self)
    _correct_section(self,self._A_ik,self._Pe_ik,self._R_ik,self._parallel_i,self._parallel_Ni,self._parallel_bi,self._parallel_Hi)
def _up_geometry(self,*args,**kwargs):
    self._parallel_native_class.upstream_hydraulic_geometry(self,*args,**kwargs)
    _correct_section(self,self._A_uk,self._Pe_uk,self._R_uk,self._parallel_k,self._parallel_Nk,self._parallel_bk,self._parallel_Hk)
def _down_geometry(self,*args,**kwargs):
    self._parallel_native_class.downstream_hydraulic_geometry(self,*args,**kwargs)
    _correct_section(self,self._A_dk,self._Pe_dk,self._R_dk,self._parallel_k,self._parallel_Nk,self._parallel_bk,self._parallel_Hk)
def _link_coeffs(self,*args,**kwargs):
    self._parallel_native_class.link_coeffs(self,*args,**kwargs)
    # g*A*h_local = K*Q*|Q|/(2*A)，本线性化系数乘Q；负流量自然反号。
    ids=self._parallel_i;A=self._A_ik[ids];Q=self._Q_ik[ids]
    term=np.divide(self._parallel_Ki*np.abs(Q),2*A,out=np.zeros_like(A),where=A>0)
    self._b_ik[ids]+=term

def attach_parallel_physics(e,c,mode='physical'):
    """c须已具有对应管长；逐次几何重算后恢复Pe/R，每步显式增加K阻力。"""
    if mode=='legacy':return e
    if mode not in {'radius_only','physical'}:raise ValueError(mode)
    m=e.model;rows=c['siphons_info'];k=[];N=[];b=[];H=[];ii=[];NN=[];bb=[];HH=[];KK=[]
    for row in rows:
        if row['shape']!='rect_closed':raise ValueError('当前模块只覆盖本工况105个矩形siphon')
        ki=int(row['superlink_id']);ids=np.flatnonzero(m._ki==ki);count=int(row['n_orifice']);width=float(row['width']);height=float(row['height'])
        if np.any(m._ctrl[ids]) or np.any(getattr(m,'_Sf_method_ik',np.zeros_like(m._n_ik,dtype=int))[ids]!=0):raise ValueError('不允许与既有内部控制或非Manning摩阻混用')
        k.append(ki);N.append(count);b.append(width);H.append(height);ii.extend(ids.tolist());NN.extend([count]*len(ids));bb.extend([width]*len(ids));HH.extend([height]*len(ids))
        # 当前105个结构均1段。若日后多段，按段长分配K，总和严格为整座K。
        weights=m._dx_ik[ids]/m._dx_ik[ids].sum();KK.extend((float(row['loss_coeff'])*weights).tolist())
    m._parallel_native_class=type(m)
    for name,value in {'k':k,'Nk':N,'bk':b,'Hk':H,'i':ii,'Ni':NN,'bi':bb,'Hi':HH,'Ki':KK}.items():setattr(m,'_parallel_'+name,np.asarray(value,dtype=int if name in {'k','i'} else float))
    m.link_hydraulic_geometry=types.MethodType(_link_geometry,m);m.upstream_hydraulic_geometry=types.MethodType(_up_geometry,m);m.downstream_hydraulic_geometry=types.MethodType(_down_geometry,m)
    if mode=='physical':m.link_coeffs=types.MethodType(_link_coeffs,m)
    m.link_hydraulic_geometry();m.upstream_hydraulic_geometry();m.downstream_hydraulic_geometry()
    e.solver_identity={**e.solver_identity,'parallel_geometry_revision':'parallel_rect_roof_v1_'+mode}
    return e

def build_engine_physics(*args,mode='physical',**kwargs):
    """与公共build_engine兼容，返回(e,source,converted)；不修改公共/冻结模块。"""
    if kwargs.get('restart') is not None:raise ValueError('物理模式须独立预热，不接受旧几何预热缓存')
    e,source,c=_legacy_build(*args,**kwargs)
    before_n=e.model._n_ik.copy();before_C=e.model._Co.copy();audit=[]
    if mode=='physical':
        c=copy.deepcopy(c)
        for row in c['siphons_info']:
            k=int(row['superlink_id']);ids=c['links'].k==k;old=float(c['links'].loc[ids,'dx'].sum());physical=float(row['physical_length']);factor=physical/old
            c['links'].loc[ids,'dx']*=factor;c['superlinks'].loc[c['superlinks'].id==k,'dx']=physical
            meta=c['channel_discretization'].get('siphon_'+row['name'],{})
            for point in meta.get('sections',[]):point['distance']*=factor
            audit.append(dict(name=row['name'],superlink_id=k,barrels=row['n_orifice'],single_width=row['width'],single_height=row['height'],legacy_length=old,physical_length=physical,loss_coeff=row['loss_coeff']))
            row['legacy_length']=row['length'];row['legacy_equivalent_length_added']=row['equivalent_length_added'];row['length']=physical;row['equivalent_length_added']=0.
        e=PipedreamHydraulicEngine(c,dt=15.,backend='numba');e.model._method='lsq'
    attach_parallel_physics(e,c,mode=mode)
    assert np.array_equal(e.model._n_ik,before_n);assert np.array_equal(e.model._Co,before_C)
    if mode=='physical':
        for row in c['siphons_info']:assert abs(e.model._dx_ik[e.model._ki==row['superlink_id']].sum()-row['physical_length'])<1e-8
    e.parallel_physics_manifest=dict(mode=mode,diagnostic_only=mode=='radius_only',physical_n_changed=False,external_gate_C_changed=False,loss_implementation='explicit K*abs(Q)/(2*A) added to momentum b coefficient once per structure' if mode=='physical' else 'legacy equivalent length',length_geometry='physical_length' if mode=='physical' else 'legacy_length_plus_Le',structures=audit,hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return e,source,c
