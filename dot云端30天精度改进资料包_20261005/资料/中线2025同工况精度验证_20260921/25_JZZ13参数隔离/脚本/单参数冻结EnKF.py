"""Local parameter EnKF: update n/C only, retain each propagated hydraulic state."""
import copy,re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra,connected_components
STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
def restore(m,s):
    m.load_state(s,compute_hydraulic_geometries=True);m.compute_storage_volumes()
def volume(m):return float(np.sum(m.A_ik*m._dx_ik)+np.sum(m.A_uk*m._dx_uk)+np.sum(m.A_dk*m._dx_dk)+np.sum(m.V_j))

class ParameterEnKF:
    def __init__(self,engine,converted,n=50,seed=20260923,radius=20000.):
        self.n=n;self.radius=radius;self.members=[copy.deepcopy(engine.model) for _ in range(n)];m=engine.model
        edges={}
        for u,v,dx in zip(m._J_uk,m._J_dk,m._dx_k):edges[(int(u),int(v))]=max(float(dx),1.)
        for u,v in zip(m._J_uo,m._J_do):edges[(int(u),int(v))]=1.
        rr,cc=zip(*edges);xj=dijkstra(csr_matrix((list(edges.values()),(rr,cc)),shape=(m.M,m.M)),directed=False,indices=engine.node_index['NODE-main-0']);xh=xj[m._J_uk[m._kI]]+np.sign(xj[m._J_dk[m._kI]]-xj[m._J_uk[m._kI]])*m.x_Ik
        self.xparts={'H_j':xj,'h_Ik':xh};self.base_n=m._n_ik.copy();self.base_C=m._Co.copy();self.parameters=[]
        # 渠池以节制闸为分界，由真实连接确定；不使用不可靠名称前缀。
        rr=np.r_[m._J_uk,m._J_dk];cc=np.r_[m._J_dk,m._J_uk]
        _,labels=connected_components(csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(m.M,m.M)),directed=False)
        components=sorted(set(labels[m._J_uk]),key=lambda z:xj[labels==z].min())
        assert len(components)==60, '渠池拓扑数量改变，需重新审核'
        pools={}
        for row in converted['superlinks'].itertuples():
            if re.match(r'^POOL',row.name,re.I):
                pool=components.index(labels[m._J_uk[int(row.id)]])+1
                pools.setdefault(pool,[]).append(int(row.id))
        for pool,ks in sorted(pools.items()):
            ids=np.flatnonzero(np.isin(m._ki,ks));lo=float(min(xj[m._J_uk[ks]].min(),xj[m._J_dk[ks]].min()));hi=float(max(xj[m._J_uk[ks]].max(),xj[m._J_dk[ks]].max()));v=self.base_n[ids]
            self.parameters.append(dict(name=f'POOL{pool}_n',kind='n',indices=ids.tolist(),xmin=lo,xmax=hi,lower=max(.5,.008/float(v.min())),upper=min(1.5,.05/float(v.max()))))
        stations=converted['orifices'].name.str.split('-').str[0]
        for st in sorted(stations.unique(),key=lambda s:int(s[3:])):
            ids=np.flatnonzero(stations==st);v=self.base_C[ids];xx=float(np.mean((xj[m._J_uo[ids]]+xj[m._J_do[ids]])/2))
            self.parameters.append(dict(name=st+'_C',kind='C',indices=ids.tolist(),xmin=xx,xmax=xx,lower=max(.5,.1/float(v.min())),upper=min(1.5,1./float(v.max()))))
        self.lower=np.log([p['lower'] for p in self.parameters])[:,None];self.upper=np.log([p['upper'] for p in self.parameters])[:,None];assert (self.lower<=0).all() and (self.upper>=-1e-10).all()
        self.rng=np.random.default_rng(seed);self.analysis_rng=np.random.default_rng(seed+1);z=self.rng.normal(size=(len(self.parameters),n));z-=z.mean(axis=1,keepdims=True);self.theta=np.clip(.05*z,self.lower,self.upper);self.log=[];self.apply()
    def values(self,op):
        ids=np.asarray(op['indices']);w=np.asarray(op['weights']);return np.array([getattr(m,op['state'])[ids]@w+op.get('offset',0) for m in self.members])
    def apply(self):
        # This method must not write H/h/Q, time, or geometric storage.
        for i,m in enumerate(self.members):
            m._n_ik[:]=self.base_n;m._Co[:]=self.base_C
            for j,p in enumerate(self.parameters):
                ids=p['indices'];factor=np.exp(self.theta[j,i])
                if p['kind']=='n':m._n_ik[ids]=self.base_n[ids]*factor
                else:m._Co[ids]=self.base_C[ids]*factor
            m._n_uk[:]=m._n_ik[m._i_1k];m._n_dk[:]=m._n_ik[m._i_nk]
    def rejuvenate(self):
        # Restore collapsed parameter spread only. No direct hydraulic noise.
        z=self.rng.normal(size=self.theta.shape);z-=z.mean(axis=1,keepdims=True);sd=self.theta.std(axis=1,ddof=1);low=(sd<.01)&np.array([getattr(self,"update_kind","both") in ("both",p["kind"]) and p["name"] not in getattr(self,"frozen",set()) for p in self.parameters])
        noise=np.sqrt(np.maximum(.05**2-sd**2,0))[:,None]*z;self.theta=np.clip(self.theta+noise*low[:,None],self.lower,self.upper);self.apply();return int(low.sum())
    def analysis(self,observations,time_s):
        if not observations:return []
        Y=np.stack([self.values(o) for o in observations]);y=np.array([o['value'] for o in observations]);sigma=np.array([o.get('sigma',.03) for o in observations]);innovation=y-Y.mean(axis=1);ok=np.isfinite(y)&(y>0)&(abs(innovation)<=6*np.sqrt(Y.var(axis=1,ddof=1)+sigma**2));trace=[]
        for j,o in enumerate(observations):trace.append(dict(time_s=time_s,station=o['station'],metric=o['metric'],observed=float(y[j]),forecast=float(Y[j].mean()),innovation=float(innovation[j]),status='used_for_parameters' if ok[j] else 'rejected_innovation'))
        if not ok.any():self.log.extend(trace);return trace
        observations=[o for o,k in zip(observations,ok) if k];Y=Y[ok];y=y[ok];sigma=sigma[ok]
        xpos=np.array([self.xparts[o['state']][np.array(o['indices'])]@np.array(o['weights']) for o in observations]);A=Y-Y.mean(axis=1,keepdims=True);P=self.theta-self.theta.mean(axis=1,keepdims=True);noise=self.analysis_rng.normal(size=Y.shape);noise-=noise.mean(axis=1,keepdims=True);proposal=self.theta.copy()
        for j,p in enumerate(self.parameters):
            if getattr(self,'update_kind','both') not in ('both',p['kind']) or p['name'] in getattr(self,'frozen',set()):continue
            distance=np.maximum(np.maximum(p['xmin']-xpos,xpos-p['xmax']),0);sel=distance<self.radius
            if not sel.any():continue
            loc=.5*(1+np.cos(np.pi*distance[sel]/self.radius));sd=sigma[sel]/np.sqrt(np.maximum(loc,1e-12));B=A[sel];cov=B@B.T/(self.n-1)+np.diag(sd**2);cross=P[j]@B.T/(self.n-1);gain=np.linalg.solve(cov,cross)
            proposal[j]+=gain@(y[sel,None]+noise[sel]*sd[:,None]-Y[sel])
        # Gentle correction: at most 1% per member/parameter per 30min cycle.
        change=np.clip(proposal-self.theta,-np.log(1.01),np.log(1.01));self.theta=np.clip(self.theta+change,self.lower,self.upper);self.apply();self.log.extend(trace);return trace
