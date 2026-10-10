"""串行确定性平方根 EnKF，固定水力参数，全线集合传播，流量断面观测。

水位观测使用明确同侧位置算子；保留空间留出站，analysis 通过原生 load_state 联合装载各类水动力状态。
"""
import copy, json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
STATE_KEYS=('H_j','h_Ik','Q_ik','Q_uk','Q_dk','Q_o')
def snapshot(m):return {k:getattr(m,k).copy() for k in STATE_KEYS}
def restore(m,s):
    m.load_state(s,compute_hydraulic_geometries=True);m.compute_storage_volumes()
def volume(m):return float(np.sum(m.A_ik*m._dx_ik)+np.sum(m.A_uk*m._dx_uk)+np.sum(m.A_dk*m._dx_dk)+np.sum(m.V_j))

class LocalEnKF:
    def __init__(self,engine,n=20,seed=20260916,radius=20000.):
        self.engine=engine;self.n=n;self.seed=seed;self.radius=radius
        m=engine.model
        self.sizes={k:getattr(m,k).size for k in STATE_KEYS}
        pos=0;self.slices={}
        for k,size in self.sizes.items():self.slices[k]=slice(pos,pos+size);pos+=size
        edges=[]
        for a,b,length in zip(m._J_uk,m._J_dk,m._dx_k):edges.extend([(a,b,max(float(length),1.)),(b,a,max(float(length),1.))])
        for a,b in zip(m._J_uo,m._J_do):edges.extend([(a,b,1.),(b,a,1.)])
        row,col,val=zip(*edges)
        # 同一闸站多孔只需一条通路，避免 scipy 累加并行孔。
        graph={}
        for a,b,v in edges:graph[(a,b)]=min(graph.get((a,b),float('inf')),v)
        row,col=zip(*graph)
        source=int(m._J_uk[0])
        xj=dijkstra(csr_matrix((list(graph.values()),(row,col)),shape=(m.M,m.M)),directed=False,indices=source)
        direction=np.sign(xj[m._J_dk]-xj[m._J_uk])
        xh=xj[m._J_uk[m._kI]]+direction[m._kI]*m.x_Ik
        xq=(xh[m._Ik]+xh[m._Ip1k])/2
        self.xparts=dict(H_j=xj,h_Ik=xh,Q_ik=xq,Q_uk=xj[m._J_uk],Q_dk=xj[m._J_dk],Q_o=(xj[m._J_uo]+xj[m._J_do])/2)
        self.x=np.concatenate([self.xparts[k] for k in STATE_KEYS])
        self.mask=np.isfinite(self.x)
        self.mask[self.slices['H_j']]&=~m.bc
        self.heads=np.zeros(pos,dtype=bool);self.heads[self.slices['H_j']]=True;self.heads[self.slices['h_Ik']]=True
        rng=np.random.default_rng(seed)
        base=self.pack(m)
        modes=np.stack([np.ones(pos),np.sin(self.x/10000),np.cos(self.x/10000),np.sin(self.x/3000),np.cos(self.x/3000)])
        modes=np.nan_to_num(modes)
        coeff=np.clip(rng.normal(size=(5,n)),-2,2);coeff-=coeff.mean(axis=1,keepdims=True)
        raw=modes.T@coeff/np.sqrt(5)
        # 有界初始水面线与流量扰动，参数、开度、边界保持不变。
        scale=np.where(self.heads,.02,np.maximum(np.abs(base)*.01,.2))
        delta=np.clip(raw*scale[:,None],-2*scale[:,None],2*scale[:,None])*self.mask[:,None]
        delta-=delta.mean(axis=1,keepdims=True)
        self.members=[copy.deepcopy(m) for _ in range(n)]
        for i,member in enumerate(self.members):self.unpack(member,base+delta[:,i])
        self.log=[]
    def pack(self,m):return np.concatenate([getattr(m,k) for k in STATE_KEYS])
    def unpack(self,m,v):restore(m,{k:v[s].copy() for k,s in self.slices.items()})
    def propagate(self,kw):
        for member in self.members:
            member.step(**kw)
            if not all(np.isfinite(getattr(member,k)).all() for k in STATE_KEYS):raise FloatingPointError('集合传播出现非有限状态')
    def values(self,op):
        ids=np.asarray(op['indices'],int);weights=np.asarray(op['weights'],float)
        return np.array([getattr(m,op.get('state','Q_ik'))[ids]@weights + op.get('offset',0.) for m in self.members])
    def analysis(self,observations,time_s):
        trace=[]
        for obs in observations:
            y=float(obs['value']);pred=self.values(obs);sigma=float(obs.get('sigma',max(1.,abs(y)*.05)))
            var=float(pred.var(ddof=1));innovation=y-pred.mean()
            row=dict(time_s=time_s,station=obs['station'],observed=y,forecast=float(pred.mean()),forecast_spread=float(np.sqrt(var)),sigma=sigma,innovation=float(innovation))
            if not np.isfinite(y) or y<0 or abs(innovation)>6*np.sqrt(var+sigma*sigma):
                row.update(status='rejected_innovation',analysis=row['forecast'],delta_volume_mean_m3=0.);trace.append(row);continue
            X=np.stack([self.pack(m) for m in self.members],axis=1);mean=X.mean(axis=1);A=X-mean[:,None];dy=pred-pred.mean()
            state=obs.get('state','Q_ik');wpos=np.abs(np.asarray(obs['weights'],float));wpos/=wpos.sum();xc=float(self.xparts[state][np.asarray(obs['indices'],int)]@wpos)
            dist=np.abs(self.x-xc)/self.radius
            # 连续紧支撑余弦局地化，半径外增量严格为零。
            loc=np.where(dist<1,.5*(1+np.cos(np.pi*np.minimum(dist,1))),0)*self.mask
            gain=(A@dy)/(self.n-1)/(var+sigma*sigma)*loc
            alpha=1/(1+np.sqrt(sigma*sigma/(var+sigma*sigma)))
            delta=gain[:,None]*(innovation-alpha*dy[None,:])
            # 单次水位更新≤2 cm，流量≤3 m³/s；按成员整体缩放保留联合增量方向。
            bound=np.where(self.heads,.02,3.)
            factors=np.minimum(1.,np.min(bound[:,None]/np.maximum(np.abs(delta),1e-30),axis=0))
            delta*=factors[None,:]
            dvs=[]
            for i,m in enumerate(self.members):
                before=volume(m);new=X[:,i]+delta[:,i]
                if np.min(new[self.slices['h_Ik']])<m.min_depth:raise FloatingPointError('分析后内部水深低于下限')
                if np.min(new[self.slices['H_j']]-m._z_inv_j)<m.min_depth:raise FloatingPointError('分析后节点水深低于下限')
                self.unpack(m,new);dvs.append(volume(m)-before)
            row.update(status='applied',analysis=float(self.values(obs).mean()),delta_volume_mean_m3=float(np.mean(dvs)),delta_volume_min_m3=float(min(dvs)),delta_volume_max_m3=float(max(dvs)),increment_scale_min=float(min(factors)))
            trace.append(row)
        self.log.extend(trace);return trace
