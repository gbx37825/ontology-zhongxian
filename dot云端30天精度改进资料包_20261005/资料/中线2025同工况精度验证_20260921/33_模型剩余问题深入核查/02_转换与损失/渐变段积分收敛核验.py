from pathlib import Path
import sys,json,numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'26_冻结基线与后续归因';sys.path.insert(0,str(R/'30_分通道方法逐项测试/03_稳态与分通道试验'))
from 试验水力工具 import build
e,s,c,f,params=build();a=pd.read_csv(O/'沿程中点近似明细.csv');a=a[(a['case']=='B02三十天末')&a.link_id.isin([1067,1445])];rows=[]
with np.load(B/'B02_渠池归属修复后的参数同化诊断基线/同化运行/末态集合.npz') as z:
 theta=z['parameter_log_ratios'];Q=np.mean([z[f'm{i}_Q_ik'] for i in range(50)],axis=0)
 for r in a.itertuples():
  j=next(j for j,p in enumerate(params) if p['kind']=='n' and r.link_id in p['indices']);n=e.model._n_ik[r.link_id]*np.exp(theta[j]).mean();q=Q[r.link_id]
  def resistance(t):
   h=(1-t)*r.left_depth_m+t*r.right_depth_m;b=(1-t)*r.left_width_m+t*r.right_width_m;ss=(1-t)*r.left_slope+t*r.right_slope;A=h*(b+ss*h);Rh=A/(b+2*h*np.sqrt(1+ss**2));return 1/(A**2*Rh**(4/3))
  x,w=np.polynomial.legendre.leggauss(64);reference=float((w/2)@resistance((x+1)/2));factor=n*n*q*abs(q)*r.length_m
  for splits in [1,2,4,8,16]:
   approx=float(np.mean(resistance((np.arange(splits)+.5)/splits)));rows.append(dict(link_id=r.link_id,channel=r.channel,splits=splits,mean_Q_m3s=q,mean_n=n,headloss_approx_m=factor*approx,headloss_reference_m=factor*reference,headloss_difference_m=factor*(approx-reference),relative_error=approx/reference-1))
pd.DataFrame(rows).to_csv(O/'渐变段固定状态积分收敛.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
