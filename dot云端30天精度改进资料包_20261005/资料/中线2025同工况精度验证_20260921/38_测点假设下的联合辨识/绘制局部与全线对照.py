"""同图呈现条件性局部检验与全线转移，保留完整30天及未用于率定的后15天。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
local=R/'05_固定参数复核';full=R/'06_全线联合参数转移'
for case in ['G00_冻结原参数','G16_联合参数转移']:
    s=json.loads((full/case/'进度.json').read_text())
    assert s['status']=='COMPLETED' and s['completed_cycles']==1440
base=pd.read_csv(local/'A_原参数/内部水位检验.csv')
fit=pd.read_csv(local/'A_联合辨识参数/内部水位检验.csv')
g0=pd.read_csv(full/'G00_冻结原参数/实际位置水位.csv')
g1=pd.read_csv(full/'G16_联合参数转移/实际位置水位.csv')
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13.2,8.0),sharex=True,layout='constrained')
x=base.time_s.to_numpy()/86400.
colors=['#17212b','#277da8','#d76a29']
for col,(label,metric,target) in enumerate([('闸前','forebay_m','JZZ16_forebay_m'),('闸后','afterbay_m','JZZ16_afterbay_m')]):
    for row,scope in enumerate(['A假设下局部复算','全线固定参数转移']):
        ax=axes[row,col]
        yobs=base[target+'_observed'].to_numpy()
        if row==0:y0=base[target].to_numpy();y1=fit[target].to_numpy()
        else:
            a=g0[(g0.station=='JZZ16')&(g0.metric==metric)].sort_values('time_s')
            b=g1[(g1.station=='JZZ16')&(g1.metric==metric)].sort_values('time_s')
            assert np.array_equal(a.time_s.to_numpy(),base.time_s.to_numpy()) and np.array_equal(b.time_s.to_numpy(),base.time_s.to_numpy())
            y0=a.mapped_sim.to_numpy();y1=b.mapped_sim.to_numpy()
        ax.axvspan(15,30,color='#f3ead8',alpha=.65,zorder=0)
        for y,color,name,width in [(yobs,colors[0],'实测水位',1.35),(y0,colors[1],'原参数',1.25),(y1,colors[2],'固定联合参数',1.25)]:
            ax.plot(x,y,color=color,label=name,lw=width)
        ax.axvline(15,color='#888888',ls='--',lw=.8)
        ax.set_title(f'{scope} · 兰河{label}',loc='left',fontweight='bold')
        ax.set_ylabel('水位（m）')
        ax.grid(axis='y',color='#dedede',lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0,30)
        a0=np.abs(y0[x>15]-yobs[x>15]).mean();a1=np.abs(y1[x>15]-yobs[x>15]).mean()
        ax.text(.02,.04,f'后15天 MAE  {a0:.3f} → {a1:.3f} m',transform=ax.transAxes,fontsize=10,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
        if row==1:ax.set_xlabel('起算后天数（2025-07-22 12:00 起）')
axes[0,0].legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
fig.suptitle('同一组联合参数：局部保留检验与全线转移',fontsize=17,fontweight='bold')
fig.supxlabel('浅色区为后15天保留检验。全程锁参、无内部水位状态纠正；已知历史边界，不代表未知未来预报。',fontsize=10)
fig.savefig(R/'局部与全线30天对照.png',dpi=240)
fig.savefig(R/'局部与全线30天对照.svg')
plt.close(fig)
