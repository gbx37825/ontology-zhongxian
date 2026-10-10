"""仅从全批终态汇总绘图；两种输入轨道及状态等效水量不混用。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'条件性在线对照汇总.json').read_text())
assert summary['status'] in ['COMPLETED','COMPLETED_WITH_FAILED_CASES']
issues=summary['common_completed_issue_times_s'];assert issues
df=pd.read_csv(HERE/'冻结更新预测分组误差.csv')
aggregate=df[df.population=='全部共同完成起点']
colors={'F':'#666666','P':'#2878b5','W':'#27a58c','S':'#e07a25','H':'#8656a2'}
labels={'F':'固定参数','P':'逐次参数微调','W':'因果窗口参数','S':'仅状态校正','H':'参数与状态联合'}
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(12.8,8.1),sharex=True)
fig.subplots_adjust(left=.085,right=.98,bottom=.18,top=.86,hspace=.27,wspace=.23)
for row,group in enumerate(['旧留出9','安装位置43']):
    for col,(track,name) in enumerate([('frozen_at_issue','起报时冻结输入'),('known_historical_inputs','已知历史输入回放')]):
        ax=axes[row,col]
        for method in colors:
            a=aggregate[(aggregate.score_group==group)&(aggregate.track==track)&(aggregate.method==method)].sort_values('lead_s')
            assert len(a)==4
            ax.plot(a.lead_s/3600,a.MAE,color=colors[method],label=labels[method],marker='o',lw=1.8,ms=4.5)
        ax.set_title(f'{group} · {name}',loc='left',fontweight='bold');ax.set_ylabel('水位 MAE（m）')
        ax.set_xticks([.5,1,3,6]);ax.grid(axis='y',color='#dedede',lw=.6);ax.spines[['top','right']].set_visible(False)
        if row==1:ax.set_xlabel('预测提前量（h）')
handles,names=axes[0,0].get_legend_handles_labels()
fig.legend(handles,names,ncol=5,loc='upper center',bbox_to_anchor=(.52,.945),frameon=False,fontsize=9.5)
fig.suptitle('条件性在线方法 · 冻结更新后的水位预测',fontsize=17,fontweight='bold',y=.995)
fig.text(.5,.025,f'{len(issues)}个共同起点，50成员。每提前量留出误差样本{len(issues)*9}个、43位置误差样本{len(issues)*43}个；起点与空间点并非独立重复样本。\n起报后均无观测、参数或状态更新；短期共同起算比较，不代表连续30天在线效果。',ha='center',fontsize=9.5)
fig.savefig(HERE/'五方法预测提前量对照.png',dpi=240);fig.savefig(HERE/'五方法预测提前量对照.svg');plt.close(fig)

water=pd.DataFrame(json.loads((HERE/'各起点独立六小时同化等效水量.json').read_text()))
fig,axes=plt.subplots(1,2,figsize=(13.2,5.1));fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.82,wspace=.24)
x=np.arange(len(issues));width=.35
for j,method in enumerate(['S','H']):
    a=water[(water.method==method)&water.issue_s.isin(issues)].set_index('issue_s').loc[issues]
    mean=a.cumulative_delta_volume_mean_m3.to_numpy()/1e4
    low=a.cumulative_delta_volume_min_member_m3.to_numpy()/1e4;high=a.cumulative_delta_volume_max_member_m3.to_numpy()/1e4
    axes[0].bar(x+(j-.5)*width,mean,width,label=labels[method],color=colors[method],yerr=np.vstack([mean-low,high-mean]),capsize=2,error_kw={'lw':.8})
axes[0].set_xticks(x,[f'{t/86400:.2f}' for t in issues],rotation=25)
axes[0].set_xlabel('起报时刻（起算后天数）');axes[0].set_ylabel('6h内模型等效蓄量变化（万m³）')
axes[0].set_title('各起点分别计账，不能相加称整月补水',loc='left',fontweight='bold',fontsize=11)
axes[0].legend(frameon=False,fontsize=9)
for method in colors:
    a=aggregate[(aggregate.score_group=='兰河测流Q16')&(aggregate.track=='frozen_at_issue')&(aggregate.method==method)].sort_values('lead_s')
    axes[1].plot(a.lead_s/3600,a.MAE,color=colors[method],label=labels[method],marker='o',lw=1.6,ms=4)
axes[1].set_xticks([.5,1,3,6]);axes[1].set_xlabel('预测提前量（h）');axes[1].set_ylabel('同位置测流 MAE（m³/s）')
axes[1].set_title('兰河300091m测流口 · 冻结输入预测',loc='left',fontweight='bold',fontsize=11)
axes[1].legend(frameon=False,fontsize=8.5,ncol=2)
for ax in axes:ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('状态校正的模型蓄量变化与独立流量检验',fontsize=16,fontweight='bold',y=.985)
fig.text(.5,.05,'左图柱为成员均值、线为50成员范围；不是现场补水或置信区间。F/P/W没有直接状态校正蓄量。\n右图每提前量仅有各共同起点的单站测流样本；该流量未参与观测更新。原工程验收失败记录保留。',ha='center',fontsize=9)
fig.savefig(HERE/'状态等效水量与兰河流量检验.png',dpi=240);fig.savefig(HERE/'状态等效水量与兰河流量检验.svg');plt.close(fig)
