from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

R=Path(__file__).resolve().parent
folders={'FF':R/'06_独立接口复算/G16_DFF',**{k:R/'07_边界四组合'/k for k in ['HF','FH','HH']}}
frames={k:pd.read_csv(p/'半小时局部全线接口对照.csv') for k,p in folders.items()}
obs=pd.read_csv(R.parent/'26_冻结基线与后续归因/共享输入/作者已核实修正观测层.csv')
terms=pd.read_csv(R/'07_边界四组合/半小时边界响应分解.csv')
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13.4,8.3),sharex=True)
fig.subplots_adjust(left=.075,right=.98,bottom=.115,top=.875,hspace=.23,wspace=.20)
colors={'FF':'#277da8','HF':'#51a4b0','FH':'#d56a29','HH':'#619548'}
labels={'FF':'全线边界','HF':'仅换历史流量','FH':'仅换历史尾水','HH':'同时换两边界'}
x=frames['FF'].time_s/86400
for col,(label,metric,target) in enumerate([('闸前','forebay_m','JZZ16_forebay_m'),('闸后','afterbay_m','JZZ16_afterbay_m')]):
    y=obs[(obs.station_code=='JZZ16')&(obs.metric==metric)&(obs.time_s>0)].sort_values('time_s')
    ax=axes[0,col];ax.plot(x,y.value,color='#17212b',lw=1.5,label='实测水位',zorder=6)
    for code in ['FF','HF','FH','HH']:ax.plot(x,frames[code][target],color=colors[code],label=labels[code],lw=1.2)
    ax.set_title(f'兰河{label}水位 · 四组合',loc='left',fontweight='bold');ax.set_ylabel('水位（m）')
    ax=axes[1,col]
    for key,name,color,style in [('只换入口流量_HF减FF','入口流量作用','#277da8','-'),('只换尾水_FH减FF','尾水作用','#d56a29','-'),('交互项_HH减HF减FH加FF','交互项','#777777','--')]:
        ax.plot(terms.time_s/86400,terms[key+'_'+metric],label=name,color=color,ls=style,lw=1.3)
    ax.axhline(0,color='#999999',lw=.6);ax.set_title(f'{label}响应分解 · 相对全线边界',loc='left',fontweight='bold');ax.set_ylabel('模型水位变化（m）')
    ax.set_xlabel('起算后天数（2025-07-22 12:00 起）')
for ax in axes.flat:
    ax.axvspan(15,30,color='#f3ead8',alpha=.55,zorder=0);ax.axvline(15,color='#888888',ls='--',lw=.8)
    ax.set_xlim(0,30);ax.grid(axis='y',color='#dedede',lw=.6);ax.spines[['top','right']].set_visible(False)
handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.52,.945),ncol=5,frameon=False,fontsize=10)
axes[1,0].legend(loc='upper left',ncol=3,frameon=False,fontsize=9)
fig.suptitle('兰河局部与全线差异的边界诊断',fontsize=17,fontweight='bold',y=.99)
fig.text(.5,.035,'固定同一组 n/C、相同投影初态和时间步；浅色区为后15天诊断。模型内作用分解，不是现场原因判定或基线替换验收。',ha='center',fontsize=10)
fig.savefig(R/'边界四组合与响应分解.png',dpi=240)
fig.savefig(R/'边界四组合与响应分解.svg')
plt.close(fig)
