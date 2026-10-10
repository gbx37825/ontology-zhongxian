"""三项来源修复各自对同一原参考的精度变化，不叠加、不画部分过程。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
R42=HERE.parent
geometry=json.loads((HERE/'主进程宽坡复算独立核验.json').read_text())
chain=json.loads((R42/'05_梅河桩号锁参验证/主进程原始结果独立核验.json').read_text())
assert geometry['status']==chain['status']=='VERIFIED'
records=geometry['metrics']+chain['metrics']
for folder in [HERE/'q22',HERE/'q26',R42/'05_梅河桩号锁参验证/chainage']:
    status=json.loads((folder/'结果.json').read_text())
    assert status['status']=='COMPLETED' and status['completed_cycles']==1440

def value(case,group,period):
    found=[r['MAE'] for r in records if r['case']==case and r['group']==group and r['period']==period]
    assert found and all(abs(v-found[0])<1e-12 for v in found)
    return found[0]

plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],
                    'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(13,5.8),sharey=True)
fig.subplots_adjust(left=.09,right=.985,bottom=.24,top=.79,wspace=.10)
x=np.arange(3);width=.23;groups=['118','43','9']
for ax,period,title in zip(axes,['all30','post15'],['全30天','后15天']):
    for i,(case,label,color) in enumerate([('chainage','梅河桩号','#888888'),('q22','十八里河宽坡','#de9a39'),('q26','穿黄宽坡','#2878b5')]):
        y=np.array([(value(case,g,period)-value('reference',g,period))*1000 for g in groups])
        pos=x+(i-1)*width;ax.bar(pos,y,width,color=color,label=label)
        for xi,yi in zip(pos,y):
            ax.text(xi,yi+(0.20 if yi>=0 else -0.20),f'{yi:+.2f}',ha='center',va='bottom' if yi>=0 else 'top',fontsize=9)
    ax.axhline(0,color='#555555',lw=.9)
    ax.set_xticks(x,['118闸节点','43安装位置','旧9留出位置'])
    ax.set_ylim(-7.1,4.2);ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
axes[0].set_ylabel('MAE变化（候选 − 原参考，mm）\n负值为改善，正值为退化')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False)
fig.suptitle('三项修复各自完成30天，均未通过预定精度门槛',fontsize=17,fontweight='bold',y=.975)
fig.text(.5,.065,'原n/C、边界和开度保持，不校正内部水位；同一初始水位配各自几何，初始容积差另列。\n这些分组有重叠，不能相加；穿黄候选后15天还有28个节点与7个安装位置表示退化超过1cm。',ha='center',fontsize=10)
fig.savefig(HERE/'本轮单项修复精度变化.png',dpi=240)
fig.savefig(HERE/'本轮单项修复精度变化.svg')
plt.close(fig)
