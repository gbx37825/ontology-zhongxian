"""只在配对全月结果完整且核验后绘制；不读取部分时段评分。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
for case in ['reference','endpoint']:
    status = json.loads((HERE/case/'结果.json').read_text())
    assert status['status'] == 'COMPLETED' and status['completed_cycles'] == 1440
assert json.loads((HERE/'reference/与06逐值复现核验.json').read_text())['all_exact']
scores = pd.read_csv(HERE/'端面几何锁参_分组精度.csv')
groups = ['118闸节点水位','原88可比节点水位','43安装位置水位','旧更新34水位','旧9留出水位']
labels = ['118闸节点','原88可比点','43安装位置','旧34更新点','旧9留出点']
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],
                    'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes = plt.subplots(1,2,figsize=(13.5,5.6),sharey=True)
fig.subplots_adjust(left=.065,right=.99,bottom=.23,top=.80,wspace=.10)
x=np.arange(len(groups));width=.35
for ax,period in zip(axes,['全30天','后15天']):
    for case,offset,color,label in [('reference',-width/2,'#777777','相同严格判据的原端面取值'),
                                  ('endpoint',width/2,'#2878b5','修正端面几何来源')]:
        part=scores[(scores.case==case)&(scores.period==period)].set_index('score_group')
        y=part.loc[groups,'MAE'].to_numpy()*100
        ax.bar(x+offset,y,width,color=color,label=label)
        for xi,yi in zip(x+offset,y):
            ax.text(xi,yi+.55,f'{yi:.2f}',ha='center',fontsize=8.7)
    ax.set_xticks(x,labels,rotation=12)
    ax.set_title(period,loc='left',fontsize=13,fontweight='bold')
    ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,float(scores[(scores.unit=='m') & scores.score_group.isin(groups)].MAE.max())*100+5)
axes[0].set_ylabel('水位平均绝对误差 MAE（cm）')
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False)
fig.suptitle('只修端面几何来源：锁定原参数、无内部水位校正',fontsize=17,fontweight='bold',y=.985)
fig.text(.5,.058,'两分支均完成30天，原输入、原六态及原n/C相同；存在逐点退化，候选未通过预定精度门槛。\n五组测点存在重叠，不能相加；后15天为已查看历史资料的描述性复核。原断面域外问题仍单独保留。',
         ha='center',fontsize=10)
fig.savefig(HERE/'端面几何锁参全月水位精度.png',dpi=240)
fig.savefig(HERE/'端面几何锁参全月水位精度.svg')
plt.close(fig)
