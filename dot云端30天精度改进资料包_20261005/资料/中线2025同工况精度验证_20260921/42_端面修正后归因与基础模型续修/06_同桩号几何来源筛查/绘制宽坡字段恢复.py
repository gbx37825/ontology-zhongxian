"""两组原表支持的宽坡恢复示意，图形横纵比例不同，不作现场断面测量图。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'02_退化渠段结构核查'
data=[json.loads((SRC/name).read_text())['points'][0] for name in
      ['q22s14宽坡恢复候选.json','q26四断面宽坡恢复候选.json']]
plt.rcParams.update({'font.sans-serif':['PingFang SC','Arial Unicode MS'],
                    'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(13.2,5.4))
fig.subplots_adjust(left=.065,right=.985,bottom=.26,top=.74,wspace=.22)
for ax,item,title in zip(axes,data,['十八里河出口 q22s14','穿黄出口 q26s1（四个修复断面之一）']):
    for key,label,color,style in [('old_yz','原输入','#777777','--'),
                                  ('candidate_yz','按原表恢复','#2878b5','-')]:
        yz=np.asarray(item[key]);yz[:,0]-=(yz[:,0].min()+yz[:,0].max())/2;yz[:,1]-=yz[:,1].min()
        ax.plot(yz[:,0],yz[:,1],color=color,ls=style,lw=2.1,label=label)
    ymax=item['preserved_height_m'];xmax=max(p[0] for p in item['old_yz'])/2
    ax.set_xlim(-xmax*1.08,xmax*1.08);ax.set_ylim(-.55,ymax+1)
    old=item['old_dimensions']
    ax.text(0,1.025,f"原：底宽 {old['b']:g} m / 边坡 {old['m']:g}",color='#666666',ha='left',transform=ax.transAxes)
    ax.text(1,1.025,f"恢复：底宽 {item['source_b_m']:g} m / 边坡 {item['source_slope']:g}",
            color='#2878b5',ha='right',transform=ax.transAxes,fontsize=10)
    ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=32)
    ax.set_xlabel('横向坐标（m，中心对齐）');ax.set_ylabel('距同一渠底的高度（m）')
    ax.grid(axis='y',color='#dddddd',lw=.5);ax.spines[['top','right']].set_visible(False)
fig.suptitle('恢复写反的底宽与边坡，全部高程保持不变',fontsize=17,fontweight='bold',y=.975)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
fig.text(.5,.075,'这是输入轮廓对照，横纵比例不同；边坡为水平与竖直之比。\n原表渠道与相接渐变段字段一致，仅恢复横向坐标；不改桩号、底高程、顶部、长度、糙率或闸流系数。过程精度另行验证。',
         ha='center',fontsize=10)
fig.savefig(HERE/'两组宽坡字段恢复对照.png',dpi=240)
fig.savefig(HERE/'两组宽坡字段恢复对照.svg')
plt.close(fig)
