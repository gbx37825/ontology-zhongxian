from pathlib import Path
import json,pandas as pd
R=Path(__file__).resolve().parents[1];d=pd.read_csv(R/'02_输入/2025分退水原始窗口.csv');d['时间']=pd.to_datetime(d['时间']);out=[]
def record(row):return {k:None if pd.isna(v) else str(v) if isinstance(v,pd.Timestamp) else v for k,v in row.to_dict().items()}
for name,target in [('白庄分水口','FSK54-1-pool29'),('袁庄分水口','FSK63-1-pool33'),('吴庄分水口','FSK82-2-pool39')]:
 g=d[d['闸站名称']==name].sort_values('时间').reset_index(drop=True);bad=g['总瞬时流量']<=-100;groups=(bad!=bad.shift()).cumsum();segments=[]
 for _,v in g[bad].groupby(groups[bad]):
  if v['时间'].max()<pd.Timestamp('2025-07-22 12:00'):continue
  lo,hi=v.index.min(),v.index.max();cols=['源行号','时间','孔1瞬时流量','孔2瞬时流量','孔3瞬时流量','总瞬时流量']
  segments.append({'start':str(v['时间'].min()),'end':str(v['时间'].max()),'n':len(v),'before':record(g.loc[lo-1,cols]) if lo else None,'after':record(g.loc[hi+1,cols]) if hi+1<len(g) else None,'hole_values':{c:v[c].dropna().unique().tolist() for c in cols[2:]}})
 good=g[(~bad)&(g['时间']>=pd.Timestamp('2025-07-22 12:00'))]
 out.append({'name':name,'target':target,'valid_n':len(good),'valid_min':float(good['总瞬时流量'].min()),'valid_max':float(good['总瞬时流量'].max()),'valid_mean':float(good['总瞬时流量'].mean()),'bad_segments':segments,'alternative':'原孔1流量与总量同为-300；孔2/3为空，无可替代分孔；窗口无同名第二代码或分支'})
(R/'07_对比分析/2025三站坏值逐段审核.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
