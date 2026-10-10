"""实际/目标差异段只读筛查；目标稳定不代表目标等于真实执行。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'07_对比分析'
f=pd.read_csv(R/'02_输入/2025节制闸原始窗口.csv',parse_dates=['时间']);mp=pd.read_csv(R/'02_输入/2025节制闸站名映射.csv');start=pd.Timestamp('2025-07-22 12:00:00');end=pd.Timestamp('2025-08-21 12:00:00');records=[];checks=[]
for _,mapping in mp[mp.station_code.str.startswith('JZZ')].iterrows():
 g=f[f.站点名称==mapping.source_name].sort_values('时间').set_index('时间')
 for hole in str(mapping.model_holes).split(';'):
  h=int(hole);a=pd.to_numeric(g[f'孔{h}实际开度'],errors='coerce');t=pd.to_numeric(g[f'孔{h}目标开度'],errors='coerce');inside=(g.index>=start)&(g.index<=end)
  checks.append(dict(station_code=mapping.station_code,hole=h,n=int(inside.sum()),missing=int(t[inside].isna().sum()),nonfinite=int((~np.isfinite(t[inside])).sum()),negative=int(t[inside].lt(0).sum()),minimum=float(t[inside].min()),maximum=float(t[inside].max())))
  # 中心±1h共5点，要求连续30min；末端不足两侧观测时不宣称目标稳定。
  spread=t.rolling(5,center=True,min_periods=5).max()-t.rolling(5,center=True,min_periods=5).min()
  seq=pd.Series(g.index,index=g.index).diff().dt.total_seconds().eq(1800).rolling(4,min_periods=4).sum().shift(-2).eq(4)
  hit=inside & a.gt(0)&t.ge(0)&(a-t).abs().gt(1000)&spread.lt(200)&seq
  edge=np.diff(np.r_[False,hit.to_numpy(),False].astype(int));beg=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)-1
  for lo,hi in zip(beg,ends):
   left=max(0,lo-1);right=min(len(g)-1,hi+1);seg=g.iloc[lo:hi+1];duration=(g.index[hi]-g.index[lo]).total_seconds()/3600;before=a.iloc[left];after=a.iloc[right];targetmed=float(t.iloc[lo:hi+1].median());count=hi-lo+1
   boundary_return=bool(abs(before-t.iloc[left])<=200 and abs(after-t.iloc[right])<=200 and left<lo and right>hi)
   # 与邻近实际值分离的编码候选仍非自动修正许可。
   kind='孤立单点' if count==1 else ('短段（不超过2h采样跨度）' if duration<=2 else '长段持续不一致')
   rec=dict(station_code=mapping.station_code,name=mapping.source_name,hole=h,start=str(g.index[lo]),end=str(g.index[hi]),source_row_start=int(g.iloc[lo]['源行号']),source_row_end=int(g.iloc[hi]['源行号']),points=count,sample_span_h=duration,support_duration_h=count*.5,classification=kind,actual_before_mm=float(before),actual_first_mm=float(a.iloc[lo]),actual_min_mm=float(a.iloc[lo:hi+1].min()),actual_max_mm=float(a.iloc[lo:hi+1].max()),actual_last_mm=float(a.iloc[hi]),actual_after_mm=float(after),target_before_mm=float(t.iloc[left]),target_median_mm=targetmed,target_min_mm=float(t.iloc[lo:hi+1].min()),target_max_mm=float(t.iloc[lo:hi+1].max()),target_after_mm=float(t.iloc[right]),max_abs_actual_target_difference_mm=float((a-t).iloc[lo:hi+1].abs().max()),boundary_actual_returns_within_200mm_target=boundary_return)
   for col,prefix in [('瞬时流量','q'),('闸前水位','forebay'),('闸后水位','afterbay')]:
    v=pd.to_numeric(g[col],errors='coerce');rec.update({prefix+'_before':float(v.iloc[left]),prefix+'_first':float(v.iloc[lo]),prefix+'_min':float(v.iloc[lo:hi+1].min()),prefix+'_max':float(v.iloc[lo:hi+1].max()),prefix+'_after':float(v.iloc[right]),prefix+'_change_across':float(v.iloc[right]-v.iloc[left])})
   rec['recommendation']='保留原值；孤立或短段且前后回归目标，仅作为现场记录/原始采集核查候选，不足以确认编码坏点' if count<=5 and boundary_return else '保留原值；实际与目标可为执行差、停机或量程差，不能默认目标是真实开度'
   records.append(rec)
res=pd.DataFrame(records)
res['same_hole_segment_count']=res.groupby(['station_code','hole']).points.transform('size')
res['repeated_pattern_note']=np.where(res.same_hole_segment_count>=10,'同孔多次反复段；不能逐段当成独立编码坏点','仍需采集编码或运行记录证实')
res.to_csv(O/'2025实际开度异常段候选.csv',index=False);pd.DataFrame(checks).to_csv(O/'2025目标开度完整性.csv',index=False)
summary=dict(rule='实际正值；abs(实际-目标)>1000mm；中心±1h五个30min目标值完整且极差<200mm。未用转换上限b判异常。相邻命中点合并为段，段外可能仍有差异但目标不稳，因此不等同完整物理故障段。',end_boundary='正式末端8月21日12时附近缺未来±1h，不能满足中心窗口，未强行判定稳定；使用7月20日先导观测支持起点±1h。',segments=len(records),stations=int(res.station_code.nunique()) if len(res) else 0,holes=int(res[['station_code','hole']].drop_duplicates().shape[0]) if len(res) else 0,classification_counts=res.classification.value_counts().to_dict() if len(res) else {},target_holes=len(checks),target_expected_points_per_hole=1441,target_missing=sum(x['missing'] for x in checks),target_nonfinite=sum(x['nonfinite'] for x in checks),target_negative=sum(x['negative'] for x in checks),target_branch_constructible=all(x['n']==1441 and x['nonfinite']==0 and x['negative']==0 for x in checks),recommendation='目标分支可作独立输入假设，不代表实测执行；正值候选不自动替目标，不追加所谓明确编码修正，除非采集原始记录或明确编码规则证实。')
(O/'2025实际开度异常段审核.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False));print(res[['station_code','hole','start','end','points','classification','actual_min_mm','actual_max_mm','target_median_mm','boundary_actual_returns_within_200mm_target']].to_string(index=False))
lines=['# 2025实际开度异常段候选','',summary['rule'],'',summary['end_boundary'],'',f'共{len(records)}段，{summary["stations"]}站{summary["holes"]}孔。类别统计：{summary["classification_counts"]}。','',f'目标开度183孔×1441点，缺失{summary["target_missing"]}，非有限值{summary["target_nonfinite"]}，负值{summary["target_negative"]}。可直接构造目标开度独立输入分支：{summary["target_branch_constructible"]}。','',summary['recommendation'],'','JZZ8孔3和JZZ9孔3存在持续23h、46.5h不一致；JZZ27四孔合计67个重复短段，应整体核查是否检修、采集或执行记录问题。JZZ42孔3三次0→5551/7787/5002→0mm而目标0，属于优先核查的孤立尖点，但本站总流量和水位不能证明单孔没动作，仍不能确认为编码坏点。','', '逐段证据见2025实际开度异常段候选.csv，包含前后实际开度、目标开度、源行号、本站流量及闸前后水位变化。实际与目标差异本身不能证明传感器错误；实际轨迹可与控制目标长期不同。']
(R/'08_报告/2025实际开度异常段审核.md').write_text('\n'.join(lines))
