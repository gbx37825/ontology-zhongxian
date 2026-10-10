from pathlib import Path
import numpy as np,pandas as pd,json
O=Path(__file__).resolve().parent;B=O.parent/'26_冻结基线与后续归因';rows=[];summ=[]
for label,folder,phase in [('改水位同化','B01_允许修改水位的状态同化参考基线','analysis_after_update'),('改参数同化','B02_渠池归属修复后的参数同化诊断基线','forecast_before_update')]:
 d=pd.read_csv(B/folder/'同化运行/预报与分析逐点.csv');d=d[(d.operator=='闸端全线')&(d.phase==phase)].copy();d['error']=d.sim-d.observed
 for met,title in [('forebay_m','闸前'),('afterbay_m','闸后')]:
  p=d[d.metric==met].pivot(index='time_s',columns='station',values='error').sort_index();assert p.shape==(1440,59) and np.isfinite(p).all().all();windows=np.lib.stride_tricks.sliding_window_view(p.to_numpy(),12,axis=0);mae=np.abs(windows).mean(axis=(1,2));rmse=np.sqrt((windows**2).mean(axis=(1,2)));mx=np.abs(windows).max(axis=(1,2));perstation=np.abs(windows).mean(axis=2)
  for k in range(len(mae)):rows.append(dict(baseline=label,position=title,start_s=int(p.index[k]-1800),end_s=int(p.index[k+11]),mae=float(mae[k]),rmse=float(rmse[k]),max_error=float(mx[k]),station_window_mae_le_0p1=int((perstation[k]<=.1).sum())))
  summ.append(dict(baseline=label,position=title,windows=len(mae),min_window_mae=float(mae.min()),median_window_mae=float(np.median(mae)),max_window_mae=float(mae.max()),global_window_mae_le_0p1=int((mae<=.1).sum()),best_window_start_s=int(p.index[int(mae.argmin())]-1800),worst_window_start_s=int(p.index[int(mae.argmax())]-1800)))
pd.DataFrame(rows).to_csv(O/'滚动六小时窗口逐窗精度.csv',index=False);s=pd.DataFrame(summ);s.to_csv(O/'滚动六小时窗口摘要.csv',index=False);print(s.to_string(index=False))
(O/'滚动六小时统计说明.md').write_text('''# 滚动6小时窗口统计

每半小时滑动一次，30天共有1429个重叠窗口；每个窗口按(start,end]取12个半小时结果。没有重新初始化，也没有停止同化或重跑水力计算。B01使用修正后分析水位，B02使用参数同化持续计算的水位，不能互称独立6小时预报。

逐窗精度按同化与既定有效实测的差统计，闸前和闸后各59个测点；每窗包含708个误差样本。摘要中的窗口MAE≤0.1m只指该窗口全体测点平均，不代表所有测点全时刻达标；相邻窗口大量重叠，不能作为独立试验数。

文件：滚动六小时窗口逐窗精度.csv、滚动六小时窗口摘要.csv。
''')
