from pathlib import Path
import json,numpy as np,pandas as pd
p=Path('outputs/中线2025同工况精度验证_20260921/23_五十成员参数同化');report=[]
for code,old in [('P01','P01_50成员30天参数同化'),('P02','P02_50成员30天同集合无同化')]:
 new=p/(code+'_恢复一致性验证');status=json.loads((new/'进度.json').read_text());assert status['status']=='COMPLETED'
 row={'job':code,'checks':{}}
 for file in ('预报与分析逐点.csv','参数轨迹.csv','同化等效水量.csv','逐观测更新记录.csv'):
  if (new/file).stat().st_size<2:continue
  aa=pd.read_csv(p/old/file);bb=pd.read_csv(new/file)
  aa=aa[(aa.time_s>604800)&(aa.time_s<=626400)].reset_index(drop=True);bb=bb[bb.time_s>604800].reset_index(drop=True)
  assert aa.shape==bb.shape,(file,aa.shape,bb.shape)
  diff={}
  for col in aa:
   if pd.api.types.is_numeric_dtype(aa[col]) and aa[col].dtype!=bool:
    delta=float(np.nanmax(np.abs(aa[col].to_numpy()-bb[col].to_numpy()))) if len(aa) else 0
    assert delta<1e-9,(code,file,col,delta);diff[col]=delta
   else:assert aa[col].fillna('').equals(bb[col].fillna('')),(file,col)
  row['checks'][file]={'rows':len(bb),'max_differences':diff}
 ledger=pd.read_csv(new/'同化等效水量.csv');assert ledger.state_unchanged.all() and (ledger.analysis_delta_volume_mean_m3==0).all()
 row['status']='PASS';report.append(row)
(p/'恢复一致性验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
