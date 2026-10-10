from pathlib import Path
import json,re,collections
import openpyxl,numpy as np,pandas as pd
from openpyxl.utils import get_column_letter as col
R=Path(__file__).resolve().parents[1];w=openpyxl.load_workbook(R/'01_原始资料/中线全线建模数据.xlsx',read_only=True,data_only=True);m=next(iter(json.loads((R/'02_输入/B04_分水身份修正30天.json').read_text()).values()))['baseData'];rows=[]
def add(typ,target,field,source,actual,ref):
 equal=source is not None and actual is not None and abs(float(source)-float(actual))<1e-6
 rows.append(dict(type=typ,target=target,field=field,excel_value=source,json_value=actual,status='一致' if equal else '差异或缺失',evidence=ref))
def nam(s):return re.sub(r'^DHX\d*-','',str(s)).split('-pool')[0]
fields={3:'inlet_zb',4:'inlet_b',5:'inlet_m',7:'outlet_zb',8:'outlet_b',9:'outlet_m',10:'n_orifice',11:'b_orifice',12:'h_orifice',13:'l_orifice',14:'nc',15:'loss_coeff'}
matched=[]
for ri,v in enumerate(w['倒虹吸'].values,1):
 if ri==1:continue
 hits=[(k,z) for k,z in m['siphons'].items() if nam(k)==nam(v[1])]
 if len(hits)>1:hits=[(k,z) for k,z in hits if abs(z['l_orifice']-v[13])<1e-7]
 if len(hits)!=1:raise ValueError((ri,v[1],len(hits)))
 k,z=hits[0];matched.append(k)
 for c,f in fields.items():add('倒虹吸',k,f,v[c],z.get(f),f'倒虹吸!{col(c+1)}{ri}')
assert len(set(matched))==105
for ri,v in enumerate(w['节制闸'].values,1):
 if ri==1:continue
 prefix=v[1];hits=[(k,z) for k,z in m['gates'].items() if k.rsplit('-',1)[0]==prefix]
 add('节制闸',prefix,'hole_count',v[4],len(hits),f'节制闸!E{ri}')
 for k,z in hits:
  for c,f in [(5,'b'),(6,'zb')]:add('闸孔',k,f,v[c],z[f],f'节制闸!{col(c+1)}{ri}')
for ri,v in enumerate(w['分水口'].values,1):
 if ri==1:continue
 hits=[(k,z) for k,z in m['bleeders'].items() if k.startswith(f'FSK{int(v[0])}-')];assert len(hits)==1
 k,z=hits[0];add('分水口',k,'chainage_m',float(v[2])*1000,z['channel'][1],f'分水口!C{ri}')
# Geometry comparison at source channel/transition endpoints, preserving coincident different sections.
endpoints=[]
for sheet in ['渠道','渐变段']:
 for ri,v in enumerate(w[sheet].values,1):
  if ri==1:continue
  for side in ['in','out']:
   if sheet=='渠道':
    xc=2 if side=='in' else 3;zc=6 if side=='in' else 7;bc=4;mc=5;tc=8 if side=='in' else 9
   else:xc,zc,bc,mc=(2,3,4,5) if side=='in' else (6,7,8,9);tc=None
   if v[xc] is None:continue
   endpoints.append(dict(sheet=sheet,row=ri,side=side,chainage=round(float(v[xc])*1000,6),zb=float(v[zc]),b=float(v[bc]),slope=float(v[mc]),top=float(v[tc]) if tc is not None else None,evidence=f'{sheet}!{col(xc+1)}{ri}:{col((tc if tc is not None else mc)+1)}{ri}'))
def features(sec):
 yz=np.array(sec['yz']);zb=yz[:,1].min();ids=np.isclose(yz[:,1],zb,atol=1e-7,rtol=0);b=np.ptp(yz[ids,0]);slopes=[]
 for a,z in zip(yz,yz[1:]):
  if abs(z[1]-a[1])>1e-7:slopes.append(abs((z[0]-a[0])/(z[1]-a[1])))
 return dict(zb=float(zb),b=float(b),slopes=slopes,top=float(yz[:,1].max()))
sections=[dict(key=k,x=round(float(s['location']),6),**features(s)) for k,s in m['sections'].items()];byp=collections.defaultdict(list)
for s in sections:byp[s['x']].append(s)
geom=[]
for e in endpoints:
 ss=byp[e['chainage']];same=[s for s in ss if abs(s['zb']-e['zb'])<1e-6 and abs(s['b']-e['b'])<1e-6 and all(abs(v-e['slope'])<1e-6 for v in s['slopes'])]
 geom.append({**e,'status':'几何一致' if same else '同桩号几何不一致' if ss else '该端点未显式保留','json_sections':[s['key'] for s in ss],'geometry_candidates':ss})
res={'source':'01_原始资料/中线全线建模数据.xlsx','json':'02_输入/B04_分水身份修正30天.json','counts':{'siphons':105,'siphon_field_checks':105*len(fields),'gate_stations':59,'gate_holes':183,'bleeders':128,'json_sections':len(sections),'source_channel_rows':688,'source_transition_rows':605},'checks':rows,'numeric_differences':[x for x in rows if x['status']!='一致'],'rounding_note':'JZZ30单孔宽9.6066666667与9.606667差3.3e-7m，按1e-6m容差归一致；FSK46桩号0.13m差另列，不与大幅物理差异混淆。','geometry_endpoint_counts':dict(collections.Counter(x['status'] for x in geom)),'geometry_endpoints':geom,'coefficient_semantics':{'说明_sheet':[list(v) for v in w['说明'].values],'finding':'说明页只解释类型编码和边界类型，没有定义节制闸系数1/2/3公式；不可把0/0/0.5直接映射Pipedream C。JSON另有c1..c4=0.9及独立Pipedream闸流系数文件，属不同契约，未经公式来源确认不能互换。'},'matching_note':'倒虹吸按设施名匹配；两处同名沁河由单孔长度消歧，不以表序号对应JSON局部编号。源单孔尺寸按单孔对照，未将进口总宽当单孔宽。','transition_loss':'渐变段K/L为局部水力损失，源表605段；JSON无独立渐变段对象，合并sections/channels不含对应局部损失字段。几何匹配不代表损失已在求解器实现。'}
pd.DataFrame(rows).to_csv(R/'07_对比分析/原建模资料与JSON核对.csv',index=False);pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in z.items()} for z in geom]).to_csv(R/'07_对比分析/原建模渠道渐变段端点核对.csv',index=False);(R/'07_对比分析/原建模资料与JSON核对.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print('SUMMARY',res['counts'],res['geometry_endpoint_counts']);print('DIFF',res['numeric_differences'])
(R/'08_报告/原建模资料与JSON核对简报.md').write_text('# 原建模资料与JSON核对\n\n'+f'核对105座倒虹吸12字段、59闸183孔孔数/宽度/底高程、128分水口桩号；直接数值差异{len(res["numeric_differences"])}项。逐单元格证据见CSV/JSON。\n\n渠道及渐变段端点核对：{res["geometry_endpoint_counts"]}。端点未显式保留和同桩号几何不一致仅作为结构审核线索，不能自动判成计算错误，需核对是否被合并/插值以及结构节点含义。\n\n'+res['coefficient_semantics']['finding']+'\n\n'+res['transition_loss']+'\n\n刁河前关键例：渠道第7—9行与JSON q1s10—q1s13差异非单一端点双值可解释，源14610m进口应底140.7m、宽26m、边坡0，JSON该桩号为底138.828m、宽13.5m、边坡3.5。需单独回查原构模截面选取，不能直接用源端点覆盖全线。JZZ30宽度小数差已按1e-6m容差归舍入一致。\n\n本审核未改模型、未跑仿真。')

report=R/'08_报告/原建模资料与JSON核对简报.md'
with report.open('a') as out:
 out.write('\n\n## 数值差异\n\n|对象|字段|源表|JSON|证据|\n|---|---|---:|---:|---|\n')
 for x in res['numeric_differences']:out.write(f'|{x["target"]}|{x["field"]}|{x["excel_value"]}|{x["json_value"]}|{x["evidence"]}|\n')
