"""只读复核原Excel和C07断面，结果写本目录；不修改任何源文件。"""
from pathlib import Path
import hashlib
import json
import openpyxl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODEL = ROOT / '17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json'
BOOK = Path('/Users/gaobingxiang/code/hydroclaw0418/datasources/中线/中线全线建模数据.xlsx')
LEDGER = Path('/Users/gaobingxiang/code/十四五/节制闸+分水闸/64座节制闸+54座退水闸+97个分水口 +61个控制闸(2023-4).xlsx')
model = json.loads(MODEL.read_text())['中线全线']['baseData']
w = openpyxl.load_workbook(BOOK, read_only=True, data_only=True)
l = openpyxl.load_workbook(LEDGER, read_only=True, data_only=True)

def row(book, sheet, number):
    return {c.coordinate:c.value for c in book[sheet][number] if c.value is not None}

def geom(name):
    s = model['sections'][name]
    z = [p[1] for p in s['yz']]
    bottom = min(z)
    xs = [p[0] for p in s['yz'] if p[1] == bottom]
    return dict(section=name, location_m=s['location'], bottom_m=bottom,
                edge_m=max(z), height_m=max(z)-bottom, bottom_width_m=max(xs)-min(xs), yz=s['yz'])

checks = []
for name, sheet, xcol,zcol,bcol,r in [
    ('q26s5','渐变段','G','H','I',233),
    ('q13s38','渐变段','G','H','I',118),
    ('q30s23','渠道','C','G','E',321),
    ('q30s24','渠道','D','H','E',321),
    ('q15s3','渠道','C','G','E',153),
    ('q15s4','渠道','D','H','E',153),
]:
    g = geom(name)
    g['source'] = {'sheet':sheet,'range':f'A{r}:L{r}', 'cells':row(w,sheet,r)}
    g['source_geometry_equal'] = (
        abs(g['location_m']-w[sheet][f'{xcol}{r}'].value*1000)<1e-6 and
        abs(g['bottom_m']-w[sheet][f'{zcol}{r}'].value)<1e-10 and
        abs(g['bottom_width_m']-w[sheet][f'{bcol}{r}'].value)<1e-10)
    checks.append(g)
assert all(x['source_geometry_equal'] for x in checks)

def jump(a,b):
    ga,gb=geom(a),geom(b)
    return dict(start=a,end=b,length_m=gb['location_m']-ga['location_m'],
                bottom_change_m=gb['bottom_m']-ga['bottom_m'],
                edge_change_m=gb['edge_m']-ga['edge_m'])

audit = json.loads((ROOT/'30_分通道方法逐项测试/02_输入与测点核查/断面边缘语义核验.json').read_text())
snapshots=[]
for s in audit['baseline_mean_state_checks']:
    for ch in ['POOL15-1-JJZ15DHX20','POOL30-5-DHX53DHX54']:
        v=[x['above_model_section_edge_m'] for x in s['details'] if x['channel']==ch]
        snapshots.append(dict(case=s['case'],channel=ch,above_count=len(v),min_above_m=min(v),max_above_m=max(v)))

result = {
    'source_hashes': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL,BOOK,LEDGER]},
    'section_comparisons':checks,
    'geometry_headers': {s:row(w,s,1) for s in ['渠道','渐变段','倒虹吸']},
    'nearby_transition_rows':{f'渐变段!A{r}:L{r}':row(w,'渐变段',r) for r in [118,131,132,133,233,287,288,289]},
    'nearby_siphon_rows':{f'倒虹吸!A{r}:P{r}':row(w,'倒虹吸',r) for r in [17,20,21,38,54,55]},
    'ledger_rows':{f'{s}!A{r}:AA{r}':row(l,s,r) for s,r in [('节制闸',16),('节制闸',18),('节制闸',28),('汇总',97),('汇总',129),('汇总',130)]},
    'geometric_jumps':[jump(a,b) for a,b in [('q15s2','q15s3'),('q15s3','q15s4'),('q15s5','q15s6'),('q30s22','q30s23'),('q30s23','q30s24'),('q30s24','q30s25')]],
    'frozen_snapshot_above_range':snapshots,
    'endpoint_height_relations':[
        {'section':'q26s5','json_edge_m':105.397,'siphon_bottom_plus_hole_height_m':98+7.2,'upstream_channel_height_m':109.457-102.06,'json_height_m':105.397-98},
        {'section':'q13s38','json_edge_m':128.47,'siphon_bottom_plus_hole_height_m':120.05+7,'upstream_channel_height_m':131.114-122.694,'json_height_m':128.47-120.05},
    ],
    'scope':'只读查源，未认定实际墙顶/堤顶/有压顶板；不改数据、不运行水力过程。'
}
(OUT/'断面来源复核证据.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'matched_sections':len(checks),'geometry_checks_passed':all(x['source_geometry_equal'] for x in checks),'jumps':result['geometric_jumps'],'snapshot_summary':snapshots},ensure_ascii=False,indent=2))
w.close();l.close()
