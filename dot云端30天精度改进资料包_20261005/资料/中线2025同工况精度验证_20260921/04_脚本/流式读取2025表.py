"""只读大Excel，保留源行号，流式提取作者时间段及前置观测。"""
from pathlib import Path
import csv,json,hashlib,zipfile,posixpath,datetime
from lxml import etree as E
from openpyxl.utils.datetime import from_excel,to_excel
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
RNS='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
R=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def column_index(ref):
 n=0
 for ch in ref:
  if not ch.isalpha():break
  n=n*26+ord(ch)-64
 return n-1

def extract(path,sheet_name,out,start='2025-07-20T00:00:00',end='2025-08-21T12:00:00'):
 path=Path(path);out=Path(out);startdt=datetime.datetime.fromisoformat(start);enddt=datetime.datetime.fromisoformat(end);stats={'source':str(path),'sha256':sha(path),'sheet':sheet_name,'start':str(startdt),'end':str(enddt),'rows_scanned':0,'kept':0,'date_order_violations':0};lo=to_excel(startdt);hi=to_excel(enddt)
 with zipfile.ZipFile(path) as z:
  ss=[]
  if 'xl/sharedStrings.xml'in z.namelist():
   root=E.fromstring(z.read('xl/sharedStrings.xml'));ss=[''.join(n.itertext()) for n in root] # 此批共享字符串为少量站名/表头
  w=E.fromstring(z.read('xl/workbook.xml'));sheets=w.find('{'+NS+'}sheets');entry=next(s for s in sheets if s.get('name')==sheet_name);rid=entry.get('{'+RNS+'}id')
  rel=E.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(x.get('Target') for x in rel if x.get('Id')==rid);target=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
  old=None;writer=None
  with z.open(target) as xml,out.open('w',newline='',encoding='utf-8-sig') as dest:
   for _,row in E.iterparse(xml,events=('end',),tag='{'+NS+'}row'):
    ri=int(row.get('r'));cells=list(row);stats['rows_scanned']+=1
    def value(cell):
     v=cell.find('{'+NS+'}v');t=cell.get('t')
     if t=='inlineStr':return ''.join(cell.find('{'+NS+'}is').itertext())
     if v is None or v.text is None:return None
     if t=='s':return ss[int(v.text)]
     if t in ['str','e']:return v.text
     try:return float(v.text)
     except ValueError:return v.text
    if ri==1:
     head={column_index(c.get('r')):value(c) for c in cells};headers=[head.get(i,f'列{i+1}') for i in range(max(head)+1)];writer=csv.writer(dest);writer.writerow(['源行号']+headers);stats['headers']=headers
    else:
     cell=next((c for c in cells if c.get('r','').startswith('C') and not c.get('r','')[1:2].isalpha()),None);t=value(cell) if cell is not None else None
     if isinstance(t,(int,float)):
      if old is not None and t<old:stats['date_order_violations']+=1
      old=t
      if lo<=t<=hi:
       vals={column_index(c.get('r')):value(c) for c in cells};vals[2]=from_excel(t).isoformat(sep=' ');writer.writerow([ri]+[vals.get(i) for i in range(len(headers))]);stats['kept']+=1
    row.clear()
    while row.getprevious() is not None:del row.getparent()[0]
    if stats['rows_scanned']%200000==0:print(path.name,'rows',stats['rows_scanned'],'kept',stats['kept'],flush=True)
 assert stats['kept']>0;assert stats['sha256']==sha(path),'输入读取期间发生变化'
 out.with_suffix('.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n');return stats
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('sheet');p.add_argument('out');a=p.parse_args();print(extract(a.path,a.sheet,a.out))
