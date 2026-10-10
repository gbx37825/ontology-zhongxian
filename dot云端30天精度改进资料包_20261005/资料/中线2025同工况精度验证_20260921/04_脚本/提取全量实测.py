"""只读xlsx到压缩CSV，保存工作表/行号/原值，不填补不修正。"""
from pathlib import Path
import sys,csv,json,gzip,zipfile,posixpath,datetime
from lxml import etree as E
from openpyxl.utils.datetime import from_excel
from 流式读取2025表 import NS,RNS,column_index,sha
SRC=Path('/Users/gaobingxiang/code/hydroclaw0418/datasources/中线/2025年数据');OUT=Path(__file__).resolve().parents[1]/'12_全量实测质量核查'
def extract(name):
 p=SRC/name;dest=OUT/(p.stem+'_全量原值.csv.gz');stats={'file':str(p),'sha256':sha(p),'sheets':[]};first=True
 with zipfile.ZipFile(p) as z,gzip.open(dest,'wt',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);strings=[]
  if 'xl/sharedStrings.xml' in z.namelist():
   root=E.fromstring(z.read('xl/sharedStrings.xml'));strings=[''.join(n.itertext()) for n in root]
  wb=E.fromstring(z.read('xl/workbook.xml'));rels=E.fromstring(z.read('xl/_rels/workbook.xml.rels'));rel={x.get('Id'):x.get('Target') for x in rels}
  def value(c):
   typ=c.get('t');v=c.find('{'+NS+'}v')
   if typ=='inlineStr':return ''.join(c.find('{'+NS+'}is').itertext())
   if v is None or v.text is None:return None
   if typ=='s':return strings[int(v.text)]
   if typ in ['str','e']:return v.text
   return v.text # 数值原文不改写；日期列单独格式化。
  for sheet in wb.find('{'+NS+'}sheets'):
   name=sheet.get('name');target=rel[sheet.get('{'+RNS+'}id')];target=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target));n=0;lo=None;hi=None;badtime=0;lastdate=None;laststr=None
   with z.open(target) as xml:
    for _,row in E.iterparse(xml,events=('end',),tag='{'+NS+'}row'):
     vals={column_index(c.get('r')):value(c) for c in row};ri=int(row.get('r'))
     if ri==1:
      heads=[vals.get(i,f'列{i+1}') for i in range(max(vals)+1)]
      if first:writer.writerow(['源工作表','源行号']+heads);canonical=heads;first=False
      assert heads==canonical,(name,heads,canonical)
     else:
      a=[vals.get(i) for i in range(len(heads))]
      try:
       serial=float(a[2]);lo=serial if lo is None else min(lo,serial);hi=serial if hi is None else max(hi,serial)
       if serial!=lastdate:lastdate=serial;laststr=from_excel(serial).isoformat(sep=' ')
       a[2]=laststr
      except (ValueError,TypeError,OverflowError):badtime+=1
      writer.writerow([name,ri]+a);n+=1
     row.clear()
     while row.getprevious() is not None:del row.getparent()[0]
     if n and n%200000==0:print(p.name,name,n,flush=True)
   stats['sheets'].append(dict(sheet=name,rows=n,start=None if lo is None else str(from_excel(lo)),end=None if hi is None else str(from_excel(hi)),bad_time=badtime,headers=heads))
 assert stats['sha256']==sha(p)
 (OUT/(p.stem+'_提取清单.json')).write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False),flush=True)
if __name__=='__main__':extract(sys.argv[1])
