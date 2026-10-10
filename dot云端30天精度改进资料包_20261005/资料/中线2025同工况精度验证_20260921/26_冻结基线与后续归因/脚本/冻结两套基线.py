from pathlib import Path
import shutil,json,hashlib,datetime
R=Path(__file__).resolve().parents[2];O=R/'26_冻结基线与后续归因';records=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sets=[('B01_允许修改水位的状态同化参考基线','20_连续状态同化','E01_20成员30天连续水位同化','E02_同集合30天无同化对照','连续同化运行.py','连续EnKF核心.py'),('B02_渠池归属修复后的参数同化诊断基线','24_参数同化根因核查','F03_拓扑分组修复30天','F04_拓扑分组匹配无同化30天','拓扑修复试验运行.py','拓扑参数EnKF.py')]
for label,folder,run,control,runner,filter_file in sets:
 target=O/label;target.mkdir(exist_ok=False);items=[]
 for role,name in [('同化运行',run),('匹配无同化对照',control)]:
  src=R/folder/name;status=json.loads((src/'进度.json').read_text());assert status['status']=='COMPLETED' and status['completed_cycles']==1440 and not (src/'失败.json').exists();shutil.copytree(src,target/role)
  for p in src.iterdir():
   if p.is_file():q=target/role/p.name;assert digest(p)==digest(q);items.append(dict(role=role,source=str(p),archived=str(q.relative_to(O)),sha256=digest(q),bytes=q.stat().st_size))
 config=json.loads((R/folder/run/'运行配置.json').read_text())
 for name,key in [(runner,'runner_sha256'),(filter_file,'filter_sha256')]:
  src=R/folder/'脚本'/name;assert digest(src)==config[key],f'源码指纹变化:{src}';(target/'运行源码').mkdir(exist_ok=True);shutil.copy2(src,target/'运行源码'/name)
 records.append(dict(baseline=label,source_run=run,source_control=control,role='状态同化精度参考，不满足不改水位约束' if folder.startswith('20') else '后续根因试验起点，不表示精度验收或生产采用',files=items,config=config))
# 依赖快照供审计，不将目录移位后的脚本冒充可直接运行的独立交付包。
for source,dest in [(R/'23_五十成员参数同化/脚本',O/'共享依赖/参数并行运行脚本'),(R/'17_求解器修复/运行脚本',O/'共享依赖/水力运行脚本'),(R/'04_脚本/并联结构物理修复.py',O/'共享依赖/并联结构物理修复.py'),(R/'17_求解器修复/C07_黄金河闭闸重建候选_2025同工况30天.json',O/'共享输入/C07.json'),(R/'02_输入/作者已核实修正观测层.csv',O/'共享输入/作者已核实修正观测层.csv'),(R/'01_原始资料/Pipedream闸流系数.json',O/'共享输入/Pipedream闸流系数.json'),(R/'03_运行参数/2025开度归一化上限.json',O/'共享输入/2025开度归一化上限.json'),(R/'06_结果/A03R_C07物理加初态自适应30天/初始化后状态_0s.npz',O/'共享输入/初始化后状态_0s.npz')]:
 dest.parent.mkdir(parents=True,exist_ok=True)
 if source.is_dir():shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__'))
 else:shutil.copy2(source,dest)
manifest={str(p.relative_to(O)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in O.rglob('*') if p.is_file()}
(O/'冻结文件校验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(O/'基线登记.json').write_text(json.dumps(dict(created=datetime.datetime.now().isoformat(),baselines=records),ensure_ascii=False,indent=2));print('冻结完成',len(manifest),'文件')
