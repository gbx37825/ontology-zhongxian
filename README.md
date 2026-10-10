# 中线全线30天水动力模型精度改进资料

本仓库交付2026年10月5日冻结的云端运行资料包，于2026年10月10日上传。它不是10月10日最新模型，也不包含京石段附件8的36步复验工作。

## 下载完整资料

**云端运行需要先下载完整资料包。只克隆仓库不能运行完整试验。** Git中保留代码、说明和验证证据；原始工程表、实测数据、初态及完整参考结果位于Release压缩包。

- [下载完整资料包，约935 MB](https://github.com/gbx37825/ontology-zhongxian/releases/download/cloud-data-20261005-v1/dot-cloud-30day-20261005.zip)
- [Release交付页](https://github.com/gbx37825/ontology-zhongxian/releases/tag/cloud-data-20261005-v1)
- [先读我：给dot的任务说明](dot云端30天精度改进资料包_20261005/先读我_给dot的任务说明.md)
- [历史聊天正文](dot云端30天精度改进资料包_20261005/历史对话/对话正文.md)

## 在云端运行

完整压缩包解压后会生成 `dot云端30天精度改进资料包_20261005` 目录。在仓库根目录解压可以补齐Git未跟踪的数据文件，已有同名文件与压缩包保持一致。

```bash
curl -L --fail -o dot-cloud-30day-20261005.zip https://github.com/gbx37825/ontology-zhongxian/releases/download/cloud-data-20261005-v1/dot-cloud-30day-20261005.zip
python3 - <<'PYCODE'
import hashlib
from pathlib import Path
p = Path('dot-cloud-30day-20261005.zip')
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == '2871eeb8e5d8040b53ca5c31e1a252744763517d8f174bf874db065e15eac2a4'
print('SHA256 verified')
PYCODE
unzip -q dot-cloud-30day-20261005.zip -d 完整运行资料
cd 完整运行资料/dot云端30天精度改进资料包_20261005
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python 校验资料包.py
python 云端运行.py smoke
python 云端运行.py reference
```

新候选完整运行后，使用 `云端评分.py` 按固定口径评分。具体命令、资料索引及改进要求见先读我。Release下载无需借助本机路径；若运行环境无法访问GitHub，可手动上传下载后的压缩包。

## 已有验证及边界

资料包内2858项文件的清单校验通过；原运行入口50项来源指纹通过。30分钟短程复现覆盖59座闸站，水位与流量最大差约2.84e-14；原参考自身配对评分的MAE、RMSE差值为零。完整入口启动检查通过。

上述验证是在原本机环境完成，本次打包未重新跑完整30天，也未宣称Linux或其他云端平台已通过验证。原完整30天参考结果已附。基础模型仍未通过精度改进门槛，候选未替换基线或部署。

旧逐成功子步残差分块、缓存、日志及重复历史压缩包未纳入，详见资料包内逐项清单；运行代码可重新生成分块。原始工程资料、模型输入、完整参考过程及历史核验结论已保留。
