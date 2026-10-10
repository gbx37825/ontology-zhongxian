"""复用原评分程序；本入口只做精度统计，不代替水量与物理检查。"""
from pathlib import Path
import argparse
import importlib.util
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'资料/中线2025同工况精度验证_20260921'
SAVED = DATA/'41_基础模型深入检查与修复/10_端面几何锁参验证'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=SAVED/'reference')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(DATA):
        parser.error('输出不能覆盖历史资料')
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('_cloud_scores', SAVED/'汇总端面几何锁参验证.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observations = module.stats07.observations()
    frames = []
    for label, folder in [('reference', args.reference), ('endpoint', args.candidate)]:
        result = json.loads((folder/'结果.json').read_text())
        assert result['status'] == 'COMPLETED' and result['completed_cycles'] == 1440, '只对完整成功30天评分'
        meters = pd.read_csv(folder/'测流位置算子流量.csv').rename(columns={'model_flow_m3s': 'sim'})
        module.stats07.check_grid(meters, ['station'], 44)
        meters['metric'] = 'flow_m3s'
        frames.append(module.stats07.load_scores(folder, label, observations, meters))
    module.HERE = output
    report = module.write_accuracy(frames, json.loads((SAVED/'运行前约定.json').read_text()))
    report['candidate_label_note'] = '为复用原程序，endpoint列在本入口代表所传candidate；不表示已安装端面修复。'
    report['folders'] = {'reference': str(args.reference.resolve()), 'candidate': str(args.candidate.resolve())}
    report['engineering_acceptance'] = False
    report['numerical_and_physical_audits_performed_by_this_entry'] = False
    (output/'精度比较.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report['conditional_accuracy_gate'], ensure_ascii=False, indent=2))
