"""FSK46唯一字段转录候选；只解析输入及重建Forcing数组，不构造/推进求解器。"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from decimal import Decimal
import ast
import copy
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / '26_冻结基线与后续归因'
DEPS = FROZEN / '共享依赖'
sys.dont_write_bytecode = True
sys.path.insert(0, str(DEPS))
from wxq_pipedream.wxq_to_superlink import parse_wxq_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changes(old, new, location=()):
    if isinstance(old, dict):
        assert old.keys() == new.keys()
        return sum((changes(v, new[k], location + (k,)) for k, v in old.items()), [])
    if isinstance(old, list):
        assert len(old) == len(new)
        return sum((changes(a, b, location + (i,)) for i, (a, b) in enumerate(zip(old, new))), [])
    return [] if old == new else [dict(path=list(location), before=old, after=new)]


def main():
    baseline = FROZEN / '共享输入/C07.json'
    proof_path = HERE / '原始表行与分水坐标证据.json'
    proof = json.loads(proof_path.read_text())
    for path, value in proof['source_sha256'].items():
        p = Path(path)
        if not p.is_absolute():
            p = ROOT.parents[1] / p
        assert sha(p) == value
    original_hash = sha(baseline)
    raw = json.loads(baseline.read_text())
    candidate = copy.deepcopy(raw)
    value = float(Decimal('478906.13'))
    field = candidate['中线全线']['baseData']['bleeders']['FSK46-2-pool25']['channel']
    assert field == ['POOL25-2-DHX35DHX36', 478906]
    field[1] = value
    diff = changes(raw, candidate)
    assert diff == [dict(path=['中线全线', 'baseData', 'bleeders', 'FSK46-2-pool25', 'channel', 1],
                         before=478906, after=value)]
    common = DEPS / '水力运行脚本/试验公共.py'
    definitions = [x for x in ast.parse(common.read_text()).body
                   if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in ['meshes', 'weights_at', 'Forcing']]
    binding = ModuleType('_41_frozen_forcing')
    binding.__dict__.update(np=np, pd=pd, re=re, Path=Path, R=ROOT)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(common), 'exec'), binding.__dict__)
    normalization_path = FROZEN / '共享输入/2025开度归一化上限.json'
    normalization = json.loads(normalization_path.read_text())
    forcing = []
    for model in [raw['中线全线'], candidate['中线全线']]:
        limits = {k: float(v.get('b', 1) or 1) for k, v in model['baseData']['gates'].items()}
        limits.update(normalization)
        converted = parse_wxq_json(model, gate_max_openings=limits, section_geometry_mode='safe', minimum_initial_depth_m=7.)
        m = SimpleNamespace(M=len(converted['superjunctions']), NIk=len(converted['junctions']),
                            n_o=len(converted['orifices']), bc=converted['superjunctions'].bc.to_numpy(bool))
        e = SimpleNamespace(model=m, node_index=dict(zip(converted['superjunctions'].name, converted['superjunctions'].id)),
                            gate_index=dict(zip(converted['orifices'].name, converted['orifices'].id)))
        forcing.append(binding.Forcing(e, model, converted))
    array_checks = {}
    for key in ['times', 'h', 'q', 'q0', 'u']:
        a, b = [getattr(f, key) for f in forcing]
        assert np.array_equal(a, b, equal_nan=True)
        ah = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        bh = hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
        assert ah == bh
        array_checks[key] = dict(shape=list(a.shape), exact_equal_including_nan=True, original_sha256=ah, candidate_sha256=bh)
    series = raw['中线全线']['timeSeries']
    sid = next(k for k, target in series['seriesLink'].items() if target == ['bleeders', 'FSK46-2-pool25'])
    values = np.asarray(series['seriesData'][sid], float)[:, 1]
    assert len(values) == 1441 and np.array_equal(values, np.zeros(1441))
    candidate_path = HERE / 'C07_分水口桩号转录修正候选.json'
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    assert changes(raw, json.loads(candidate_path.read_text())) == diff and sha(baseline) == original_hash
    record = dict(status='ISOLATED_SINGLE_FIELD_CANDIDATE_VERIFIED', changes=diff,
                  source=dict(file='01_原始资料/中线全线建模数据.xlsx', sheet='分水口', cell='C47',
                              raw_numeric_token=proof['FSK46_decimal_check']['xml_numeric_token'],
                              source_scale_to_model_m=1000, decimal_check=proof['FSK46_decimal_check'],
                              unit_basis=proof['coordinate_scale_note']),
                  forcing_arrays=array_checks, FSK46_zero_flow_samples=1441,
                  routing_before=next(x for x in forcing[0].routing if x['target'] == 'FSK46-2-pool25'),
                  routing_after=next(x for x in forcing[1].routing if x['target'] == 'FSK46-2-pool25'),
                  solver_constructed=False, hydraulic_steps=0, assimilation=False, full_month_run_needed_for_zero_effect=False,
                  interpretation='旧已知分水口桩号转录差异现完成隔离修复候选；本月FSK46全零，因此执行输入逐值不变，不是主要精度修复，不覆盖基线。',
                  baseline_unchanged=True, candidate_path=str(candidate_path),
                  source_sha256={str(p): sha(p) for p in [baseline, proof_path, common, normalization_path, Path(__file__)]},
                  candidate_sha256=sha(candidate_path))
    (HERE / '分水口桩号字段差分与执行回归.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(dict(status=record['status'], candidate=str(candidate_path), field_changes=len(diff),
                          all_Forcing_arrays_exact_equal=True, hydraulic_steps=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
