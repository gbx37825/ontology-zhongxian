"""只适配路径与输出目录；水力算法、输入、评分口径保持原样。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parent
DATA = ROOT / '资料/中线2025同工况精度验证_20260921'
OLD = Path('/Users/gaobingxiang/code/HydroOS-hangzhou/outputs/中线2025同工况精度验证_20260921')
RUNNER = DATA / '41_基础模型深入检查与修复/10_端面几何锁参验证/运行端面几何配对.py'


def relocated(path):
    path = Path(path)
    return DATA / path.relative_to(OLD) if path.is_relative_to(OLD) else path


def load_runner(output):
    source = RUNNER.read_text()
    old = 'out=HERE/case;out.mkdir(parents=True,exist_ok=False)'
    assert source.count(old) == 1
    source = source.replace(old, 'out=CLOUD_OUTPUT/case;out.mkdir(parents=True,exist_ok=False)')
    module = types.ModuleType('_cloud_original_runner')
    module.__file__ = str(RUNNER)
    module.CLOUD_OUTPUT = output
    exec(compile(source, str(RUNNER), 'exec'), module.__dict__)
    # 历史配置中的原绝对路径仅用于读取同一份原文件并核对原 SHA256。
    module.sha = lambda p: hashlib.sha256(relocated(p).read_bytes()).hexdigest()
    return module


def smoke(module, output):
    import numpy as np
    import pandas as pd
    shared = None
    try:
        shared, common, advance, restore, engine, source, converted = module.build_runtime()
        model = engine.model
        cfg = json.loads((module.REFERENCE06 / '运行配置.json').read_text())
        with np.load(module.B / '共享输入/初始化后状态_0s.npz') as z:
            restore(model, {k: z[k].copy() for k in module.STATE_KEYS})
        model.t = 0.
        model.iter_count = 0
        assert {k: module.arrsha(getattr(model, k)) for k in module.STATE_KEYS} == cfg['initial_state_sha256']
        assert {k: module.arrsha(getattr(model, k)) for k in module.PARAM_KEYS} == cfg['parameter_sha256']
        forcing = common.Forcing(engine, source, converted)
        assert {k: module.arrsha(getattr(forcing, k)) for k in ['times', 'h', 'q', 'q0', 'u']} == cfg['forcing_array_sha256']
        native = model.step
        def step(**kw):
            kw.update(flow_tol=1e-9, head_tol=.0015)
            return native(**kw)
        model.step = step
        advance(model, forcing, 0., 1800.)
        assert model.t == 1800. and all(np.isfinite(getattr(model, k)).all() for k in module.STATE_KEYS)
        rows = []
        for station, group in converted['orifices'].assign(st=converted['orifices'].name.str.split('-').str[0]).groupby('st'):
            rows.append(dict(station=station, forebay_m=float(model.H_j[int(group.iloc[0].sj_0)]),
                afterbay_m=float(model.H_j[int(group.iloc[0].sj_1)]), gate_flow_m3s=float(model.Q_o[group.id.to_numpy(int)].sum())))
        ref = pd.read_csv(module.REFERENCE06 / '半小时水位流量.csv')
        ref = ref[ref.time_s == 1800.].set_index('station').sort_index()
        got = pd.DataFrame(rows).set_index('station').sort_index()
        assert list(got.index) == list(ref.index)
        columns = ['forebay_m', 'afterbay_m', 'gate_flow_m3s']
        errors = {k: float(abs(got[k] - ref[k]).max()) for k in columns}
        assert max(errors.values()) < 1e-8, errors
        output.mkdir(parents=True, exist_ok=False)
        report = dict(status='PASSED', simulated_seconds=1800, stations=len(rows),
            initial_state_parameter_forcing_hashes_match=True, max_absolute_errors=errors,
            smoke_tolerance=1e-8, full_30day_run_performed=False, runtime=module.runtime_identity())
        (output/'短程验证.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if shared is not None:
            shutil.rmtree(shared.CACHE, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'reference', 'endpoint'])
    parser.add_argument('--output', type=Path, default=ROOT/'云端新结果')
    args = parser.parse_args()
    output = args.output.resolve()
    # 不允许新结果写入历史证据目录。
    if output.is_relative_to(DATA):
        parser.error('输出目录必须在历史资料目录之外')
    module = load_runner(output)
    if args.mode == 'smoke':
        smoke(module, output/'短程验证')
    else:
        module.run_branch(args.mode)
