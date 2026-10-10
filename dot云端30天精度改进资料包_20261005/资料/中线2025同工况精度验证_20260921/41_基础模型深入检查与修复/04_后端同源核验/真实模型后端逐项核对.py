"""相同真实模型/状态下对账两后端；不修改冻结源码或作参数率定。"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/pipedream_frozen_41_numba')
sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FULL = ROOT / '38_测点假设下的联合辨识/06_全线联合参数转移/全线固定参数转移.py'
spec = importlib.util.spec_from_file_location('full38_audit41', FULL)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
api = full.api
started = time.time()
old = api._physics._legacy_build
try:
    api._physics._legacy_build = full.frozen_full_build
    numba_engine, source, converted = api._physics.build_engine_physics(gate_max_normalization=api.data()[2])
finally:
    api._physics._legacy_build = old
standard_engine = api.PipedreamHydraulicEngine(copy.deepcopy(converted), dt=15., backend='standard')
standard_engine.model._method = 'lsq'
api._physics.attach_parallel_physics(standard_engine, converted)
forcing = api.Forcing(numba_engine, source, converted)
models = [standard_engine.model, numba_engine.model]
assert models[0].M == models[1].M and models[0].NIk == models[1].NIk
assert all(np.array_equal(getattr(models[0], key), getattr(models[1], key))
           for key in ['_n_ik', '_Co', '_z_inv_j', '_z_inv_Ik', '_dx_ik'])

KEYS = ['_A_ik', '_B_ik', '_Pe_ik', '_R_ik', '_A_uk', '_B_uk', '_Pe_uk', '_R_uk',
        '_A_dk', '_B_dk', '_Pe_dk', '_R_dk', '_A_sj', '_V_sj',
        '_u_ik', '_u_Ik', '_u_uk', '_u_dk', '_a_ik', '_b_ik', '_c_ik', '_P_ik',
        '_E_Ik', '_D_Ik', '_U_Ik', '_V_Ik', '_W_Ik', '_X_Ik', '_Y_Ik', '_Z_Ik',
        '_alpha_uk', '_beta_uk', '_chi_uk', '_alpha_dk', '_beta_dk', '_chi_dk',
        '_alpha_o', '_beta_o', '_chi_o', '_D_j', 'b', 'H_j', 'h_Ik', 'Q_ik', 'Q_uk', 'Q_dk', 'Q_o']


def differences(stage):
    result = []
    for key in KEYS:
        if not all(hasattr(m, key) for m in models):
            continue
        a, b = [np.asarray(getattr(m, key)) for m in models]
        assert a.shape == b.shape
        if a.size == 0:
            continue
        difference = np.abs(a - b)
        tol = 1e-9 + 1e-10 * np.maximum(np.abs(a), np.abs(b))
        result.append(dict(stage=stage, array=key, size=a.size,
                           all_finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                           max_absolute_difference=float(np.max(difference)),
                           index_of_max=int(np.argmax(difference)),
                           count_above_numerical_tolerance=int(np.count_nonzero(difference > tol))))
    return result


cases = []
paths = [(0., api.FROZEN / '共享输入/初始化后状态_0s.npz'),
         (2592000., ROOT / '38_测点假设下的联合辨识/06_全线联合参数转移/G00_冻结原参数/末态.npz')]
for timestamp, path in paths:
    with np.load(path) as z:
        states = {k: z[k].copy() for k in api.STATE_KEYS}
    for m in models:
        for key, values in states.items():
            getattr(m, key)[:] = values
        m.t = timestamp
        m.iter_count = 0
    kw = forcing.at(timestamp)
    kw.pop('num_iter')
    for m in models:
        m._setup_step(**kw, first_iter=True)
    rows = differences('same_state_setup')
    for m in models:
        m._solve_step(**kw, first_iter=True)
    rows.extend(differences('one_linearized_solve'))
    cases.append(dict(time_s=timestamp, state_path=str(path),
                      state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows))

result = dict(status='COMPLETED', scope='真实全线共同状态代数与单次线性化核验，非独立真值证明/非精度评分',
              cases=cases, runtime=api.runtime_identity(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              full_builder_sha256=hashlib.sha256(FULL.read_bytes()).hexdigest(),
              original_inputs_modified=False, baseline_replaced=False, wall_seconds=time.time()-started)
(OUT / '真实模型后端逐项核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'status': 'COMPLETED', 'wall_seconds': result['wall_seconds'],
                  'differences': [{ 'time_s':c['time_s'], **row} for c in cases for row in c['rows']
                                  if row['count_above_numerical_tolerance']]}, ensure_ascii=False))
