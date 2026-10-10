"""有限的代数/接口自检；借用冻结滤波器方法，不构建或运行水力求解器。"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json

import numpy as np


HERE = Path(__file__).resolve().parent
B = HERE.parents[1] / "26_冻结基线与后续归因"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mechanism = load(HERE / "条件性同化机制.py", "conditional40")
parameter = load(B / "B02_渠池归属修复后的参数同化诊断基线/运行源码/拓扑参数EnKF.py", "parameter26")
state = load(B / "B01_允许修改水位的状态同化参考基线/运行源码/连续EnKF核心.py", "state26")


class AlgebraModel:
    """仅提供滤波接口和可重算几何，明确没有 step 方法。"""
    def __init__(self):
        self.H_j = np.array([120., 120.2, 120.4])
        self.h_Ik = np.linspace(5., 5.4, 61)
        self.Q_ik = np.linspace(20., 21., 60)
        self.Q_uk = np.array([20.])
        self.Q_dk = np.array([21.])
        self.Q_o = np.full(59, 20.)
        self._n_ik = np.full(60, .015)
        self._Co = np.full(59, .84)
        self._i_1k, self._i_nk = np.array([0]), np.array([59])
        self._n_uk, self._n_dk = self._n_ik[self._i_1k], self._n_ik[self._i_nk]
        self._dx_ik = np.full(60, 1000.)
        self._dx_uk = self._dx_dk = np.zeros(1)
        self._z_inv_j = np.full(3, 115.)
        self.min_depth = .01
        self.t = 1800.
        self.iter_count = 120
        self.load_state({})
        self.compute_storage_volumes()

    def load_state(self, states, compute_hydraulic_geometries=True):
        for key, value in states.items():
            setattr(self, key, value.copy())
        if compute_hydraulic_geometries:
            self.A_ik = 15. * (self.h_Ik[:-1] + self.h_Ik[1:]) / 2.
            self.A_uk = np.array([15. * self.h_Ik[0]])
            self.A_dk = np.array([15. * self.h_Ik[-1]])

    def compute_storage_volumes(self):
        self.V_j = 2. * (self.H_j - self._z_inv_j)


def fixture():
    ens = parameter.ParameterEnKF.__new__(parameter.ParameterEnKF)
    ens.n, ens.radius = 50, 20000.
    base = AlgebraModel()
    ens.members = [copy.deepcopy(base) for _ in range(50)]
    ens.base_n, ens.base_C = base._n_ik.copy(), base._Co.copy()
    ens.parameters = [dict(name=f"POOL{i+1}_n", kind="n", indices=[i],
                           xmin=i * 1000., xmax=(i+1) * 1000., lower=.5333333333333, upper=1.5)
                      for i in range(60)]
    ens.parameters += [dict(name=f"JZZ{i+1}_C", kind="C", indices=[i],
                            xmin=(i+.5) * 1000., xmax=(i+.5) * 1000., lower=.5, upper=1.1897194722116355)
                       for i in range(59)]
    ens.lower = np.log([p["lower"] for p in ens.parameters])[:, None]
    ens.upper = np.log([p["upper"] for p in ens.parameters])[:, None]
    ens.theta = np.zeros((119, 50))
    ens.log = []
    view = state.LocalEnKF.__new__(state.LocalEnKF)
    view.n, view.radius, view.members = 50, 20000., ens.members
    view.sizes = {k: getattr(base, k).size for k in mechanism.STATE_KEYS}
    view.slices = {}
    offset = 0
    for key, size in view.sizes.items():
        view.slices[key] = slice(offset, offset + size)
        offset += size
    view.xparts = dict(H_j=np.array([0., 30000., 60000.]), h_Ik=np.linspace(0, 60000, 61),
                       Q_ik=np.linspace(500, 59500, 60), Q_uk=np.array([0.]),
                       Q_dk=np.array([60000.]), Q_o=np.linspace(500, 58500, 59))
    view.x = np.concatenate([view.xparts[k] for k in mechanism.STATE_KEYS])
    view.heads = np.zeros(offset, bool)
    view.heads[view.slices["H_j"]] = True
    view.heads[view.slices["h_Ik"]] = True
    view.mask = np.ones(offset, bool)
    view.mask[[0, 2]] = False
    ens.xparts = {k: view.xparts[k].copy() for k in ("H_j", "h_Ik")}
    view.log = []
    before = [view.pack(m) for m in ens.members]
    physical_lower, physical_upper = ens.lower.copy(), ens.upper.copy()
    init = mechanism.initialize(ens, view,
                                {"POOL16_n": 1.3866234488251865, "JZZ16_C": 1.188325359804508},
                                20260923)
    assert np.array_equal(ens.physical_lower, physical_lower)
    assert np.array_equal(ens.physical_upper, physical_upper)
    assert all(np.array_equal(view.pack(m), old) for m, old in zip(ens.members, before))
    assert init["clipped_entries"] > 0 and view.members is ens.members
    # 模拟一个已有先验的相关性，仅代数赋值，无水力推进或实测数据。
    latent = (ens.theta[15] - ens.theta[15].mean()) / .025
    for i, model in enumerate(ens.members):
        model.h_Ik += .05 * latent[i]
        model.H_j[1] += .04 * latent[i]
        model.Q_ik += 25. * latent[i]
        model.Q_o += 15. * latent[i]
        model.load_state({})
        model.compute_storage_volumes()
        # 故意留下不同于当前几何的旧缓存，验证更新前体积不是直接读此缓存。
        model.A_ik *= 7.
        model.V_j *= 9.
    return ens, view, init


def packed(ens, view):
    return np.stack([view.pack(m) for m in ens.members], axis=1)


def main():
    ens, view, init = fixture()
    obs = [dict(station=f"T{i}", metric="forebay_m", state="h_Ik", indices=[j],
                weights=[1.], offset=115., sigma=.03, time_s=1800.)
           for i, j in enumerate([15, 55, 35, 45])]
    Y = np.stack([ens.values(o) for o in obs])
    for o, value in zip(obs, Y.mean(axis=1) + np.array([.12, 50., -.08, .04])):
        o["value"] = float(value)
    noise = np.random.default_rng(91).normal(size=Y.shape)
    base_X = packed(ens, view)
    base_theta = ens.theta.copy()
    runs = {}
    for name, do_p, do_s in [("P", True, False), ("S", False, True), ("PS", True, True)]:
        e, v = copy.deepcopy((ens, view))
        audit = mechanism.analysis(e, v, obs, noise, do_p, do_s, elapsed_cycles=2)
        runs[name] = (e, v, audit)
        assert audit["accepted_observation_indices"] == [0, 2, 3]
    p, pv, pa = runs["P"]
    s, sv, sa = runs["S"]
    joint, jv, ja = runs["PS"]
    assert np.array_equal(p.theta, joint.theta)
    assert np.array_equal(packed(s, sv), packed(joint, jv))
    assert np.array_equal(packed(p, pv), base_X)
    assert np.array_equal(s.theta, base_theta)
    assert ja["parameter_dirty"] and ja["state_dirty"]
    assert np.array_equal(packed(joint, jv)[~jv.mask], base_X[~view.mask])
    assert np.all(joint.theta >= joint.lower) and np.all(joint.theta <= joint.upper)
    assert abs(joint.theta - base_theta).max() <= 2*np.log(1.01) + 1e-12
    assert ja["state_analysis"]["max_head_change_m"] <= .02 + 1e-12
    assert ja["state_analysis"]["max_flow_change_m3s"] <= 3. + 1e-12
    assert min(ja["state_analysis"]["member_increment_scales"]) < 1.
    assert pa["state_dirty"] is False and sa["parameter_dirty"] is False
    # 拒收中间行后，保留下来的观测仍使用它们原来的噪声行。
    keep = [0, 2, 3]
    e, v = copy.deepcopy((ens, view))
    mechanism.analysis(e, v, [obs[i] for i in keep], noise[keep], True, True, elapsed_cycles=2)
    assert np.array_equal(e.theta, joint.theta)
    assert np.array_equal(packed(e, v), packed(joint, jv))
    # 常量窗口 Y 无集合协方差，应不更新；不能暗中再读取当前成员 Y。
    e, v = copy.deepcopy((ens, view))
    e.values = lambda _: (_ for _ in ()).throw(AssertionError("Y_override 被忽略"))
    override = np.broadcast_to(Y.mean(axis=1)[:, None], Y.shape).copy()
    supplied = mechanism.analysis(e, v, obs, noise, True, True, Y_override=override)
    assert np.allclose(e.theta, base_theta, rtol=0, atol=1e-14)
    assert np.allclose(packed(e, v), base_X, rtol=0, atol=1e-12)
    assert supplied["Y_source"] == "provided_window_forecast"
    # 以逐行原公式交叉检查生产代码的分组/批量 solve。
    rng = np.random.default_rng(5)
    prior = rng.normal(size=(8, 50)); forecast = rng.normal(size=(3, 50))
    values = np.array([.1, -.2, .3]); z = noise[keep].copy(); z -= z.mean(axis=1, keepdims=True)
    sigma = np.full(3, .03); xmin = np.linspace(0, 70000, 8); xmax = xmin+1000
    xpos = np.array([5000., 22000., 46000.]); mask = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    got = mechanism._local_increment(prior, forecast, values, z, sigma, xmin, xmax, xpos, 20000., mask)
    expected = np.zeros_like(got); A = forecast-forecast.mean(axis=1, keepdims=True)
    P = prior-prior.mean(axis=1, keepdims=True)
    for j in range(len(prior)):
        distance = np.maximum(np.maximum(xmin[j]-xpos, xpos-xmax[j]), 0)
        select = (distance < 20000.) & mask[j]
        if not select.any():
            continue
        loc = .5*(1+np.cos(np.pi*distance[select]/20000.))
        sd = sigma[select]/np.sqrt(np.maximum(loc, 1e-12)); Bsel = A[select]
        cov = Bsel @ Bsel.T/49 + np.diag(sd**2); cross = P[j] @ Bsel.T/49
        gain = np.linalg.solve(cov, cross)
        expected[j] = gain @ (values[select, None] + z[select]*sd[:, None] - forecast[select])
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)
    # 同口径几何体积核算，原始旧缓存体积不应被当作 before。
    correct_before = []
    for m in ens.members:
        c = copy.deepcopy(m); view.unpack(c, view.pack(c)); correct_before.append(state.volume(c))
    assert np.array_equal(correct_before, ja["state_analysis"]["geometric_volume_before_m3"])
    after = [state.volume(m) for m in joint.members]
    assert np.array_equal(np.asarray(after)-correct_before, ja["state_analysis"]["delta_volume_members_m3"])
    assert not np.array_equal([state.volume(m) for m in ens.members], correct_before)
    # 散度恢复可超过分析 1% 限制，但不得越有效域或改水力状态。
    e, v = copy.deepcopy((ens, view)); e.theta[:] = np.log(e.center_ratios)[:, None]; e.apply()
    before = packed(e, v)
    rejuv = mechanism.rejuvenate(e, np.random.default_rng(19).normal(size=(119, 50)))
    assert rejuv["rejuvenated_parameters"] == 119
    assert rejuv["max_abs_log_change"] > np.log(1.01)
    assert np.array_equal(before, packed(e, v))
    assert np.all(e.theta >= e.lower) and np.all(e.theta <= e.upper)
    report = dict(
        status="PASS_SYNTHETIC_ALGEBRA_AND_INTERFACE_CHECKS",
        hydraulic_simulation_started=False, frozen_filter_methods_used=True,
        checks=["joint_parameter_block_equals_P_exactly", "joint_state_block_equals_S_exactly",
                "P_six_states_unchanged", "S_theta_unchanged", "mask_and_domain_and_batch_caps",
                "noise_rows_preserved_after_rejection", "Y_override_used_without_current_Y",
                "batched_solve_matches_original_row_formula", "volume_before_refreshed_only_on_copy",
                "rejuvenation_not_mislabelled_as_1percent", "initialize_keeps_physical_bounds_and_states"],
        formula_max_abs_difference=float(abs(got-expected).max()),
        prior_clipped_entries=init["clipped_entries"],
        parameter_max_log_change=ja["parameter_analysis"]["max_abs_log_change"],
        state_max_head_change_m=ja["state_analysis"]["max_head_change_m"],
        state_max_flow_change_m3s=ja["state_analysis"]["max_flow_change_m3s"],
        delta_volume_mean_m3=ja["state_analysis"]["delta_volume_mean_m3"],
        mechanism_sha256=hashlib.sha256((HERE/"条件性同化机制.py").read_bytes()).hexdigest(),
    )
    target = HERE / "条件性同化机制自检结果.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
