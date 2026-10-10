"""40 条件性实验的批次局地更新；不构模、不推进水力、不读取数据文件。

ens 使用冻结 B02 拓扑 ParameterEnKF，view 使用冻结 B01 LocalEnKF。
两者共享 50 个成员。参数和状态增量均由同一先验和同一观测噪声计算；
状态上限是本实验整批次每成员 0.02 m / 3 m3/s，不是 B01 的逐观测上限。
观测时刻、窗口因果性、求解器和并行状态同步由调用方管理。
"""
import copy
import hashlib

import numpy as np


STATE_KEYS = ("H_j", "h_Ik", "Q_ik", "Q_uk", "Q_dk", "Q_o")
MEMBERS = 50
PARAMETERS = 119
RADIUS_M = 20000.0
SIGMA_M = 0.03


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _states(ens):
    return [{key: getattr(m, key).copy() for key in STATE_KEYS} for m in ens.members]


def _same_states(ens, before):
    return all(np.array_equal(getattr(m, key), saved[key])
               for m, saved in zip(ens.members, before) for key in STATE_KEYS)


def _check_ensemble(ens, view=None):
    if ens.n != MEMBERS or len(ens.members) != MEMBERS:
        raise ValueError("本试验要求 50 个成员")
    if len(ens.parameters) != PARAMETERS or ens.theta.shape != (PARAMETERS, MEMBERS):
        raise ValueError("本试验要求 B02 的 119 x 50 参数集合")
    if not np.isfinite(ens.theta).all():
        raise ValueError("参数集合包含非有限值")
    if float(ens.radius) != RADIUS_M:
        raise ValueError("参数局地化半径须为 20000 m")
    for bound in (ens.lower, ens.upper):
        if np.shape(bound) != (PARAMETERS, 1) or not np.isfinite(bound).all():
            raise ValueError("参数界形状或数值无效")
    if (ens.lower > ens.upper).any():
        raise ValueError("参数界交集为空")
    if (ens.theta < ens.lower - 1e-12).any() or (ens.theta > ens.upper + 1e-12).any():
        raise ValueError("当前参数已超出有效域，不能在分析中暗中修正")
    if view is not None:
        if view.n != MEMBERS or view.members is not ens.members:
            raise ValueError("view.members 必须绑定 ens.members，且为 50 成员")
        if float(view.radius) != RADIUS_M:
            raise ValueError("状态局地化半径须为 20000 m")


def _center_noise(noise, shape):
    z = np.array(noise, dtype=float, copy=True)
    if z.shape != shape or not np.isfinite(z).all():
        raise ValueError(f"噪声须为有限数组 {shape}")
    # 在拒收之前按完整观测身份去均值；不因接受集合改变随机数对应关系。
    z -= z.mean(axis=1, keepdims=True)
    return z


def _parameter_audit(ens, before, raw_delta, step_limit=None):
    delta = ens.theta - before
    ratio_change = np.expm1(delta)
    return dict(
        dirty=bool(np.any(delta != 0)),
        max_abs_log_change=float(np.abs(delta).max()),
        max_abs_relative_change=float(np.abs(ratio_change).max()),
        raw_max_abs_log_change=float(np.abs(raw_delta).max()),
        analysis_log_step_limit=step_limit,
        step_limited_entries=(int((np.abs(raw_delta) > step_limit).sum())
                              if step_limit is not None else None),
        clipped_to_domain_entries=int(np.count_nonzero(
            before + (np.clip(raw_delta, -step_limit, step_limit)
                      if step_limit is not None else raw_delta) != ens.theta)),
        at_lower_entries=int((ens.theta <= ens.lower + 1e-12).sum()),
        at_upper_entries=int((ens.theta >= ens.upper - 1e-12).sum()),
        max_abs_log_change_by_parameter=np.abs(delta).max(axis=1).tolist(),
        mean_log_change_by_parameter=delta.mean(axis=1).tolist(),
        before_theta_sha256=_sha(before), after_theta_sha256=_sha(ens.theta),
    )


def initialize(ens, view, center_by_name, seed):
    """建立共同先验和原物理界内的 ±5% 信任域；不改变任何 H/Q 或时钟。"""
    view.members = ens.members
    _check_ensemble(ens, view)
    names = [p["name"] for p in ens.parameters]
    if len(set(names)) != PARAMETERS or set(center_by_name) - set(names):
        raise ValueError("参数名称重复或中心含未知参数")
    required = {"POOL16_n", "JZZ16_C"}
    if not required.issubset(center_by_name):
        raise ValueError("须显式提供 POOL16_n 和 JZZ16_C 的既定中心倍率")
    center = np.array([center_by_name.get(name, 1.0) for name in names], float)
    if not np.isfinite(center).all() or (center <= 0).any():
        raise ValueError("中心倍率须为有限正数")
    if any(center[j] != 1.0 for j, name in enumerate(names) if name not in required):
        raise ValueError("仅 POOL16_n/JZZ16_C 可以使用非 1 的中心")
    physical_lower = np.array(getattr(ens, "physical_lower", ens.lower), copy=True)
    physical_upper = np.array(getattr(ens, "physical_upper", ens.upper), copy=True)
    log_center = np.log(center)[:, None]
    if (log_center < physical_lower - 1e-12).any() or (log_center > physical_upper + 1e-12).any():
        raise ValueError("先验中心不在原物理倍率界内")
    lower = np.maximum(physical_lower, np.log(center * .95)[:, None])
    upper = np.minimum(physical_upper, np.log(center * 1.05)[:, None])
    if (lower > upper).any():
        raise ValueError("有效信任域为空")
    before = _states(ens)
    clocks = [m.t for m in ens.members]
    rng = np.random.default_rng(seed)
    z = _center_noise(rng.normal(size=(PARAMETERS, MEMBERS)), (PARAMETERS, MEMBERS))
    raw = log_center + .05 * z
    ens.physical_lower, ens.physical_upper = physical_lower, physical_upper
    ens.lower, ens.upper = lower, upper
    ens.center_ratios = center.copy()
    ens.theta = np.clip(raw, lower, upper)
    ens.rng = rng
    ens.analysis_rng = np.random.default_rng(seed + 1)
    ens.apply()
    if not _same_states(ens, before) or clocks != [m.t for m in ens.members]:
        raise AssertionError("初始化参数时改变了水力状态或时钟")
    return dict(
        seed=int(seed), members=MEMBERS, parameters=PARAMETERS,
        initial_log_std=.05, trust_ratio_interval=[.95, 1.05],
        centers=dict(zip(names, center.tolist())),
        physical_lower_ratios=np.exp(physical_lower[:, 0]).tolist(),
        physical_upper_ratios=np.exp(physical_upper[:, 0]).tolist(),
        effective_lower_ratios=np.exp(lower[:, 0]).tolist(),
        effective_upper_ratios=np.exp(upper[:, 0]).tolist(),
        clipped_entries=int(np.count_nonzero(raw != ens.theta)),
        clipped_entries_by_parameter=np.count_nonzero(raw != ens.theta, axis=1).tolist(),
        prior_log_std_by_parameter=ens.theta.std(axis=1, ddof=1).tolist(),
        centered_noise_sha256=_sha(z), raw_prior_sha256=_sha(raw),
        prior_theta_sha256=_sha(ens.theta), states_unchanged=True,
        clocks_unchanged=True, shared_members=True,
    )


def rejuvenate(ens, noise):
    """仅恢复参数散度。noise 对应全部 119 x 50 项，不使用观测或内部 RNG。"""
    _check_ensemble(ens)
    z = _center_noise(noise, ens.theta.shape)
    before = ens.theta.copy()
    hydraulic = _states(ens)
    clocks = [m.t for m in ens.members]
    sd = before.std(axis=1, ddof=1)
    low = sd < .01
    delta = np.sqrt(np.maximum(.05**2 - sd**2, 0))[:, None] * z * low[:, None]
    ens.theta = np.clip(before + delta, ens.lower, ens.upper)
    ens.apply()
    if not _same_states(ens, hydraulic) or clocks != [m.t for m in ens.members]:
        raise AssertionError("参数散度恢复改变了水力状态或时钟")
    return dict(
        mechanism="parameter_rejuvenation_outside_analysis_step_limit",
        spread_floor=.01, target_log_std=.05,
        rejuvenated_parameters=int(low.sum()),
        rejuvenated_parameter_mask=low.tolist(),
        spread_before=sd.tolist(), spread_after=ens.theta.std(axis=1, ddof=1).tolist(),
        centered_noise_sha256=_sha(z), states_unchanged=True, clocks_unchanged=True,
        **_parameter_audit(ens, before, delta),
    )


def _positions(parts, observations):
    return np.array([np.asarray(parts[o["state"]])[np.asarray(o["indices"], int)]
                     @ np.asarray(o["weights"], float) for o in observations])


def _local_increment(prior, Y, y, z, sigma, xmin, xmax, xpos, radius, mask):
    """同邻域分组、分块批量求解；没有逐水力状态行的 Python 求解循环。"""
    result = np.zeros_like(prior)
    if not len(y):
        return result
    A = Y - Y.mean(axis=1, keepdims=True)
    P = prior - prior.mean(axis=1, keepdims=True)
    distance = np.maximum(np.maximum(xmin[:, None] - xpos, xpos - xmax[:, None]), 0.)
    selected = (distance < radius) & mask[:, None]
    patterns, inverse = np.unique(selected, axis=0, return_inverse=True)
    residual = y[:, None] - Y
    for group, sel in enumerate(patterns):
        if not sel.any():
            continue
        rows = np.flatnonzero(inverse == group)
        B = A[sel]
        covariance = B @ B.T / (prior.shape[1] - 1)
        diagonal = np.arange(int(sel.sum()))
        for start in range(0, len(rows), 256):
            ids = rows[start:start + 256]
            loc = .5 * (1. + np.cos(np.pi * distance[np.ix_(ids, sel)] / radius))
            sd = sigma[sel][None, :] / np.sqrt(np.maximum(loc, 1e-12))
            cov = np.broadcast_to(covariance, (len(ids), *covariance.shape)).copy()
            cov[:, diagonal, diagonal] += sd**2
            cross = P[ids] @ B.T / (prior.shape[1] - 1)
            gain = np.linalg.solve(cov, cross[..., None])[..., 0]
            result[ids] = gain @ residual[sel] + (gain * sd) @ z[sel]
    return result


def _volume(m):
    # 与冻结 B01/B02 volume 完全相同的几何口径。
    return float(np.sum(m.A_ik * m._dx_ik) + np.sum(m.A_uk * m._dx_uk)
                 + np.sum(m.A_dk * m._dx_dk) + np.sum(m.V_j))


def _number(value):
    return float(value) if np.isfinite(value) else None


def analysis(ens, view, observations, noise_full, update_parameters, update_states,
             Y_override=None, elapsed_cycles=1):
    """同一先验的一次随机局地 EnKF 两块更新；不消费 RNG、不读取观测文件。

    noise_full/Y_override 的列固定为 50 成员，行顺序为全部 observations。
    Y_override 必须由调用方提供与 observations 同时段、同权重的已结束窗均值。
    elapsed_cycles 是本批参数更新允许覆盖的半小时周期数。
    """
    _check_ensemble(ens, view)
    if not np.isfinite(elapsed_cycles) or elapsed_cycles < 1 or int(elapsed_cycles) != elapsed_cycles:
        raise ValueError("elapsed_cycles 须为正整数")
    observations = list(observations)
    count = len(observations)
    z_full = _center_noise(noise_full, (count, MEMBERS))
    X = np.stack([view.pack(m) for m in ens.members], axis=1)
    if not np.isfinite(X).all():
        raise FloatingPointError("更新前水力集合非有限")
    if any(np.shape(a) != (len(X),) for a in (view.x, view.mask, view.heads)):
        raise ValueError("状态坐标、mask、heads 与 pack 维数不一致")
    if Y_override is None:
        Y_full = np.stack([ens.values(o) for o in observations]) if count else np.empty((0, MEMBERS))
    else:
        Y_full = np.array(Y_override, dtype=float, copy=True)
    if Y_full.shape != (count, MEMBERS) or not np.isfinite(Y_full).all():
        raise ValueError("先验观测集合 Y 形状错误或非有限")
    sigma_full = np.array([o.get("sigma", SIGMA_M) for o in observations], float)
    if not np.all(sigma_full == SIGMA_M):
        raise ValueError("本试验水位 sigma 固定为 0.03 m，不按窗口样本数缩小")
    y_full = np.array([o["value"] for o in observations], float)
    means = Y_full.mean(axis=1)
    spreads = Y_full.std(axis=1, ddof=1)
    innovations = y_full - means
    limits = 6. * np.sqrt(spreads**2 + sigma_full**2)
    accepted = np.isfinite(y_full) & (y_full > 0) & (np.abs(innovations) <= limits)
    trace = [dict(
        observation_index=i, station=o.get("station"), metric=o.get("metric"),
        observation_time_s=o.get("time_s"), observed=_number(y_full[i]),
        forecast=float(means[i]), forecast_spread=float(spreads[i]),
        sigma=SIGMA_M, innovation=_number(innovations[i]), innovation_limit=float(limits[i]),
        status="accepted" if accepted[i] else "rejected_innovation",
        used_for_parameters=bool(accepted[i] and update_parameters),
        used_for_states=bool(accepted[i] and update_states),
    ) for i, o in enumerate(observations)]
    used = [o for o, keep in zip(observations, accepted) if keep]
    Y, y, z, sigma = Y_full[accepted], y_full[accepted], z_full[accepted], sigma_full[accepted]
    theta_before = ens.theta.copy()
    clocks = [m.t for m in ens.members]
    parameter_raw = np.zeros_like(theta_before)
    state_raw = np.zeros_like(X)
    if update_parameters and used:
        parameter_raw = _local_increment(
            theta_before, Y, y, z, sigma,
            np.array([p["xmin"] for p in ens.parameters]),
            np.array([p["xmax"] for p in ens.parameters]),
            _positions(ens.xparts, used), ens.radius, np.ones(PARAMETERS, bool))
    if update_states and used:
        state_raw = _local_increment(
            X, Y, y, z, sigma, np.asarray(view.x), np.asarray(view.x),
            _positions(view.xparts, used), view.radius, np.asarray(view.mask, bool))
    if not np.isfinite(parameter_raw).all() or not np.isfinite(state_raw).all():
        raise FloatingPointError("局地批次增量非有限；尚未提交")
    step_limit = int(elapsed_cycles) * np.log(1.01)
    theta_after = (np.clip(theta_before + np.clip(parameter_raw, -step_limit, step_limit),
                           ens.lower, ens.upper) if update_parameters else theta_before.copy())
    heads = np.asarray(view.heads, bool)
    bounds = np.where(heads, .02, 3.)
    scales = np.minimum(1., np.min(bounds[:, None] / np.maximum(abs(state_raw), 1e-30), axis=0))
    candidate = X + state_raw * scales[None, :]
    candidate[~np.asarray(view.mask, bool)] = X[~np.asarray(view.mask, bool)]
    state_delta = candidate - X
    dirty_members = np.any(state_delta != 0, axis=0)
    state_dirty = bool(dirty_members.any())
    volume_before, volume_after = [], []
    if state_dirty:
        # 所有候选先在副本中检查；不能为统计口径重算原模型更新前的缓存。
        for i, member in enumerate(ens.members):
            if (candidate[view.slices["h_Ik"], i] < member.min_depth).any():
                raise FloatingPointError("批次状态分析后内部水深低于下限；尚未提交")
            if (candidate[view.slices["H_j"], i] - member._z_inv_j < member.min_depth).any():
                raise FloatingPointError("批次状态分析后节点水深低于下限；尚未提交")
            trial = copy.deepcopy(member)
            view.unpack(trial, X[:, i])
            volume_before.append(_volume(trial))
            view.unpack(trial, candidate[:, i])
            volume_after.append(_volume(trial))
        if not np.isfinite(volume_before + volume_after).all():
            raise FloatingPointError("候选状态几何体积非有限；尚未提交")
    try:
        if update_parameters:
            ens.theta = theta_after
            ens.apply()
        if state_dirty:
            for i, member in enumerate(ens.members):
                view.unpack(member, candidate[:, i])
            actual_volumes = [_volume(m) for m in ens.members]
            if not np.array_equal(actual_volumes, volume_after):
                raise AssertionError("候选副本与提交后的几何体积不一致")
        actual_X = np.stack([view.pack(m) for m in ens.members], axis=1)
        if not np.array_equal(actual_X, candidate) or clocks != [m.t for m in ens.members]:
            raise AssertionError("提交结果与候选或时钟不一致")
        if not update_parameters and not np.array_equal(ens.theta, theta_before):
            raise AssertionError("状态分析改变了 theta")
    except BaseException:
        # 非预期提交失败不能留下只更新一部分成员的集合供继续推进。
        ens.theta = theta_before
        ens.apply()
        if state_dirty:
            for i, member in enumerate(ens.members):
                view.unpack(member, X[:, i])
                member.t = clocks[i]
        raise
    parameter_audit = _parameter_audit(ens, theta_before, parameter_raw,
                                        step_limit if update_parameters else None)
    dv = np.asarray(volume_after) - np.asarray(volume_before) if state_dirty else np.zeros(MEMBERS)
    state_audit = dict(
        dirty=state_dirty, dirty_members=dirty_members.tolist(),
        max_head_change_m=float(abs(state_delta[heads]).max(initial=0)),
        max_flow_change_m3s=float(abs(state_delta[~heads]).max(initial=0)),
        raw_max_head_change_m=float(abs(state_raw[heads]).max(initial=0)),
        raw_max_flow_change_m3s=float(abs(state_raw[~heads]).max(initial=0)),
        member_increment_scales=scales.tolist(),
        masked_rows_unchanged=bool(np.array_equal(actual_X[~view.mask], X[~view.mask])),
        volume_definition="same_geometry_recomputed_before_on_copy_and_after_unpack",
        geometry_volume_audited=state_dirty,
        geometric_volume_before_m3=volume_before if state_dirty else None,
        geometric_volume_after_m3=volume_after if state_dirty else None,
        delta_volume_members_m3=dv.tolist(),
        delta_volume_mean_m3=float(dv.mean()),
        delta_volume_min_m3=float(dv.min()), delta_volume_max_m3=float(dv.max()),
        prior_state_sha256=_sha(X), posterior_state_sha256=_sha(actual_X),
    )
    return dict(
        observations=count, accepted_observations=int(accepted.sum()),
        rejected_observations=int((~accepted).sum()), trace=trace,
        update_parameters=bool(update_parameters), update_states=bool(update_states),
        elapsed_cycles=int(elapsed_cycles),
        Y_source="provided_window_forecast" if Y_override is not None else "current_prior_members",
        prior_Y_sha256=_sha(Y_full), centered_full_noise_sha256=_sha(z_full),
        accepted_observation_indices=np.flatnonzero(accepted).tolist(),
        parameter_dirty=parameter_audit["dirty"], state_dirty=state_dirty,
        parameter_analysis=parameter_audit, state_analysis=state_audit,
        clocks_unchanged=True, single_prior_single_observation_batch=True,
    )
