"""只供理论和系数最小例核对的未集成候选；不挂接模型，不允许作为水力运行入口。

已知问题：动态适用域mask切换会使storage_B出现有限跳变。
这里故意不平滑、不扩大轮廓、不改变被排除路径，也不隐藏最后组装值。
"""
import numpy as np

NOT_APPROVED_FOR_HYDRAULIC_RUN = True
TRAPEZOIDAL_CODE = 5  # 26/nSuperLink 的真实几何分派编号。


def propose_storage_B(h0, h_assembly, left, right, g1, bottom_width,
                      side_slope, geom_codes, raw_node_heights, native_B):
    """从步初和组装迭代水深产生候选；不写输入数组或真实水力状态。"""
    h0, h = np.asarray(h0), np.asarray(h_assembly)
    left, right = np.asarray(left, int), np.asarray(right, int)
    g1, b, slope = map(np.asarray, (g1, bottom_width, side_slope))
    codes, tops, B = map(np.asarray, (geom_codes, raw_node_heights, native_B))
    shape = left.shape
    if any(x.shape != shape for x in (right, g1, b, slope, codes, B)) or h0.shape != h.shape or tops.shape != h.shape:
        raise ValueError("连接/节点数组维数不一致")
    if not np.isfinite(h0).all() or not np.isfinite(h).all() or not np.isfinite(B).all():
        raise ValueError("非有限水深或原顶宽")
    y0, ya = (h0[left]+h0[right])/2., (h[left]+h[right])/2.
    trapezoid = codes == TRAPEZOIDAL_CODE
    known_geometry = (np.isfinite(g1) & np.isfinite(b) & np.isfinite(slope)
                      & (g1 > 0) & (b >= 0) & (slope >= 0) & ((b > 0) | (slope > 0)))
    known_node_contours = np.isfinite(tops[left]) & np.isfinite(tops[right])
    old_ends_inside = ((h0[left] >= 0) & (h0[right] >= 0)
                       & (h0[left] <= tops[left]) & (h0[right] <= tops[right]))
    assembly_ends_inside = ((h[left] >= 0) & (h[right] >= 0)
                            & (h[left] <= tops[left]) & (h[right] <= tops[right]))
    means_inside = (y0 >= 0) & (y0 <= g1) & (ya >= 0) & (ya <= g1)
    # 精确比较，不加缓冲带或平滑权重以掩盖切换面。
    mask = trapezoid & known_geometry & known_node_contours & old_ends_inside & assembly_ends_inside & means_inside
    storage_B = B.copy()
    storage_B[mask] = b[mask]+slope[mask]*(y0[mask]+ya[mask])
    return dict(storage_B_ik=storage_B, native_B_ik=B.copy(), used_mask=mask,
                h0=h0.copy(), h_assembly=h.copy(), y0=y0, y_assembly=ya,
                trapezoidal=trapezoid, known_node_contours=known_node_contours,
                old_endpoints_inside=old_ends_inside,
                assembly_endpoints_inside=assembly_ends_inside,
                connection_means_inside=means_inside)


def node_coefficients_for_review(frozen_numba_node_coeffs, native_args, proposal):
    """真实26系数函数的纯副本调用；只把第四个B参数替换为候选，不集成step。"""
    args = {name: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for name, value in native_args.items()}
    args["_B_ik"] = proposal["storage_B_ik"].copy()
    frozen_numba_node_coeffs(**args)
    # 此记录就是传入E/D组装的数组；之后不能用最终状态的理想割线覆盖它。
    last_assembly = {key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in proposal.items()}
    last_assembly.update(E_Ik=args["_E_Ik"].copy(), D_Ik=args["_D_Ik"].copy(),
                         dt_s=float(args["_dt"]))
    return last_assembly


def check_accepted_domain(last_assembly, accepted_h, geometry):
    """只有审计，不修正接受状态、不改实际已用mask；越域必须显式停止候选。"""
    final = propose_storage_B(last_assembly["h0"], accepted_h,
                              geometry["left"], geometry["right"], geometry["g1"],
                              geometry["bottom_width"], geometry["side_slope"],
                              geometry["geom_codes"], geometry["raw_node_heights"],
                              last_assembly["native_B_ik"])
    violation = last_assembly["used_mask"] & ~final["used_mask"]
    return dict(stop_candidate=bool(violation.any()),
                assembled_inside_but_accepted_outside=np.flatnonzero(violation),
                final_eligibility_mask=final["used_mask"],
                actual_used_storage_B_ik=last_assembly["storage_B_ik"].copy(),
                final_state_ideal_storage_B_ik=[float(value) if eligible else None
                    for value,eligible in zip(final["storage_B_ik"],final["used_mask"])],
                final_ideal_B_defined_only_on_final_eligible_links=True,
                actual_assembly_values_not_overwritten=True)
