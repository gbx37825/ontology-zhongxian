"""只恢复有原表证据的梅河四孔闸底；不改节点基准、参数、开度或水力状态。

调用 restore_meihe_gate_bottom(model, converted, source)，返回更改清单。
model 为已按冻结26构建、尚未安装本候选的模型；converted 为该模型转换表。
函数不构模、不重算几何、不调用闸流或推进时间。调用者须单独授权过程试验。
"""
import numpy as np

GATE_NAMES = tuple(f"JZZ20-梅河倒虹吸出口节制闸-{i}" for i in range(1, 5))
SOURCE_BOTTOM_M = 115.521
EXPECTED_NODE_BOTTOM_M = 116.723


def _changes(converted, source):
    table = converted["orifices"]
    sj = converted["superjunctions"].set_index("id", drop=False)
    changes = []
    for name in GATE_NAMES:
        hit = table.loc[table["name"] == name]
        if len(hit) != 1:
            raise ValueError(f"梅河门孔不能唯一定位: {name}")
        row = hit.iloc[0]
        raw = source["baseData"]["gates"][name]
        up = sj.loc[int(row.sj_0)]
        checks = (
            str(up["name"]) == "NODE-main-37",
            str(sj.loc[int(row.sj_1), "name"]) == "NODE-main-38",
            raw["node1"] == "NODE-main-37",
            raw["node2"] == "NODE-main-38",
            abs(float(raw["zb"]) - SOURCE_BOTTOM_M) <= 1e-12,
            abs(float(up.z_inv) - EXPECTED_NODE_BOTTOM_M) <= 1e-12,
            float(row.z_o) == 0.0,
            float(row.g2) == float(raw["b"]) == 7.0,
            row.flow_model == "rectangular_pressure",
        )
        if not all(checks):
            raise ValueError(f"梅河原输入/转换状态偏离已审核身份: {name}")
        changes.append(dict(name=name, table_index=hit.index[0],
                            orifice_id=int(row.id), upstream_node_id=int(row.sj_0),
                            downstream_node_id=int(row.sj_1), old_z_o_m=0.0,
                            new_z_o_m=float(raw["zb"]) - float(up.z_inv),
                            old_absolute_gate_bottom_m=float(up.z_inv),
                            new_absolute_gate_bottom_m=float(raw["zb"])))
    return changes


def restore_meihe_gate_bottom(model, converted, source):
    """验证全部前提后，仅更新四孔运行时_z_o和对应orifices.z_o。"""
    changes = _changes(converted, source)
    ids = np.asarray([r["orifice_id"] for r in changes], dtype=int)
    if len(model._z_o) != len(converted["orifices"]):
        raise ValueError("运行时门孔总数与转换表不一致")
    runtime_rows = []
    for row in changes:
        i, up, down = row["orifice_id"], row["upstream_node_id"], row["downstream_node_id"]
        hit = model.orifices.loc[model.orifices["name"] == row["name"]]
        if (len(hit) != 1 or int(hit.iloc[0].id) != i or float(hit.iloc[0].z_o) != 0.0
                or model._z_o[i] != 0.0 or model._J_uo[i] != up or model._J_do[i] != down
                or abs(model._z_inv_j[up] - EXPECTED_NODE_BOTTOM_M) > 1e-12):
            raise ValueError(f"运行时梅河门孔与冻结转换表不一致: {row['name']}")
        runtime_rows.append(hit.index[0])
    # 所有可预见验证在写入前完成；模型状态、系数、开度和节点床高都不写。
    values = np.asarray([r["new_z_o_m"] for r in changes])
    model._z_o[ids] = values
    model.orifices.loc[runtime_rows, "z_o"] = values
    converted["orifices"].loc[[r["table_index"] for r in changes], "z_o"] = values
    return dict(status="ISOLATED_MEIHE_FOUR_GATE_BOTTOM_RESTORED",
                changed_variable="z_o", gates=changes,
                hydraulic_step_executed=False, node_bed_changed=False,
                n_or_C_changed=False, opening_changed=False,
                engineering_acceptance=False, baseline_replaced=False)
