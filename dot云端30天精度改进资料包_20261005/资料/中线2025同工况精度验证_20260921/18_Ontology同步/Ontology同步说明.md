# Ontology闸流修复同步

已更新本地`water/hydros-agent-ontology`的Pipedream源码。只移植已验证的矩形闸流方法、宽高传参和收敛检查，没有用独立runner整目录覆盖ontology适配层，也没有切换其标准Python后端。

同步内容为两个求解器、共享gate_hydraulics.py、WXQ转换器、保留诊断日志的引擎封装，以及simulation.py的缓存版本。矩形闸门时变默认最多20次迭代、严格检查，失败回退；稳态准备保持独立判据。缓存schema升级为pipedream-superlink-restart-state-v2-rectangular-pressure，适配器版本0.4.1，拒绝旧缓存、不同步长或闸门参数缓存。

## 验证

- 核心方法AST与已验证runner一致，共享公式文件逐字节相同。
- ontology真实导入解析到src/engines/pipedream_solver，并未误用独立runner包。
- 4项针对性测试通过，覆盖矩形参数、两后端、跨水面连续/等水位、严格步进失败回退、缓存拒绝和恢复、真实异步执行入口。
- 旧Pipedream测试同步前12项：2失败、4错误；同步后加3项回归共15项：仍相同2失败、4错误。旧失败是第0步调用不符合当前第1步起算契约及固定7m初始水深预期，不是本轮引入，未为了通过测试改平台现有语义。
- 按仓库既有CRLF行尾检查diff，无新增空白错误。

独立runner的30天验证不冒充ontology平台30天验收，本轮未启动数据库/MQTT/OPC UA完整平台任务、未重启或部署服务、未提交Git。原有pyproject.toml及hydro1dWxqSimulation.py改动保留。

## 文件和提交注意

源文件及哈希在《同步文件校验.json》，完整增量在《Ontology闸流同步.patch》，修改前副本保留用于回退。回归测试延伸现有tests/test_pipedream_flow_direction.py；该路径被仓库本地.git/info/exclude忽略，因此普通git diff不展示它，本交付patch已显式包含，若后续提交需一并纳入。共享gate_hydraulics.py是新增文件，同样须随源码提交。
