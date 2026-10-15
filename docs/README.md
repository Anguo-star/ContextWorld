# ContextWorld 文档导航

**先读 [ContextWorld 主报告](ContextWorld_ICL_Benchmark.md)。** 它说明任务、数据、评分与训练对照；[结果附录](reference/ContextWorld_Expanded_Training_Results.md)保存完整数值，冻结 v3 另见[参考附录](reference/Benchmark_Result_Provenance.md)。

## 公开使用

1. [项目首页](../README.md)：任务概览、安装和最短运行示例；
2. [主报告](ContextWorld_ICL_Benchmark.md)：数据、指标、九任务训练对照、评分协议、结果解释与适用边界；[结果附录](reference/ContextWorld_Expanded_Training_Results.md)保存任务明细与补充诊断；
3. [数据生成方法](Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
4. [HF 数据集指南](HF_Dataset_Export.md)：数据包格式与加载方式；稳定公共下载尚未发布；
5. [外部模型 Adapter 规范](External_Model_Adapter_Contract.md)：接入新模型所需的统一接口；
6. [Stable-WorldModel 训练](StableWM_Training.md)：内置 LeWM、PLDM 和 PreJEPA 的训练入口。

九项任务按隐藏动力学类型组织。主表只列各任务的原协议主指标和训练重复数；附录分别列出匹配历史胜率、响应 NRE、多步完整误差比 E/B 与原环境 CEM。正确／错误历史对照用于解释历史作用，不将辨别、预测与规划混成一个总分。

Public Test 任务成绩与 Development 对照见[九任务总表](ContextWorld_ICL_Benchmark.md#training-comparison)。补充多步评测覆盖九任务主分布的 2,436 个 Development 场景；Speed 限于未见速度插值；这套扩展动作的多步诊断尚未评测 Test。相同配置的全部可用训练重复参与汇总，[JSON](research/data/multistep_prediction_coverage_v2.json) 与 [CSV](research/data/multistep_prediction_coverage_v2.csv) 提供逐次与汇总结果。公开汇总可检查，但数据面板与权重尚无稳定下载 revision，外部端到端复现尚不可用。指标计算、正反例、物理校准和规划诊断见[ICL 评分验证与训练机制研究](ICL_Metric_Study.md)。

<a id="指标研究"></a>

## 评分验证与训练机制

[ICL 研究报告](ICL_Metric_Study.md) 维护评分验证、训练干预和机制解释的边界。[跨任务证据表](ICL_Metric_Study.md#root-cause-evidence-synthesis)综合已有数据扩量、模型容量、初始化与冻结、训练目标对照；九任务结果和具体实验由表内链接展开。主报告只纳入已验证现象，候选公式不作为正式协议，未证实的机制不作为根因结论。

## 任务导航

| 能力类型 | 任务 | 说明 | 机器配置 | 任务命令 |
|---|---|---|---|---|
| 即时连续响应 | 速度 | [速度](ContextWorld_ICL_Benchmark.md#611-速度) | `configs/benchmark/tworoom_speed_icl_release_v1.yaml` | `contextworld-speed` |
| 即时连续响应 | 推手移动幅度 | [推手移动幅度](ContextWorld_ICL_Benchmark.md#612-推手移动幅度) | `configs/benchmark/pusht_action_strength_icl_release_v1.yaml` | `contextworld-action-strength` |
| 即时连续响应 | 机械臂质量 | [机械臂质量](ContextWorld_ICL_Benchmark.md#613-机械臂质量) | `configs/benchmark/reacher_arm_mass_icl_release_v1.yaml` | `contextworld-reacher-arm-mass` |
| 时间延迟动力学 | 动作延迟 | [动作延迟](ContextWorld_ICL_Benchmark.md#621-动作延迟) | `configs/benchmark/tworoom_action_delay_icl_release_v1.yaml` | `contextworld-action-delay` |
| 接触或附着条件动力学 | 接触摩擦 | [接触摩擦](ContextWorld_ICL_Benchmark.md#631-接触摩擦) | `configs/benchmark/pusht_contact_friction_icl_release_v1.yaml` | `contextworld-contact-friction` |
| 接触或附着条件动力学 | 运动阻尼 | [运动阻尼](ContextWorld_ICL_Benchmark.md#632-运动阻尼) | `configs/benchmark/pusht_motion_damping_icl_release_v1.yaml` | `contextworld-motion-damping` |
| 接触或附着条件动力学 | Cube 夹爪携带规则 | [6.3.3 Cube 夹爪携带规则](ContextWorld_ICL_Benchmark.md#633-cube-夹爪携带规则) | `configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml` | `contextworld-cube-gripper-carry` |
| 隐藏结构转移 | 门通行规则 | [门通行规则](ContextWorld_ICL_Benchmark.md#641-门通行规则) | `configs/benchmark/tworoom_door_icl_release_v1.yaml` | `contextworld-door` |
| 隐藏结构转移 | 传送门出口位置 | [传送门出口位置](ContextWorld_ICL_Benchmark.md#642-传送门出口位置) | `configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml` | `contextworld-portal-exit` |

## 结果复现

- [固定基准验收](reference/Baseline_Final_Acceptance_2026-09-10.md)：v3 的覆盖范围、验证证据与后续比较规则；
- [参考结果复现附录](reference/Benchmark_Result_Provenance.md)：检查点来源、训练种子、评测预算和机器可读结果；
- `protocols/`：执行前确定的任务协议；
- `archive/`：已经结束的实验阶段材料；
- `reference/`：第三方工程、运行环境和结果来源说明。

这些材料用于复核已报告结果。冻结 v3、当前训练对照的 Development / Public Test 以及扩量发布候选的
证据范围和发布状态见[主报告的状态表](ContextWorld_ICL_Benchmark.md#8-可复现性与结果报告)。

## 仓库维护与发布

[v3 发布需求与验收](ContextWorld_Public_v1_Release_Readiness.md) 记录候选包、数据卡、
文件校验和上传后固定 revision 的下载验证。独立外部复现与新增模型覆盖单独维护，
不修改已冻结的历史基线。

v3 与九任务扩量数据均已组装为本地发布候选；扩量包的文件与加载检查见[发布说明](Expanded_Training_Release.md)，检查范围不等同于独立端到端复现。
两者均未上传 HF，缺少公共固定 revision 和公开权重，当前无法从公开材料端到端复跑。

扩量训练数据的打包与发布准备见[发布说明](Expanded_Training_Release.md)。研究结果仍以统一技术报告为准。

## 文档写作与更新规范

- 主报告只维护确定的任务、评分协议、结果与解释边界；指标研究过程、候选方案和验证结论集中放在独立研究文档，研究完成后再更新主报告。
- 按研究问题、实验设置、结果和解释组织材料；不按日期或排错过程写成实验日志。先说明当前结论及适用条件，再给必要证据；明确已完成验证与尚缺证据，已有验证不重复扩写。
- 首次出现时定义术语。表格注明划分、样本数、单位、方向、分母和统计单位。
- 区分观测结果、机制解释与待验证问题；主指标、诊断指标和规划收益分开报告，不将相关指标当作独立证据或因果分解。
- 规划诊断先用模拟器真实代价检查可达性和候选区分度；已知动力学控制及真实观测替换只作验证，不算模型规划成绩。
- 数值以机器可读结果为准并由脚本生成；保留历史版本，不混合数据或协议。完整表格不等于正式冻结基准。
- 扩展实验时优先引用已有定义；长方法和复算细节放在相应指南，更新时同步校对表头、公式、正文、链接和数据范围。
- 新结果按“要验证什么、如何比较、观察到什么、能解释到哪里”组织；主报告只保留理解结论所需的表格，完整数值提供 CSV / JSON，不追加排错过程或内部执行日志。
- 区分指标的解释边界与已证实的误判；不得把单项校准失败写成整个 benchmark 无效，也不得用已有论文采用某指标替代本任务的适用性验证。
- 表头、正文、导航保持同一评分用途和版本状态；未冻结的诊断不能称为“新主分”。不同测量不能互相替代，不等于已证明统计独立；保留旧指标有效含义，不用新名称重复计算证据。
- 根因研究先明确要解释的表现和可区分的机制；评分验证达到该对照所需范围后即可推进，不把统一物理重建或通用综合分设为无关前置任务。

- 机制分析分别写清观测、已排除解释和待验证原因；“可区分”不等于“表示充分”。首步、末步必须注明物理时间端点，模型与训练方案的范围须与表格一致。

新增评分或数据分布须标明验证阶段与覆盖范围。根因对照固定场景、动作、目标表示和误差分母，并说明一次干预同时改变了什么。
