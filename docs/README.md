# ContextWorld 文档导航

**先读 [ContextWorld 统一技术报告](ContextWorld_ICL_Benchmark.md)。** 任务定义、数据版本、完整模型与训练方案对比、主要发现均在这一页维护。其他文档只承担数据、训练、接入和复现操作说明；历史协议不是使用 benchmark 的前置知识。

## 公开使用

1. [项目首页](../README.md)：任务概览、安装和最短运行示例；
2. [统一技术报告](ContextWorld_ICL_Benchmark.md)：当前数据、指标、九任务总表与任务子表、九任务动作选择与预测诊断及结果解释；
3. [数据生成方法](Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
4. [HF 数据集指南](HF_Dataset_Export.md)：数据包格式与加载方式；稳定公共下载尚未发布；
5. [外部模型 Adapter 规范](External_Model_Adapter_Contract.md)：接入新模型所需的统一接口；
6. [Stable-WorldModel 训练](StableWM_Training.md)：内置 LeWM、PLDM 和 PreJEPA 的训练入口。

九项任务按隐藏动力学类型组织。主表只展示多步完整预测分，任务子表报告区间和训练重复数；CEM 与历史选择、响应指标折叠单列。

最新结果见[九任务主表](ContextWorld_ICL_Benchmark.md#training-comparison)。当前覆盖九任务主分布的 2,436 个 Development 场景；Speed 限于未见速度插值，Test 尚未评测。相同配置的全部可用训练重复参与汇总，[JSON](research/data/multistep_prediction_coverage_v2.json) 与 [CSV](research/data/multistep_prediction_coverage_v2.csv) 提供逐次与汇总结果。公开汇总可检查，但数据面板与权重尚无稳定下载 revision，外部端到端复现尚不可用。评分边界与 PushT 可观测性限制见[主分解释范围与辅助物理读出](ContextWorld_ICL_Benchmark.md#main-score-interpretation-and-auxiliary-physical-readout)。

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

这些材料用于复核已报告结果。历史 v3 的 Training / Development 来源为 `ContextWorld-v1`，
Test 来源为 `ContextWorld-v1-full`；`ContextWorld-v3-hf` 是汇集这些冻结来源的发布候选。
**当前训练比较的六项配对任务采用扩量 Training，其余三项沿用基础版；Development / Test 固定。**
具体规模以技术报告为准，扩量包的发布准备见下文，不将当前结果重新标为历史 v3。

## 仓库维护与发布

[v3 发布需求与验收](ContextWorld_Public_v1_Release_Readiness.md) 记录候选包、数据卡、
文件校验和上传后固定 revision 的下载验证。独立外部复现与新增模型覆盖单独维护，
不修改已冻结的历史基线。

Training、Development 和 Test 已在本地组装为发布候选，但稳定公共下载地址与固定 revision 尚未公布。
仓库中的汇总结果可供检查；缺少公开面板和权重，当前无法端到端复跑。Test 使用离线评分入口作最终报告。

扩量训练数据的打包与发布准备见[发布说明](Expanded_Training_Release.md)。研究结果仍以统一技术报告为准。

## 文档写作与更新规范

- 按研究问题、实验设置、结果和解释组织材料；不按日期或排错过程写成实验日志。
- 首次出现时定义术语。表格注明划分、样本数、单位、方向、分母和统计单位。
- 区分观测结果、机制解释与待验证问题；主指标、诊断指标和规划收益分开报告，不将相关指标当作独立证据或因果分解。
- 规划诊断先用模拟器真实代价检查可达性和候选区分度；已知动力学控制及真实观测替换只作验证，不算模型规划成绩。
- 数值以机器可读结果为准并由脚本生成；保留历史版本，不混合数据或协议。完整表格不等于正式冻结基准。
- 扩展实验时优先引用已有定义；长方法和复算细节放在相应指南，更新时同步校对表头、公式、正文、链接和数据范围。

新增评分或数据分布须标明验证阶段与覆盖范围。根因对照固定场景、动作、目标表示和误差分母，并说明一次干预同时改变了什么。
