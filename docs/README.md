# ContextWorld 文档导航

**先读 [ContextWorld 主报告](ContextWorld_ICL_Benchmark.md)。** 它完整说明任务定义、数据构建、评分、训练对照与高层机制证据；[结果附录](reference/ContextWorld_Expanded_Training_Results.md)保存完整数值。

## 公开使用

1. [项目首页](../README.md)：任务概览、安装和最短运行示例；
2. [主报告](ContextWorld_ICL_Benchmark.md)：任务定义、构建流程、评分公式、九任务对照与[高层根因分析](ContextWorld_ICL_Benchmark.md#training-mechanism-findings)；[结果附录](reference/ContextWorld_Expanded_Training_Results.md)保存任务明细与补充诊断；
3. [数据生成方法](Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
4. [HF 数据集指南](HF_Dataset_Export.md)：数据包格式与加载方式；稳定公共下载尚未发布；
5. [外部模型 Adapter 规范](External_Model_Adapter_Contract.md)：接入新模型所需的统一接口；
6. [Stable-WorldModel 训练](StableWM_Training.md)：内置 LeWM、PLDM 和 DINO-WM（实现名称 `prejepa`）的训练入口。

九项任务按隐藏动力学类型组织。主表只列各任务主指标和训练重复数；附录分别列出匹配历史胜率、响应 NRE、多步完整误差比 E/B 与原环境 CEM。正确／错误历史对照用于解释历史作用，不将辨别、预测与规划混成一个总分。

Public Test 任务成绩与 Development 对照见[九任务总表](ContextWorld_ICL_Benchmark.md#training-comparison)。多步预测诊断覆盖九任务主分布的 2,436 个 Development 场景；Speed 限于未见速度插值。相同配置的全部可用训练重复参与汇总，[JSON](research/data/multistep_prediction_coverage_v2.json) 与 [CSV](research/data/multistep_prediction_coverage_v2.csv) 提供逐次与汇总结果。公开汇总可检查，但评测数据与权重尚无稳定公共下载，外部端到端复现尚不可用。指标计算、正反例、物理校准和规划诊断见[ICL 评分验证与训练机制研究](ICL_Metric_Study.md)。

<a id="指标研究"></a>

## 评分验证与训练机制

[主报告的高层根因分析](ContextWorld_ICL_Benchmark.md#training-mechanism-findings)给出理解结论所需的训练对照表；[ICL 研究报告](ICL_Metric_Study.md)提供评分验证、完整曲线与具体干预细节。主要发现是：部分模型的整体训练损失持续下降，但训练查询上的条件响应仍不足；冻结编码器等训练选择的作用又随任务改变。[结论与证据表](ICL_Metric_Study.md#root-cause-evidence-synthesis)集中列出数据扩量、模型容量、初始化、冻结与损失干预的支持证据和反例。九任务比较给出问题范围，具体机制不超出直接对照所覆盖的任务和模型；评分方法的探索性对照与任务主指标分开呈现。

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

## 数据与结果复现

- [数据集使用指南](HF_Dataset_Export.md)：数据包格式、加载和评测示例；
- [数据包与结果复现](Expanded_Training_Release.md)：九任务数据构成、文件验证及 Test 汇总复算；
- [Public Test 结果](research/data/current_reference_test_v1/overview.md)：检查点集合、逐次成绩和汇总；
- [结果附录](reference/ContextWorld_Expanded_Training_Results.md)：训练方案、数据规模、模型容量与预测诊断的完整比较。

结果 JSON / CSV 已公开。九任务数据包已完成本地文件与加载验证，完整数据和权重尚无稳定公共下载；获取状态统一见[主报告](ContextWorld_ICL_Benchmark.md#8-可复现性与结果报告)。

<a id="文档写作与更新规范"></a>

文档贡献者请遵循[写作与更新规范](AGENTS.md)。
