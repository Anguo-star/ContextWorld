# 实验复现索引

本目录保存实验前冻结的协议、旧配置身份和逐文件审计证据。它服务于复现，不维护另一套
对外结论。当前任务定义、数据规模、评分、结果和使用方法统一以
[ContextWorld ICL Benchmark](../ContextWorld_ICL_Benchmark.md)为准。

## Suite v2 候选配置中的九个组件

| 能力 | 正式配置 | 主要复现材料 |
|---|---|---|
| TwoRoom Speed | [`tworoom_speed_icl_release_v1.yaml`](../../configs/benchmark/tworoom_speed_icl_release_v1.yaml) | [History-3 速度协议](TwoRoom_History3_Speed_Benchmark_v2_Protocol.md)、[范围外与多步协议](TwoRoom_History3_Speed_Extrapolation_Multistep_v1_Protocol.md) |
| TwoRoom Door Rule | [`tworoom_door_icl_release_v1.yaml`](../../configs/benchmark/tworoom_door_icl_release_v1.yaml) | [History-3 门规则复现附录](TwoRoom_History3_Hidden_Passage_Feasibility_v1.md) |
| TwoRoom Action Delay | [`tworoom_action_delay_icl_release_v1.yaml`](../../configs/benchmark/tworoom_action_delay_icl_release_v1.yaml) | [统一主文档中的动作延迟](../ContextWorld_ICL_Benchmark.md#63-动作延迟) |
| PushT Action Strength | [`pusht_action_strength_icl_release_v1.yaml`](../../configs/benchmark/pusht_action_strength_icl_release_v1.yaml) | [统一主文档中的动作力度](../ContextWorld_ICL_Benchmark.md#64-动作力度) |
| PushT Contact Friction | [`pusht_contact_friction_icl_release_v1.yaml`](../../configs/benchmark/pusht_contact_friction_icl_release_v1.yaml) | [统一主文档中的接触摩擦](../ContextWorld_ICL_Benchmark.md#65-接触摩擦) |
| PushT Motion Damping | [`pusht_motion_damping_icl_release_v1.yaml`](../../configs/benchmark/pusht_motion_damping_icl_release_v1.yaml) | [统一主文档中的运动阻尼](../ContextWorld_ICL_Benchmark.md#66-运动阻尼) |
| Reacher Arm Mass | [`reacher_arm_mass_icl_release_v1.yaml`](../../configs/benchmark/reacher_arm_mass_icl_release_v1.yaml) | [统一主文档中的机械臂质量](../ContextWorld_ICL_Benchmark.md#67-机械臂质量) |
| TwoRoom Portal Exit | [`tworoom_portal_exit_icl_release_v1.yaml`](../../configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml) | [统一主文档中的传送门出口位置](../ContextWorld_ICL_Benchmark.md#68-传送门出口位置) |
| Cube Gripper-Carry | [`cube_gripper_carry_h3_v4r1_icl_release_v1.yaml`](../../configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml) | [统一主文档中的 Cube 夹爪携带规则](../ContextWorld_ICL_Benchmark.md#69-cube-夹爪携带规则)、[Public v1 失败与恢复边界](Cube_Gripper_Carry_History3_v4r1_Public_v1_Generation_Failure.md) |

当前统一 Suite 候选配置是
[`contextworld_icl_suite_v2_recovery_v2.yaml`](../../configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml)；
此前的 `contextworld_icl_suite_v2.yaml` 是未提交 registration v1 staging 中的历史候选快照。
`contextworld-benchmark audit --full` 会按九个组件的正式配置检查代码、数据、结果和
哈希。

## Cube 历史 recovery 轨道

下面的材料记录 Cube 从失败的旧 Public v1 迁移到独立 recovery v1 的历史；技术内容以
Suite v2、v4r1 release config 和主文档为准，成员资格仅由 canonical registration
decision 激活。

| 能力 | 当前状态 | 冻结材料 |
|---|---|---|
| Cube Gripper-Carry（History=3） | 独立 Public recovery v1：LeWM Public 3/3 与 CEM 留存 3/3 通过；PLDM 未获 Public 授权；旧失败命名空间保持封存 | [Public v1 失败与恢复边界](Cube_Gripper_Carry_History3_v4r1_Public_v1_Generation_Failure.md)、[Public 前历史交接](Cube_Gripper_Carry_History3_v4r1_Pre_Public_Handoff.md)、[v4r1 数据恢复协议](Cube_Gripper_Carry_History3_Development_v4r1_Recovery_Protocol.md)、[参考训练 v3 协议](Cube_Gripper_Carry_History3_v4r1_Reference_Training_v3_Protocol.md)、[CEM 留存 v2 协议](Cube_Gripper_Carry_History3_v4r1_Original_Task_Retention_v2_Protocol.md) |

Cube 的早期 `cube_gripper_carry_icl_release_v1.yaml` 仍是已失败的旧 Development 快照，
不能替代 v4r1 recovery 冻结链，也不授权重跑 Public。Public v1 原命名空间已经消耗；
Suite v2 的正式结果来自新的 recovery 预注册、freeze、一次性评分和独立注册审计。

## 历史材料怎样阅读

以下材料解释某个设计为何形成，不代表额外发布组件：

| 材料 | 回答的问题 |
|---|---|
| [原始能力重建](TwoRoom_OriginalAbility_Reconstruction_Protocol.md) | 合成单速、原始混训和多速度混训如何影响基础能力 |
| [速度 5 跨 Eval](TwoRoom_Speed5_CrossEval_Protocol.md) | 原始与合成 Eval 差异来自哪里 |
| [CEM 配置影响](TwoRoom_CEM_Planning_Config_Impact_Protocol.md) | horizon、候选数和迭代数如何影响规划结果 |
| [速度敏感距离](TwoRoom_Speed_ICL_Sensitive_Eval_Protocol.md) | 怎样避免过远目标造成成功率地板 |
| [快慢历史对照](TwoRoom_Speed_Context_Direction_Eval_Protocol.md) | 为什么有限执行预算下较快历史可能占优 |
| [早期 SpeedFull 数据](TwoRoom_SpeedFull_Data_Protocol.md) | 多速度数据的早期生成与训练 |
| [早期 SpeedClean 数据](TwoRoom_SpeedClean_Data_Protocol.md) | 单速控制数据的早期生成与训练 |
| [早期 SpeedTask 数据](TwoRoom_SpeedTask_Data_Protocol.md) | 更早版本的速度任务设计 |

历史文件保留当时的字段名、预注册门槛和阶段性结果，便于核对已有产物；若其文字与统一
主文档不同，应以当前正式组件配置和统一主文档为准。
