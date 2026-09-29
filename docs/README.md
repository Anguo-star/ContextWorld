# ContextWorld 文档导航

**先读 [ContextWorld 统一技术报告](ContextWorld_ICL_Benchmark.md)。** 任务定义、数据版本、完整模型与训练方案对比、主要发现均在这一页维护。其他文档只承担数据、训练、接入和复现操作说明；历史协议不是使用 benchmark 的前置知识。

## 公开使用

1. [项目首页](../README.md)：任务概览、安装和最短运行示例；
2. [统一技术报告](ContextWorld_ICL_Benchmark.md)：当前数据、指标、九任务总表与任务子表、动作选择实验及结果解释；
3. [数据生成方法](Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
4. [HF 数据集指南](HF_Dataset_Export.md)：数据包格式与加载方式；稳定公共下载尚未发布；
5. [外部模型 Adapter 规范](External_Model_Adapter_Contract.md)：接入新模型所需的统一接口；
6. [Stable-WorldModel 训练](StableWM_Training.md)：内置 LeWM、PLDM 和 PreJEPA 的训练入口。

九项任务按隐藏动力学类型组织。主表按任务横向展示 ICL / CEM，任务详情报告条件响应与历史利用等诊断指标。

最新结果直接见[九任务主表](ContextWorld_ICL_Benchmark.md#training-comparison)。各任务详情展开完整指标、标准差和重复数；Speed 分布与数据扩量分别展示。全部可用的同配置训练重复参与汇总，表格由同一份 [JSON](research/data/icl_training_study_v2.json) 生成，并提供汇总 [CSV](research/data/icl_training_study_v2.csv)。

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
不重开已封板的 baseline。

目前 Training、Development 和 Test 数据包已在本地完成发布候选组装，但稳定的公开数据集
修订尚未公布；Test 通过离线评分入口用于最终报告。

扩量训练数据的打包与发布准备见[发布说明](Expanded_Training_Release.md)。研究结果仍以统一技术报告为准。

## 文档写作与更新规范

- 面向首次接触项目的读者，按研究问题、实验设置、结果、解释的顺序组织；不按执行日期或排错过程记录。
- 术语与训练方案先定义后使用；表前交代数据划分、比较条件、单位、方向和统计方式。
- 结果解释对应具体表格，区分观测事实、机制假设与待验证问题；不同对照的收益不能擅自解释为因果分解。
- 规划评测先用模拟器真实代价验证可达性与决策区分度，再评测模型；未来不同不等于最优动作不同，真实未来的 latent 距离也不能代替物理代价校验。已知动力学的验证控制只用于可达性或机制对照，不计入模型规划成绩。
- 总览保留核心发现，任务详情保留完整指标；通用限制集中说明，复现细节放入附录或折叠说明。
- 数值以机器可读结果为来源，统一生成表格；历史版本不覆盖，不将不同数据或协议的结果混合汇总。
- 更新时同步检查表头、图例、公式、正文、链接及配套数据指南；当前数据与历史版本明确区分，删去重复解释与内部简称，保留复现所需信息。
- 新增实验先说明它与已有评测的关系；显式交代样本数、分母与配对单位，不混用成功数、成功率和重复次数。已有说明准确时直接引用，避免每次更新都重复堆叠背景与限制。
