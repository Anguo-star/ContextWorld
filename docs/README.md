# 文档导航

## 公开阅读入口

外部读者只需要阅读两份文件：

1. 项目首页 [`README.md`](../README.md)：一分钟了解 Benchmark 和运行入口；
2. 正式文档 [ContextWorld ICL Benchmark](ContextWorld_ICL_Benchmark.md)：完整任务定义、
   数据、评分、原始模型与训练后模型的对比结果，以及使用方法。

八个正式能力不再分别维护公开说明，避免重复内容和版本漂移。每项能力在正式文档中都使用
“任务目标、数据构成、评测方法、基线表现、适用范围”五段统一结构。

> [ContextWorld ICL Benchmark](ContextWorld_ICL_Benchmark.md)

| 能力 | 主文档 | 机器配置 | 独立命令 |
|---|---|---|---|
| 速度 | [6.1 速度](ContextWorld_ICL_Benchmark.md#61-速度) | `configs/benchmark/tworoom_speed_icl_release_v1.yaml` | `contextworld-speed` |
| 门通行规则 | [6.2 门通行规则](ContextWorld_ICL_Benchmark.md#62-门通行规则) | `configs/benchmark/tworoom_door_icl_release_v1.yaml` | `contextworld-door` |
| 动作延迟 | [6.3 动作延迟](ContextWorld_ICL_Benchmark.md#63-动作延迟) | `configs/benchmark/tworoom_action_delay_icl_release_v1.yaml` | `contextworld-action-delay` |
| 推手移动幅度 | [6.4 推手移动幅度](ContextWorld_ICL_Benchmark.md#64-推手移动幅度) | `configs/benchmark/pusht_action_strength_icl_release_v1.yaml` | `contextworld-action-strength` |
| 接触摩擦 | [6.5 接触摩擦](ContextWorld_ICL_Benchmark.md#65-接触摩擦) | `configs/benchmark/pusht_contact_friction_icl_release_v1.yaml` | `contextworld-contact-friction` |
| 运动阻尼 | [6.6 运动阻尼](ContextWorld_ICL_Benchmark.md#66-运动阻尼) | `configs/benchmark/pusht_motion_damping_icl_release_v1.yaml` | `contextworld-motion-damping` |
| 机械臂质量 | [6.7 机械臂质量](ContextWorld_ICL_Benchmark.md#67-机械臂质量) | `configs/benchmark/reacher_arm_mass_icl_release_v1.yaml` | `contextworld-reacher-arm-mass` |
| 传送门出口位置 | [6.8 传送门出口位置](ContextWorld_ICL_Benchmark.md#68-传送门出口位置) | `configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml` | `contextworld-portal-exit` |

## 未发布研发候选

研发候选沿用相同的五段文档模板，但不计入上述八项 Suite，也不提供公开命令或 Public
分数。

| 能力 | 主文档 | 冻结进展 | 发布状态 |
|---|---|---|---|
| Cube 夹爪携带规则 | [6.9 Cube 夹爪携带规则](ContextWorld_ICL_Benchmark.md#69-cube-夹爪携带规则public-生成失败待恢复) | [Public v1 失败与恢复边界](protocols/Cube_Gripper_Carry_History3_v4r1_Public_v1_Generation_Failure.md) | Development 与 CEM 留存通过；Public v1 生成在发布前因元数据缺陷失败，原命名空间已封存 |

## 复现材料

- `protocols/`：实验执行前冻结的协议与审计说明；
- `archive/`：已经结束的阶段材料；
- `reference/`：第三方工程与运行环境说明。

这些材料解释结果怎样得到，但不是发布说明，也不替代统一主文档中的当前结论。
