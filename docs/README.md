# 文档导航

## 公开阅读入口

普通读者从两份面向研究结论的材料开始：

1. 项目首页 [`README.md`](../README.md)：了解 Benchmark、安装方式和命令入口；
2. 正式文档 [ContextWorld ICL Benchmark](ContextWorld_ICL_Benchmark.md)：了解能力地图、
   任务定义、评分方法和参考结果。

九项任务在正式文档中按要识别的隐藏规律组织，而不是按环境数量组织。每项任务均采用
“任务目标、数据构成、评测方法、基线表现、适用范围”的统一结构，便于横向比较。

> [ContextWorld ICL Benchmark](ContextWorld_ICL_Benchmark.md)

## 能力导航与命令

| 能力类型 | 任务 | 主文档 | 机器配置 | 独立命令 |
|---|---|---|---|---|
| 即时连续响应 | 速度 | [6.1.1 速度](ContextWorld_ICL_Benchmark.md#611-速度) | `configs/benchmark/tworoom_speed_icl_release_v1.yaml` | `contextworld-speed` |
| 即时连续响应 | 推手移动幅度 | [6.1.2 推手移动幅度](ContextWorld_ICL_Benchmark.md#612-推手移动幅度) | `configs/benchmark/pusht_action_strength_icl_release_v1.yaml` | `contextworld-action-strength` |
| 即时连续响应 | 机械臂质量 | [6.1.3 机械臂质量](ContextWorld_ICL_Benchmark.md#613-机械臂质量) | `configs/benchmark/reacher_arm_mass_icl_release_v1.yaml` | `contextworld-reacher-arm-mass` |
| 时间延迟动力学 | 动作延迟 | [6.2.1 动作延迟](ContextWorld_ICL_Benchmark.md#621-动作延迟) | `configs/benchmark/tworoom_action_delay_icl_release_v1.yaml` | `contextworld-action-delay` |
| 接触或附着条件动力学 | 接触摩擦 | [6.3.1 接触摩擦](ContextWorld_ICL_Benchmark.md#631-接触摩擦) | `configs/benchmark/pusht_contact_friction_icl_release_v1.yaml` | `contextworld-contact-friction` |
| 接触或附着条件动力学 | 运动阻尼 | [6.3.2 运动阻尼](ContextWorld_ICL_Benchmark.md#632-运动阻尼) | `configs/benchmark/pusht_motion_damping_icl_release_v1.yaml` | `contextworld-motion-damping` |
| 接触或附着条件动力学 | Cube 夹爪携带规则 | [6.3.3 Cube 夹爪携带规则](ContextWorld_ICL_Benchmark.md#633-cube-夹爪携带规则) | `configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml` | `contextworld-cube-gripper-carry` |
| 隐藏结构转移规则 | 门通行规则 | [6.4.1 门通行规则](ContextWorld_ICL_Benchmark.md#641-门通行规则) | `configs/benchmark/tworoom_door_icl_release_v1.yaml` | `contextworld-door` |
| 隐藏结构转移规则 | 传送门出口位置 | [6.4.2 传送门出口位置](ContextWorld_ICL_Benchmark.md#642-传送门出口位置) | `configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml` | `contextworld-portal-exit` |

## Public v1 发布状态

[Public v1 发布准备与多模型验证矩阵](ContextWorld_Public_v1_Release_Readiness.md) 说明许可证、
公共下载、干净环境复现、外部模型验证和结果纳入规则等尚未完成的发布条件。它不是
Benchmark 的主叙事，也不改变主文档中的参考结果或提供新的 Public Test 访问权限。
仓库已具备本地技术接口；冻结 Public Test 不是可下载数据，Public v1 尚未正式发布。
Cube 的外部模型工作只是首批试点；其他三类能力的外部结果仍待补齐，因此不得宣称全
Suite 已完成跨架构验证。

## 复现材料

- `protocols/`：实验执行前冻结的协议与审计说明；
- `archive/`：已经结束的阶段材料；
- `reference/`：第三方工程与运行环境说明。

这些材料解释结果怎样得到，但不是发布说明，也不替代统一主文档中的当前结论。
