# LeWM / PLDM：2026-09-03 之前的参考结果（历史归档）

本文归档 [ContextWorld ICL Benchmark](../ContextWorld_ICL_Benchmark.md) 主文档 §5.1
在 2026-09-03 更新为标准 `joint_scratch_v1`/`native` 完整三训练种子结果之前，各任务
引用过的 LeWM / PLDM 结果。这些结果本身仍然有效——检查点、训练日志和评测输出都
存在，没有被判定为错误——但它们分别来自与当前主参考不同的训练方法或训练协议，
**不能与当前主参考直接相减计算“训练前后变化”**，因此从主表移到本文单独保存。

每条记录标注其训练方法/协议身份，便于研究者判断能否与自己的实验直接比较。

## 1. 推手移动幅度、机械臂质量、动作延迟、门通行规则、传送门出口位置

这五项任务此前的“组件训练后”数值同样来自 `joint_scratch_v1`/`native` 配方，但属于
**同一配方更早的一批检查点**（训练种子同为 3072/3073/3074，但检查点内容与
2026-09-03 批次不同；参见主文档 §5.3 关于批次间差异的说明）：

| 任务 | 模型 | ICL 主分数（Public Test，3/3 或 0/3） | 训练后原任务 CEM |
|---|---|---:|---|
| 推手移动幅度 | LeWM | 96.61% ± 0.41pp（3/3 通过） | 74.67% / 72.00% / 77.33%（保持） |
| 推手移动幅度 | PLDM | 94.27% ± 0.30pp（0/3，仅差临界值） | 补跑：75.67% / 76.67% / 76.33%（保持） |
| 机械臂质量 | LeWM | 76.11% ± 0.49pp（3/3 通过） | 54.33% / 54.33% / 53.33%（保持） |
| 机械臂质量 | PLDM | 63.15% ± 0.41pp（0/3） | 补跑：76.67% / 75.33% / 77.00%（未保持） |
| 动作延迟 | LeWM | 32.43% ± 0.50pp（0/3，最弱响应组 0%） | 97.33% / 97.33% / 97.33%（保持） |
| 动作延迟 | PLDM | 93.36% ± 0.39pp（3/3 通过） | 95.00% / 97.33% / 96.33%（保持） |
| 门通行规则 | LeWM | 100.00% ± 0.00pp（3/3 通过） | 65.67% / 63.00% / 62.33%（未保持） |
| 门通行规则 | PLDM | 99.33% ± 0.00pp（3/3 通过） | 45.00% / 44.00% / 48.67%（未保持） |
| 传送门出口位置 | LeWM | 83.92% ± 1.41pp（0/3） | 92.00% / 90.33% / 88.67%（保持） |
| 传送门出口位置 | PLDM | 59.31% ± 0.63pp（0/3） | 补跑：94.67% / 96.00% / 95.33%（保持） |

机器可读来源：`artifacts/evaluation/complete_reference_comparison_v1/complete_comparison_v2.json`，
`artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/public_scoreboard_spec.json`。

## 2. 接触摩擦、运动阻尼

此前唯一的“组件训练后”结果来自 **`CW_METHOD=coja_v1`** 方法（检查点根
`coja-contact-friction-lewm-full-v1`、`coja-motion-damping-lewm-full-v1`），且各自只有
一个训练种子：

| 任务 | 模型 | 方法 | 主分数 | 正确历史 | 未通过原因 |
|---|---|---|---:|---:|---|
| 接触摩擦 | LeWM | `coja_v1`（单种子） | 96.09% | 90.23% | 正确历史未过 95% 门槛 |
| 运动阻尼 | LeWM | `coja_v1`（单种子） | 97.46% | 52.93% | 正确历史未过 95% 门槛（接近随机） |
| 接触摩擦 | PLDM | `native`（单种子） | 52.73% | 69.14% | 主分数等多项门槛未过 |
| 运动阻尼 | PLDM | `native`（单种子） | 51.95% | 57.42% | 主分数等多项门槛未过 |

另有更早的 2,048 对、4,096 步 legacy 配方三训练种子对照（不是当前 8,192 对配方的
证据）：

| 任务 | 模型 | Public Test ICL | 原 PushT CEM |
|---|---|---:|---|
| 接触摩擦 | LeWM | 49.35% ± 0.11pp（0/3） | 71.00% / 74.67% / 73.33%（未保持） |
| 接触摩擦 | PLDM | 50.00% ± 0.00pp（0/3） | 69.33% / 75.67% / 70.33%（未保持） |
| 运动阻尼 | LeWM | 50.07% ± 0.11pp（0/3） | 73.33% / 72.33% / 72.67%（未保持） |
| 运动阻尼 | PLDM | 50.26% ± 0.11pp（0/3） | 66.00% / 67.00% / 67.67%（未保持） |

机器可读来源：`artifacts/evaluation/history3/pusht_contact_friction_h3_strict_v3/development_decision.json`、
`artifacts/evaluation/history3/pusht_motion_damping_release_v1/failed_development.json`、
`artifacts/evaluation/pldm_reference_completion_v1/pusht_contact_friction_current_v3/pilot_seed13313/development_evaluation_decision_v1.json`、
`artifacts/evaluation/pldm_reference_completion_v1/pusht_motion_damping_current_v4/pilot_seed14321/development_evaluation_decision_v1.json`。

## 3. Cube 夹爪携带规则

LeWM 此前的 PASS 结果来自专门的 **`v4r1` 协议**（见 `docs/protocols/
Cube_Gripper_Carry_History3_v4r1_*`），不是标准 `joint_scratch_v1` 配方：

| 模型 | 身份 | ICL 主分数（Public Test） | 训练后 CEM |
|---|---|---:|---|
| LeWM | `v4r1` 协议，3 训练种子 | 77.73% / 79.10% / 78.52%（3/3 通过） | 186 / 183 / 185 / 300（相对基线 198/300，保持） |
| PLDM | `v4r1` 协议，Development | 50.13% ± 0.11pp（0/3） | 未运行（未满足 Development 准入条件） |
| PLDM（外部补充身份） | `v4r1` 检查点，独立登记 | Public Test 50.98% ± 0.20pp（0/3） | 55.11% ± 0.77pp（保持） |

PLDM 的外部补充结果使用与正式参考流程相同的 `v4r1` 检查点，但以独立结果身份登记，
不改写正式 scoreboard（工件标记 `external_three_seed_method`、
`formal_scoreboard_eligible: false`）。

## 如何解读

- 本文档的数值**不是错误**，也不是被撤回的结果——检查点和评测记录仍然可查。
- 它们**不能**与 [主文档 §5.1](../ContextWorld_ICL_Benchmark.md#51-当前标准参考2026-09-03-冻结快照)
  的当前参考直接相减，因为方法、协议或所属批次不同（`coja_v1` ≠ `native`；`v4r1` ≠
  `joint_scratch_v1`；同一 `native` 配方的不同批次也不隐含相同的训练/数据快照，见
  主文档 §5.3）。
- 如果某个历史结果明显强于当前 `native` 主参考（如 Cube LeWM 的 `v4r1` 结果），
  它仍然是一个有价值的强对照：后续新方法如果只对比“弱化的新 native 基线”，并不能
  说明改进了这个任务本身已知可达到的最好结果。
