# ContextWorld 文档导航

## 从哪里开始

如果只想知道“现在做到什么、证据是什么、下一步是什么”，请读：

[TwoRoom History-3 速度 Benchmark 报告](TwoRoom_Speed_Benchmark_Report.md)

如果要审核 Benchmark 是否合理，再读：

[ContextWorld Benchmark 设计规范](ContextWorld_Benchmark_Design.md)

如果要复现实验或核对冻结计数，再读：

[History-3 速度 Benchmark v2 执行协议](protocols/TwoRoom_History3_Speed_Benchmark_v2_Protocol.md)

三份文档分工固定：

| 文档 | 只负责什么 |
|---|---|
| 结果报告 | 当前数据、阶段结论、失败边界和下一步 |
| 设计规范 | 跨任务通用的 Benchmark 原则 |
| 执行协议 | 本实验冻结的数据、模型、计数、指标和命令 |

结果不再分散写入多个阶段记录。旧实验只保留复现和决策背景，不作为当前结论入口。

## 当前状态

History-3 速度 v2 已完成：

- 608 对严格匹配的单速/多速度训练 scenario 审计；
- 三个成对训练种子的 M1/M2 训练；
- 7 个 checkpoint 的时间因果掩码审计；
- 真实下一状态、固定候选和 CEM 闭环的完整 50×6 矩阵；
- 新模型在原始 heldout 与速度 5 合成 Eval 上的能力保持评测；
- 136 项代码测试。

Validation 已完整结束，所有评测任务均成功产出；封存 Test 尚未打开。正式判定只
通过 A 级“历史敏感”，B–E 级尚未通过。M2 的原能力保持为 3/3 训练种子通过，
因此失败原因不是原始能力退化。

当前不会提前写成“完整速度自适应”。已经稳定建立的是：多速度模型会根据
History-3 改变相同 action 下的下一状态步长，而且该效应强于匹配单速控制；尚未
完全通过的是所有 query 速度和多步 rollout 上的真实动力学校准。

## 阅读用名称

当前报告只使用：

| 名称 | 含义 |
|---|---|
| M0 原始单速基线 | 只用原始速度 5 数据训练 |
| M1 匹配单速控制 | 原始数据加速度 5 合成数据 |
| M2 匹配多速度模型 | 原始数据加多速度合成数据 |

历史条件称为低速、中速和高速历史。只有在“历史与 query 属于同一稳定环境”的
任务假设下，才用“同速历史”描述二者速度相同。当前文档不使用带有对错含义的
历史名称。

`agent.speed` 与 `frameskip/action block` 也必须分开：

- 本轨道改变 `agent.speed`；
- `frameskip/action block` 始终为 5；
- 改变观测频率或动作重复次数属于独立 Benchmark。

## 复现材料

`protocols/` 保存已执行或冻结的专项协议，包括数据生成、能力重建、CEM 资源影响
和 History-3 v2。它们用于复现，不重复维护当前结论。

主要机器汇总写入：

```text
artifacts/evaluation/history3/speed_isolated_v2/final_summary.json
```

`reference/` 保存仍被配置或工具引用的边界说明。`archive/` 保存旧计划和阶段
快照；若其内容与当前报告冲突，以当前报告、设计规范和 v2 执行协议为准。
