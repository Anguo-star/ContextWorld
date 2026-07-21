# ContextWorld 文档导航

## 从哪里开始

如果只想知道“现在做到什么、证据是什么、下一步是什么”，请读：

[TwoRoom History-3 速度 Benchmark 报告](TwoRoom_Speed_Benchmark_Report.md)

如果要审核 Benchmark 是否合理，再读：

[ContextWorld Benchmark 设计规范](ContextWorld_Benchmark_Design.md)

如果要复现实验或核对冻结计数，再读：

[History-3 速度 Benchmark v2 执行协议](protocols/TwoRoom_History3_Speed_Benchmark_v2_Protocol.md)

旧实验协议按用途统一索引在：

[TwoRoom 速度实验协议导航](protocols/README.md)

文档分工固定：

| 文档 | 只负责什么 |
|---|---|
| 结果报告 | 当前数据、正式结论、失败边界和下一步 |
| 设计规范 | 跨任务通用的 Benchmark 原则 |
| v2 执行协议 | 本实验冻结的数据、模型、计数、指标和命令 |
| 协议导航 | 旧实验分别回答什么问题，以及它们在当前报告中的位置 |

结果不再分散写入多个阶段记录。旧实验只保留复现和决策背景，不作为当前结论入口。

## 当前状态放在哪里

README 不重复维护实验数字，避免同一结论散落在多个文件。训练数据配方、速度 ICL
指标、CEM 成功率和最终判断均统一维护在主报告第 1–8 节。

## 阅读用名称

当前报告只使用：

| 名称 | 含义 |
|---|---|
| 原始数据模型 | 只用原始速度 5 数据训练 |
| 单速合成混训模型 | 原始数据加速度 5 合成数据，抽样比 1:1 |
| 多速度合成混训模型 | 原始数据加 32 个速度的合成数据，抽样比 1:1 |

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
