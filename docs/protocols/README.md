# TwoRoom 速度实验协议导航

本目录只保存“实验怎样执行”的技术材料，不维护当前结论。当前数据、模型对比、
能力等级和下一步统一见
[TwoRoom History-3 速度 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

## 当前正式协议

| 文档 | 状态 | 用途 |
|---|---|---|
| [History-3 速度 Benchmark v2](TwoRoom_History3_Speed_Benchmark_v2_Protocol.md) | Validation 已执行，Test 封存 | 当前 M0/M1/M2 隔离训练、完整速度矩阵、统计门和复现入口 |

## 支持性协议

以下文档记录了形成 v2 设计之前的专项实验。它们仍用于复现，但其中的阶段表述不再
作为当前结论入口。

| 文档 | 回答的问题 | 当前报告对应位置 |
|---|---|---|
| [原始能力重建](TwoRoom_OriginalAbility_Reconstruction_Protocol.md) | 速度 5 合成单训能否重建基础能力 | 0.6、2.4 节 |
| [速度 5 跨 Eval](TwoRoom_Speed5_CrossEval_Protocol.md) | 原始与合成 Eval 的分数差来自哪里 | 0.6、1.3 节 |
| [CEM 规划配置影响](TwoRoom_CEM_Planning_Config_Impact_Protocol.md) | horizon、候选数和迭代数如何影响速度分数 | 0.6、6.3 节 |
| [上下文敏感距离校准](TwoRoom_Speed_ICL_Sensitive_Eval_Protocol.md) | 怎样避免远目标地板 | 1.3 节 |
| [双向速度上下文评测](TwoRoom_Speed_Context_Direction_Eval_Protocol.md) | 早期低/中/高速历史现象能否复现 | 6.3 节 |
| [SpeedFull 数据与训练](TwoRoom_SpeedFull_Data_Protocol.md) | 早期多速度数据怎样生成和训练 | 历史复现材料 |
| [SpeedClean 数据与训练](TwoRoom_SpeedClean_Data_Protocol.md) | 早期单速控制怎样生成和训练 | 历史复现材料 |
| [SpeedTask 数据与训练](TwoRoom_SpeedTask_Data_Protocol.md) | 更早期速度任务怎样生成和评测 | 历史复现材料 |

## 阅读规则

- 只判断“当前得到什么结论”时，只读主报告第 0 节；
- 审核当前实验是否公平时，读 v2 正式协议；
- 只有需要复现旧对照或追溯设计原因时，才打开支持性协议；
- 支持性协议与主报告表述不一致时，以主报告和 v2 正式协议为准。
