# TwoRoom 速度实验协议导航

本目录只保存“实验怎样执行”的技术材料，不维护当前结论。当前数据、模型对比、
能力等级和下一步统一见
[TwoRoom History-3 速度 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

## 当前正式协议

| 文档 | 状态 | 用途 |
|---|---|---|
| [History-3 速度 Benchmark v2](TwoRoom_History3_Speed_Benchmark_v2_Protocol.md) | Validation 已执行，Test 封存 | 原始数据、单速合成混训和多速度合成混训的隔离实验、完整速度矩阵与复现入口 |
| [速度范围外与多步预测](TwoRoom_History3_Speed_Extrapolation_Multistep_v1_Protocol.md) | Validation 已执行，Test 封存 | 低端/高端范围外速度和 1/2/3/5-step 离线真实未来 latent |

## 支持性协议

以下文档记录了形成 v2 设计之前的专项实验。它们仍用于复现，但其中的阶段表述不再
作为当前结论入口。

| 文档 | 回答的问题 | 当前报告对应位置 |
|---|---|---|
| [原始能力重建](TwoRoom_OriginalAbility_Reconstruction_Protocol.md) | 速度 5 合成单训能否重建基础能力 | 6.2 节 |
| [速度 5 跨 Eval](TwoRoom_Speed5_CrossEval_Protocol.md) | 原始与合成 Eval 的分数差来自哪里 | 6.3 节 |
| [CEM 规划配置影响](TwoRoom_CEM_Planning_Config_Impact_Protocol.md) | horizon、候选数和迭代数如何影响速度分数 | 5.3 节 |
| [上下文敏感距离校准](TwoRoom_Speed_ICL_Sensitive_Eval_Protocol.md) | 怎样避免远目标地板 | 6.3 节 |
| [双向速度上下文评测](TwoRoom_Speed_Context_Direction_Eval_Protocol.md) | 早期低/中/高速历史现象能否复现 | 5 节 |
| [SpeedFull 数据与训练](TwoRoom_SpeedFull_Data_Protocol.md) | 早期多速度数据怎样生成和训练 | 历史复现材料 |
| [SpeedClean 数据与训练](TwoRoom_SpeedClean_Data_Protocol.md) | 早期单速控制怎样生成和训练 | 历史复现材料 |
| [SpeedTask 数据与训练](TwoRoom_SpeedTask_Data_Protocol.md) | 更早期速度任务怎样生成和评测 | 历史复现材料 |

## 阅读规则

- 只判断“训练数据怎么配、速度 ICL 表现怎样、CEM 成功率怎样”时，读主报告
  第 1–5 节；
- 审核隔离训练、规划和基础能力时读 v2；审核范围外与多步评分时读多步协议；
- 只有需要复现旧对照或追溯设计原因时，才打开支持性协议；
- 支持性协议与主报告表述不一致时，以主报告和两份当前正式协议为准。
