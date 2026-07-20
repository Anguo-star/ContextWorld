# ContextWorld 文档导航

如果只读一份文档，请读
[TwoRoom 速度上下文学习 Benchmark 报告](TwoRoom_Speed_Benchmark_Report.md)。
它统一维护当前阶段的问题、实验设计、数据结果、结论边界和下一步工作。

## 当前文档

| 文档 | 作用 | 何时阅读 |
|---|---|---|
| [TwoRoom 速度 Benchmark 报告](TwoRoom_Speed_Benchmark_Report.md) | 当前实验结果与阶段结论 | 想知道“做到了什么、证据是什么、下一步是什么” |
| [ContextWorld Benchmark 设计规范](ContextWorld_Benchmark_Design.md) | 跨任务的评测原则、指标和冻结流程 | 想设计或审核 Benchmark |
| [History-3 速度 Benchmark v2 方案](protocols/TwoRoom_History3_Speed_Benchmark_v2_Protocol.md) | 下一阶段完整模型、数据、指标和执行矩阵 | 想审核下一步如何形成正式 History-3 结论 |

当前证据属于单训练种子的 Validation 机制验证，正式 Test 尚未启用。报告中的
50/75/100 步是 TwoRoom Validation 的当前执行预算阶梯，25/50 步是本阶段的
模型视野对照；这些数值都不应直接套用到其他任务。

当前文档统一使用“慢速历史、同速历史、快速历史”。它们只描述历史速度相对查询
环境速度的关系，不表示正确或错误。旧配置和结果中的 `correct`、`wrong_slow`、
`wrong_fast` 是冻结机器字段，不再作为概念名称。

v2 的训练/Eval 速度集合、三个成对训练种子、`3×3` 速度矩阵、物理下一状态指标
和主 planner profile 已预注册；模型尚未重训，Validation 尚未评分。

## 复现材料

`protocols/` 保存已经执行或冻结的数据、训练和评测协议。它们用于说明某项结果
如何得到，不负责更新当前结论：

- [原始能力重建对照](protocols/TwoRoom_OriginalAbility_Reconstruction_Protocol.md)
- [速度 5 跨 Eval 对照](protocols/TwoRoom_Speed5_CrossEval_Protocol.md)
- [多速度训练数据](protocols/TwoRoom_SpeedFull_Data_Protocol.md)
- [速度上下文双向评测](protocols/TwoRoom_Speed_Context_Direction_Eval_Protocol.md)
- [上下文敏感距离校准](protocols/TwoRoom_Speed_ICL_Sensitive_Eval_Protocol.md)
- [CEM 规划配置影响实验](protocols/TwoRoom_CEM_Planning_Config_Impact_Protocol.md)
- [History-3 速度 Benchmark v2 方案](protocols/TwoRoom_History3_Speed_Benchmark_v2_Protocol.md)
- [SpeedClean 数据与训练](protocols/TwoRoom_SpeedClean_Data_Protocol.md)
- [SpeedTask 数据与训练](protocols/TwoRoom_SpeedTask_Data_Protocol.md)

`reference/` 保存仍可能被配置或工具引用的说明材料：

- [TwoRoom Step 1 数据卡](reference/TwoRoom_Benchmark_Step1_Data_Card_v1.md)
- [StableWorldModel 执行边界](reference/StableWorldModel_Execution_Boundary.md)

## 历史材料

`archive/` 是只读历史快照，包括早期实施计划、工具评估、阶段报告和旧路线图。
这些文件保留决策背景，但不再作为当前状态或结论来源。发现归档内容与当前报告
不一致时，以当前报告和设计规范为准。
