# ContextWorld 实验复现索引

本目录保存冻结配置、审计规则和运行方法。它面向需要复现实验的读者，不负责讲述
当前结论；结果请直接阅读
[TwoRoom History-3 速度学习报告](../TwoRoom_Speed_Benchmark_Report.md)或
[TwoRoom 可见门位置实验报告](../TwoRoom_Door_Benchmark_Design.md)。

## 当前协议

| 协议 | 覆盖内容 | 状态 |
|---|---|---|
| [History-3 速度 Benchmark v2](TwoRoom_History3_Speed_Benchmark_v2_Protocol.md) | 隔离训练、区间内速度、固定候选、CEM 资源阶梯、能力保持 | Validation 已执行，Test 封存 |
| [范围外速度与多步预测](TwoRoom_History3_Speed_Extrapolation_Multistep_v1_Protocol.md) | 低端和高端范围外速度、1/2/3/5 步真实未来 latent | Validation 已执行，Test 封存 |

复现完整当前结论时，两份都需要：第一份建立训练归因和规划证据，第二份补全多步
预测与能力边界。协议中的旧字段名和预注册计算式为了机器结果兼容保持不变；面向
读者的名称和指标解释以主报告为准。

## 可见门位置泛化

门位置 v1 已完成单门与多门数据、六个配对模型和验证集评测。正式通过标准未满足，
封存测试集（sealed Test）继续锁定。复现时使用以下冻结入口：

| 文件 | 作用 |
|---|---|
| [`tworoom_door_visual_generalization_v1.yaml`](../../configs/benchmark/tworoom_door_visual_generalization_v1.yaml) | Eval 划分、指标、样本数和判定门槛 |
| [`tworoom_door_training_v2.yaml`](../../configs/benchmark/tworoom_door_training_v2.yaml) | 六模型训练矩阵和训练审计 |
| [`tworoom_door_sealed_test_gate_v1.json`](../../configs/benchmark/tworoom_door_sealed_test_gate_v1.json) | 封存测试集锁定状态 |
| [可见门位置实验报告](../TwoRoom_Door_Benchmark_Design.md) | 实际结果、解释边界和运行入口 |

门在当前 query 画面中可见，所以该协议测视觉几何泛化，不把结果称为门位置 ICL。
机器汇总的顶层 `status=passed` 只表示矩阵完整；正式结论读取预测报告中的能力判定字段。

## 隐藏通行规则 ICL

门的外观保持不变，只改变它在碰撞计算中能否通过。第一阶段已经证明 History-3 可以
在动作相同、query 状态和像素完全相同的前提下，先留下规则证据，再让同一 query 动作
产生不同的真实未来：

| 文件 | 作用 | 状态 |
|---|---|---|
| [History-3 隐藏通行规则可行性协议](TwoRoom_History3_Hidden_Passage_Feasibility_v1.md) | 32 对连续轨迹、物理结果、序列化、路径和动作泄露审计 | 可行性通过，端到端 pilot 待执行 |
| [`tworoom_hidden_passage_h3_feasibility_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_feasibility_v1.yaml) | 环境、动作、样本覆盖和通过标准 | 已冻结 |

这项结果只开放 `World → Lance → 重载 → 规则恢复 → 真实模型输入` pilot，不代表模型
已经学会门规则 ICL。当前 History-3 恢复还依赖碰撞器的 8.5 px 投影；pilot 通过并决定
保留该限定或改用更长 History 后，才能开始正式训练。正式阶段仍须使用单规则控制和
双规则训练的三个配对种子，并让每个模型、真实规则和历史条件各自拥有独立
`50×6=300` 个 query。

## 专项支持协议

这些文件保存形成当前设计之前的专项对照。只有复现相应子问题时才需要阅读。

| 协议 | 子问题 |
|---|---|
| [原始能力重建](TwoRoom_OriginalAbility_Reconstruction_Protocol.md) | 速度 5 合成单训能否重建基础能力 |
| [速度 5 跨 Eval](TwoRoom_Speed5_CrossEval_Protocol.md) | 原始与合成 Eval 的分数差来自哪里 |
| [CEM 规划配置影响](TwoRoom_CEM_Planning_Config_Impact_Protocol.md) | horizon、候选数和迭代数如何影响成功率 |
| [上下文敏感距离校准](TwoRoom_Speed_ICL_Sensitive_Eval_Protocol.md) | 怎样避免远目标造成成功率地板 |
| [双向速度历史评测](TwoRoom_Speed_Context_Direction_Eval_Protocol.md) | 较慢、同速和较快历史的早期规划现象 |
| [SpeedFull 数据与训练](TwoRoom_SpeedFull_Data_Protocol.md) | 早期多速度数据的生成和训练 |
| [SpeedClean 数据与训练](TwoRoom_SpeedClean_Data_Protocol.md) | 早期单速控制的生成和训练 |
| [SpeedTask 数据与训练](TwoRoom_SpeedTask_Data_Protocol.md) | 更早期速度任务的生成和评测 |

## 阅读顺序

- 只想理解结果：不要从协议开始，阅读主报告；
- 想检查 Benchmark 为什么这样设计：阅读
  [ContextWorld Benchmark 设计指南](../ContextWorld_Benchmark_Design.md)；
- 想复现速度实验：先读速度 v2，再读范围外与多步协议；想复现门实验：从门报告和门
  主配置开始；最后按需查专项协议；
- 专项协议中的阶段性判断若与主报告不同，以主报告的当前结论为准。
