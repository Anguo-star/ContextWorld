# ContextWorld 文档入口

这里的公开文档只保留四条阅读路径。读者不需要从实验记录中自行拼结论。

## 我想使用 Benchmark

阅读 [TwoRoom Speed ICL Benchmark 使用指南](TwoRoom_Speed_ICL_Benchmark_Release.md)。

它提供数据布局、完整性校验、Stable-WorldModel 训练、单模型评分、完整方法汇总、
规划评测和其他模型 adapter 契约。

## 我只想知道结果

阅读 [TwoRoom History-3 速度学习报告](TwoRoom_Speed_Benchmark_Report.md)。

这份报告回答：

- 用什么训练数据；
- 模型是否真的会根据历史调节速度预测；
- 这种能力能保持多少步；
- 训练范围外是否有效；
- CEM 成功率为什么不能代替预测误差；
- 当前结论的边界和下一阶段是什么。

## 我想设计新的 Benchmark

阅读 [ContextWorld Benchmark 设计指南](ContextWorld_Benchmark_Design.md)。

它说明怎样区分隐藏动力学与可见几何，怎样准备训练对照、Eval 数据、预测指标和
规划指标，以及怎样避免把一个难解释的百分数当成“模型准确率”。

当前下一阶段的门能力设计集中在
[TwoRoom 门能力 Benchmark 设计](TwoRoom_Door_Benchmark_Design.md)。这份文档明确区分
可见门位置泛化、隐藏通道规则 ICL 和速度×门组合，并给出尚未执行的训练数据、Eval、
指标和完整执行顺序。

## 我需要复现实验

阅读 [复现实验索引](protocols/README.md)。

其中两份当前协议分别负责：

- [隔离训练、区间内评测、规划与能力保持](protocols/TwoRoom_History3_Speed_Benchmark_v2_Protocol.md)；
- [训练范围外速度与多步真实未来评测](protocols/TwoRoom_History3_Speed_Extrapolation_Multistep_v1_Protocol.md)。

协议保存配置、样本数、哈希、判定门和运行命令。它们服务于复现，不重复讲一遍
当前结论。

## 文档分工

| 文档 | 面向谁 | 只负责什么 |
|---|---|---|
| 结果报告 | 所有读者 | 当前数据、结果、结论和下一步 |
| 设计指南 | Benchmark 设计者 | 可复用的设计原则和指标解释 |
| 门能力设计 | TwoRoom 下一阶段实施者 | 门部分的三阶段顺序与第 1 阶段冻结方案 |
| 使用指南 | Benchmark 使用者 | 数据准备、训练、评分和提交格式 |
| 执行协议 | 实验复现者 | 冻结配置、审计规则和命令 |
| `archive/` | 需要追溯历史的人 | 旧计划和阶段快照，不代表当前结论 |

如果文件之间出现表述差异，以结果报告中的当前结论、设计指南中的通用规则和当前
执行协议中的冻结配置为准。

## 名称约定

| 名称 | 含义 |
|---|---|
| 原始数据模型 | 只用原始速度 5 数据训练 |
| 单速合成混训模型 | 原始数据与速度 5 合成数据按 1:1 抽样 |
| 多速度合成混训模型 | 原始数据与 32 个速度的合成数据按 1:1 抽样 |

历史条件只称为较慢历史、同速历史和较快历史，不使用“正确速度”“错误速度”或
“反事实速度”。`agent.speed` 是环境动力学参数；`action block=5` 是时间聚合方式，
两者不是同一个变量。
