# ContextWorld Documentation

## Benchmark guide

[ContextWorld：视觉世界模型的上下文规则学习基准](ContextWorld_ICL_Benchmark.md)
是 v1 的公开入口，包含：

- Benchmark 的研究问题与任务定义；
- Speed 和 Door Rule 的 Training、Validation 与隔离方式；
- 评分指标、参考结果和报告规范；
- 安装、数据审计、训练与评测命令；
- 版本范围与扩展接口。

## Technical references

以下资料用于复现实验或追溯设计过程，不是使用 Benchmark 的前置阅读：

- `protocols/`：冻结实验协议与审计证据；
- `reference/`：数据卡与实现参考；
- Speed、Door Rule 专项报告：更完整的组件分析；
- `archive/`：历史计划和阶段快照。

若技术附录中的阶段性表述与公开指南不同，以公开指南及对应 release YAML 为准。
