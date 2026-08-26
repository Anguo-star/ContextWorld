# ContextWorld 参考结果复现附录

本文记录主 Benchmark 文档中参考结果的来源、评测预算和机器可读文件。普通使用者运行
Training 或 Development 评测时不需要阅读本附录；复核论文表格、比较训练配方或重新生成
参考结果时再使用这些信息。

## 1. 结果分层

主文档中的结果分为三类：

1. 原始模型在 ContextWorld 任务上的 ICL 起点；
2. 原始模型在原环境中的 CEM 规划起点；
3. 使用组件训练数据后的 ICL 与原任务规划保持结果。

这三类结果使用不同输入和不同结论范围。原始环境 CEM 不能复制到多个能力任务后再次
计分，Development 结果也不能作为 Public Test 成绩。

## 2. 原始 LeWM 与 PLDM 检查点

TwoRoom、PushT、Reacher 和 Cube 各包含一套 LeWM 与 PLDM 原始基线。检查点使用公开原
环境数据，按相应 Stable-WorldModel 基线配方从头训练 10 epochs；不使用 ContextWorld
组件数据，不继续训练已有权重，也不按评测分数选择检查点。

用于族级 CEM 统计的训练种子为 3072、3073 和 3074。主文档报告三个训练种子的平均成功
率和样本标准差；单个训练种子的完整计数保存在机器可读结果中。

原始 ICL 表按能力任务评分。同一原环境检查点可以用于多个能力任务，但每个任务仍只产生
一个独立分数。接触摩擦和运动阻尼使用 Development；其他已报告行来自此前完成的 Public
Test 参考评测。Public Test 数据本身仍保持封存。

动作延迟要求 History=7，而原始 LeWM 与 PLDM 检查点使用 History=3。原始结果只输入
History=7 查询中对齐的最后三帧和相应动作，因此只能解释为 History=3 模型在该评分器下
的零样本起点。

## 3. 原始环境 CEM

每个检查点完成 300 次规划评测。当前标准执行方式为六个评测种子（42–47），每个种子
50 次。部分较早完成的 Reacher 与 Cube 结果使用三个评测种子、每个种子 100 次；总预算
仍为 300 次，主文档已明确区分。新的训练后自动评测统一使用 6 × 50。

共同规划参数为：

| 参数 | 值 |
|---|---:|
| candidates | 300 |
| iterations | 30 |
| top-k | 30 |
| history | 3 |
| action block | 5 |
| goal offset | 25 |
| episode budget | 50 |

规划前后记录模型 state-dict SHA-256，以确认评测过程没有改变模型权重。

机器可读汇总：

- `artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json`
- `configs/benchmark/contextworld_original_baseline_seed_completion_results_freeze_v1.json`
- `artifacts/evaluation/complete_reference_comparison_v1/complete_comparison_v2.json`

## 4. DINO-WM / PreJEPA 起点

四个原环境各训练三枚 DINO-WM / PreJEPA 检查点，训练种子为 3072、3073 和 3074，均只
使用原环境数据并训练 10 epochs。

这些原始检查点的 predictor 还需要 `observation` 或 `proprio`，而 ContextWorld v1 的正式
ICL 接口只提供图像和动作。因此，它们不能获得正式 ICL 分数。主文档中的辅助结果把缺失
输入固定为模型归一化空间中的零，不读取模拟器状态，也不改变模型权重。动作延迟另使用
对齐的最后三帧。该辅助分析只用于说明原始检查点的数值起点。

机器可读汇总：

- `artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json`

DINO-WM 原始环境 CEM 使用相同的每检查点 300 次预算。后续使用 ContextWorld 组件训练的
PreJEPA 检查点应只依赖图像和动作，并直接使用正式 ICL 接口。

## 5. 组件训练后的结果

训练后参考方法通常使用三个独立训练种子。方法只有在三个检查点分别通过任务的主分数和
全部附加条件时才记为 3/3 通过。平均分达到门槛不能替代逐检查点判定。

训练后 CEM 将每个检查点与其对应的原始检查点比较，而不是与模型族平均值比较。表中的
“保持”表示性能变化没有超过该任务预先规定的允许范围，不表示成功率完全没有下降。

部分方法在 Development 或 ICL 阶段未满足继续评测的条件，因此没有训练后 CEM。此类
“未运行”只描述训练后流程，不表示相应原始检查点或原始环境 CEM 缺失。

统一结果入口：

```bash
contextworld-benchmark results
python scripts/audit_contextworld_original_baseline_matrix_freeze_v1.py
```

第一个命令读取当前机器可读参考结果。第二个命令验证原始 LeWM/PLDM ICL 矩阵的检查点、
输入和结果文件是否一致。

## 6. 版本与完整性

复现实验应记录：代码提交、Stable-WorldModel 版本、数据 manifest SHA-256、训练配方、训练
种子、检查点 SHA-256、Adapter 版本、评测划分、评测种子和输出文件 SHA-256。

`docs/protocols/` 保存执行前确定的任务协议，`docs/archive/` 保存已经结束的实验阶段材料。
这些材料用于核对已报告结果，不是公开 Training 或 Development 工作流的运行依赖。
