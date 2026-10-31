# ContextWorld 论文

英文主稿为 `main_en.tex`，可直接阅读[英文 PDF](build/main_en.pdf)。`main.tex` 保留中文稿作为写作参考。两份源文件均记录中英文题目。

## 文件

| 文件 | 用途 |
|---|---|
| `main.tex` | 中文参考稿与编译入口 |
| `main_en.tex` | 英文主稿与编译入口 |
| `build/main_en.pdf` | 英文预览，随源稿同步更新 |
| `references.bib` | 参考文献 |
| `figures/` | 图片资源 |
| `cvpr.sty`、`ieeenat_fullname.bst` | CVPR 样式与参考文献格式 |

## 写作资料

英文主稿是当前写作入口；中文稿仅作内容参考，不与英文稿逐句同步。论文的方法、实验与分析主要依据以下文档：

| 来源 | 用途 |
|---|---|
| [Benchmark 主报告](../docs/ContextWorld_ICL_Benchmark.md) | 任务定义、数据构建、评分、训练方案、主要结果与高层机制结论 |
| [完整结果附录](../docs/reference/ContextWorld_Expanded_Training_Results.md) | 九任务详细对比、数据规模与模型容量实验、训练曲线、多步误差及 CEM |
| [评分验证与训练机制](../docs/ICL_Metric_Study.md) | 指标验证、训练干预、支持证据与结论的适用范围 |
| [数据生成方法](../docs/Data_Generation.md) | 轨迹与配对构建、数据划分及方法附录细节 |
| [模型训练说明](../docs/StableWM_Training.md) | 训练入口、初始化、冻结模块及实验设置 |

上述文档链接到的结果文件和图表均保存在仓库中。制作表格或绘图时，使用对应的 [JSON／CSV 结果](../docs/research/data/)，无需搬运原始轨迹、模型权重或重新运行训练。九任务正式主分见 [Test 结果](../docs/research/data/current_reference_test_v1/overview.md)；多步预测诊断及部分机制实验使用 Development，应按各结果注明的划分报告。

参考文献在 `references.bib` 中维护。引用已有工作的具体结论时，应核对原文；研究文档提供相关背景，但不能替代文献本身。完整复跑实验所需的数据与权重不包含在论文工程中。

写作与逐段整改遵循 [论文写作要求](AGENTS.md)。

## 图表与结果复算

正文实验按实验设置、任务适应结果、训练选择、预测损失与适应四个小节组织。完整九任务 Test 主表保留在正文；多步预测、数据与容量对照、训练损失与条件响应分别由图表展示。详细结果与评测设置保留在附录。

数据与容量图采用单栏：上图展示数据规模，下图直接显示模型扩容后的准确率变化（百分点）。训练图按任务将 loss 与 NRE 并排，保留训练集与 Development 对照；NRE 只有两个实测时点，连线不表示中间时点已评测。

`generate_results.py` 从仓库中已保存的 Test／Development 研究结果生成附录表格，`generate_figures.py` 生成协议示意图、多步预测图、数据与容量对照图，以及训练损失／条件响应图。它们不训练模型、不重新评测检查点。在仓库根目录运行：

```bash
python paper/generate_results.py
python paper/generate_figures.py
```

结果表读取 `docs/research/data/icl_training_study_v2.json`、`multistep_prediction_coverage_v2.json` 和 `model_capacity_scaling_v1/final_comparison.json`；训练图读取 `native_training_loss_v1/summary.csv` 与 `native_epoch_checkpoints_v1/summary.csv`。九任务 Test 主表依据 `current_reference_test_v1/summary.json`。多步图从 `multistep_prediction_coverage_v2_queries.json.gz` 复算逐时域误差比：各时域分别汇总预测误差与均值未来参照误差，再在检查点内取比值，最后对训练重复取均值。该曲线与先汇总所有时域再取比值的整体 `E/B` 不同。新图使用的数值、运行标识和区间保存于 `figures/experiment_plot_data.json`，可随脚本重新生成。

图形生成需要 Matplotlib 和 NumPy；编译论文直接使用已保存的 PDF 图，无需执行这些脚本。

本地 `contextworld-cvpr.zip` 保存编译所需论文源文件与图形。复算图表需要完整仓库中的上述结果文件，单独编译源包不需要原始数据或权重。

## 编译与预览

使用 TeX Live 2025 和 `latexmk`，在本目录编译英文稿：

```bash
make en
```

英文稿使用 pdfLaTeX，输出为 `build/main_en.pdf`。中文参考稿使用 XeLaTeX，通过 `make zh` 编译，输出为 `build/main.pdf`。

当前环境若未设置编译器路径，可先执行：

```bash
export PATH="/opt/texlive/2025/bin/x86_64-linux:$PATH"
```

在 Overleaf 中，中文稿选择 `main.tex` 和 XeLaTeX；英文稿选择 `main_en.tex` 和 pdfLaTeX。两者均使用 TeX Live 2025。中文字体使用 TeX Live 自带的 Fandol。

## 模板来源

样式文件原样取自 [CVPR 官方 author-kit](https://github.com/cvpr-org/author-kit/tree/CVPR2026-v1(latex)) 的 `CVPR2026-v1(latex)` 发布版。工程的目标会议年份为 2027；正式投稿时需按 CVPR 2027 官方模板与要求设置。中文双栏预览的篇幅不用于判断英文投稿页数。
