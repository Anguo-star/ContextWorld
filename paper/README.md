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

上述文档链接到的结果文件和图表均保存在仓库中。制作表格或绘图时，使用对应的 [JSON／CSV 结果](../docs/research/data/)，无需搬运原始轨迹、模型权重或重新运行训练。九任务完整 Test 准确率、History 与 NRE 见[三项指标汇总](../docs/research/data/current_reference_test_v1/metrics.md)，并提供 [CSV](../docs/research/data/current_reference_test_v1/metrics.csv) 与 [JSON](../docs/research/data/current_reference_test_v1/metrics.json)；正式评测统一使用 Test，训练过程分析使用 Training；Development 用于方法与检查点选择。多步 Test 结果同样覆盖九个任务、136 次运行和 86 个模型／任务／训练方案组合，见[完整汇总](../docs/research/data/multistep_prediction_test_v1/summary.csv)。Gripper carry 使用全部 256 个 Test 来源样本连续重渲染历史与多步未来；配对约束及与原始图像的重放差异均有[可复现记录](../docs/research/data/multistep_prediction_test_v1/cube_replay/README.md)，三个主指标仍使用原始 Test 数据。

参考文献在 `references.bib` 中维护。引用已有工作的具体结论时，应核对原文；研究文档提供相关背景，但不能替代文献本身。完整复跑实验所需的数据与权重不包含在论文工程中。

写作与逐段整改遵循 [论文写作要求](AGENTS.md)。

## 图表与结果复算

主图见 [主图预览](design/response_fidelity_preview.pdf)，训练图见 [设计预览](design/figure_design.pdf)；[设计文件](design/README.md)包含可编辑 SVG、真实轨迹渲染、数据来源和复绘脚本。

正文实验按实验设置、任务适应结果、训练选择、预测损失与适应四个小节组织。完整九任务 Test 主表以模型和训练方案为行、任务为列，每个单元格从上到下列出 Accuracy、History 和 NRE 的均值，指标顺序只在表注中统一说明；附录完整表同时列出三项指标的标准差及训练重复次数。多步预测、数据扩量与编码器冻结对照、训练损失与条件响应分别由图表展示；模型容量结果保留在附录。详细结果与评测设置保留在附录。

Figure 3 两部分均使用 Test：上图比较 LeWM 与 PLDM 在推手位移任务上使用 2,048 与 32,768 对 ContextWorld 训练数据的结果；下图对两模型统一比较摩擦、质量和阻尼三个任务上的联合训练与冻结编码器训练。两部分采用相同准确率刻度，柱顶标注准确率；数据扩量时原环境数据集保持相同。Test 端点的三项指标与训练重复次数见附录，其他数据规模和模型容量结果也使用 Test，原环境 CEM 结果单独报告。正文训练图只展示 PLDM 摩擦任务的联合训练案例：左图显示六个 epoch 的预测损失，右图显示两个实测端点的训练集 NRE，并标明零响应参考；两个任务、两种训练方案及 Test 对照保留在附录。两张训练图共用 `design/training_style.py`：蓝色表示 loss，橙色表示 NRE；圆点／方点表示 Joint／Frozen Encoder，实心／空心 NRE 点分别表示 Training／Test。降幅和 NRE 接近 1 的标注放在对应曲线与端点附近。主图使用真实 PushT 推手位移配对轨迹，展示不同历史在相同当前观测与后续动作下对应不同未来；B 中三个子图分别说明未来辨别、匹配历史的收益和响应保真度。场景图片来自原始观测，潜空间点位与向量为概念示意。主图由 `design/benchmark_overview_pusher_response.pdf` 导出，复绘入口会同步到 `figures/protocol_overview.pdf`。多步预测图展示 Original 与冻结编码器方案的逐步归一化误差。

`scripts/summarize_test_metrics.py` 从 136 次已保存的 Test 原始结果生成三项指标汇总；`generate_test_tables.py` 生成正文与附录的 Test 表格，`generate_results.py` 调用它并生成多步、数据扩量和模型容量的 Test 附录表格。`generate_figures.py` 生成协议示意图、多步预测图、数据扩量与编码器冻结对照图，以及训练损失／条件响应图。它们不训练模型、不重新评测检查点。在仓库根目录运行：

```bash
python scripts/summarize_test_metrics.py
python paper/generate_results.py
python paper/generate_figures.py
```

结果表读取 `docs/research/data/current_reference_test_v1/metrics.json`、`icl_training_study_v2.json`、`multistep_prediction_test_v1/summary.json`、`data_scaling_test_v1/summary.json` 和 `capacity_scaling_test_v1/final_comparison.json`；训练图读取 `native_training_loss_v1/summary.csv` 与 `native_epoch_checkpoints_v1/summary.csv`，附录图的 Test 对照读取 `figure5_checkpoint_test_v1/summary.json`。Figure 3 的 Test 数据扩量端点读取 `figure3_data_scaling_test_v1/metrics.json`，其中 PLDM 2k 联合微调端点来自匹配 32k 配方的补跑，其他 2k 端点使用已有权重补评，32k 端点复用主 Test 归档；编码器冻结对比读取 `current_reference_test_v1/metrics.json`。九任务 Test 主表与附录完整表依据 `current_reference_test_v1/metrics.json`；准确率与原 `summary.json` 保持一致，History 和 NRE 按同一检查点的原始 Test 结果汇总。多步图从 `multistep_prediction_test_v1/summary_queries.json.gz` 复算逐时域误差比：各时域分别汇总预测误差与均值未来参照误差，再在检查点内取比值，最后对训练重复取均值。图中横轴为 rollout 第 1–5 步，每步对应 5 个仿真步。该曲线与先汇总所有时域再取比值的整体 `E/B` 不同。新图使用的数值、运行标识和区间保存于 `figures/experiment_plot_data.json`，可随脚本重新生成。

评分控制使用 `scoring_controls_test_v1/results.json`：在两个 LeWM 检查点的表示中，对同一批 256 对 Test 查询验证构造预测的评分。上述 Test 结果目录保存复现脚本、原始分数或逐查询能量、数据来源和检查点身份；不以 Development 结果替代缺项。

全部图形生成需要 Matplotlib、NumPy，以及用于 SVG 导出的 Node.js、Playwright 和 Chromium；配置见 [图形说明](design/README.md)。编译论文直接使用已保存的 PDF 图，无需执行这些脚本。

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
