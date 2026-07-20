# ContextWorld TwoRoom 速度轨道项目状态与路线图（已归档）

> 本文保留 2026-07-20 文档合并前的状态快照，不再更新。当前进度、结论和后续
> 工作统一维护在
> [TwoRoom 速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

**版本**：v5.1  
**日期**：2026-07-20  
**当前阶段**：Validation 机制验证完成

## 1. 当前状态

TwoRoom 速度轨道已经完成数据合成、四模型训练、基础能力评测、速度上下文规划
归因，以及预测—候选排序—闭环结果的机制验证。

当前阶段可以正式表述为：

1. 速度 5 合成数据能够独立重建原始 TwoRoom 的基础规划能力。
2. 只有多速度混训的 SpeedFull 形成稳定的规划层速度上下文效应。
3. SpeedFull 的 Correct 上下文在规划 horizon 上具有最低的平均预测误差。
4. Fast 在 50 步 CEM 中的高成功率包含明显的有限执行截止效应。
5. 增加到 100 步后 Correct 基本追平 Fast，但 Fast−Slow 仍然存在。
6. 当前结论属于单训练种子的 Validation 机制证据，不是正式 Test。
7. Speed Benchmark 的结果结构已经确定：预测、候选排序、三档执行预算、连续
   轨迹、真实效率和能力保持必须联合报告。

当前只冻结了这套结果结构。正式 Test 的 Tight/Standard/Relaxed 具体步数仍需在
新 Calibration split 上确定，不能直接沿用已经看过结果的 Validation 配置。

完整数据和解释见
[TwoRoom 速度上下文学习阶段报告](TwoRoom_Task_Naming_and_Eval_Comparison.md)。

## 2. 已完成的交付

| 模块 | 已完成内容 | 当前判断 |
|---|---|---|
| 数据与划分 | 原始 episode-heldout、速度 5 合成 v2、多速度数据、冻结 normalizer | 回放与完整性通过 |
| 模型训练 | 原始单训、合成单训、单速混训、多速度混训 | 四个 checkpoint 固定 |
| 基础能力 | 原始 heldout、速度 5 matched、非劣效、rollout retention | 合成单训可重建能力 |
| 上下文信息 | K=2 可辨识性、一步与多步预测 | SpeedFull 预测层 ICL 已建立 |
| Eval 有效性 | 旧 E4 模板分解、距离和 geometry 校准 | 旧 E4 缺少上下文敏感度 |
| 四模型归因 | 每模型 1,200 条 directional v2，合计 4,800 条 | 仅 SpeedFull 通过冻结门 |
| 固定候选诊断 | 2 模型、600 query、1,800 次上下文评估 | SpeedFull 会改变顶部候选排序 |
| 精确动力学正对照 | Slow/Correct/Fast 同候选 bank | Correct 的物理末端结果最优 |
| 闭环轨迹 | 逐步状态、动作、距离、到达步数、AUC、路径效率 | 不再用重规划次数代替到达时间 |
| Speed 预算曲线 | SpeedFull 的 50/75/100 步完整 `50×6` | 已升级为 Speed 核心评测层 |
| 自动审计 | 48 闭环文件、4,800 条、2,700 个轨迹前缀检查 | 全部通过 |

## 3. 关键阶段数据

### 3.1 四模型 50 步规划

| 模型 | Slow | Correct | Fast | Fast−Slow |
|---|---:|---:|---:|---:|
| 原始单训 | 51.67% | 52.00% | 52.00% | +0.33 pp |
| 速度 5 合成单训 | 43.33% | 43.67% | 44.67% | +1.33 pp |
| 单速混训 | 32.67% | 32.67% | 31.67% | −1.00 pp |
| 多速度混训 | 50.67% | 58.33% | **64.67%** | **+14.00 pp** |

### 3.2 SpeedFull 预测误差

5/10/15/25 步的 query 内平均 latent MSE：

| Slow | Correct | Fast |
|---:|---:|---:|
| 0.10972 | **0.03528** | 0.04716 |

Correct−Fast 的配对 bootstrap 95% 区间为 `[-0.01760,-0.00630]`。Correct 的
整段预测总体更准。

### 3.3 SpeedFull 执行预算曲线

| Validation 预算档 | 原始步数 | Slow | Correct | Fast | Fast−Correct | Fast−Slow |
|---|---:|---:|---:|---:|---:|---:|
| Tight | 50 | 50.67% | 58.33% | 64.67% | +6.33 pp | +14.00 pp |
| Standard | 75 | 60.00% | 68.00% | 70.00% | +2.00 pp | +10.00 pp |
| Relaxed | 100 | 63.00% | 70.67% | 71.67% | +1.00 pp | +8.67 pp |

Fast−Correct 从显著差异降到 `p=0.549`。有限执行截止是 Fast 高分的重要来源，
但不能解释 100 步时仍存在的 Fast−Slow。

## 4. 已解决与未解决

### 4.1 已解决

- “模型是否读取速度上下文”：是，且只有 SpeedFull 在规划层稳定出现。
- “Fast 高分是否代表预测更准确”：否，Correct 的整段平均预测误差更低。
- “Fast 高分是否主要受有限执行步数影响”：是，Fast−Correct 的大部分优势随
  预算增加而消失。
- “Fast 是否让共同成功任务真实到达更快”：否；50 步共同成功集上 Fast 平均
  多用 1.40 步。
- “旧 E4 没差是否说明没有 ICL”：否；旧 E4 被任务难度地板和天花板锁定。
- “Speed 是否应只看一个执行预算”：否；多预算曲线已经成为核心评测层。

### 4.2 尚未解决

1. 100 步下剩余 Fast−Slow 由 model horizon、CEM 搜索预算还是重规划主导。
2. 当前配方效应是否跨训练种子稳定。
3. 当 geometry、loader 和有效曝光量完全匹配时，速度多样性本身的独立贡献。
4. 全新 Test catalog 上的一次性结果。

## 5. 后续工作顺序

### 5.1 下一步：CEM 规划配置影响实验

**目标**：查清 100 步时剩余的 Fast−Slow，主要是因为模型规划时看得不够远，
还是因为 CEM 候选搜得不够充分。

**变量**：

- model rollout horizon；
- CEM candidate 数；
- CEM optimization iterations。

**固定项**：

- query、goal、catalog 和上下文；
- checkpoint、normalizer 和成功半径；
- 真实执行预算 100；
- action block、随机计划和配对方式。

**执行要求**：

1. 先做不读正式分数的显存、速度和正对照预检；
2. 再冻结最小单因素矩阵与主判定；
3. 每个 Eval、每个条件独立 `50×6=300`；
4. 同时报告 prediction、rank、success、AUC 和 paired steps；
5. 不把任何有限配置称为无限 CEM。

**验收问题**：

- 延长 horizon 是否优先消除 Slow 的可达性劣势？
- 增加 samples 或 iterations 是否消除自由搜索放大？
- 哪个资源变化最先让 Correct 同时不劣于 Slow 和 Fast？

### 5.2 后续：多训练种子与单变量复现

在 planner 协议冻结后：

- 训练多个 SpeedFull 和单速混训种子；
- 匹配优化步数、原始/合成曝光量、geometry、分层和 loader；
- 预注册训练种子层级统计；
- 不按 Validation 最优种子选模型。

### 5.3 最后：正式 Benchmark Test

启动条件：

- CEM 规划配置影响实验和多训练种子复现完成；
- 原始能力保持门通过；
- 新 Test catalog 未评分；
- 模型、planner、指标、阈值和停止规则全部冻结。

Test 每个 Eval、每个条件仍为独立 `50×6=300`，只运行一次并完整发布。
三档执行预算中的每个预算点都有完整 300 个配对观测，不能把 300 次均分给三档。

## 6. 当前正式产物

- 阶段报告：
  [TwoRoom_Task_Naming_and_Eval_Comparison.md](TwoRoom_Task_Naming_and_Eval_Comparison.md)
- 技术报告：
  [当前 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)
- Benchmark 设计规范：
  [ContextWorld_Benchmark_Design.md](../ContextWorld_Benchmark_Design.md)
- 机制验证预注册：
  `configs/benchmark/tworoom_planner_mechanism_attribution_v1.yaml`
- 机制验证统一汇总：
  `artifacts/evaluation/history3/planner_mechanism_v1/final_summary_n50x6.json`
- 四模型统一结果：
  `artifacts/evaluation/history3/icl_sensitive_v2_directional/four_model_attribution_summary_n50x6.json`
- 固定候选执行：
  `scripts/run_tworoom_fixed_candidate_mechanism.sh`
- 闭环预算执行：
  `scripts/run_tworoom_planner_mechanism_eval.sh`
- 审计与汇总：
  `scripts/analyze_tworoom_planner_mechanism.py`
