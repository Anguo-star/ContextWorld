# TwoRoom 速度上下文学习 Benchmark 报告

**版本**：v5.2  
**日期**：2026-07-20  
**证据级别**：Validation 机制验证；单训练种子；正式 Test 尚未启用

## 摘要

本报告研究 History-3 LeWM 能否从两段历史交互中识别隐藏速度，并在不更新权重
的情况下，把识别结果用于多步预测和 CEM 闭环规划。

四模型双向上下文实验已经证明：只有原始数据加多速度合成数据训练的
`H3-SpeedFull` 稳定出现 Fast 大于 Slow 的规划效应，50 步成功率差为
`14.00` 个百分点，六个评测种子方向一致。三个原始或单速控制模型均未出现
同等效应。

机制验证把“预测是否准确”和“有限 CEM 是否成功”拆开。主要结果如下：

1. 在固定动作和真实 query-speed 轨迹上，SpeedFull 的 Correct 上下文具有最低
   的规划 horizon 平均预测 MSE；单速混训控制没有这一模式。
2. 精确动力学 oracle 在 Correct 速度下实现 100% 的 horizon 末端成功，证明
   正确速度在物理和评测定义上有效。
3. 固定 300 条候选动作时，SpeedFull 的 Correct 与 Fast 真实末端结果基本相当；
   Fast 的优势主要在自由 CEM 搜索和闭环执行后出现。
4. 把真实执行预算从 50 步增加到 100 步后，Fast−Correct 从 `+6.33` 降至
   `+1.00` 个百分点并失去显著性，说明有限截止是 Fast 高分的重要来源。
5. 在 50 步内两者都成功的任务上，Fast 平均反而多用 `1.40` 个真实步。因此，
   Fast 高分不等于真实到达更快。

阶段结论是：SpeedFull 已形成速度条件化的预测和规划能力；正确速度的预测总体
更准确，而错误快速上下文在有限规划与执行资源下具有截止收益。预测准确率、
截止前成功率和真实执行效率必须分别报告。

因此，Speed Benchmark 的正式结果不采用单一 Eval score，而是固定报告速度
可辨识性、预测校准、候选动作、执行预算曲线、连续轨迹、真实效率和能力保持。

本文是 TwoRoom 速度轨道的唯一当前结果报告。数据生成与历史实验协议保留在
`protocols/`，旧版阶段总结保留在 `archive/`，但不再作为当前结论入口。

## 1. 研究问题

### 1.1 模型看到了什么

History-3 输入由两段历史转移和一个当前查询组成：

```text
历史 1：observation → action → next observation
历史 2：observation → action → next observation
查询：当前 observation + 待评估 action sequence
```

模型不接收速度标签。它只能根据历史动作和画面位移推断隐藏速度。CEM 使用模型
预测候选动作的结果，执行选中的动作后再根据新画面重新规划。

### 1.2 Speed Benchmark 的判断层级

| 层级 | 要回答的问题 | 主要证据 |
|---|---|---|
| 速度可辨识性 | 两段历史是否提供足够速度信息 | RGB+action 速度恢复 |
| 速度条件化 | 模型是否随上下文速度稳定变化 | 双向上下文效应、控制模型 |
| 预测校准 | Correct 是否更接近真实 query-speed 轨迹 | 固定动作 rollout MSE |
| 候选动作 | 上下文如何改变规划选择 | cost rank、top-k、argmin |
| 预算曲线 | 高成功率是否依赖执行截止 | Slow/Correct/Fast 的 `S(B)` |
| 连续轨迹 | 未成功时是否仍接近目标 | final/best distance、AUC |
| 真实效率 | 同一成功任务是否更快、更省 | 配对 steps、path efficiency |
| 能力保持 | 新训练是否损伤原能力 | 原始 ID/OOD Eval |

“Fast 分数更高”只能直接说明它在当前规划协议下更容易过成功线，不能单独证明
预测更准或真实到达更快。

## 2. 数据与模型

### 2.1 数据

| 数据 | 规模与内容 | 用途 |
|---|---|---|
| 原始 TwoRoom | 9,000 train / 1,000 episode-heldout，速度为 5 | 原始能力基线 |
| 速度 5 同分布合成 v2 | 10,000 episodes，904,228 transitions | 合成能力重建与单速控制 |
| SpeedFull 多速度数据 | 16,384 episodes，多速度和多几何分层 | 学习隐藏速度变化 |

速度 5 合成数据的状态、目标、像素和终止状态均通过模拟器回放，mismatch 为 0。
所有模型共享从原始训练 split 计算的冻结 normalizer。

### 2.2 模型矩阵

| 本文简称 | 实验标识 | 训练数据 | 优化步数 | 实验作用 |
|---|---|---|---:|---|
| 原始单训 | `H3-OrigHeldout` | 原始数据 | 6,420 | 原始 LeWM 基线 |
| 单速合成单训 | `H3-Synth5Matched` | 仅速度 5 合成数据 | 6,420 | 合成数据来源控制 |
| 单速混训 | `H3-OrigPlusSynth5` | 原始 + 速度 5 合成 | 12,840 | 单速混训控制 |
| 多速度混训 | `H3-SpeedFull` | 原始 + 多速度合成 | 12,840 | 多速度训练目标 |

`H3-Synth5Matched` 不是一个新任务，而是“只用我们合成的速度 5 数据训练”；
`H3-SpeedFull` 也不是纯合成单训，它使用原始数据和多速度合成数据混合训练。
正文优先使用中文简称，实验标识只用于对应配置、checkpoint 和产物路径。

四模型共享 History-3、action-block-5、模型种子 3072、normalizer 和 checkpoint
选择规则。

`H3-OrigPlusSynth5` 是 SpeedFull 的主要控制，但两套合成数据的 geometry、
分层和 loader 并非完全相同。因此，训练层结论归属于当前完整多速度配方，不能
写成只改变速度字段的单变量因果效应。

## 3. 评测协议

### 3.1 双向速度上下文

查询环境真实速度为 5.0 或 5.1：

```text
Slow：上下文速度 3.1
Correct：上下文速度与查询一致
Fast：上下文速度 7.0
```

Slow 和 Fast 两个 Eval 共享 query、goal、geometry、正确上下文、模拟器种子和
CEM 随机计划，只改变错误上下文内容。

每个 Eval、每个条件均独立执行：

```text
50 次 × 6 个评测种子 = 300 次
```

相同的 Correct 条件在两个 Eval 中也分别完整运行，不能把多个任务合计成 300。

### 3.2 冻结的 Planner Profile 与执行预算

CEM 搜索配置在三个执行预算间保持不变：

| 参数 | 数值 |
|---|---:|
| Action block | 5 个原始步 |
| Model rollout horizon | 5 个 action blocks，即 25 个原始步 |
| Receding horizon | 5 个 action blocks |
| CEM samples / iterations / top-k | 300 / 30 / 30 |
| 成功半径 | 16 px |

TwoRoom 当前 Validation 的执行预算阶梯为：

| 预算档 | 原始步数 | 作用 |
|---|---:|---|
| Tight | 50 | 检查紧截止下的上下文响应 |
| Standard | 75 | 观察中等资源下的成功变化 |
| Relaxed | 100 | 检查放宽截止后差异是否保留 |

每个预算、每个 Eval、每个条件均有完整 `50×6=300` 个配对观测。短预算轨迹必须
是长预算轨迹的严格前缀。50/75/100 是当前 Validation 的冻结实现，不自动成为
其他任务或正式 Test 的通用预算。

四模型归因使用 Tight 预算，每个模型执行 1,200 条闭环记录，合计 4,800 条。
机制验证同时报告 SpeedFull 的 Tight/Standard/Relaxed 曲线。

### 3.3 固定候选与预测校准

机制协议在出分前冻结于
`configs/benchmark/tworoom_planner_mechanism_attribution_v1.yaml`。

#### 固定候选动作

每个 query 生成 300 条、每条 25 个原始步的动作序列：

- 239 条冻结随机候选；
- 60 条朝向目标、角度和幅度分层的候选；
- 1 条全零候选。

同一候选 bank 在 Slow、Correct、Fast 和两个模型间完全共享。记录每条候选的
模型代价、排序、top-30、选中候选和真实动力学结果。

候选有效性预检要求 Correct-speed 精确动力学能够稳定到达目标。纯随机候选未
通过该门，因此在任何正式模型结果生成前替换为上述混合候选；预注册配置保留了
修订原因。

#### 真实 rollout 预测误差

精确 Correct-speed oracle 在候选 bank 中选择一条与任务相关的 probe action。
真实模拟器以 query speed 执行该动作，保存第 5、10、15、20、25 个原始步的
真实帧。两个模型分别在三种上下文下预测同一动作，并在各自原生 latent 空间中
计算 MSE。

不同 checkpoint 的 latent 尺度不可直接比较，因此只做同一模型内的配对比较。

### 3.4 计数与完整性审计

| 审计项 | 结果 |
|---|---:|
| 固定候选文件 | 12 |
| 固定候选 query 记录 | 600 |
| 固定候选上下文评估 | 1,800 |
| 闭环文件 | 48 |
| 闭环原始记录 | 4,800 |
| 去除两个 Eval 的重复 Correct 后的条件记录 | 3,600 |
| 50/75/100 轨迹前缀一致性检查 | 2,700/2,700 通过 |

Checkpoint、normalizer、catalog、StableWorldModel commit、随机 schedule 和
candidate bank 哈希全部匹配。所有主条件都是独立 `50×6`，没有多任务均分。

### 3.5 Horizon 记录说明

预注册的宽泛 horizon 列表曾写入原始步 1/2/3，但该模型每 5 个原始动作输出一个
预测帧，实际可观测点为 5/10/15/20/25。预注册主分析要求的
5/10/15/25 全部存在，第 20 步作为描述指标。该记录差异不影响主分析，但在正式
Test 配置中应直接按 action block 表达，避免再次混淆。

## 4. 既有四模型结果

### 4.1 基础能力

| 模型 | 原始 episode-heldout | 速度 5 matched |
|---|---:|---:|
| 原始单训 | 91.00% | 93.67% |
| 速度 5 合成单训 | 91.67% | 94.00% |
| 单速混训 | 87.67% | 91.33% |
| 多速度混训 | 96.33% | 95.00% |

速度 5 合成单训在两个评测域均通过非劣效判定，说明合成数据能够独立重建基础
规划能力。

### 4.2 50 步双向规划

| 模型 | Slow | Correct | Fast | Fast−Slow | 配对 p |
|---|---:|---:|---:|---:|---:|
| 原始单训 | 51.67% | 52.00% | 52.00% | +0.33 pp | 1.0 |
| 速度 5 合成单训 | 43.33% | 43.67% | 44.67% | +1.33 pp | 0.125 |
| 单速混训 | 32.67% | 32.67% | 31.67% | −1.00 pp | 0.375 |
| 多速度混训 | 50.67% | 58.33% | **64.67%** | **+14.00 pp** | **5.12e-12** |

SpeedFull 的 Fast-only/Slow-only 为 43/1，六个种子的 Fast−Slow 为
`+10/+14/+20/+12/+14/+14 pp`。三个控制模型没有通过冻结的 5 个百分点效应门。

这建立了规划层速度条件化，但尚未解释 Fast 为什么高于 Correct。

## 5. 机制结果

### 5.1 多步预测校准

SpeedFull 的各 horizon 平均 MSE 如下，数值越低越好：

| 原始步 | Slow | Correct | Fast |
|---:|---:|---:|---:|
| 5 | 0.02691 | **0.01027** | 0.01395 |
| 10 | 0.08058 | **0.02640** | 0.04038 |
| 15 | 0.13180 | **0.03907** | 0.06061 |
| 20 | 0.16768 | **0.04898** | 0.06881 |
| 25 | 0.19957 | **0.06538** | 0.07369 |

按预注册主 horizon 5/10/15/25 先在每个 query 内平均：

| 模型 | Slow | Correct | Fast | Correct−Fast 95% 区间 |
|---|---:|---:|---:|---:|
| SpeedFull | 0.10972 | **0.03528** | 0.04716 | `[-0.01760,-0.00630]` |
| 单速混训控制 | 0.05420 | 0.05247 | **0.05078** | `[+0.00151,+0.00189]` |

SpeedFull 的 Correct 在整段 horizon 上具有更低的平均预测误差；单速控制没有
匹配速度最优的模式。这是预测层速度 ICL 的直接证据。

末端第 25 步需要单独限定：Correct 的 pooled mean 较低，但 Correct−Fast 的
bootstrap 区间为 `[-0.02292,+0.00543]`，且 Correct/Fast 更低误差的 query 数为
125/175。不能声称 Correct 在每个 query 的末端都更准确。

### 5.2 精确动力学正对照

| Oracle 假设速度 | 真实 query-speed 第 25 步平均距离 | 曾进入成功半径 | 第 25 步仍成功 |
|---|---:|---:|---:|
| Slow | 23.90 px | 100.00% | 34.33% |
| Correct | **2.85 px** | **100.00%** | **100.00%** |
| Fast | 24.17 px | 6.00% | 6.00% |

Correct 对 300/300 个配对的真实末端距离都不劣于 Fast，对 Slow 为 291 个更优、
9 个并列。候选 bank 的 Correct 末端可达率为 100%，正对照通过。

Slow 的“曾成功”与“末端成功”分离是因为固定动力学 rollout 不在首次进入成功
半径时停止。较慢假设会选择较大的动作；在真实较快速度执行时，轨迹可能穿过
目标后继续前进。

### 5.3 固定候选的模型代价

#### SpeedFull

| 上下文 | 选中候选真实末端距离 | 曾成功 | 末端成功 |
|---|---:|---:|---:|
| Slow | 6.64 px | 100.00% | 97.00% |
| Correct | **4.81 px** | 100.00% | 100.00% |
| Fast | 5.07 px | 100.00% | 100.00% |

Correct−Fast 真实末端距离为 `−0.26 px`，95% 区间 `[-0.72,+0.20]`。固定候选
下两者没有稳定差异。

| 排名比较 | 完整排序 Spearman | top-30 重合率 | argmin 相同率 |
|---|---:|---:|---:|
| Slow−Correct | 0.990 | 94.28% | 71.33% |
| Correct−Fast | 0.987 | 92.72% | 28.00% |
| Slow−Fast | 0.961 | — | 22.00% |

整体排序非常接近，但顶部候选的细小代价变化足以频繁改变 argmin。

#### 单速混训控制

控制模型三种上下文的排序 Spearman 约为 `0.9998–0.9999`，Slow 与 Correct 的
argmin 100% 相同，Correct 与 Fast 为 94.33% 相同。它几乎不根据速度上下文
改变候选排序，与 SpeedFull 形成清楚对照。

### 5.4 执行预算曲线

| 真实执行预算 | Slow | Correct | Fast | Fast−Correct | Fast−Slow |
|---:|---:|---:|---:|---:|---:|
| 50 | 50.67% | 58.33% | **64.67%** | +6.33 pp | +14.00 pp |
| 75 | 60.00% | 68.00% | **70.00%** | +2.00 pp | +10.00 pp |
| 100 | 63.00% | 70.67% | 71.67% | +1.00 pp | +8.67 pp |

Fast−Correct：

- 50 步：Fast-only/Correct-only 为 21/2，`p=6.60e-5`；
- 75 步：10/4，`p=0.180`；
- 100 步：7/4，`p=0.549`。

从 50 到 100 步，Fast−Correct 缩小 `5.33 pp`，配对 bootstrap 95% 区间为
`[-8.33,-2.33]`。有限真实执行预算对 Fast 优势有实质贡献。

Fast−Slow 从 `14.00 pp` 降到 `8.67 pp`，变化区间同样为
`[-8.33,-2.33]`，但 100 步差异仍显著，`p=2.16e-7`。因此真实执行预算不是
Fast−Slow 的完整解释。

### 5.5 连续轨迹指标

| 预算 | 条件 | 平均最终距离 | 平均归一化进度 | 平均归一化距离 AUC |
|---:|---|---:|---:|---:|
| 50 | Slow | 55.04 px | 0.387 | 0.834 |
| 50 | Correct | 46.53 px | 0.491 | 0.771 |
| 50 | Fast | **43.43 px** | **0.525** | **0.730** |
| 100 | Slow | 50.69 px | 0.434 | 0.827 |
| 100 | Correct | 42.91 px | 0.530 | 0.761 |
| 100 | Fast | **40.64 px** | **0.556** | **0.726** |

AUC 越低表示整段轨迹总体离目标更近。连续指标和成功率方向一致，但它们仍是
规划器—模型联合指标，不是纯预测精度。

### 5.6 真实到达步数

50 步时，173 个 Fast 与 Correct 都成功的配对任务中：

```text
Fast steps-to-success − Correct steps-to-success = +1.40 步
bootstrap 95% CI = [0.49, 2.30]
```

Fast 没有让共同成功的任务更快完成。它的成功率更高来自成功集合变化：一些更难
的任务在 Fast 下赶在截止前成功。直接比较各条件“成功样本的平均步数”也会受到
成功集合不同的选择偏差，因此主报告采用共同成功配对。

## 6. 机制解释

现有证据支持以下链路：

```text
速度上下文
  → SpeedFull 的预测位移和 rollout error 改变
  → 顶部候选代价出现小幅重排
  → 自由 CEM 的采样分布、argmin 和滚动重规划改变
  → 有限执行截止前的成功集合改变
```

Fast 上下文不会提高真实模拟器速度。它让模型内部认为相同动作能走得更远，因此
更多候选可能在固定 25 步 model horizon 内看起来接近目标。这个变化可以促使
CEM 选择更激进或不同的动作，并在 50 步真实执行截止内增加成功数。

同时，Correct 在固定真实 rollout 上具有更低的整段平均预测误差，精确动力学
oracle 也明确偏好 Correct。两者共同证明 Fast 的高 endpoint score 不是
“Fast 更符合真实动力学”。

预算曲线显示 Correct 在宽松预算下基本追平 Fast；Fast−Slow 仍残留，说明
固定 25 步 model horizon、CEM candidate 数或重规划过程还可能放大慢速上下文
的可达性偏差。

## 7. 结论边界

### 7.1 已经建立

1. 速度 5 合成数据可以重建基础规划能力。
2. SpeedFull 形成了预测层和规划层的速度条件化上下文学习。
3. 四模型控制支持当前完整多速度训练配方的区分作用。
4. Correct 在规划 horizon 上的平均预测误差低于 Slow 和 Fast。
5. Fast 的 50 步 endpoint 优势包含显著的有限执行截止成分。
6. Endpoint success、prediction error 和 steps-to-success 必须分开评价。
7. 多执行预算曲线是 Speed Benchmark 的核心结果层。

### 7.2 尚未建立

1. Fast−Slow 的剩余效应由 horizon、采样预算或重规划中的哪一项主导。
2. Correct 在所有资源配置和每个 query 上都优于双向错误上下文。
3. 当前训练配方归因可以跨训练种子复现。
4. 速度支持是两个训练配方间唯一变化的单变量因果结论。
5. 正式 Test 或跨任务外推。

## 8. 下一阶段

Speed Benchmark 的结果层级已经确定。下一优先级是 CEM 规划配置影响实验，用于
判断 Relaxed 执行预算下仍存在的 Fast−Slow，主要来自规划视野不足还是搜索不
充分：

1. 先验证模型和显存支持更长 rollout horizon；
2. 在未评分配置上冻结 horizon 与 CEM sample/iteration 的单因素对照；
3. 保持 query、执行预算、随机计划、cost 和成功半径不变；
4. 每个 Eval、每个条件继续独立执行 `50×6=300`；
5. 判断 100 步下剩余 Fast−Slow 差异随 horizon 还是搜索预算缩小。

机制闭合后，增加多个 `H3-SpeedFull` 与 `H3-OrigPlusSynth5` 训练种子，匹配
geometry、数据曝光量和 loader。正式 Test 的 Tight/Standard/Relaxed 预算只能
在新 Calibration split 上确定，随后与模型、planner 和统计方法一并冻结。

## 9. 复现信息

逻辑路径 `artifacts/...` 默认映射到
`/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/`。

- 机制验证预注册：
  `configs/benchmark/tworoom_planner_mechanism_attribution_v1.yaml`
  （SHA-256 `eb9141b8df30caad3b9403a8873c4063673a7f5c5c2f90bc329e672a58c5b190`）
- 机制验证汇总：
  `artifacts/evaluation/history3/planner_mechanism_v1/final_summary_n50x6.json`
  （SHA-256 `f3daeb2c5c50386634c1d966a6ee3d20a62de6021856861a7e1e6abdd0ec4663`）
- 固定候选执行：
  `scripts/run_tworoom_fixed_candidate_mechanism.sh`
- 闭环预算执行：
  `scripts/run_tworoom_planner_mechanism_eval.sh`
- 审计与汇总：
  `scripts/analyze_tworoom_planner_mechanism.py`
- 四模型归因汇总：
  `artifacts/evaluation/history3/icl_sensitive_v2_directional/four_model_attribution_summary_n50x6.json`
