# TwoRoom 原始能力重建对照协议

> **文档角色**：支持性实验，仅用于复现基础能力对照。当前数据比较和正式结论
> 统一见[Benchmark 主文档中的速度章节](../ContextWorld_ICL_Benchmark.md#41-速度)。

**版本**：v1.6
**日期**：2026-07-19  
**状态**：已完成；能力重建、固定 speed=5 跨 Eval 与四模型归因均已完成

> **阅读命名**：`Synth5Matched-v2` 是训练数据，不是新任务。本文称其为
> **单速 5 同分布合成数据 v2**；四个模型依次简称为**原始单训**、
> **单速 5 合成单训**、**原始+单速 5 混训**和**原始+多速度混训**。
> 机器 ID 仅用于复现。阶段结论与后续路线图见
> [ContextWorld ICL Benchmark：速度](../ContextWorld_ICL_Benchmark.md#41-速度)。
> 旧结果字段 `correct` 和 `wrong` 分别按同速历史和另一档速度历史解释，不表示
> 速度正确或错误。

## 1. 目标

本协议检验 ContextWorld synthetic 数据能否在 speed=5、任务分布匹配时重建
原始 TwoRoom 的 prediction 与 planning 能力，并区分三个潜在瓶颈：

1. synthetic 轨迹/状态分布本身不足；
2. original 与 synthetic 混合训练产生优化干扰；
3. 多速度 factor diversity 引入容量竞争或长程 rollout 退化。

该实验不直接判定 ICL。原始能力 non-inferiority 是规划评测的必要能力 gate；
速度条件化 ICL 还需检验结果是否随历史速度系统变化；规划校准则进一步要求
同速历史在真实查询动力学下优于慢速和快速历史。

## 2. 模型矩阵

| 中文显示名 | 机器 ID | 训练数据 | Exposure | 状态 |
|---|---|---|---:|---|
| 原始单训 | `H3-OrigHeldout` | 原始 9,000 train episodes | 6,574,080 original draws | 已完成 |
| 单速 5 合成单训 | `H3-Synth5Matched` | 9,000 单速 5 同分布合成 train episodes | 6,574,080 synthetic draws | v2 已完成 |
| 原始+单速 5 混训 | `H3-OrigPlusSynth5` | 原始 + 单速 5 同分布合成数据 v2 | 两组各 6,574,080 draws | 已完成 |
| 原始+多速度混训 | `H3-SpeedFull` | 原始 + 多速度合成数据 | 两组各 6,574,080 draws | 已完成 |

前三个模型使用同一 model seed 3072、history-3 LeWM、global batch 1,024、
optimizer、learning-rate schedule、warmup 比例和 final-fixed-step checkpoint
selection。Original-only 与 Synth5-only 各执行 6,420 optimizer steps；两个
additive models 各执行 12,840 optimizer steps。

历史 legacy H3-Orig 不作为 OrigHeldout 的训练替代，因为其训练使用全部原始
episodes，无法对当前 1,000-episode heldout split 提供严格泛化比较。

## 3. 单速 5 同分布合成数据 v2 协议

机器数据集 ID 为 `tworoom_synth5_matched_v2`。它仍执行原始 TwoRoom
双房间目标导航任务，只把速度固定为 5 并重新生成独立 geometry；不是新的
task 或 eval。

### 3.1 固定条件

- `agent.speed=5.0`，不加入速度混合；
- 10,000 个独立 episode geometries，按固定 seed 划分 9,000 train / 1,000 dev；
- 100% cross-room，与原始 TwoRoom episode 任务语义一致；
- 每个 reset seed group 只生成一个 episode，不做 factor cross；
- 100 raw-step horizon、相同 ExpertPolicy、action noise 和 repeat probability；
- lossless PNG，模型读取 decoded RGB，不读取 factor label；
- 所有 transition 通过 simulator state/pixel/termination 精确重放。

### 3.2 分布匹配

Synth5Matched 根据 original 9,000 train episodes 的统计量冻结生成约束，不复制
heldout states 或 goals。至少匹配：

- start/goal room direction 与 14 px grid occupancy；
- initial-distance distribution；
- model-visible state-goal cell coverage；
- episode length、termination success 与 goal-side row share；
- action norm、action saturation/repeat、collision residual、goal progress 和路径效率。

不得通过 success-only filtering 达到分布匹配。所有数值门槛在正式生成前由
original train partition 计算并写入 synthesis config；heldout partition 不参与
约束拟合。

### 3.3 归一化与采样

四个模型统一使用 original 9,000 train episodes 拟合的 normalization，不从
original heldout、synthetic dev 或 E4 拟合统计量。Original-only 与
Synth5-only 的总 draws 相同；mixed models 使用 50/50 batch composition，但
采用 additive budget，使 original exposure 不因加入 synthetic 而减少。

### 3.4 v1 失败与 v2 执行修订（在 v2 生成前冻结）

`tworoom_synth5_matched_v1` 完成了 10,000 episode 采集和逐轨迹重放，但未通过
预注册分布门，因此不得进入训练。失败项为 initial-distance、episode
length/mean rows、termination success 和 goal-grid TV；其余 controller、action、
collision、progress、path efficiency 与 state-goal coverage gates 均通过。

审计 StableWM 历史提交 `37f34bb` 后确认，original H5 的生成器曾启用
`_constrain_target_by_min_steps`，默认 `min_steps=25`、speed=5，即最短过门路径
至少 125 px；当前 pinned 环境仍保留实现，但 target `constrain_fn` 已被关闭。
同时 original 100-transition episode 保存 101 行，而 v1 只保存最多 100 行。

因此 v2 只做两项实现修复：

- 恢复 speed-independent 的 `minimum_door_path_distance=125 px`；
- 将采集上限改为 101 行，以保留 100 个可训练 transition。

所有 v1 已冻结的 distribution thresholds、original train reference、模型配方和
评测 gate 均不修改；v1 数据和失败报告保留作审计证据，v2 使用全新 seeds 与
全新 artifact 名称。此修订不使用 heldout 数据，也不依据 trajectory success
筛选 episode。

v2 最终包含 10,000 episodes、914,228 rows 和 904,228 transitions；逐帧
pixel bytes、decoded pixels、state、goal 与 termination mismatch 均为 0。
train split 的 9,000 episodes 产生 651,692 个 history-3/frameskip-5 raw clips。
长度、initial distance、start/goal grid、success、action、collision、progress、
path efficiency 和 state-goal coverage 等全部冻结分布门均通过。失败的 v1
没有进入任何训练 loader。

## 4. 评测协议

全部 planning eval 使用 seeds 42–47、每 seed 50 次，即每个 condition 300 次，
并复用 evaluation IDs 和 CEM 子 seed。

| 中文显示名 | 机器旧称 | 目的 | 主要指标 |
|---|---|---|---|
| 原始留出导航 | Original episode-heldout ID | 原始任务泛化 | success、final distance |
| 单速 5 同分布导航 | Frozen speed=5 matched catalog | 同动力学、同任务分布能力 | success、final distance、1/2/3/5-step error |
| 多速度上下文导航（无上下文） | Frozen E4 no-context | 一般多速度 OOD 能力 | success、final distance |
| 多速度历史对照导航 | Frozen E4 `correct/wrong` 字段 | history-to-planning | paired success、distance、sign test |

prediction 另外报告 one-step latent MSE；planning 报告 room/template strata，
不得只使用 pooled success 掩盖某个任务层全部失败。

## 5. 预注册能力 gate

所有 non-inferiority 比较使用相同 evaluation IDs 的 paired bootstrap CI。正式
执行前冻结以下 margin：

- success-rate difference 下界不低于 -5 percentage points；
- mean final-distance difference 上界不高于 +5 px；
- 不允许 OrigHeldout 可解的 room/template stratum 在候选模型上降为 0 success。

`H3-Synth5Matched` 只有同时通过 original heldout ID 和 frozen speed=5 catalog
才视为重建原始能力。只在 synthetic dev 上得分高不构成通过。

## 6. 结果解释规则

| 观察结果 | 正式解释 |
|---|---|
| 单速 5 合成单训未通过，原始单训通过 | synthetic 数据分布/轨迹质量是基础能力瓶颈 |
| 单速 5 合成单训通过，原始+单速 5 混训未通过 | mixed-domain optimization 或 gradient interference |
| 原始+单速 5 混训通过，原始+多速度混训在能力指标下降 | 多速度组成、factor balancing 或容量竞争 |
| 原始+多速度混训能力通过但 E4 同速≈另一档历史 | history-to-rollout/cost/planner 链路瓶颈 |
| 原始+多速度混训能力通过且结果随 context speed 系统变化 | 建立速度条件化 planning ICL |
| 同速历史同时优于慢速和快速历史 | 进一步建立按查询动力学校准的规划收益 |

单一失败模式不能证明更细的机制。例如 Synth5Matched 失败后，需要结合
state-goal coverage、rollout-horizon error 和 action statistics 再区分轨迹覆盖
与训练实现差异。

## 7. 产物命名

- 数据：`tworoom_synth5_matched_v2`（v1 失败产物只作审计）
- 模型：`h3_origheldout_s3072`、`h3_synth5matched_s3072`、
  `h3_origplus_synth5_s3072`
- 统一报告目录：`artifacts/evaluation/history3/original_ability_reconstruction/`
- 汇总：`original_ability_reconstruction_n50x6.json`

## 8. 正式执行结果

### 8.1 训练审计

| 中文显示名 | 机器 ID | Optimizer steps | Final checkpoint SHA-256 | 精确重载 |
|---|---|---:|---|---|
| 原始单训 | `H3-OrigHeldout` | 6,420 | `7d141b86...e962b54` | 通过 |
| 单速 5 合成单训 | `H3-Synth5Matched` | 6,420 | `565de1fe...c36283` | 通过 |
| 原始+单速 5 混训 | `H3-OrigPlusSynth5` | 12,840 | `79e1c63a...1b3474` | 通过 |
| 原始+多速度混训 | `H3-SpeedFull` | 12,840 | `79e2b2d1...8a778` | 通过 |

`H3-Synth5Matched` 的训练组只有 `synth5_matched`；original H5 只提供四模型
共用的冻结 normalization statistics，不提供训练 clip。`H3-OrigPlusSynth5`
对 original 与 synthetic 各执行 6,574,080 draws，两组 raw clips 均完整
exposure。

### 8.2 两域 planning

每个单元为 `success/300；mean final distance (px)`。

| 训练模型 | 原始留出导航 | 单速 5 同分布导航 |
|---|---:|---:|
| 原始单训 | 273/300（91.00%）；19.91 | 281/300（93.67%）；18.31 |
| 单速 5 合成单训 | 275/300（91.67%）；19.82 | 282/300（94.00%）；17.84 |
| 原始+单速 5 混训 | 263/300（87.67%）；23.43 | 274/300（91.33%）；19.03 |
| 原始+多速度混训 | 289/300（96.33%）；16.89 | 285/300（95.00%）；16.59 |

### 8.3 预注册 non-inferiority 判定

下列差值均为 candidate − `H3-OrigHeldout`，括号为 paired-bootstrap 95% CI。

| 候选模型 | Eval | Success-rate difference | Final-distance difference | 判定 |
|---|---|---:|---:|---|
| 单速 5 合成单训 | 原始留出导航 | +0.67 pp `[-1.33,+2.67]` | -0.08 `[-1.80,+1.61]` | 通过 |
| 单速 5 合成单训 | 单速 5 同分布导航 | +0.33 pp `[-2.00,+3.00]` | -0.48 `[-2.42,+1.32]` | 通过 |
| 原始+单速 5 混训 | 原始留出导航 | -3.33 pp `[-6.33,-0.33]` | +3.53 `[+1.12,+6.08]` | 失败 |
| 原始+单速 5 混训 | 单速 5 同分布导航 | -2.33 pp `[-5.67,+0.67]` | +0.71 `[-1.50,+2.89]` | success gate 失败 |
| 原始+多速度混训 | 原始留出导航 | +5.33 pp `[+2.67,+8.33]` | -3.01 `[-4.99,-1.12]` | 通过 |
| 原始+多速度混训 | 单速 5 同分布导航 | +1.33 pp `[-1.33,+4.00]` | -1.72 `[-3.59,+0.04]` | 通过 |

所有比较均无可解 stratum 降为 0 success。正式机器可读判定为：

- `synthetic_only_reconstructs_original_ability=true`；
- `mixed_training_preserves_original_ability=false`。

### 8.4 Rollout 与 context 诊断

native-latent RMSE 的 `1/2/3/5-step` 绝对值如下：

| 训练模型 | 原始留出导航 | 单速 5 同分布导航 |
|---|---|---|
| 原始单训 | .104/.130/.155/.196 | .105/.139/.166/.194 |
| 单速 5 合成单训 | .128/.155/.178/.214 | .130/.161/.188/.210 |
| 原始+单速 5 混训 | .116/.139/.161/.194 | .114/.141/.159/.173 |
| 原始+多速度混训 | .108/.135/.163/.214 | .109/.137/.162/.201 |

因此 synthetic-only 虽通过 planning non-inferiority，其跨模型 latent rollout
统计并非完全等同于 original-only；两个域的 1/2/3/5-step RMSE 均更高。不能把
“重建规划能力”扩写为“所有 representation/prediction 指标等价”。

E1 K=2 prediction gain 分别为 OrigHeldout `0.00229`、Synth5Matched
`0.00145`、OrigPlusSynth5 `0.00156`、SpeedFull `0.03523`。四模型的 E4
无历史/同速历史/另一档历史 success 均为 `73/300`，后两者均无 discordant
success，paired sign-test `p=1.0`。

### 8.5 正式结论

1. **speed=5 matched synthetic 数据单独训练足以重建预注册的原始规划能力。**
   因此不支持把先前的绝对能力差距正式归因为 synthetic 轨迹/状态分布质量。
2. **本次 fixed-speed mixed recipe 未建立能力保持。** 结果支持
   mixture-specific optimization 或 gradient interference，但单个训练 seed
   不能证明所有混合训练都会退化；无 stratum collapse，点估计退化也小于
   margin，失败来自 paired CI 越界。
3. **多速度竞争不是本次能力差距的解释。** `H3-SpeedFull` 在公平的
   OrigHeldout 两域上均通过，并在 original heldout 上显著更高；旧 legacy
   `H3-Orig` 的优势不应继续当作 episode-heldout 能力证据。
4. **旧 E4 没有建立规划层历史效应。** SpeedFull 虽有强 prediction-level
   history effect，但旧 E4 的同速/另一档历史 success 完全一致。规划层 ICL
   需要由独立的双向上下文协议判定，不属于本能力重建协议的结论范围。

### 8.6 固定 speed=5 跨 Eval 补充

为检验 E4 低分是否主要由八个速度混合造成，执行前另行冻结
[speed=5 跨 Eval 协议](TwoRoom_Speed5_CrossEval_Protocol.md)。前三个模型的
训练速度均为 5，三套 Eval 的 query 速度也统一为 5；SpeedFull 只作控制。

| 训练模型 | 原始 future-25 | 合成速度 5 future-25 | E4-speed5 无/同速/快速历史 |
|---|---:|---:|---:|
| 原始单训 | 91.00% | 93.67% | 25.00% / 25.00% / 25.00% |
| 单速 5 合成单训 | 91.67% | 94.00% | 25.00% / 25.00% / 25.00% |
| 原始+单速 5 混训 | 87.67% | 91.33% | 25.00% / 25.00% / 25.00% |
| 原始+多速度混训 | 96.33% | 95.00% | 25.00% / 25.00% / 25.00% |

三项预注册归因得到：

- 原始与合成 future-25 的最大模型内差为 3.67 pp，轨迹来源成分次要；
- future-25 到 E4-speed5 的最小下降仍为 62.67 pp，固定几何/规划成分主要；
- speed5 E4 相对多速度 E4 只高 0.67 pp，且完全来自 s3 抽样数
  `75 vs 73`；按模板条件化后速度成分为 0。

E4-speed5 的 s0/s1/s2 在四模型与三种历史条件下均为零成功，s3 均为全成功。
所以 fixed-speed 结果进一步排除了“多速度混合压低 pooled score”这一解释，
并把当前根因收窄到 E4 固定远目标的任务几何与 planner 链路。它不改变
8.5 的训练数据结论，也不能单独区分 cost、CEM search、动作边界和 rollout
误差各自的因果份额。

原始/matched future-25 的平均距离 42.87/41.43 px 在数值上接近 E4-s3 的
45.18 px，但它们分别仍包含 23.67%/26.33% 跨房间 query，且目标来自同一
真实轨迹 25 步后的已实现状态。故能力重建结论针对原始分布匹配的局部可达
planning，不把原始/matched Eval 等同为固定 same-room s3，也不把 E4 失败
归因成“synthetic Eval 整体距离分布有问题”。
