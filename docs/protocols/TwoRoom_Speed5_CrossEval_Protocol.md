# TwoRoom speed=5 跨 Eval 对照协议

> **文档角色**：支持性实验，仅用于复现 Eval 分布诊断。当前数据比较和正式结论
> 统一见[主报告](../TwoRoom_Speed_Benchmark_Report.md)。

**版本**：v1.4
**日期**：2026-07-19  
**状态**：已完成；协议与阈值在执行前冻结

> 当前阶段结论统一见
> [TwoRoom 速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)；
> 本文只保留本次 speed=5 归因的预注册规则、结果与限制。

本协议中的 speed 是 `agent.speed`，不是 frameskip。旧机器字段 `correct` 和
`wrong` 分别表示同速历史和快速历史，不表示速度正确或错误。

## 1. 意图

本实验固定 `agent.speed=5.0`，把原始 LeWM Eval 与 ContextWorld 合成 Eval
放到同一动力学速度下，分离三个可能造成分数差异的成分：

1. 原始轨迹与合成轨迹的数据来源差异；
2. future-25 局部目标与 E4 固定远目标的任务几何/规划难度差异；
3. E4 同时混合八个速度本身造成的差异。

本实验不训练新模型。主归因只使用三个训练速度同为 5 的公平模型：

- 原始单训 `H3-OrigHeldout`；
- 单速 5 合成单训 `H3-Synth5Matched`；
- 原始+单速 5 混训 `H3-OrigPlusSynth5`。

`H3-SpeedFull` 训练时包含多速度，只作为稳健性对照，不进入“训练速度也相同”
的主归因。

## 2. 三套同速 Eval

| Eval | 速度 | 起点/目标规则 | 用途 |
|---|---:|---|---|
| 原始 future-25 heldout | 5 | 原始 heldout 轨迹中相隔 25 步 | 原始来源基线 |
| 合成速度 5 future-25 | 5 | Synth5Matched-v2 轨迹中相隔 25 步 | 只改变轨迹来源 |
| E4 fixed-geometry speed5 | 5 | 固定 s0/s1/s2/s3 到 `(190,190)` | 改变目标几何与规划难度 |

前两套结果已经完成并原样复用。第三套从冻结 E4 catalog 中只选择 speed=5 的
四个 query；对每个模型执行 seeds 42–47、每 seed 50 次。

E4-speed5 同时运行：

- `none`：不加固定 context，用于与前两套基础 planning Eval 比较；
- 同速历史：历史 speed=5；
- 快速历史：冻结 catalog 中的历史 speed=7。

三种条件复用相同 evaluation schedule 和 CEM seed。全部使用原始 9,000 train
episodes 的冻结 normalizer、50 步预算、5×5 raw-step horizon、CEM
`300 samples × 30 iterations`。

## 3. 预注册归因规则

正式看结果前冻结以下口径：

- 合成速度 5 与原始 future-25 的模型内成功率差绝对值都不超过 10 pp：
  数据来源成分记为次要；
- speed5 E4 no-context 相对两套 future-25 Eval 至少下降 40 pp：
  固定几何/规划成分记为主要；
- speed5 E4 相对既有多速度 E4 至少提高 20 pp，才支持“多速度混合是主要原因”；
- 同速历史与快速历史或无历史的成功率至少相差 5 pp，才记为
  planning-level context effect；
- 必须报告 s0/s1/s2/s3，不允许只看 pooled success。

## 4. 限制

E4-speed5 的 300 次评测来自四个冻结 base queries 在不同 CEM seeds 下重复，
不是 300 个独立 geometry。因此它能检验“固定速度后，当前 E4 是否仍被模板
和规划器锁死”，不能替代一套大规模、独立 geometry 的 speed5 OOD benchmark。

机器可读冻结协议：
`configs/benchmark/tworoom_speed5_cross_eval_v1.yaml`。

## 5. 结果

### 5.1 四模型同速结果

每个单元均为 6 个 seeds、每 seed 50 次，共 300 次。前两列复用既有
future-25 结果；后三列为本协议新执行的 E4-speed5。

| 训练模型 | 原始 future-25 | 合成速度 5 future-25 | E4-speed5 无历史 | E4-speed5 同速历史 | E4-speed5 快速历史 |
|---|---:|---:|---:|---:|---:|
| 原始单训 | 273/300（91.00%） | 281/300（93.67%） | 75/300（25.00%）；106.96 px | 75/300；106.95 px | 75/300；106.73 px |
| 单速 5 合成单训 | 275/300（91.67%） | 282/300（94.00%） | 75/300（25.00%）；111.14 px | 75/300；112.39 px | 75/300；111.84 px |
| 原始+单速 5 混训 | 263/300（87.67%） | 274/300（91.33%） | 75/300（25.00%）；104.97 px | 75/300；104.33 px | 75/300；104.87 px |
| 原始+多速度混训 | 289/300（96.33%） | 285/300（95.00%） | 75/300（25.00%）；104.07 px | 75/300；102.51 px | 75/300；102.69 px |

主归因的三个 fixed-speed-trained 模型，在“合成速度 5 − 原始”上的成功率差
分别为 `+2.67/+2.33/+3.67 pp`，最大绝对差 3.67 pp，低于预注册的 10 pp
阈值。相反，E4-speed5 无上下文相对两套 future-25 的下降为
`62.67–69.00 pp`，远高于预注册的 40 pp 阈值。

### 5.2 Eval 分布差异

| Eval | 平均起点—目标距离 | 中位数 | P90 | 跨房间比例 |
|---|---:|---:|---:|---:|
| 原始 future-25 | 42.87 px | 42.82 px | 67.70 px | 71/300（23.67%） |
| 合成速度 5 future-25 | 41.43 px | 40.15 px | 64.83 px | 79/300（26.33%） |
| E4 fixed-geometry speed5 | 121.97 px | 131.31 px | 180.62 px | 150/300（50.00%） |

原始与合成 future-25 在距离和跨房间比例上接近。E4-speed5 的平均距离约为
两者的三倍，跨房间占比约为两倍，并使用固定远目标。因此“原始来源换成合成
来源”只改变了很小一部分；真正的大变化是从有真实 25-step 后续轨迹的局部目标，
换成固定远目标和当前 CEM 规划问题。

E4 四个 speed=5 base templates 的固定几何为：

| 模板 | 起点 → 目标 | 直线距离 | 房间关系 |
|---|---|---:|---|
| s0 | `(55,70) → (190,190)` | 180.62 px | 跨房间 |
| s1 | `(55,150) → (190,190)` | 140.80 px | 跨房间 |
| s2 | `(169,70) → (190,190)` | 121.82 px | 同房间 |
| s3 | `(169,150) → (190,190)` | 45.18 px | 同房间 |

原始 future-25 的平均距离 42.87 px、合成速度 5 的 41.43 px 在数值上接近
s3 的 45.18 px，但二者不能被重命名为 s3：它们从许多 episodes 的真实轨迹
采样，分别仍有 23.67%/26.33% 跨房间，而且目标是同轨迹 25 步后的已实现状态。
s3 则是单个固定 same-room geometry。最新结论针对的是 **E4 fixed-geometry
压力测试与 planner 的组合**，不是“所有 synthetic Eval 距离分布有问题”。

### 5.3 速度混合与模板锁定

既有多速度 E4 无上下文为 73/300（24.33%），固定为 speed=5 后为
75/300（25.00%），只增加 0.67 pp，未达到 20 pp 预注册阈值。进一步按模板
条件化后，四个模型以及无历史、同速历史、快速历史三种条件都完全相同：

| 模板 | E4-speed5 成功 |
|---|---:|
| s0 | 0/74 |
| s1 | 0/76 |
| s2 | 0/75 |
| s3 | 75/75 |

原多速度 E4 也是 s0/s1/s2 全失败、s3 全成功；其 73 次成功只是冻结 schedule
抽到了 73 个 s3，而 speed5 schedule 抽到 75 个。因此 0.67 pp 不是速度变为 5
后的模型收益，而是模板样本数差异。按模板条件化后，速度混合可解释的二值
成功率成分为 0。

三种历史条件的成功集合也完全一致，所有模型的 success effect 都是
0 pp、discordant pairs 都是 0。同速历史对平均最终距离有小幅连续变化
（原始单训 -0.02 px、合成单训 +1.26 px、单速混训 -0.63 px、
SpeedFull -1.56 px，相对无上下文），但没有跨过任何成功阈值。

### 5.4 预注册判定与结论

- `trajectory_source_component_minor=true`；
- `fixed_geometry_planner_component_major=true`；
- `speed_mixture_is_primary=false`；
- `planning_context_success_effect_detected=false`；
- `e4_speed5_outcomes_locked_by_template=true`。

因此，本实验支持用户提出的分解思路，并给出明确答案：**把训练与 Eval 都固定
为 speed=5 后，原始 future-25 与合成速度 5 future-25 仍都在约 88%–94%，
而 E4 仍只有 25%；主要分数落差来自 E4 的固定远目标几何与当前规划器，
不是合成 Eval 的多速度混合，也不是原始/合成轨迹来源。**

该结论把根因定位到“任务构造 × planner”层，但本实验本身还不能把其中的
goal cost、CEM 搜索维度、动作边界和模型 rollout 各自分成独立因果比例。
后续模型归因、规划器诊断和项目路线不在本协议中维护，统一见
[速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

机器可读汇总：
`artifacts/evaluation/history3/original_ability_reconstruction/speed5_cross_eval_n50x6.json`
（SHA-256 `d6e2e959...fdabbf`）。冻结配置 SHA-256 为
`e23de93c...432f2`。
