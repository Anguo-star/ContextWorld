# TwoRoom History-3 速度 Benchmark v2 执行协议

**版本**：v1.4
**日期**：2026-07-21
**状态**：Validation 已执行；Test 保持封存
**用途**：固定数据、模型、评测、统计和能力声明边界

本文件回答“结果是怎样得到的”，适合复现者和审计者。只想了解训练配方、关键
数字和能力边界的读者，请直接阅读
[ContextWorld ICL Benchmark：速度](../ContextWorld_ICL_Benchmark.md#611-速度)。

## 1. 研究问题

本协议检验：

> 在不更新权重的情况下，History-3 LeWM 能否从前两段动作—画面转移中识别
> `agent.speed`，并据此调整相同动作下的未来状态预测和规划？

History-3 包含两个历史转移和一个当前 query：

```text
历史 1：observation → action block → next observation
历史 2：observation → action block → next observation
query：一张当前 observation + 待评估的 action sequence
```

模型只接收像素和动作，不接收速度、状态坐标或 query 的真实未来。评测期间不更新
权重。

## 2. 术语和任务假设

记：

```text
v_reference：评测器用于生成参考未来的速度，不输入模型
v_history：生成两段历史转移的速度
```

结果文件为了兼容旧代码，仍把 `v_reference` 存在字段 `query_speed` 中。Query 只有
一张静态图像，从图像中无法知道这个值，模型输入也不包含它。

每个参考未来速度比较低速、中速和高速三种历史。若历史速度与 `v_reference` 相同，
称为“同速历史”；其余条件只称为“较慢历史”或“较快历史”。速度和历史本身没有
“正确、错误”之分。

把同速历史作为准确性参照，依赖一个明确的任务假设：历史和 query 来自同一个、
短期内速度稳定的环境。若取消该假设，本协议只能判断模型是否随历史速度响应，不能
判断哪种历史应最接近参考未来。

### 2.1 速度不是 frameskip

本实验只改变 `agent.speed`，始终固定：

```text
frameskip/action block = 5 个原始环境步
```

一个模型时间步接收块内 5 个有序动作，但只在块边界接收图像。无碰撞时：

```text
block 位移 ≈ agent.speed × Σ(块内 5 个动作)
```

因此结论限定为“固定 action block 下的物理速度响应”。改变 frameskip、动作重复
次数或观测频率属于另一项 Benchmark。

### 2.2 时间因果性

所有模型使用相同的逐模型帧 causal attention。执行版额外要求真实 checkpoint
通过未来 token 扰动测试：

- 改变某个时间边界之后的 observation/action token；
- 边界及以前的 predictor 输出不得变化；
- 边界后的输出必须能够变化；
- 审计前后模型权重哈希必须一致。

图像内部 ViT 的空间 attention 不属于这里的时间 attention。

## 3. 隔离训练

### 3.1 三类模型

| 模型名称 | 训练数据 | 数量 | 作用 |
|---|---|---:|---|
| 原始数据模型 | 原始 TwoRoom，速度 5 | 1 | 公共参考 |
| 单速合成混训模型 | 原始数据 + 速度 5 合成数据 | 3 | 控制额外数据与训练 |
| 多速度合成混训模型 | 原始数据 + 多速度合成数据 | 3 | 目标方法 |

单速合成混训模型与多速度合成混训模型使用成对训练种子
`3072/4096/5120`。原始数据模型只有一个 checkpoint，不用它估计训练方差。

### 3.2 唯一允许变化的训练因素

单速合成混训模型和多速度合成混训模型复用相同的：

- 608 个 scenario request，其中 512 个训练、96 个训练期监控；
- geometry seed group、reset 约束和采集策略；
- episode 数、像素编码和数据质量门；
- 原始/合成各 50% 的 batch 抽样；
- 原始训练 split normalizer；
- History-3 架构、初始化种子和优化器；
- 有效 global batch 1024；
- 12,840 个 optimizer step；
- 固定最后一步 checkpoint，不按 Validation 选最好模型。

两组之间只改变合成数据的速度支持：

```text
单速合成混训：所有合成 scenario 的 agent.speed = 5.0
多速度合成混训：32 个训练速度
```

每个模型实际抽样：

```text
原始数据 6,574,080 次
合成数据 6,574,080 次
```

速度导致 episode 长度不同，因而两组可用原始 clip 数可以不同；正式归因匹配的是
scenario、采集机制和实际训练抽样次数。

### 3.3 速度集合

```text
原始训练：
  5.0

多速度合成训练：
  2.6/2.7/2.8/2.9/3.1/3.3/3.5/3.7/3.8/3.9/4.1/4.2/4.4/4.5/
  4.7/5.0/5.1/5.3/5.5/5.7/5.9/6.1/6.2/6.3/6.6/6.7/6.8/7.0/
  7.2/7.3/7.8/7.9

训练期监控：
  2.5/3.2/4.3/4.6/4.9/5.4/5.6/5.8

Planner Calibration：
  3.0/5.2/6.5

Validation—多速度合成训练见过的速度：
  3.1/5.1/7.0

Validation—所有训练均未见、但位于多速度训练范围内：
  3.4/4.8/6.9

封存 Test：
  3.6/6.0/7.5
```

速度集合按数值和 `1e-6` 容差审计。未见速度 Validation 与原始训练、多速度合成
训练及训练期监控均无交集。Test 与 Train、监控、Calibration 和 Validation 均
无交集；本阶段不打开 Test。

## 4. 配对 Eval 数据

Validation 分两条轨道：

| 轨道 | 三档速度 | 回答的问题 |
|---|---|---|
| 训练见过速度 | 3.1 / 5.1 / 7.0 | 是否会根据历史调用学过的速度规律 |
| 训练未见速度 | 3.4 / 4.8 / 6.9 | 是否能对连续速度做区间内插值 |

每条轨道构造完整 `3×3` 矩阵：

| 参考未来速度 | 低速历史 | 中速历史 | 高速历史 |
|---|---|---|---|
| 低速 | 同速 | 较快 | 较快 |
| 中速 | 较慢 | 同速 | 较快 |
| 高速 | 较慢 | 较慢 | 同速 |

三个参考未来速度使用相同的静态 query 像素；速度只会影响动作执行后的未来。三种
历史使用相同历史初始状态和动作，后续状态按各自速度自然变化。

每个参考未来速度行、历史条件、模型和评测种子都使用完整样本：

```text
50 个 query × 6 个评测种子 = 300
```

下一帧 latent 推理是确定性的，因此每个种子中的 50 个 query 必须互不相同，六个
种子之间也不得重复。同一批 300 个静态 query 在三个参考未来速度之间严格配对；
不能像随机 CEM 那样重复一个 query 多次再把重复项算作独立样本。

因此：

```text
每个 checkpoint、每条轨道：3 × 3 × 300 = 2,700 个条件轨迹
7 个 checkpoint、两条轨道：37,800 个条件轨迹
```

不能在多个 query 行、历史条件、模型或轨道之间均分这 300 次。

## 5. 证据顺序

### 5.1 数据与上下文可辨识性

评分前必须通过：

- scenario、episode、query 和 payload 哈希审计；
- `agent.speed` 与 action block 分开读取；
- 静态 query 像素在三种参考未来速度间完全相同；
- 模型输入字段仅为像素和动作；
- 历史状态与块内动作能恢复生成速度；
- Train、Validation 和封存 Test 的速度支持符合第 3.3 节。

若真实状态和动作都无法恢复速度，模型无响应不能解释为模型能力不足。

### 5.2 参考未来 Latent Loss：主要证据

Eval payload 在构建时已经离线保存：

- `query_pixels`：当前 query 图像；
- `query_action`：随后执行的一个 5-step action block；
- `target_pixels`：按 `v_reference` 真实执行该 action 后的下一帧；
- 低速、中速和高速三种 History-3 上下文。

评分时禁止启动环境重新生成目标。每张 `target_pixels` 使用当前 checkpoint 自己的
冻结 encoder 编码：

`target_pixels` 是当前 query 执行 `query_action` 后的真实下一帧，不是已经出现在
上下文输入中的 history next frame。后者不得作为当前 query 的评分目标。

```text
reference_latent(v) = encoder(frozen_target_pixels(v))

loss(v_history → v_reference)
= MSE(predicted_next_latent(v_history), reference_latent(v_reference))
```

主指标为：

```text
matching_loss(v) = loss(v → v)

matching_advantage(v)
= mean(loss(other_history → v)) − matching_loss(v)

relative_loss_reduction(v)
= matching_advantage(v) / mean(loss(other_history → v))
```

这组指标不把 latent 反推成速度或二维坐标。正的 `matching_advantage` 表示：在生成
同一个参考未来时，同速历史比另外两种历史产生了更准确的预测。只报
`matching_loss` 不足以隔离 ICL，因为普通画面预测误差也会进入该 loss。

所有比较必须在同一个 checkpoint 内完成。不同 checkpoint 的 encoder 空间和 MSE
尺度可能不同，因此跨模型只比较归一化后的相对 loss 降低、成对训练种子效应和方向
一致性，不直接比较原始 latent MSE。

正式直接指标只评测这个冻结的一步转移，不保留从 latent 反推的速度、像素位置或
oracle 网格指标。若未来需要多步准确性，必须先离线生成并冻结多步目标帧，再发布
新的协议版本，不能在正式评分期间临时运行环境。

该扩展现已由独立的
[速度范围外与多步预测协议](TwoRoom_History3_Speed_Extrapolation_Multistep_v1_Protocol.md)
执行。v2 的原始一步数据和判断门保持不变。

### 5.3 固定候选动作

每个 query 冻结 300 条、每条 10 个 action block 的候选序列。三种历史共享同一
candidate bank。模型对候选排序，冻结的机制评测按 `v_reference` 判断候选真实代价。

主指标：

```text
固定候选选择误差
= 被选候选的真实末端代价 − bank 中真实最优候选的末端代价
```

固定候选隔离模型代价和候选选择，不混入 CEM 采样分布或滚动重规划。

### 5.4 CEM 闭环

冻结主规划配置：

| 参数 | 数值 |
|---|---:|
| Action block | 5 个原始步 |
| Model rollout horizon | 10 blocks，即 50 个原始步 |
| Receding horizon | 5 blocks |
| CEM candidates / iterations / top-k | 300 / 30 / 30 |
| 最大真实执行预算 | 100 个原始步 |
| 成功半径 | 16 px |

一次 100 步最大轨迹同时读取 50/75/100 步 deadline。短预算是同一轨迹的严格
前缀，不重新抽样。每个矩阵单元在每个截止点仍有完整 300 个配对观测。

必报指标为：

- 三个截止点的成功率；
- final/best distance 和 normalized distance AUC；
- steps-to-success；
- CEM solve 数、候选评估量和运行时间。

CEM 成功率是冻结规划资源下的效用，不是下一状态预测准确性的替代指标。

### 5.5 基础能力保持

单速合成混训模型、多速度合成混训模型均与原始数据模型在以下任务配对比较：

- 原始 episode-heldout Eval；
- 速度 5 合成同分布 Eval；
- 1/2/3/5-step rollout error。

每个规划 Eval 对每个模型均为独立 `50×6=300`。规划非劣效门冻结为：

- 成功率差 95% 区间下界不低于 `−5` 个百分点；
- final distance 差 95% 区间上界不高于 `+5 px`；
- 原始数据模型可解的任务分层不得在新模型中完全坍塌。

Rollout error 必须报告，但不同 checkpoint 的原生 latent 尺度不直接作为跨模型
能力门。

## 6. 统计

下一帧 latent 与固定候选比较均在同一 query、action 和评测种子内配对。独立 cluster
定义为：

```text
静态 query geometry × action-probe family
```

正式汇总使用：

- 以静态 query 为 cluster 的 10,000 次 bootstrap 95% 区间；
- 六个评测种子的逐种子方向；
- 单速与多速度混训三个相同训练种子的成对方法效应。

v4 的正式门是预注册的方向一致性和成对训练种子效应，不把事后新增的显著性检验
写成正式门。若后续协议同时冻结多个 p-value 主检验，再按通用设计规范预注册多重
比较校正。

训练方法是否稳定，以三个训练种子为最高层复现单位，不以 300 个 query 重复代替
训练方差。

## 7. 结论判定顺序

| 能力 | 必须满足 | 允许的结论 |
|---|---|---|
| 下一帧速度 ICL | 三个参考速度上，同速历史的离线下一帧 latent loss 低于另外两种历史的平均 loss；六个评测种子方向一致；多速度相对单速的提升在三个成对训练种子上稳定 | 模型会根据 History-3 提高相应速度下一帧的预测准确性 |
| 候选校准 | 下一帧速度 ICL 通过；三行固定候选选择误差门通过；多速度相对单速的提升稳定 | 下一帧校准改善真实候选选择 |
| 完整训练方法 | 候选校准通过；三种归因效应稳定；基础能力保持 | 多速度训练稳定带来完整能力且不损伤原任务 |
| 未见速度插值 | 完整训练方法通过；未见速度复现下一帧 latent、候选校准和归因门 | 能力推广到训练未见的区间内速度 |

能力按顺序判定。后续能力的某个平均指标改善，不能绕过前置条件。
闭环 endpoint score 在所有等级中均为必报结果，但不作为物理预测门。

额外报告“同速历史是否分别优于另外两种历史”，但它是严格诊断，不替代上表冻结的
平均 loss 主门。实际通过情况统一见主报告，本协议不重复维护结果数字。

## 8. 产物与复现入口

当前下一帧 latent 评分配置：

- `configs/benchmark/tworoom_speed_next_latent_v4.yaml`

旧的速度网格和位置 probe 只保留为历史产物，不再是本协议的评分入口。当前离线
评分与汇总入口为：

```bash
python scripts/build_tworoom_speed_next_latent_catalogs.py
python scripts/eval_tworoom_speed_next_latent.py \
  --model MODEL_SLUG --output MODEL_RESULT.json --device cuda:0
python scripts/analyze_tworoom_speed_next_latent.py
```

已有训练、规划和审计入口：

```bash
python scripts/audit_tworoom_speed_isolated_v2.py
python scripts/build_tworoom_speed_cube_catalog.py
python scripts/run_tworoom_speed_isolated_eval.py --mode fixed
python scripts/run_tworoom_speed_isolated_eval.py --mode planning
python scripts/run_tworoom_speed_isolated_ability.py --mode all
python scripts/audit_tworoom_temporal_causality.py
python scripts/analyze_tworoom_speed_isolated_v2.py
```

下一帧 latent 统一机器汇总写入：

```text
artifacts/evaluation/history3/speed_next_latent_v4/final_summary.json
```

范围外与多步扩展机器汇总写入：

```text
artifacts/evaluation/history3/speed_multistep_extrap_v5/final_summary.json
```

已有规划与能力保持汇总写入：

```text
artifacts/evaluation/history3/speed_isolated_v2/final_summary.json
```

封存 Test 只有在 Validation 结论、planner、指标、阈值和报告格式全部冻结后才能
执行一次。本阶段不以 Validation 结果反向修改 Test。
