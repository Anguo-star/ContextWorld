# TwoRoom History-3 速度 Benchmark v2 执行协议

**版本**：v1.0
**日期**：2026-07-20
**状态**：Validation 已执行；Test 保持封存
**用途**：固定数据、模型、评测、统计和能力声明边界

本文件回答“结果是怎样得到的”。当前数据、结果和结论统一见
[TwoRoom 速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

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
v_query：评测器固定、决定 query 后续真实转移的速度
v_history：生成两段历史转移的速度
```

每个 query 行比较低速、中速和高速三种历史。若历史速度与 `v_query` 相同，称为
“同速历史”；其余条件只称为“较慢历史”或“较快历史”。速度和历史本身没有
“正确、错误”之分。

Query 只有一张静态图像，从图像中无法知道 `v_query`。把同速历史作为物理校准
参照，依赖一个明确的任务假设：历史和 query 来自同一个、短期内速度稳定的环境。
若取消该假设，本协议只能判断模型是否随历史速度响应，不能判断哪种历史应最接近
query 的真实未来。

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

| 组别 | 中文名称 | 训练数据 | 数量 | 作用 |
|---|---|---|---:|---|
| M0 | 原始单速基线 | 原始 TwoRoom，速度 5 | 1 | 公共参考 |
| M1 | 匹配单速控制 | 原始数据 + 速度 5 合成数据 | 3 | 控制额外数据与训练 |
| M2 | 匹配多速度模型 | 原始数据 + 多速度合成数据 | 3 | 目标方法 |

M1 与 M2 使用成对训练种子 `3072/4096/5120`。M0 只有一个原始 checkpoint，不用
它估计训练方差。

### 3.2 唯一允许变化的训练因素

M1 和 M2 复用相同的：

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
M1：所有合成 scenario 的 agent.speed = 5.0
M2：32 个训练速度
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

M2 合成训练：
  2.6/2.7/2.8/2.9/3.1/3.3/3.5/3.7/3.8/3.9/4.1/4.2/4.4/4.5/
  4.7/5.0/5.1/5.3/5.5/5.7/5.9/6.1/6.2/6.3/6.6/6.7/6.8/7.0/
  7.2/7.3/7.8/7.9

训练期监控：
  2.5/3.2/4.3/4.6/4.9/5.4/5.6/5.8

Planner Calibration：
  3.0/5.2/6.5

Validation—M2 训练见过的速度：
  3.1/5.1/7.0

Validation—所有训练均未见、但位于 M2 训练范围内：
  3.4/4.8/6.9

封存 Test：
  3.6/6.0/7.5
```

速度集合按数值和 `1e-6` 容差审计。未见速度 Validation 与原始训练、M2 合成
训练及训练期监控均无交集。Test 与 Train、监控、Calibration 和 Validation 均
无交集；本阶段不打开 Test。

## 4. 配对 Eval 数据

Validation 分两条轨道：

| 轨道 | 三档速度 | 回答的问题 |
|---|---|---|
| 训练见过速度 | 3.1 / 5.1 / 7.0 | 是否会根据历史调用学过的速度规律 |
| 训练未见速度 | 3.4 / 4.8 / 6.9 | 是否能对连续速度做区间内插值 |

每条轨道构造完整 `3×3` 矩阵：

| query 速度 | 低速历史 | 中速历史 | 高速历史 |
|---|---|---|---|
| 低速 | 同速 | 较快 | 较快 |
| 中速 | 较慢 | 同速 | 较快 |
| 高速 | 较慢 | 较慢 | 同速 |

三个 query 速度使用相同的静态 query 像素；速度只会影响动作执行后的未来。三种
历史使用相同历史初始状态和动作，后续状态按各自速度自然变化。

每个 query 速度行、历史条件、模型和评测种子都使用完整样本：

```text
50 个 query × 6 个评测种子 = 300
```

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
- 静态 query 像素在三种 query 速度间完全相同；
- 模型输入字段仅为像素和动作；
- 历史状态与块内动作能恢复生成速度；
- Train、Validation 和封存 Test 的速度支持符合第 3.3 节。

若真实状态和动作都无法恢复速度，模型无响应不能解释为模型能力不足。

### 5.2 真实下一状态与位移：主要证据

对相同 query 单帧和相同原始动作序列，精确模拟器按 `v_query` 生成真实未来。
模型分别输入三种历史，在 `1/2/3/5/10` 个 action block 上报告：

- 推断速度；
- 预测位置与真实位置误差，单位 px；
- 预测位移长度、真实位移长度及误差；
- 位移方向误差；
- 预测 latent 到真实 query future latent 的 MSE。

LeWM 没有坐标解码头。评测器在 `2.5–8.0`、步长 `0.05` 的冻结速度网格上生成
精确未来，经同一冻结 encoder 编码；与预测 latent 最接近的 oracle 速度提供
“推断速度”和对应物理位置。直接 latent MSE 同时保留，避免只依赖网格映射。

Action probe 包含恒定方向、变幅和转向三类，并限制在无碰撞区域。这样测量的是
单位动作位移，不把撞墙截断混入主结论。

两个概念必须分开：

```text
历史响应：
  高速历史预测 − 低速历史预测

物理校准：
  其他历史对真实 query future 的误差
  − 同速历史对真实 query future 的误差
```

历史响应大于零说明模型读取了速度线索；只有同速历史在三个 query 行都具有最低
误差，才说明响应已经校准到 query 的真实动力学。

### 5.3 固定候选动作

每个 query 冻结 300 条、每条 10 个 action block 的候选序列。三种历史共享同一
candidate bank。模型对候选排序，精确模拟器按 `v_query` 执行被选候选。

主指标：

```text
真实动力学 regret
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

### 5.5 原能力保持

M1、M2 均与 M0 在以下任务配对比较：

- 原始 episode-heldout Eval；
- 速度 5 合成同分布 Eval；
- 1/2/3/5-step rollout error。

每个规划 Eval 对每个模型均为独立 `50×6=300`。规划非劣效门冻结为：

- 成功率差 95% 区间下界不低于 `−5` 个百分点；
- final distance 差 95% 区间上界不高于 `+5 px`；
- M0 可解的任务分层不得在新模型中完全坍塌。

Rollout error 必须报告，但不同 checkpoint 的原生 latent 尺度不直接作为跨模型
能力门。

## 6. 统计

物理与固定候选比较均在同一 query、action probe 和评测种子内配对。独立 cluster
定义为：

```text
静态 query geometry × action-probe family
```

正式汇总使用：

- 10,000 次分层 bootstrap 95% 区间；
- 六个评测种子的逐种子方向；
- cluster sign test；
- 预注册主比较的 Holm 校正，familywise α=0.05；
- M1/M2 三个相同训练种子的成对方法效应。

训练方法是否稳定，以三个训练种子为最高层复现单位，不以 300 个 query 重复代替
训练方差。

## 7. 阶段结论等级

| 等级 | 必须满足 | 允许的结论 |
|---|---|---|
| A 历史敏感 | M2 在见过速度的三行均稳定随历史速度响应 | 模型会读取 History-3 中的速度线索 |
| B 物理校准 | A + 三行的一步及多步真实未来误差门 + M2−M1 稳定 | 响应已校准到 query 动力学 |
| C 候选校准 | B + 三行固定候选 regret 门 + M2−M1 稳定 | 校准改善真实候选选择 |
| D 训练方法 | C + 三种归因效应稳定 + 原能力保持 | 多速度训练稳定带来完整能力且不损伤原任务 |
| E 未见速度插值 | D + 未见速度轨道复现 A–C 和归因门 | 能力推广到训练未见的区间内速度 |

等级按顺序判定。较高层的某个平均指标改善，不能绕过较低层失败的必要条件。
闭环 endpoint score 在所有等级中均为必报结果，但不作为物理预测门。

## 8. 产物与复现入口

冻结配置：

- `configs/benchmark/tworoom_speed_isolated_v2.yaml`
- `configs/benchmark/tworoom_speed_cube_eval_v2.yaml`

主要入口：

```bash
python scripts/audit_tworoom_speed_isolated_v2.py
python scripts/build_tworoom_speed_cube_catalog.py
python scripts/run_tworoom_speed_isolated_eval.py --mode physical
python scripts/run_tworoom_speed_isolated_eval.py --mode fixed
python scripts/run_tworoom_speed_isolated_eval.py --mode planning
python scripts/run_tworoom_speed_isolated_ability.py --mode all
python scripts/audit_tworoom_temporal_causality.py
python scripts/analyze_tworoom_speed_isolated_v2.py
```

统一机器汇总写入：

```text
artifacts/evaluation/history3/speed_isolated_v2/final_summary.json
```

封存 Test 只有在 Validation 结论、planner、指标、阈值和报告格式全部冻结后才能
执行一次。本阶段不以 Validation 结果反向修改 Test。
