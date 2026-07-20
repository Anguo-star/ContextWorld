# TwoRoom History-3 速度 Benchmark v2 设计方案

**版本**：v0.5
**日期**：2026-07-20
**状态**：catalog 已生成；模型尚未评分；下一状态主判据已在评分前冻结
**范围**：先完成 History-3 的单因子速度结论；更长历史、其他因子和因子组合后置

v0.5 是评分前修订。v0.4 catalog 生成后尚未加载任何模型、也没有产生模型分数。
本次只把物理下一状态的主次关系和执行预算读取方式写得更明确：第 1 个 action
block 的位置与位移误差是首要门槛，完整轨迹继续检查 1/2/3/5/10 blocks；一次
100 步闭环轨迹同时读取 50/75/100 步 deadline，避免为每个预算重复采样。

## 1. 目标

本方案回答一个明确问题：

> History-3 LeWM 能否只根据两段历史动作—观测转移，识别当前环境不可见的速度
> 动力学，并把识别结果用于预测和规划？

当前 query 只有一帧 RGB 图像，不含速度状态或速度标签。评测器会固定一个决定
query 后续转移的隐藏物理速度，模型只能根据历史动作—位移关系推断它。评测期间
不更新权重。

本阶段不研究：

- History-1、History-5 等历史长度比较；
- 门位置、摩擦、质量等其他 ICL 因子；
- 多因子同时变化；
- 正式 Test 上的最终排行榜。

这些工作只有在 History-3 速度轨道形成稳定结论后才启动。

## 2. 任务定义与术语

记：

```text
v_query：评测器固定、决定 query 未来的隐藏动力学速度
v_history：生成两段历史转移的速度
```

正式术语为：

| 条件 | 定义 | 含义 |
|---|---|---|
| 慢速历史 | `v_history < v_query` | 历史速度比查询环境慢 |
| 同速历史 | `v_history = v_query` | 历史与查询环境速度相同 |
| 快速历史 | `v_history > v_query` | 历史速度比查询环境快 |

速度值本身没有对错。旧代码中的 `correct`、`wrong_slow` 和 `wrong_fast` 只作为
兼容字段保留。新配置和报告使用 `same_speed_history`、`slower_history` 和
`faster_history`。

### 2.1 物理速度与 frameskip 不是同一参数

TwoRoom 当前同时有两个数值恰好为 5、但含义不同的参数：

| 参数 | 当前值 | 含义 | 本轨道是否改变 |
|---|---:|---|---|
| `agent.speed` | 训练基线为 5.0 | 每个原始环境步中，动作产生的像素位移增益 | 是 |
| `frameskip/action_block` | 5 | 一个模型时间步跨 5 个原始环境步，并接收这 5 个稠密动作 | 否 |

无碰撞时，一个模型时间步的位移满足：

```text
Δx_block ≈ agent.speed × Σ(action_1, …, action_5)
```

因此，本方案检验的是：

> 模型能否在固定 `action_block=5` 的表示下，从 History-3 推断
> `agent.speed`，并据此调整 rollout 和规划。

快速历史会让模型内部预测“同一组动作走得更远”，所以在有限 model horizon 下
可能让目标看起来更早可达。这是合理的规划效应，但不表示真实 Eval 环境被历史
加速；真实环境始终按该行固定的 `v_query` 执行动作。

当块内 5 个动作完全相同、轨迹无碰撞时，提高 `agent.speed` 与增加动作重复次数
在末端位移上会近似等价。为避免误解，主速度轨道必须：

- 在训练、历史、query future 和规划中固定 `action_block=5`；
- 保存并审计每个 block 的 5 个原始动作，不能只保存动作和或末端帧；
- 在可辨识性 probe 中包含块内动作方向或幅度变化；
- 分别记录 `agent.speed` readback 与 observation stride；
- 把结论写成“固定 action block 下的物理速度 ICL”。

改变 frameskip/action repeat 属于独立的时间聚合 Benchmark，需要匹配训练和模型
输入维度后另行设计，不与本阶段混跑。

### 2.2 时间注意力约束

当前 LeWM predictor 在模型帧之间使用 causal attention：

```text
frame 1 只能读取 frame 1
frame 2 只能读取 frame 1–2
frame 3 只能读取 frame 1–3
```

单张图像内部的 ViT 空间 attention 不受此限制。M0、M1、M2 必须使用相同的 causal
时间结构；任何 full-temporal-attention 变体都应另列模型，不能混入训练数据归因。
执行前还需用 token 扰动测试确认：改变未来模型帧或未来动作不能改变更早位置的
输出。

## 3. 模型矩阵

首页结果至少比较两类模型：

1. 原始 LeWM：没有接受多速度合成训练；
2. 多速度模型：接受我们生成的多速度数据训练。

但仅有这两个模型不足以把差异归因于“速度多样性”。正式训练归因至少需要三个
模型：

| 模型 | 训练数据 | 作用 |
|---|---|---|
| M0 原始 LeWM | 原始 TwoRoom | 公开基线和原始能力参照 |
| M1 单速训练控制 | 原始 + 单速合成 | 控制额外数据量、训练步数和合成流程 |
| M2 多速度目标 | 原始 + 多速度合成 | 检验多速度训练是否形成速度 ICL |

M1 与 M2 必须匹配：

- geometry、起点、目标和采集策略；
- episode、transition 和有效训练曝光量；
- loader、batch mixture、normalizer 和优化步数；
- 模型结构、初始化种子和 checkpoint 选择规则。

两者只允许速度支持不同。现有 `H3-OrigPlusSynth5` 与 `H3-SpeedFull` 可用于
runner 预检和历史结果对照，但由于 geometry、分层和 loader 尚未完全匹配，不能
代替这组正式单变量训练对照。

v2 的单速数据复用多速度数据的 512 个训练 seed group、96 个训练期 monitor
seed group、reset 约束、采集策略和 episode 数，只把每个合成 episode 的
`agent.speed` 固定为 5.0。M1 与 M2 都使用原始数据和合成数据各 50%，每个训练
种子执行 12,840 个 optimizer step。冻结的成对训练种子为
`3072/4096/5120`。

正式方法归因要求 M1 和 M2 至少三个成对训练种子。M0 若只有公开单 checkpoint，
应明确写成参考基线，不用它估计训练方差。后续若要证明“纯合成数据也足够”，再
增加多速度 synthetic-only 模型，不阻塞本阶段。

## 4. Eval 结构

### 4.1 四类数据域

| Eval | 作用 | 是否用于速度 ICL 主结论 |
|---|---|---|
| 原始 episode-heldout、速度 5 | 检查原始能力保持 | 否 |
| 合成速度 5 同分布 Eval | 分离合成数据域差异 | 否 |
| 训练见过的速度值 × 新 geometry | 检验基本的历史速度识别与调用 | 是 |
| 训练未见的区间内速度值 × 新 geometry | 检验连续速度插值 | 是，单独结论 |

不能用多个距离分布不同、任务难度不一致的 Eval 分数直接解释速度 ICL。核心
速度 Eval 必须使用同一批 geometry、query 单帧、起点、目标和动作 probe，只改变
评测器的 query 动力学与历史速度。

### 4.2 训练速度与 Eval 速度必须分账

当前 SpeedFull 的 32 个训练速度包含旧 E4 的全部 8 个速度。Directional v2 使用
的查询速度 5.0/5.1、慢速历史 3.1 和快速历史 7.0 也都在训练集合中。因此现有
结果属于“速度值见过、geometry/query 留出”，不能解释为未见速度插值。

v2 已冻结以下互不混用的速度集合：

```text
原始训练：5.0
M2 合成训练：
  2.6/2.7/2.8/2.9/3.1/3.3/3.5/3.7/3.8/3.9/4.1/4.2/4.4/4.5/4.7/5.0/
  5.1/5.3/5.5/5.7/5.9/6.1/6.2/6.3/6.6/6.7/6.8/7.0/7.2/7.3/7.8/7.9
训练期 monitor：2.5/3.2/4.3/4.6/4.9/5.4/5.6/5.8
Planner Calibration：3.0/5.2/6.5
M2 见过速度 Validation：3.1/5.1/7.0
未见速度 Validation：3.4/4.8/6.9
封存 Test：3.6/6.0/7.5
```

未见速度 Validation 与原始训练、M2 合成训练和训练期 monitor 都没有数值交集，
但位于 M2 训练范围内。见过速度轨道的三档值来自 M2 合成训练。所有集合都按
`agent.speed` 数值和 `1e-6` 容差审计，不能只靠 split 名称。

“见过速度”轨道回答模型能否根据历史调用已学过的动力学；“未见速度插值”轨道
回答模型能否根据历史估计新的连续速度。前者成立不自动推出后者。训练范围外的
速度外推只作为后续压力测试，不与基本 ICL 门绑定。

### 4.3 两条轨道使用相同的配对矩阵

每条轨道各选择三档可辨识、物理可达的速度：

```text
v_low < v_mid < v_high
```

见过速度轨道的三档值来自 `S_train`；插值轨道的三档值不得出现在 `S_train`。
具体数值在 Calibration 完成后、Validation 出分前冻结。每个 query 生成完整
`3×3` 矩阵：

| query 隐藏动力学 | 慢速历史 `v_low` | 中速历史 `v_mid` | 快速历史 `v_high` |
|---|---|---|---|
| `v_query=v_low` | 同速 | 比查询快 | 比查询快 |
| `v_query=v_mid` | 比查询慢 | 同速 | 比查询快 |
| `v_query=v_high` | 比查询慢 | 比查询慢 | 同速 |

三个 query 动力学行必须共享相同的当前 query 像素哈希。速度不在静态画面中，只
通过执行动作后的真实未来和历史转移体现。三个历史列使用同一组历史初始状态和
动作，允许后续画面按各自速度自然变化。

### 4.4 Split 与 geometry

速度立方体分为：

- Calibration：选择速度间隔、难度带和 planner profile；
- Validation：冻结后建立 History-3 阶段结论；
- Test：独立 geometry，直到模型、指标和阈值全部冻结后才运行一次。

Validation 优先使用原始风格但 episode/scenario-heldout 的 geometry；Test 使用
独立的合成 heldout geometry，检查结论是否依赖单一数据域。两个 split 都必须
覆盖 same-room、跨门、方向和路径长度分层，但各分层中的速度矩阵保持严格配对。

### 4.5 样本量

每个 query 动力学行、每个历史条件、每个模型均独立使用：

```text
50 次 × 6 个评测种子 = 300 次
```

因此一个模型在一个主 planner profile 下，每条轨道的完整速度立方体为：

```text
3 个 query 动力学 × 3 个历史条件 × 300 = 2,700 条
```

两条轨道对一个 checkpoint 合计 5,400 条。一个 M0、一个 M1、一个 M2
checkpoint 的先导矩阵合计 16,200 条。正式训练归因要求 M1 和 M2 各三个训练
种子，因此最低总数为：

```text
M0：1 × 5,400
M1：3 × 5,400
M2：3 × 5,400
合计：37,800 条
```

如果 M0 也训练三个种子，则为 48,600 条。不能把任一矩阵单元、两条轨道或多个
训练种子均分同一批 300。若多个执行预算可从同一条最大预算轨迹读取严格前缀，
则不重复生成轨迹，但每个矩阵单元在每个预算点仍有完整 300 个观测。

## 5. 证据链与指标

### 5.1 数据与可辨识性

必须审计：

- query 单帧在三个 `v_query` 行的像素完全相同；
- 历史状态、动作、像素和速度 readback 可逐步回放；
- 每个模型步恰好包含 5 个有序原始动作，`action_block` 在所有条件中均为 5；
- `agent.speed` 与 frameskip 分别读取和记录，不从总位移反推后混用；
- 两段历史包含足以区分三档速度的动作—位移信号；
- 可辨识性 probe 不能全部由“同一动作无碰撞重复 5 次”的退化样本组成；
- query future、速度标签和 simulator readback 不进入模型输入；
- causal-mask 扰动测试通过，较早输出不依赖未来 observation/action token；
- `S_train/S_cal/S_val_interp/S_test_interp` 的数值交集符合各自用途；
- 每条结果明确标记速度值见过或未见，不能只标记 geometry heldout；
- Train、Calibration、Validation 和 Test 无 episode/scenario 泄漏。

### 5.2 历史速度条件化

对同一 query 单帧和同一动作序列，分别输入三档历史。真实模拟器同时生成三档
速度的 oracle future。对每个模型，在它自己的 latent 空间中计算预测 rollout
到三个 oracle future 的距离：

```text
inferred_speed = argmin_v error(predicted_rollout, oracle_rollout_v)
```

报告：

- 三档历史对应的 inferred-speed confusion matrix；
- 预测最接近速度与历史速度的 ordinal correlation；
- 快速历史−慢速历史的逐 horizon 方向效应；
- 多速度目标相对单速控制的配对差分之差。

这个层级回答“模型是否读取了历史速度”，不使用 CEM endpoint score。

### 5.3 物理下一状态与位移校准

只比较模型预测与最终 target 的距离不够，因为该值混入了任务距离和 planner
目标。v2 对相同 query 单帧和相同原始 action 序列，使用精确模拟器在不同
`v_query` 下生成真实未来状态，并逐 `1/2/3/5/10` 个模型步比较模型预测。

LeWM 直接预测 latent，没有像素或坐标解码器。为避免另训一个坐标头，评测器在
冻结速度网格上生成 oracle future，并全部送入同一个冻结 encoder：

```text
v_hat = argmin_v latent_error(predicted_future, encoded_oracle_future(v))
predicted_position = oracle_state(v_hat)
```

因此可以直接报告：

- `|v_hat−v_history|`：模型预测是否跟随历史速度；
- `|v_hat−v_query|`：同速历史时是否恢复查询环境真实速度；
- 预测位置与真实下一状态的欧氏误差，单位 px；
- 位移大小误差和位移方向误差；
- 逐 horizon latent MSE。

action probe 同时包含恒定方向、幅度变化和转向序列，并单列碰撞与无碰撞轨迹。
速度标签、真实状态和 simulator readback 只用于评分，不进入模型输入。物理状态
误差是“做到速度自适应”的主判据；target cost 与 CEM score 不能替代它。

### 5.4 Query 动力学校准

对速度立方体的每一行，真实目标是该行 `v_query` 生成的 future。记
`E(q,h)` 为 query 动力学 `q`、历史速度 `h` 下的物理状态预测误差。

同速校准要求：

```text
E(low, low) < E(low, mid), E(low, high)
E(mid, mid) < E(mid, low), E(mid, high)
E(high, high) < E(high, low), E(high, mid)
```

判定分两层。第 1 个 action block 的位置 px 误差和位移大小误差是首要下一状态
门槛；随后把 1/2/3/5/10 blocks 的位置误差在 query 内聚合，检查校准是否能持续
到自回归长 rollout。两层都要求三个 query 速度行分别满足上述同速关系。逐
horizon latent MSE 是同一模型内部的直接证据；不同 checkpoint 的原生 latent
尺度不直接横向比较。跨模型以预测速度、位置 px 误差、同速收益和标准化效应为主。

### 5.5 固定候选规划正确性

每个 query 使用同一冻结 candidate bank。模型在三档历史下分别选候选，再用该行
真实 query 动力学执行候选。主指标为精确动力学 regret：

```text
regret(q,h)
  = true_cost(candidate_selected_under_history_h, dynamics_q)
    − true_cost(oracle_best_candidate, dynamics_q)
```

同速历史若相对另外两档历史稳定降低 regret，才能建立“模型会根据当前查询环境
需要的速度选择候选”。同时
报告完整排序、top-k、argmin 和首动作变化。

### 5.6 闭环规划效用

闭环必须报告：

- 从同一次 100 步最大轨迹读取 50/75/100 步 deadline success curve；
- final/best distance、normalized progress 和 trajectory AUC；
- 共同成功 query 的配对 steps-to-success；
- CEM solve、候选评估量、rollout blocks 和耗时；
- 固定的 model-horizon 敏感性。

同速历史不必在每个有限 planner profile 下取得最高 endpoint success。闭环分数
用于说明预测和候选变化在冻结资源下造成什么实际结果，不作为速度 ICL 成立的
唯一门槛。

### 5.7 能力保持

M1、M2 相对 M0 必须在原始 episode-heldout 和合成速度 5 domain 上报告：

- 成功率非劣效；
- final distance 和 rollout error；
- 原始任务分层；
- 多训练种子区间。

## 6. Planner Calibration

在新的 Calibration catalog 上联合选择：

- `B_exec`：真实执行预算阶梯；
- `H_model`：模型前视长度；
- candidate population 与 iterations；
- terminal-only、horizon 内 best-distance 或时间加权 goal cost；
- receding horizon、top-k 和成功半径。

选择规则是让精确动力学和主 planner 在三档速度上均非地板、非天花板，并使最慢
速度在模型视野内具有合理可达率；不能选择最能放大 M2 与控制模型差异的配置。
`action_block=5` 是模型与数据的固定接口，不属于这轮 planner 调参项。

Validation 只使用一个冻结主 profile 跑完整 `3×3` 矩阵。主 profile 的最大
执行预算为 100 步，deadline 固定为 50/75/100 步；预算曲线从同一条最大轨迹
读取，不重新运行 CEM。额外 model-horizon 敏感性只作为资源稳健性层，不与九个
矩阵单元混成一个总分。

## 7. 预注册主假设

具体实用阈值在 Calibration 后冻结，Validation 出分后不得修改。主假设为：

1. **H1 历史速度条件化**：M2 的预测最接近速度随三档历史单调变化，且效应稳定
   大于 M1；
2. **H2 query 动力学校准**：M2 在三个 query 动力学行中，同速历史的第 1 block
   位置 px 误差分别低于另外两档历史，1/2/3/5/10-block 聚合误差保持该关系，
   并且 `v_hat` 同时跟随历史速度；
3. **H3 规划层校准**：M2 的同速历史在固定候选上的精确动力学 regret 分别低于
   另外两档历史；
4. **H4 训练归因**：M2−M1 的差分之差在至少三个成对训练种子上复现；
5. **H5 能力保持**：M2 在冻结原始能力指标上满足非劣效门；
6. **H6 未见速度插值**：M2 在训练未见的区间内速度轨道上复现 H1–H3。

H1–H5 先在见过速度轨道上判定；H6 使用相同指标和预先冻结的阈值，在未见速度
轨道上单独判定。两条轨道的 p 值校正范围在预注册时固定。

闭环 endpoint success、连续轨迹和执行效率是完整必报结果，但 H1–H3 不以某个
单一执行预算的成功率排序作为判定条件。

统计以 query 内配对为主，按评测种子和 geometry 分层 bootstrap；训练方法结论
以训练种子为更高层重复单位。多主比较采用预注册的多重性校正。

## 8. 阶段性结论等级

结果只能按实际通过的等级陈述：

| 等级 | 必须通过 | 允许的结论 |
|---|---|---|
| A 历史敏感 | H1 | History-3 输出随历史速度方向变化 |
| B 速度校准 | A + H2 | History-3 能用历史校准 query 的隐藏速度动力学 |
| C 规划校准 | B + H3 | 校准结果被用于更符合真实动力学的候选选择 |
| D 训练方法 | C + H4 + H5 | 多速度训练稳定带来能力且不损伤原任务 |
| E 速度插值 | D + H6 | 能力可推广到训练未见、但位于训练范围内的速度 |

本阶段希望达到的“真正 History-3 速度 ICL”至少是等级 C。闭环成功率用于说明该
能力在特定 planner profile 下的效用和资源依赖，不取代等级判定。

## 9. 执行顺序

1. 冻结训练、Calibration、Validation 和 Test 的速度值集合；
2. 构建并全量回放见过速度与未见速度两条轨道的配对矩阵；
3. 在 Calibration 上验证两段历史可辨识性并冻结三档速度；
4. 在 Calibration 上冻结 planner profile 和 goal cost；
5. 生成严格匹配的 M1/M2 训练数据并完成至少三个成对训练种子；
6. 冻结 Validation 配置、统计和结论门；
7. 先运行物理下一状态、位移和固定候选矩阵，再运行完整闭环矩阵；
8. 发布成功和失败结果，确定 History-3 达到的等级；
9. 满足 Test 进入门后封存并一次性运行 Test。

任何已有 Validation 结果只能帮助形成假设，不能用于选择 v2 的速度、geometry、
planner 或阈值。

## 10. 后续路线

History-3 速度等级确定后，依次进行：

1. 在 History-3 下增加一个当前帧不可判定、但历史可以识别的隐藏因子；
2. 在速度和第二个因子上比较 History-1/3/5，隔离历史长度本身；
3. 独立建立 frameskip/action-repeat 时间聚合 Benchmark；
4. 继续建立其他单因子 Benchmark；
5. 每个单因子通过可辨识性和校准门后，再测试因子组合；
6. 最后研究多因子变化下的历史长度、干扰和组合泛化。

不能在 History-3 速度任务尚未闭合时，同时改变 history 长度、因子类型和 planner
配置，否则无法判断收益来自哪里。

静态门位置若已在 query 单帧中直接可见，不属于需要历史推断的 ICL 因子。门任务
必须改造成“当前帧无法判定、历史交互能够识别”的门规则，或改选动作延迟、摩擦等
隐藏动力学。
