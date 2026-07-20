# ContextWorld 世界模型上下文学习评测设计规范

**版本**：v5.1  
**日期**：2026-07-20  
**文档性质**：通用设计规范

## 1. 目标

ContextWorld 评估世界模型能否在不更新权重的情况下，从近期交互中识别隐藏环境
规则，并立即把该规则用于预测和规划。

隐藏规则可以是速度、摩擦、质量、门位置、障碍布局或任务语义。Benchmark 需要
分别回答：

1. 上下文是否包含足够的信息；
2. 模型是否读取并使用了上下文；
3. 模型的预测是否更接近当前环境的真实规律；
4. 预测变化如何影响规划器的候选排序和动作；
5. 在冻结资源下是否完成任务，以及用了多少真实步和计算量。

最终成功率是重要指标，但它是模型、规划器、任务几何、成功半径和资源上限共同
作用的结果，不能单独代表世界模型预测质量。

不同隐藏因子与资源上限的关系不同。速度等直接改变动作位移和可达时间的因子，
必须报告多个真实执行预算；不会改变完成时间尺度的因子，可以使用单一主预算。
是否采用预算阶梯由因子的作用机制决定，不按任务名称简单划分。

## 2. 能力声明

### 2.1 因子条件化上下文学习

声明模型学会使用某个隐藏因子，至少要求：

1. 评测期间不更新模型权重；
2. 因子可从模型可见的上下文中识别；
3. 查询、目标、输入长度和规划随机计划保持不变；
4. 只改变上下文因子时，预测或规划产生稳定且方向相关的变化；
5. 变化在 heldout 场景、多个评测种子和关键任务分层上复现；
6. 单因子或无关上下文控制不能产生同等效应。

数值发生轻微变化不等于 ICL。变化必须能对应到上下文表达的规则。

### 2.2 预测正确性

声明模型正确估计当前环境规律，要求正确上下文在相同 query 和 action sequence
上具有更低的真实预测误差。

对有顺序的因子，应同时比较：

```text
wrong-lower / correct / wrong-higher
```

主结论应基于冻结的 horizon 聚合方式，并报告逐 horizon 结果。不能因为平均误差
较低，就声称每个 query 或每个 horizon 都更准确。

### 2.3 规划正确性与效率

声明正确上下文带来规划收益，至少要求：

1. Correct 优于无上下文；
2. Correct 同时优于 wrong-lower 和 wrong-higher；
3. 改善出现在真实模拟器结果，而非只出现在模型内部代价；
4. 成功率、连续进度和效率指标方向一致；
5. 结论明确绑定冻结的规划资源配置。

如果 `wrong-lower < correct < wrong-higher`，可以建立因子条件化规划，但不能建立
“正确上下文最优”。

对速度等资源耦合因子，规划正确性必须结合整条预算曲线判断。单个紧预算上
Wrong-higher 的成功率更高，不能推翻 Correct 的预测校准，也不能证明
Wrong-higher 是物理上正确的上下文。

## 3. 数据对象与来源追踪

### 3.1 核心对象

| 对象 | 定义 | 主要用途 |
|---|---|---|
| BaseTask | 状态、动作、终止规则、可变因子和合法干预 | 定义任务边界 |
| Scenario | 固定 geometry、起点、目标、因子和采集种子 | 划分与统计 |
| EpisodeInstance | 一次完整轨迹及状态、动作、像素和终止信息 | 训练与回放 |
| ContextQueryBundle | 固定 query 与成组上下文 | ICL 配对评测 |
| CandidateBank | 对同一 query 冻结的候选动作序列 | 机制诊断 |
| PlannerProtocol | horizon、采样、cost、预算和成功规则 | 规划结果解释 |
| CollectionProfile | 采集策略、分层、horizon 和有效曝光量 | 训练公平性 |

同一 geometry 只改变隐藏因子时，不得误计为多个独立 geometry 场景。

### 3.2 必须保存的来源信息

正式产物至少记录：

- 配置、代码、模型和模拟器版本；
- 数据、catalog、checkpoint、normalizer 和候选 bank 哈希；
- scenario、episode、query、evaluation 和 split 标识；
- 隐藏因子真实值及 readback；
- 训练、采集、评测和 planner 随机种子；
- 上下文像素与动作 payload 哈希；
- 原始结果、审计结果、汇总和日志。

## 4. 数据划分与泄漏控制

### 4.1 隔离层级

| 层级 | 最低要求 |
|---|---|
| Clip 隔离 | Query clip 不得出现在训练数据 |
| Episode 隔离 | 同一完整轨迹不得跨训练和评测 |
| Scenario 隔离 | 关键 reset-goal geometry 不得跨关键 split |
| 组合隔离 | 测试组合外推时，因子组合类别也必须 heldout |

报告必须明确满足的隔离级别，不能用 clip-heldout 代替 episode-heldout 或
scenario-heldout。

### 4.2 Split 角色

| Split | 用途 | 是否允许根据结果调整 |
|---|---|---|
| Train | 模型拟合和训练 normalizer | 允许 |
| Calibration | 选难度带、验证 planner 正向效度 | 允许，但不得训练主模型 |
| Validation | 比较方法和检验预注册机制 | 有限允许；新假设必须重新冻结 |
| Test | 一次性最终报告 | 不允许 |

Test catalog 在模型、规划器、指标和阈值冻结前不得评分。

### 4.3 测试时上下文边界

Support context 不得包含：

- Query 的未来帧或结果；
- 隐藏因子标签；
- 任务是否完成的特权信息；
- 超出正式协议的额外上下文长度。

## 5. 评测数据集构建

### 5.1 数据正确性

训练或评测数据必须通过：

- 状态和动作范围；
- 因子 readback；
- 模拟器逐步回放；
- 像素、状态、目标和终止状态一致性；
- split 泄漏、重复 episode 和重复 geometry 检查。

任何关键检查失败时，结果不得标为正式通过。

### 5.2 上下文成组构建

严格配对组按研究问题选择：

| 条件 | 作用 |
|---|---|
| 无上下文 | 测量默认行为 |
| Correct | 与 query 使用相同隐藏因子 |
| Wrong-lower | 因子值低于 query |
| Wrong-higher | 因子值高于 query |
| 无关上下文 | 排除长度或内容总量 |
| 打乱上下文 | 破坏时序或动作—观测对应关系 |

同一组的 query、goal、context actions、输入长度、candidate bank 和规划随机数
必须一致。

### 5.3 难度设计

欧氏距离只是难度的一部分。Catalog 还需覆盖：

- room 或 topology 关系；
- 障碍、门和边界；
- 起点到目标的可达路径长度；
- 运动方向和动作饱和；
- baseline planner 成功概率；
- 地板、天花板和上下文敏感区。

难度带只能在 Calibration 上选择。Validation 或 Test 出分后，不得回选最容易
体现方法差异的 geometry。

### 5.4 样本量与执行预算

默认确认样本量为：

```text
每个 Eval × 每个 condition × 50 次 × 6 个评测种子 = 300 次
```

两个 Eval 即使共享 Correct，也必须分别执行 300 次。不能把多个任务合并成一次
300 次均分。任务数增加时，样本量线性增加。

执行预算阶梯不是把这 300 次拆给多个预算点。每个预算点都必须有同一组
`50×6=300` 个配对观测：

- 如果 planner 看不到总执行预算，且短预算轨迹是长预算轨迹的严格前缀，可以
  一次执行最大预算，再从同一批 300 条轨迹读取各预算点；
- 如果 planner 会根据剩余预算改变动作，必须为每个预算点分别执行完整
  `50×6=300`；
- 两种实现都必须保留同一 query、condition 和随机计划间的配对关系。

## 6. 指标体系

### 6.1 通用证据链

| 层级 | 问题 | 必报指标 |
|---|---|---|
| L0 数据完整性 | 数据是否按协议生成 | replay、readback、hash、泄漏 |
| L1 可辨识性 | 上下文是否足以识别因子 | oracle accuracy/MAE、信息增益 |
| L2 一步预测 | 下一状态是否更准 | correct gain、wrong−correct、反事实预测 |
| L3 多步 rollout | 改善能否维持 | 逐 horizon error、horizon 平均 error、drift |
| L4 候选代价 | 上下文如何改变规划输入 | cost error、rank、top-k、argmin |
| L5 连续轨迹 | 未成功时是否更接近 | final/best distance、progress、trajectory AUC |
| L6 截止成功 | 是否在资源上限内完成 | success、deadline curve、A-only/B-only |
| L7 实际效率 | 成功用了多少真实资源 | steps-to-success、path efficiency、compute |
| L8 能力保持 | 新训练是否损伤原能力 | ID/OOD non-inferiority、rollout retention |

### 6.2 Speed 轨道的核心结果层级

Speed Benchmark 必须把以下结果串成一条证据链，不能只报告某个固定预算下的
Eval score：

| 结果层 | 必报内容 | 解释边界 |
|---|---|---|
| 速度可辨识性 | 从历史动作和位移恢复速度 | 证明上下文含有速度信息 |
| 预测校准 | Slow/Correct/Fast 的逐 horizon 与聚合误差 | 判断哪个上下文更符合真实动力学 |
| 候选动作 | cost rank、top-k、argmin、真实候选结果 | 定位上下文如何进入 CEM |
| 执行预算曲线 | 紧/标准/宽松预算下的三条件成功率 | 区分模型能力和截止效应 |
| 连续轨迹 | final/best distance、progress、trajectory AUC | 避免只看成功半径翻转 |
| 真实效率 | 配对 steps-to-success、path efficiency、compute | 判断成功是否更快、更省 |
| 能力保持 | 原始 ID/OOD 与 rollout retention | 排除新能力损伤旧能力 |

Speed 轨道的核心结果是一个结构化结果组，而不是一个混合总分。至少同时展示：

```text
S_slow(B), S_correct(B), S_fast(B)
Fast−Correct(B)
Correct−Slow(B)
```

其中 `S_condition(B)` 是该上下文在真实执行预算 `B` 前的成功率。可以附加归一化
success-budget AUC 作为整条曲线的摘要，但 AUC 不能替代逐预算曲线、预测误差和
效率指标。

三个预算点之间使用梯形积分，并按预算区间归一化：

```text
success_budget_auc(c)
  = integral[S_c(B), B_tight, B_relaxed]
    / (B_relaxed − B_tight)
```

结果范围为 0–1，越高表示在整个预算区间内越容易成功。上下文差异直接计算
`AUC_correct−AUC_slow` 和 `AUC_fast−AUC_correct`，不把预测误差或路径效率混入
同一个数。

### 6.3 预测误差

多步预测至少报告：

1. 每个冻结 horizon 的平均和中位误差；
2. 每个 query 先跨 horizon 聚合后的配对差；
3. 配对差区间或检验；
4. 哪些结论是 pooled mean，哪些是逐 query 多数。

如果比较 latent MSE，必须说明不同 checkpoint 的 latent 尺度是否可比。默认只
允许同一 checkpoint 内的上下文条件比较。

### 6.4 连续轨迹

推荐定义：

```text
normalized_remaining = final_distance / initial_distance
normalized_progress = 1 − normalized_remaining
```

同时保存：

- 每个原始步的 goal distance；
- best-so-far distance；
- normalized distance AUC；
- path length；
- progress per path length。

### 6.5 成功与效率

必须区分：

| 指标 | 含义 |
|---|---|
| Ever success | 是否曾进入成功半径 |
| Terminal success | 固定 horizon 末端是否仍在成功半径 |
| Deadline success | 环境是否在预算耗尽前终止成功 |
| Steps-to-success | 首次进入成功半径的真实步数 |

条件间成功集合不同时，直接比较“各自成功样本的平均步数”会产生选择偏差。效率
比较至少同时报告：

- 共同成功 query 上的配对 steps-to-success；
- 把失败视为删失或预算上限的 deadline curve；
- 各条件新增成功的任务数。

## 7. 规划资源与因果边界

### 7.1 必须冻结的资源

规划结果至少受以下四类资源影响：

| 资源 | 例子 | 影响 |
|---|---|---|
| 模型视野 | rollout horizon、action block | 候选是否在模型内看起来可达 |
| 搜索预算 | samples、iterations、top-k、采样方差 | 是否找到高质量候选 |
| 真实执行预算 | 最大环境步数、replanning 次数 | 是否赶在截止前成功 |
| 成功定义 | 半径、是否立即终止 | 二值 score 与到达时间 |

正式比较必须冻结这些参数，并把最终成功率写成“该 planner protocol 下的成功率”，
而不是模型的无条件能力。

### 7.2 Speed 执行预算阶梯

速度会直接改变单位动作位移、预计到达时间和固定 horizon 内的可达性，因此多
执行预算是 Speed Benchmark 的核心评测，而不是出分后的附加诊断。

正式 Speed 轨道使用三个预先冻结的预算档：

```text
B_tight < B_standard < B_relaxed
```

- `B_tight` 检查紧截止下的规划响应；
- `B_standard` 表示主任务配置；
- `B_relaxed` 检查放宽截止后上下文差异是否仍存在。

预算值只能在 Calibration split 上根据精确动力学可达时间、任务难度和非地板/
非天花板要求选择，并在 Validation 或 Test 出分前冻结。TwoRoom 当前
Validation 使用 50/75/100 个原始步；这组具体数字不自动推广到其他任务。

预算曲线至少回答：

```text
Fast−Correct 是否随预算增加而缩小？
Correct−Slow 是否在宽松预算下仍存在？
各条件新增成功发生在哪个预算区间？
```

如果差异随执行预算增加而消失，说明有限截止有贡献；如果仍然存在，还需检查
model horizon、搜索预算和滚动重规划。任何有限预算都不能称为“无限规划”。

### 7.3 其他因子的资源阶梯

非 Speed ICL 是否需要多执行预算，取决于该因子是否改变完成时间或可达路径：

| 因子类型 | 示例 | 资源设计 |
|---|---|---|
| 时间尺度或动力学 | 速度、摩擦、质量、动作延迟、action repeat | 完整执行预算阶梯 |
| 路径与拓扑 | 门位置、障碍布局、通道开闭 | 至少紧/宽两个预算的敏感性检查 |
| 语义或观测映射 | 颜色规则、目标语义、静态标签映射 | 通常一个冻结主预算即可 |
| 随机性与噪声 | 观测噪声、转移随机性 | 优先评估不确定性和搜索预算阶梯 |

即使不把预算阶梯列为主结果，也必须在 Calibration 上确认单一预算没有制造地板、
天花板或明显的截止偏置。

### 7.4 固定候选与自由搜索

机制诊断应同时包含：

1. **固定 candidate bank**：隔离模型代价和排序；
2. **精确动力学 oracle**：验证任务和候选 bank 的正向效度；
3. **自由 CEM**：测量搜索、argmin 和滚动重规划后的真实任务结果。

推荐分析链：

```text
context-conditioned rollout
  → goal cost
  → candidate rank / top-k
  → selected action
  → realized trajectory
  → deadline outcome
```

固定候选无差异而自由 CEM 有差异，说明搜索分布或重规划在放大顶部候选的细小
变化；不能把全部差异归于纯预测误差。

### 7.5 正对照

Calibration 必须确认：

- Candidate bank 在 Correct 动力学下具有足够可达率；
- 精确动力学或 oracle planner 能完成任务；
- 正确因子在真实动力学误差或 horizon 末端代价上优于双向错误因子；
- 主 planner 既非全失败也非全成功；
- 资源上限足以让模型差异传到至少一个连续指标。

正对照失败时，应先修复任务、候选或 planner，不能把主模型失败解释为没有 ICL。

## 8. 统计与预注册

### 8.1 配对优先

同一 query 的不同条件应共享 geometry、goal 和随机计划，并报告：

- 成功率差；
- A-only / B-only / both / neither；
- 配对精确检验；
- 配对连续指标差和区间；
- 评测种子、query 难度和 geometry 分层。

重复规划运行估计搜索方差，不是新的独立 geometry。报告应同时说明 query、
scenario、评测种子和训练种子数量。

### 8.2 预算效应

多个预算必须使用相同 query 和相同随机流，并验证短预算轨迹是长预算轨迹的严格
前缀。预算间上下文效应变化应按同一 query 计算差分之差，而不是只比较两个独立
百分比。每个预算点都报告 300 个配对观测；预算点之间有意保持配对，不把它们
误写成相互独立的新场景。

Speed 轨道至少报告：

- 每个预算的三条件成功数和成功率；
- `Fast−Correct` 与 `Correct−Slow` 的配对 flips 和检验；
- 上下文效应从紧预算到宽松预算的配对变化及区间；
- success-budget AUC；
- 共同成功 query 的配对到达步数。

### 8.3 预注册

评分前至少冻结：

- 主假设和主对比；
- 模型、checkpoint、normalizer；
- catalog、split 和 candidate 生成；
- 每个任务、条件和种子的样本量；
- planner 全部资源参数；
- 主指标、聚合方式、阈值和统计方法；
- 停止规则、正对照和失败处理；
- 允许与禁止的结论。

预检只用于验证 runner 和正对照，不得查看主模型正式分数后调整阈值。任何预检
修订都必须记录原因，并在正式全量执行前完成。

## 9. 训练公平性

最小训练矩阵包括：

| 模型 | 作用 |
|---|---|
| Original-only | 原始能力基线 |
| Synthetic-only matched | 合成数据独立能力 |
| Original + matched synthetic | 混训与能力保持 |
| Original + factor-diverse synthetic | 因子多样性训练 |

训练比较至少匹配或明确报告：

- 模型结构、初始化和训练种子；
- optimizer、学习率和优化步数；
- batch mixture 与有效曝光量；
- normalizer；
- geometry、factor 和采集策略；
- loader 与 checkpoint 选择规则。

如果同时改变因子支持、geometry、loader 和预算，结论只能归属于完整配方。正式
训练层归因需要多个训练种子，不能按 validation 最优种子挑模型。

## 10. 标准报告

### 10.1 主结果表

每个正式对比至少包含：

| 内容 | 字段 |
|---|---|
| 条件 | none/correct/wrong-lower/wrong-higher |
| 数量 | 每 Eval、条件、评测种子的计数 |
| 预测 | 逐 horizon 与聚合误差 |
| 候选 | rank、top-k、argmin、真实代价 |
| 成功 | 成功数、百分比、配对 flips |
| 轨迹 | final/best distance、progress、AUC |
| 效率 | paired steps、path efficiency、compute |
| 资源 | horizon、search budget、execution budget、radius |
| 统计 | 效应量、区间或 p 值 |
| 审计 | catalog、checkpoint、协议和代码哈希 |
| 证据级别 | Calibration、Validation 或 Test |

Speed 轨道的首页结果表按固定顺序呈现：

1. 三条件多步预测误差；
2. 三条件固定候选排序与真实代价；
3. 紧/标准/宽松预算成功率曲线；
4. 最终距离、best distance 和 trajectory AUC；
5. 共同成功 query 的配对到达步数；
6. 原始能力保持结果。

不允许把多个指标或多个预算简单平均成一个无法解释的 “ICL score”。

### 10.2 进入正式 Test 的门

正式 Test 前必须同时满足：

- 数据全量回放通过，关键 split 无泄漏；
- 上下文可辨识性通过；
- 精确动力学正对照有效；
- Validation 中存在非地板、非天花板信号；
- planner 资源敏感性已经摸清并冻结；
- 原始能力满足冻结的保持标准；
- 主模型不按最好训练种子选择；
- 模型、planner、指标、阈值和 Test catalog 全部冻结。

Test 只运行一次，成功与失败都完整发布。

## 11. TwoRoom 对本规范的验证

TwoRoom 速度实验给出六个直接经验：

1. 旧 E4 的地板与天花板说明距离和任务难度必须校准。
2. 双向错误上下文区分“读取因子”和“正确估计因子”。
3. 四模型控制区分多速度训练效应与原始模型或统一 CEM 的一般响应。
4. 正确上下文预测更准，但 Fast 的有限预算成功率更高，证明 prediction error
   和 endpoint score 不能合并。
5. 执行预算从 50 增至 100 后 Fast−Correct 基本消失，证明 success 依赖
   deadline。
6. Fast 在共同成功任务上没有更快到达，证明 success rate 和 execution
   efficiency 也必须分开。

TwoRoom 的当前结果见
TwoRoom 的当前实现、阶段结果和后续工作统一见
[速度上下文学习 Benchmark 报告](TwoRoom_Speed_Benchmark_Report.md)。
