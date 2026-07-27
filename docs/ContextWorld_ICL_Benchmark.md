# ContextWorld：视觉世界模型的上下文规则学习基准

ContextWorld 用于评测视觉世界模型能否从短期交互历史中识别当前环境的隐藏规则，并在
不更新模型参数的情况下，将该规则用于未来预测。

v1 首先在 TwoRoom 环境中发布两项 History=3 任务：

| 任务 | 隐藏规则 | 主要评测目标 |
|---|---|---|
| Speed | 智能体执行动作时移动多快 | 根据历史校准多步运动预测 |
| Door Rule | 外观相同的门能否通过 | 根据历史选择正确的穿门结果 |

本版本包含训练数据、离线 Validation、评分工具和参考结果。隐藏 Test 不包含在发布包
中。

## 1. 基准概览

### 1.1 研究问题

传统视觉预测可以直接利用当前画面中的物体、位置和几何信息。ContextWorld 关注另一类
问题：当前画面不足以确定未来，模型必须结合刚刚发生的交互才能识别环境规则。

以门任务为例，相同的门洞在一种环境中可以通过，在另一种环境中会阻挡智能体。当前
画面和待执行动作完全相同，区别只存在于历史中。模型只有在看见之前的穿门结果后，
才能正确预测下一次交互。

这里的“上下文学习”特指推理时利用历史信息。评测过程中不进行梯度更新，也不修改
checkpoint。

### 1.2 History=3

模型输入由三张连续 RGB 画面及其动作组成：

```text
观察 x0 --动作 u0--> 观察 x1 --动作 u1--> 当前观察 x2
```

模型看不到环境参数、坐标、规则标签或真实未来。所有用于评分的隐藏状态只保存在数据
构建与审计记录中，不进入模型输入。

### 1.3 共同设计原则

两项任务遵守同一套实验原则：

- 固定当前画面和未来动作，只改变历史中展示的环境规则；
- 提前运行环境并保存真实未来，模型评分阶段不再调用环境；
- 每个 Eval 条件使用 `50 个 query × 6 个 Eval seed = 300` 个样本；
- 单个 checkpoint 只产生描述性结果；
- 训练方法的正式结果需要三个训练种子全部通过；
- 预测准确性与 CEM 规划成功率分开报告。

## 2. 任务定义

### 2.1 Speed：从历史识别移动速度

Speed 任务的当前画面不包含速度字段。模型需要根据相同动作在历史中造成的画面位移，
判断当前环境移动得较慢还是较快。

评测假设短期历史与待预测未来来自同一个稳定环境。对于每个参考速度，数据同时提供：

- 与参考速度相同的历史；
- 比参考速度慢的历史；
- 比参考速度快的历史。

范围外评测使用四档历史速度。历史条件只是受控比较，不向模型提供“正确”或“错误”
标签。

模型从真实三帧历史开始，连续预测第 1、2、3 和 5 个 action block 的未来。每个
action block 对应 5 个原始环境步，中间不会用真实帧替换模型预测。

### 2.2 Door Rule：从历史识别门的通行规则

Door Rule 任务包含两种隐藏规则：

| 规则 | 执行穿门动作后的真实结果 |
|---|---|
| Passable | 智能体穿过墙 |
| Blocked | 智能体停在墙边 |

两种规则使用相同的门外观、当前画面和穿门动作。评测为同一个 query 提供三种历史：

| 历史条件 | 历史中发生的交互 | 在评分中的作用 |
|---|---|---|
| Observed passable | 智能体成功穿过门 | 应支持 Passable 预测 |
| Observed blocked | 智能体撞门后停下 | 应支持 Blocked 预测 |
| No attempt | 智能体没有尝试穿门 | 仅用于观察默认倾向 |

`No attempt` 没有展示门规则，因此没有唯一正确答案，不参与能力通过判定。

Door Rule 与可见门位置泛化不同。门的位置可以从当前画面中直接看到；本任务隐藏的是
门的物理通行规则。

## 3. 数据集

### 3.1 发布包

完整数据包约 25 GiB，顶层结构为：

```text
ContextWorld-ICL-Benchmark-v1/
├── README.md
└── benchmark/
    ├── suite.yaml
    ├── inventory.json
    ├── releases/
    │   ├── speed.yaml
    │   └── door.yaml
    ├── synthesis/
    ├── evaluation/
    ├── splits/
    ├── training/
    └── upstream/
        └── lewm-tworooms/
            └── tworoom.h5
```

`README.md` 是本说明文档。`benchmark/` 是统一数据根目录，其中的 YAML 和 JSON
记录数据身份、评分规则和文件哈希，由工具自动读取。

### 3.2 Speed 训练数据

| 数据集 | 内容 | 大小 |
|---|---|---:|
| Original TwoRoom | 原始 `agent.speed=5` 数据 | 12,775,849,984 bytes |
| Single-speed control | 合成速度固定为 5 | 6,405,310,504 bytes |
| Multi-speed target | 32 个合成速度，范围 2.6～7.9 | 6,478,493,614 bytes |

正式训练对照为：

```text
Single-speed control = 50% Original + 50% Single-speed
Multi-speed target   = 50% Original + 50% Multi-speed
```

两组使用相同的场景请求、样本量、模型结构、normalizer、优化器、训练步数和
checkpoint 选择方式。两组之间唯一有意改变的是合成数据中是否包含速度变化。

### 3.3 Speed Validation

| 评测轨道 | 参考速度 | 与 Multi-speed 训练集的关系 |
|---|---|---|
| Seen | 3.1 / 5.1 / 7.0 | 训练中精确出现 |
| Interpolation | 3.4 / 4.8 / 6.9 | 训练中未出现，但位于 2.6～7.9 内 |
| Extrapolation low | 1.75 / 1.95 / 2.15 / 2.35 | 低于训练范围 |
| Extrapolation high | 8.25 / 8.75 / 9.50 / 10.25 | 高于训练范围 |

离线预测集包含 4,200 个 query payload。每个 payload 保存 History=3 输入、未来动作
以及 h1、h2、h3、h5 的真实未来画面。

发布包还提供固定候选动作和 CEM 规划数据，用于分析预测变化是否改善动作排序与闭环
任务表现。规划结果属于支持性指标，不替代真实未来预测。

### 3.4 Door Rule 训练数据

Door Rule 数据由 TwoRoom 环境真实执行动作生成，不使用图像生成模型。每个场景成对
生成 Passable 和 Blocked 两条轨迹；一对轨迹的门位置、起点、动作和采样设置相同，
只改变隐藏通行规则。

| 数据划分 | 门位置 | Clip 数量 |
|---|---:|---:|
| Training | 96 | 每种规则 7,680；合计 15,360 |
| Loader Validation | 16 | 每种规则 1,280；合计 2,560 |
| Benchmark Validation | 42 | 300 个独立 query |

门规则模型从同一个原始 History=3 LeWM checkpoint 初始化，并沿用原始 normalizer。
门规则续训只采样成对合成数据，不混入 Original TwoRoom 样本。

### 3.5 Door Rule Validation

Validation 使用训练中完全未出现的 42 个门位置。六个 Eval seed 各包含 50 个 query，
左右两个穿门方向各占 25 个。

每个 query 保存：

- 三种 History=3 输入；
- 同一个待执行穿门动作；
- Passable 和 Blocked 两张真实下一帧。

因此，每个 checkpoint 共运行 900 次模型预测并计算 1,800 条目标 loss。评分阶段的
在线环境调用数为 0，也不使用 CEM。

### 3.6 数据隔离与完整性

发布审计检查：

- Training、Loader Validation 与 Benchmark Validation 的场景和门位置隔离；
- query 图像、模板 ID 和 payload 哈希不与训练数据重合；
- 配对轨迹除目标隐藏规则外保持一致；
- 速度参数的环境 readback 与真实未来一致；
- payload、catalog、normalizer、checkpoint 和代码文件的 SHA-256；
- 所有 Eval seed 和每个条件的样本数。

## 4. 评测方法

### 4.1 真实未来 latent 误差

每个 checkpoint 使用自己的冻结 Encoder 编码真实未来：

```text
L_h = MSE(
    模型自回归得到的预测 latent_h,
    冻结 Encoder 编码的真实未来画面_h
)
```

不同 checkpoint 的 latent 尺度可能不同，因此原始 MSE 只用于同一个 checkpoint
内部的配对比较，不能直接用于跨模型排名。

### 4.2 Speed 指标

| 指标 | 计算方式 | 解释 |
|---|---|---|
| Loss ratio | 同速度历史 loss ÷ 其他历史平均 loss | 越低越好；1 表示没有平均优势 |
| Query win rate | 同速度历史优于其他历史平均值的 query 比例 | 衡量优势覆盖多少样本 |
| Strict win rate | 同速度历史同时优于每一种其他历史的 query 比例 | 更严格的稳定性指标 |

四条速度轨道和四个预测 horizon 分开报告，不合并成一个总分。

### 4.3 Door Rule 指标

| 指标 | 计算方式 | 解释 |
|---|---|---|
| Next-frame choice accuracy | 有规则证据的历史是否选择正确真实下一帧 | 模型是否判断对门规则 |
| History-guidance accuracy | 正确历史是否比相反历史带来更低 loss | 历史是否把预测推向正确结果 |

正式通过还要求两个方向、六个 Eval seed 和 bootstrap 置信区间均满足冻结门槛，且两张
真实下一帧在该 checkpoint 的 latent 中可以区分。

### 4.4 方法级结果

单个 checkpoint 的通过结果只描述该模型。要报告一种训练方法稳定获得能力，需要训练
种子 `3072、4096、5120` 的三个 checkpoint 全部通过。

Speed 的训练归因还要求 Multi-speed target 在三个配对种子上稳定优于
Single-speed control。

## 5. 参考结果

### 5.1 Speed

下表报告 Multi-speed target 的 `loss ratio / strict win rate`：

| 评测轨道 | h1 | h5 | 判定 |
|---|---:|---:|---|
| Seen | 0.104× / 97.0% | 0.328× / 81.3% | h1/h2/h3/h5 全部通过 |
| Interpolation | 0.114× / 95.3% | 0.348× / 79.7% | h1/h2/h3/h5 全部通过 |
| Extrapolation low | 0.998× / 26.2% | 0.995× / 25.9% | 全部未通过 |
| Extrapolation high | 1.062× / 25.2% | 0.989× / 32.6% | 全部未通过 |

Multi-speed target 在训练已见速度和区间内未见速度上均通过最长 h5 评测；低端和高端
范围外速度均未通过。当前证据支持训练范围内的速度适应与插值，不支持范围外外推。

### 5.2 Door Rule

| 模型与训练方式 | 下一帧判断正确率 | 历史引导正确率 | 通过种子 |
|---|---:|---:|---:|
| Original History=3 LeWM | 50.00% | 51.33% | 0/1 |
| LeWM joint training | 50.33% | 53.83% | 0/3 |
| LeWM frozen representation | 100.00% | 100.00% | 3/3 |
| PLDM joint training | 99.33% | 99.67% | 3/3 |
| PLDM frozen representation | 100.00% | 100.00% | 3/3 |

`Frozen representation` 从原始 LeWM checkpoint 初始化，并冻结其 Encoder 和
Projector；它不是随机初始化 Encoder。

PLDM joint training 在相同初始化、数据和训练预算下 3/3 通过，说明 Door Rule 数据
与评测能够支持端到端学习。LeWM joint training 的失败属于模型训练问题，不构成
Benchmark 数据无效的证据。

## 6. 安装与数据准备

### 6.1 安装代码

```bash
git clone https://github.com/Anguo-star/ContextWorld.git
git clone https://github.com/galilai-group/stable-worldmodel ../stable-worldmodel
git -C ../stable-worldmodel checkout 5864b74980f6ed328fd0045e777b3865962eff43
pip install -e .
```

### 6.2 配置数据路径

解压数据包后：

```bash
export CONTEXTWORLD_BENCHMARK=/path/to/ContextWorld-ICL-Benchmark-v1
export CONTEXTWORLD_ARTIFACT_ROOT=$CONTEXTWORLD_BENCHMARK/benchmark
export CONTEXTWORLD_TWOROOM_H5=$CONTEXTWORLD_BENCHMARK/benchmark/upstream/lewm-tworooms/tworoom.h5
```

### 6.3 验证数据

完整审计会重新计算训练数据树、原始 H5 和全部 Eval payload 的哈希：

```bash
contextworld-benchmark \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/suite.yaml \
  audit --full
```

输出 `passed=true` 后，数据包才可用于正式 Validation。

只检查其中一个任务：

```bash
contextworld-benchmark \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/suite.yaml \
  audit --component speed

contextworld-benchmark \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/suite.yaml \
  audit --component door
```

## 7. 运行评测

### 7.1 Speed

检查冻结训练配方：

```bash
contextworld-speed \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/releases/speed.yaml \
  train-plan \
  --recipe multi_speed_target \
  --seed 3072
```

评测一个 LeWM checkpoint：

```bash
contextworld-speed \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/releases/speed.yaml \
  eval \
  --checkpoint /path/to/weights.pt \
  --model-name my-speed-model \
  --training-role multi_speed_target \
  --training-seed 3072 \
  --output speed-s3072.json
```

完整方法报告需要三个 Multi-speed target 结果和三个同种子的 Single-speed control
结果，并使用 `contextworld-speed aggregate` 汇总。

### 7.2 Door Rule

查看参考训练命令：

```bash
contextworld-door \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/releases/door.yaml \
  train-plan \
  --recipe pldm_joint \
  --training-seed 3072
```

评测一个 PLDM 或 LeWM checkpoint：

```bash
contextworld-door \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/releases/door.yaml \
  eval \
  --adapter pldm \
  --checkpoint /path/to/weights.pt \
  --model-name my-door-model \
  --training-recipe my-method \
  --training-seed 3072 \
  --output door-s3072.json
```

三个训练种子分别评测后，使用 `contextworld-door score` 重新计算并汇总方法结果。

## 8. 结果报告规范

公开结果应同时给出：

- ContextWorld release ID；
- component release ID；
- checkpoint SHA-256；
- 模型 adapter 与训练配方；
- 训练种子和 Eval seed；
- 每条轨道、每个 horizon 的独立结果；
- 单 checkpoint 结果或三种子方法结果；
- 是否运行完整 `50×6`；
- 是否包含固定候选或 CEM 支持性评测。

不应：

- 将原始 latent MSE 跨 checkpoint 直接比较；
- 将 loss ratio 称为准确率；
- 将有限预算下的 CEM 成功率单独解释为上下文学习；
- 将 Validation 结果描述为隐藏 Test 排行榜；
- 将 Speed 的区间内插值扩大为任意速度外推；
- 将 Door 的一步预测扩大为长距离导航能力。

## 9. 复现范围与版本管理

统一发布由 `benchmark/suite.yaml` 固定代码文件、组件 release、公开文档和数据清单。
`contextworld-benchmark audit` 会检查代码与数据是否属于同一版本。

v1 的结论范围为：

- TwoRoom；
- History=3；
- Speed 的 h1/h2/h3/h5 离线真实未来；
- Door Rule 的一步离线真实未来；
- Validation 数据；
- Stable-WorldModel LeWM，以及 Door 的 PLDM 参考实现。

v1 不包含隐藏 Test、History>3、Door 长距离规划或多因素组合任务。

当前代码和数据已形成可完整审计的 Validation release candidate。正式公共分发还需要
在代码仓中声明源码与自产数据许可证，并配置稳定的公开下载地址。

## 10. 扩展新的 ContextWorld 任务

新增任务作为新的 component 注册到统一 suite，不建立另一套数据根或并列总文档。
每个 component 需要提供：

1. 冻结的任务与数据清单；
2. Training、Validation 和隔离审计；
3. 模型 adapter；
4. 评分器与方法级门槛；
5. 命令行入口；
6. 本文中的任务、数据、指标和结果章节。

Action Delay、更长 History 和组合因素可以沿用这一结构加入后续版本，同时保持统一的
安装方式、数据根和结果格式。
