# ContextWorld 实施计划 v0.2（已归档）

> 本文是项目启动阶段的实施计划快照；当前设计以
> [ContextWorld Benchmark 设计规范](../ContextWorld_Benchmark_Design.md) 为准。
## 第一阶段：基于上下文的新环境下一状态预测

**日期**：2026-07-13  
**当前实施范围**：TwoRoom，状态输入，隐藏动力学参数，冻结参数的上下文条件预测  
**后续环境**：Reacher → PushT → OGBench-Cube  
**基础框架**：Stable-WorldModel；后续像素模型以 LeWorldModel 为主要起点

---

## 1. 项目目标

本项目首先研究预测型世界模型，而不是直接生成动作的 policy 模型。

给定一个环境实例中的少量历史转移：

\[
C_B=\{(s_i,a_i,s_{i+1})\}_{i=1}^{B},
\]

模型需要利用这些上下文，对同一环境实例中的新查询进行预测：

\[
\hat{s}_{t+1}=F_\theta(s_t,a_t,C_B).
\]

测试时模型参数保持冻结。环境的隐藏参数不会直接输入模型，只能通过上下文转移推断。

第一阶段关注的问题是：

> 当测试环境具有训练期间未出现过的动力学参数时，模型能否根据来自该环境的少量状态—动作—下一状态转移，调整其下一状态预测？

本阶段不包含：

- 直接生成动作的 policy ICL；
- 强化学习或奖励建模；
- MPC/CEM 规划；
- 像素编码器训练；
- 测试时梯度更新；
- episode 中途切换环境参数；
- 多环境家族共用一个 checkpoint。

规划将在世界模型预测能力稳定后作为后续下游验证加入。

---

## 2. 分阶段路线

### 阶段一：TwoRoom 状态空间最小闭环

目标：

1. 建立 train、validation 和 test 环境实例；
2. 每个环境实例拥有独立的 support episodes 和 query episodes；
3. 隐藏参数只通过 support transitions 暴露；
4. 验证上下文能够改善未见参数上的下一状态预测；
5. 建立可复用的数据 manifest、dataset loader、评测脚本和测试。

阶段一分成两个连续执行批次：

- **批次 A：数据生成与解析验证**
- **批次 B：神经网络 baseline 与上下文预测实验**

Codex 当前优先执行批次 A。批次 A 完成并通过自动检查后，再执行批次 B。

### 阶段二：Reacher 状态空间

沿用阶段一完全相同的数据协议，隐藏参数改为机械臂密度等连续动力学因素。该阶段用于确认上下文预测能力不只存在于简单线性系统中。

### 阶段三：像素潜在世界模型

在相同环境 manifest 和 support/query 协议上：

1. 用像素观测替代 privileged state；
2. 使用 LeWM encoder 和 action-conditioned latent predictor；
3. 比较无上下文、显式上下文和 oracle factor；
4. 增加多步 latent rollout 与 privileged physical-state scorer。

### 阶段四：PushT 与 Cube

在接触动力学和三维机器人操作中扩展同一协议，并逐步加入动作映射、摩擦、质量、几何和相机变化。

---

# 第一阶段详细设计

## 3. 第一阶段目标

第一阶段使用 TwoRoom，并只改变一个 episode-level 隐藏参数：

\[
v_e=\text{agent speed of environment instance }e.
\]

同一个环境实例中的所有 support 和 query episodes 使用相同速度 \(v_e\)，但具有不同的：

- episode seed；
- 初始位置；
- 动作序列。

模型不能直接读取 \(v_e\)。

在无碰撞的安全区域内，环境转移满足：

\[
s_{t+1}=s_t+v_e a_t.
\]

阶段一不是最终 benchmark，而是整个上下文预测协议的最小可运行版本。它用于验证：

- 数据是否正确表达“同环境上下文”；
- support 和 query 是否严格分离；
- action 与 transition 是否正确对齐；
- 测试参数是否与训练参数隔离；
- 模型是否能够使用 context，而不是只学习训练环境的平均动力学。

---

## 4. 环境设置

### 4.1 环境

```text
swm/TwoRoom-v1
```

### 4.2 输入与预测目标

模型使用：

```text
state[t]      : agent 二维位置，shape=(2,)
action[t]     : agent command，shape=(2,)
state[t+1]    : 下一时刻 agent 二维位置，shape=(2,)
```

第一阶段不使用：

```text
pixels
goal
reward
target position
door position
environment speed
```

其中，`environment speed` 只允许：

- 写入 manifest；
- 用于数据完整性检查；
- 输入 OracleSpeed baseline。

普通模型和上下文模型不能读取该字段。

### 4.3 固定环境结构

阶段一保持以下设置不变：

| 参数 | 设置 |
|---|---|
| wall axis | vertical |
| wall thickness | 10 |
| number of doors | 1 |
| door center | 112 |
| door half-size | 21 |
| agent radius | 7 |
| target position | 固定，不参与建模 |
| episode length | 64 environment steps |
| observation | state only |
| termination | 数据采集时忽略任务成功终止 |

### 4.4 安全运动区域

为了让第一阶段只验证隐藏速度推断，数据采集限制 agent 在墙左侧的无碰撞区域：

```text
x ∈ [28, 92]
y ∈ [28, 196]
```

采集策略在接近边界时反转相应动作分量，避免：

- 撞外墙；
- 撞中间墙；
- 进入门口；
- 因碰撞导致位移不再等于 `speed × action`。

验证脚本必须确认所有已保存 transition 满足安全区域和动力学残差要求。

---

## 5. 环境实例与数据划分

一个环境实例由一个固定速度定义：

```text
environment instance e = {speed: v_e}
```

训练集、验证集和测试集在环境实例层面划分，而不是随机拆分 transition。

### 5.1 参数范围

TwoRoom 当前 speed 的合法范围为：

```text
[1.75, 10.5]
```

本阶段使用：

| Split | 速度范围 | 环境实例数 | 用途 |
|---|---:|---:|---|
| train | `[2.5, 8.0]` | 32 | 模型训练 |
| validation_interp | `[2.5, 8.0]` | 8 | 超参数和早停 |
| test_interp | `[2.5, 8.0]` | 16 | 未见参数的范围内插值 |
| test_extrap_low | `[1.75, 2.4]` | 4 | 低端外推 |
| test_extrap_high | `[8.2, 10.25]` | 4 | 高端外推 |

train、validation 和 test 的速度值必须互不相同。

### 5.2 速度生成规则

使用确定性 manifest generator：

```python
MASTER_SEED = 20260713
TRAIN_RANGE = (2.5, 8.0)
MIN_SPEED_GAP = 0.08
```

生成方式：

1. `train`：从 `[2.5, 8.0]` 均匀分布采样 32 个值；
2. `validation_interp`：使用独立 RNG 采样 8 个值；
3. `test_interp`：使用独立 RNG 采样 16 个值；
4. 新值与所有已生成值的绝对差必须大于 `0.08`；
5. 外推值固定为：

```python
TEST_EXTRAP_LOW = [1.75, 1.95, 2.15, 2.35]
TEST_EXTRAP_HIGH = [8.25, 8.75, 9.50, 10.25]
```

manifest 中保存不少于 8 位小数的原始 speed。环境 reset 时使用 `float32`。

### 5.3 episode 数量

#### Full 数据

| Split | support episodes / env | query episodes / env | steps / episode |
|---|---:|---:|---:|
| train | 8 | 16 | 64 |
| validation_interp | 8 | 8 | 64 |
| test_interp | 8 | 8 | 64 |
| test_extrap_low | 8 | 8 | 64 |
| test_extrap_high | 8 | 8 | 64 |

总量约为：

```text
train:
32 × (8 + 16) × 64 = 49,152 rows

validation:
8 × (8 + 8) × 64 = 8,192 rows

test interpolation:
16 × (8 + 8) × 64 = 16,384 rows

test extrapolation:
8 × (8 + 8) × 64 = 8,192 rows
```

合计约 81,920 rows，不包含数据格式内部元数据。

#### Smoke 数据

Codex 首次运行先生成 smoke 数据：

| Split | env 数 | support episodes | query episodes | steps |
|---|---:|---:|---:|---:|
| train | 2 | 2 | 2 | 16 |
| validation_interp | 1 | 2 | 2 | 16 |
| test_interp | 1 | 2 | 2 | 16 |
| test_extrap_low | 1 | 2 | 2 | 16 |
| test_extrap_high | 1 | 2 | 2 | 16 |

smoke 全部验证通过后，再执行 full collection。

---

## 6. support 与 query 的定义

每个环境实例分别生成两个数据文件：

```text
support.lance
query.lance
```

二者必须满足：

- speed 完全相同；
- episode seeds 不重合；
- policy seeds 不重合；
- 初始状态不重合；
- 动作序列不重合。

### 6.1 support 数据

support 数据用于构造：

\[
C_B=\{(s_i,a_i,s_{i+1})\}_{i=1}^{B}.
\]

上下文长度：

```python
CONTEXT_SIZES = [0, 1, 2, 4, 8, 16]
```

训练时从 support episodes 中随机选一个 episode，再随机选择一个连续片段。

评测时：

- 固定 context sampling seed；
- 每个环境、每个 \(B\) 重复采样 20 个 context；
- 对 20 次结果取平均；
- 所有模型使用同一批 context indices。

### 6.2 query 数据

query 数据提供：

```text
query input  = (state[t], action[t])
query target = state[t+1]
```

query episode 永远不能同时作为 support episode。

### 6.3 transition 构造

Stable-WorldModel 保存的是 episode 行序列。dataset loader 必须显式构造：

```python
for t in range(episode_length - 1):
    state_t = state[t]
    action_t = action[t]
    state_tp1 = state[t + 1]
```

最后一行没有下一状态，因此不构成训练样本。

验证脚本必须检查 action 对齐：

\[
\left\|(s_{t+1}-s_t)-v_ea_t\right\|_2 < \epsilon.
\]

默认：

```python
ALIGNMENT_TOLERANCE = 1e-4
```

若当前 Stable-WorldModel 版本的数据 action alignment 与上述约定不同，解析器以验证结果为准进行修正，并在数据摘要中记录最终约定。

---

## 7. 采集策略

### 7.1 SafeExcitationPolicy

实现一个状态反馈的随机激励策略：

```text
class SafeExcitationPolicy(BasePolicy)
```

策略行为：

1. 每 `hold_steps=4` 步重新采样动作；
2. 动作方向从单位圆均匀采样；
3. 动作幅值从 `[0.25, 0.75]` 均匀采样；
4. 接近安全区域边界时，反转指向边界外的动作分量；
5. 最终 action clip 到 `[-1, 1]`；
6. 每个 vectorized env 使用独立 RNG；
7. seed 可复现。

建议参数：

```yaml
policy:
  hold_steps: 4
  min_magnitude: 0.25
  max_magnitude: 0.75
  safe_x: [28.0, 92.0]
  safe_y: [28.0, 196.0]
  boundary_margin: 8.0
```

### 7.2 SafeStartWrapper

实现：

```text
class SafeStartWrapper(gym.Wrapper)
```

每次 reset 时，根据 reset seed 在安全区域内采样初始位置，并写入：

```python
options["state"] = sampled_position
```

### 7.3 FixedHorizonDynamicsWrapper

实现：

```text
class FixedHorizonDynamicsWrapper(gym.Wrapper)
```

用于数据采集时忽略 goal success termination：

```python
terminated = False
```

episode 长度由 `max_episode_steps=64` 控制。

### 7.4 采集模式

阶段一使用：

```python
add_pixels=False
num_envs=8
format="lance"
```

示意：

```python
world = swm.World(
    "swm/TwoRoom-v1",
    num_envs=8,
    max_episode_steps=64,
    add_pixels=False,
    pre_wrappers=[
        SafeStartWrapper,
        FixedHorizonDynamicsWrapper,
    ],
)
```

每个 speed 单独创建并关闭 `World`，或在确认 reset options 不会跨实例污染后复用。优先选择实现更简单、可验证性更高的方式。

reset options 至少包含：

```python
options = {
    "variation": [
        "agent.speed",
        "target.position",
        "wall.axis",
        "wall.thickness",
        "door.number",
        "door.size",
        "door.position",
    ],
    "variation_values": {
        "agent.speed": np.array([speed], dtype=np.float32),
        "target.position": np.array([180.0, 180.0], dtype=np.float32),
        "wall.axis": 1,
        "wall.thickness": 10,
        "door.number": 1,
        "door.size": np.array([21, 21, 21]),
        "door.position": np.array([112, 112, 112]),
    },
}
```

具体数组 shape 以 Stable-WorldModel 当前 variation space 要求为准，并由单元测试确认。

---

## 8. 文件组织

建议新建独立模块，不直接把实验代码混入原始 LeWM 模型文件。

```text
contextworld/
├── README.md
├── pyproject.toml
├── configs/
│   └── phase1_tworoom_speed.yaml
├── contextworld/
│   ├── __init__.py
│   └── phase1/
│       ├── __init__.py
│       ├── manifest.py
│       ├── wrappers.py
│       ├── policy.py
│       ├── collect.py
│       ├── dataset.py
│       ├── validate.py
│       ├── analytic.py
│       └── metrics.py
├── scripts/
│   ├── phase1_generate_manifest.py
│   ├── phase1_collect.py
│   ├── phase1_validate.py
│   └── phase1_analytic_baselines.py
├── tests/
│   ├── test_phase1_manifest.py
│   ├── test_phase1_wrappers.py
│   ├── test_phase1_policy.py
│   ├── test_phase1_alignment.py
│   └── test_phase1_dataset.py
└── artifacts/
    └── phase1/
        ├── manifests/
        ├── data/
        ├── reports/
        └── plots/
```

数据目录：

```text
artifacts/phase1/data/
├── train/
│   ├── env_000/
│   │   ├── support.lance
│   │   └── query.lance
│   ├── env_001/
│   └── ...
├── validation_interp/
├── test_interp/
├── test_extrap_low/
└── test_extrap_high/
```

---

## 9. Manifest 格式

使用 JSON Lines。每行表示一个环境实例。

```json
{
  "version": "tworoom-speed-v1",
  "env_id": "train_env_000",
  "split": "train",
  "speed": 4.73182964,
  "support_path": "train/env_000/support.lance",
  "query_path": "train/env_000/query.lance",
  "support_episodes": 8,
  "query_episodes": 16,
  "episode_steps": 64,
  "support_seed_start": 100000,
  "query_seed_start": 200000,
  "policy_seed_offset": 500000
}
```

manifest generator 同时输出：

```text
manifest.jsonl
manifest_summary.json
```

`manifest_summary.json` 至少包含：

- version；
- generator seed；
- split ranges；
- speed values；
- minimum cross-split speed distance；
- episode counts；
- estimated transition counts；
- creation timestamp；
- git commit，如当前目录是 git repository。

---

## 10. 数据字段

每个 Lance episode 至少应包含：

```text
state
action
variation.agent.speed
step_idx
id
terminated
truncated
```

推荐同时保留：

```text
observation
goal_state
distance_to_target
```

但阶段一 dataloader 默认只读取：

```text
state
action
```

privileged speed 只能由：

```text
manifest["speed"]
variation.agent.speed
```

读取，用于校验和 OracleSpeed。

---

## 11. 数据验证

### 11.1 Manifest 验证

检查：

- env_id 唯一；
- path 唯一；
- split 合法；
- speed 在指定范围；
- train、validation、test speed 无重合；
- 任意跨 split speed gap ≥ 0.08；
- support/query seed 范围不重合；
- episode 数和 steps 合法。

### 11.2 文件验证

检查：

- 所有 manifest path 存在；
- episode 数与 manifest 一致；
- episode 长度一致；
- 必需字段存在；
- 无 NaN 或 Inf；
- state shape 为 `(2,)`；
- action shape 为 `(2,)`；
- speed 在一个文件内保持不变。

### 11.3 轨迹验证

检查：

- 所有 state 落在安全区域；
- action 位于 `[-1,1]`；
- action magnitude 满足策略约束；
- support/query episode ID 不重合；
- 不存在完全重复的 action sequence；
- 不存在完全重复的 state trajectory。

### 11.4 动力学与 action alignment 验证

对每条可用 transition：

```python
delta = state[t + 1] - state[t]
expected = speed * action[t]
residual = np.linalg.norm(delta - expected)
```

输出：

- mean residual；
- median residual；
- p95 residual；
- max residual；
- residual > tolerance 的比例。

Full 数据目标：

```text
p95 residual <= 1e-4
invalid transition ratio = 0
```

若浮点实现使 `1e-4` 过严，可调整到 `1e-3`，但必须在报告中记录实际数值。

### 11.5 数据摘要

验证脚本输出：

```text
artifacts/phase1/reports/data_validation.json
artifacts/phase1/reports/data_validation.md
artifacts/phase1/plots/speed_histogram.png
artifacts/phase1/plots/sample_trajectories.png
artifacts/phase1/plots/dynamics_residuals.png
```

---

## 12. 批次 A 的训练前 sanity check

批次 A 不训练神经网络，但必须运行三个解析型 baseline，确认数据确实表达了上下文推断问题。

### 12.1 GlobalMeanSpeed

使用训练环境速度均值：

\[
\bar v_{\mathrm{train}}=\frac{1}{N}\sum_e v_e.
\]

预测：

\[
\hat s_{t+1}=s_t+\bar v_{\mathrm{train}}a_t.
\]

它对应不使用环境上下文的固定平均动力学。

### 12.2 OracleSpeed

使用 manifest 中真实速度：

\[
\hat s_{t+1}=s_t+v_ea_t.
\]

它用于验证数据和 action alignment。

### 12.3 ContextEstimatedSpeed

根据 support transitions 估计当前环境速度。

对每个 context transition：

\[
\Delta s_i=s_{i+1}-s_i.
\]

最小二乘估计：

\[
\hat v=\frac{\sum_i a_i^\top \Delta s_i}{\sum_i \|a_i\|_2^2+\epsilon}.
\]

由于采集轨迹被限制在无碰撞区域，该估计应接近真实速度。

评测：

```python
B = [1, 2, 4, 8, 16]
```

对每个环境和每个 \(B\)：

1. 从 support pool 随机采样 \(B\) 条 transition；
2. 估计 \(\hat v\)；
3. 在全部 query transitions 上预测；
4. 重复 20 次；
5. 报告均值和标准差。

### 12.4 指标

主要指标：

```text
next_state_mse
delta_state_mse
speed_absolute_error
speed_relative_error
```

分别按以下 split 报告：

```text
validation_interp
test_interp
test_extrap_low
test_extrap_high
```

### 12.5 批次 A 完成结果

批次 A 完成后应得到：

1. 可重复生成的 manifest；
2. smoke 和 full 数据采集命令；
3. 完整数据验证报告；
4. GlobalMean、Oracle 和 ContextEstimated 三个结果；
5. context size 曲线；
6. 一份简洁实验摘要。

阶段一数据协议达到预期时，应观察到：

- OracleSpeed 的误差接近数值精度；
- ContextEstimatedSpeed 随 \(B\) 增大稳定接近 OracleSpeed；
- ContextEstimatedSpeed 在 interpolation 和 extrapolation 上均优于 GlobalMeanSpeed；
- 使用另一个环境实例的 wrong context 会使预测退化。

这些结果用于确认后续神经网络实验建立在正确的数据协议上。

---

# 批次 B：神经网络上下文预测

批次 B 在批次 A 数据通过后执行。

## 13. 模型

### 13.1 NoContextMLP

\[
\hat{\Delta s}_{t}=f_\theta(s_t,a_t).
\]

输入维度：

```text
state 2 + action 2 = 4
```

### 13.2 OracleSpeedMLP

\[
\hat{\Delta s}_{t}=f_\theta(s_t,a_t,v_e).
\]

只用于上界。

### 13.3 ContextDeepSet

每条 context transition 编码为：

\[
x_i=[s_i,a_i,s_{i+1}-s_i].
\]

transition encoder：

\[
h_i=\phi(x_i).
\]

环境 context：

\[
c_e=\frac{1}{B}\sum_{i=1}^{B}h_i.
\]

query predictor：

\[
\hat{\Delta s}_t=\rho([s_t,a_t,c_e]).
\]

建议默认结构：

```yaml
transition_encoder:
  input_dim: 6
  hidden_dims: [128, 128]
  output_dim: 128

predictor:
  input_dim: 132
  hidden_dims: [128, 128, 128]
  output_dim: 2
```

### 13.4 WrongContext 对照

模型结构与 ContextDeepSet 相同，但评测时从另一个 speed 环境采样 context。

wrong environment 满足：

```text
abs(wrong_speed - query_speed) >= 1.5
```

---

## 14. 归一化

状态：

\[
s_{\mathrm{norm}}=2s/224-1.
\]

动作已经位于：

```text
[-1,1]
```

预测目标使用 normalized delta：

\[
\Delta s_{\mathrm{norm}}=(s_{t+1}-s_t)/10.5.
\]

speed 的 oracle 输入使用训练集均值和标准差标准化。

训练集统计量保存到：

```text
artifacts/phase1/reports/normalization.json
```

validation 和 test 不重新计算统计量。

---

## 15. 训练设置

建议初始配置：

```yaml
training:
  optimizer: adamw
  learning_rate: 3.0e-4
  weight_decay: 1.0e-5
  batch_size: 512
  max_steps: 30000
  validation_interval: 500
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  seeds: [0, 1, 2]

context:
  train_sizes: [0, 1, 2, 4, 8, 16]
  eval_sizes: [0, 1, 2, 4, 8, 16]
  sampling: contiguous_segment
  source: support_only
```

loss：

\[
\mathcal L=\left\|\hat{\Delta s}_{t}-\Delta s_t\right\|_2^2.
\]

同一个 training step 中：

1. 先采样环境实例；
2. 从该环境 support pool 采样 context；
3. 从独立 query episode 采样 query；
4. context 和 query 不允许来自同一个 episode；
5. \(B=0\) 时使用全零 context 和有效 mask。

---

## 16. 神经网络评测

### 16.1 单步预测

报告：

```text
state MSE
delta-state MSE
MAE
```

### 16.2 多步 rollout

使用 query episode 中真实 action sequence，模型从真实初始状态开始自回归预测：

```text
H = [5, 10, 20]
```

报告：

```text
rollout state MSE at each horizon
final displacement error
```

### 16.3 Context Gain

\[
G(B)=\frac{E_{\mathrm{NoContext}}-E_{\mathrm{Context}}(B)}{E_{\mathrm{NoContext}}-E_{\mathrm{OracleSpeed}}+\epsilon}.
\]

其中 \(E\) 是误差，数值越小越好。

### 16.4 Context intervention

对同一个 query：

- matched context；
- wrong context；
- shuffled transition order；
- action-shuffled context；
- next-state-shuffled context。

第一阶段 DeepSet 对 transition 顺序不敏感，因此 `shuffled transition order` 只验证实现一致性；action 或 next-state 被打乱时应明显退化。

### 16.5 输出

```text
artifacts/phase1/reports/model_results.csv
artifacts/phase1/reports/model_results.json
artifacts/phase1/reports/model_results.md
artifacts/phase1/plots/context_curve_interp.png
artifacts/phase1/plots/context_curve_extrap.png
artifacts/phase1/plots/rollout_error.png
artifacts/phase1/plots/context_intervention.png
```

---

## 17. 阶段一完成标准

阶段一完成需要同时具备以下产物：

### 数据层

- train/validation/test 环境实例完全按 manifest 生成；
- support 和 query 跨 episode 隔离；
- action alignment 验证通过；
- 数据可以从空目录一条命令重新生成；
- smoke 和 full 两种模式均可运行；
- 单元测试通过。

### 解析型结果

- OracleSpeed 接近数值精度；
- ContextEstimatedSpeed 显著优于 GlobalMeanSpeed；
- context size 结果可重复；
- wrong context 的结果明显差于 matched context。

### 神经模型结果

- ContextDeepSet 在 `test_interp` 上优于 NoContextMLP；
- ContextDeepSet 在至少一个 extrapolation split 上优于 NoContextMLP；
- matched context 优于 wrong context；
- 多步 rollout 的改进方向与单步预测一致；
- 三个训练 seed 的结论一致。

阶段一结果将作为阶段二 Reacher 数据协议和模型接口的基准实现。

---

# Codex 当前执行范围

## 18. Codex 执行批次 A

### 18.1 目标

实现 TwoRoom speed-context 数据生成、验证和解析型 sanity check。

### 18.2 必须实现的文件

```text
contextworld/phase1/manifest.py
contextworld/phase1/wrappers.py
contextworld/phase1/policy.py
contextworld/phase1/collect.py
contextworld/phase1/dataset.py
contextworld/phase1/validate.py
contextworld/phase1/analytic.py
contextworld/phase1/metrics.py

scripts/phase1_generate_manifest.py
scripts/phase1_collect.py
scripts/phase1_validate.py
scripts/phase1_analytic_baselines.py

configs/phase1_tworoom_speed.yaml

tests/test_phase1_manifest.py
tests/test_phase1_wrappers.py
tests/test_phase1_policy.py
tests/test_phase1_alignment.py
tests/test_phase1_dataset.py
```

### 18.3 命令接口

#### 生成 manifest

```bash
python scripts/phase1_generate_manifest.py \
  --config configs/phase1_tworoom_speed.yaml \
  --output artifacts/phase1/manifests/tworoom_speed_v1.jsonl
```

#### 采集 smoke 数据

```bash
python scripts/phase1_collect.py \
  --config configs/phase1_tworoom_speed.yaml \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode smoke
```

#### 验证 smoke 数据

```bash
python scripts/phase1_validate.py \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode smoke \
  --output artifacts/phase1/reports/data_validation_smoke.json
```

#### 运行 smoke sanity baseline

```bash
python scripts/phase1_analytic_baselines.py \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode smoke \
  --output artifacts/phase1/reports/analytic_smoke
```

#### 采集 full 数据

```bash
python scripts/phase1_collect.py \
  --config configs/phase1_tworoom_speed.yaml \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode full
```

#### 验证 full 数据

```bash
python scripts/phase1_validate.py \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode full \
  --output artifacts/phase1/reports/data_validation_full.json
```

#### 运行 full sanity baseline

```bash
python scripts/phase1_analytic_baselines.py \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode full \
  --output artifacts/phase1/reports/analytic_full
```

#### 测试

```bash
pytest -q tests/test_phase1_*.py
```

### 18.4 配置文件最低字段

```yaml
version: tworoom-speed-v1
master_seed: 20260713

environment:
  id: swm/TwoRoom-v1
  num_envs: 8
  episode_steps: 64
  add_pixels: false
  state_key: state
  action_key: action

safe_region:
  x: [28.0, 92.0]
  y: [28.0, 196.0]
  boundary_margin: 8.0

speed_splits:
  train:
    count: 32
    range: [2.5, 8.0]
  validation_interp:
    count: 8
    range: [2.5, 8.0]
  test_interp:
    count: 16
    range: [2.5, 8.0]
  test_extrap_low:
    values: [1.75, 1.95, 2.15, 2.35]
  test_extrap_high:
    values: [8.25, 8.75, 9.50, 10.25]
  min_cross_split_gap: 0.08

episodes:
  train:
    support: 8
    query: 16
  evaluation:
    support: 8
    query: 8

smoke:
  envs:
    train: 2
    validation_interp: 1
    test_interp: 1
    test_extrap_low: 1
    test_extrap_high: 1
  support_episodes: 2
  query_episodes: 2
  episode_steps: 16

policy:
  hold_steps: 4
  min_magnitude: 0.25
  max_magnitude: 0.75

validation:
  alignment_tolerance: 1.0e-4
  require_fixed_episode_length: true

analytic:
  context_sizes: [1, 2, 4, 8, 16]
  context_repeats: 20
  epsilon: 1.0e-8
```

### 18.5 实现约束

- 不把 speed 输入普通 dataset sample；
- Oracle baseline 通过单独接口读取 speed；
- support/query 文件分开；
- 不从 query 数据抽取 context；
- 不在 train 时读取 validation/test；
- 所有随机源显式接收 seed；
- 不依赖 notebook；
- CLI 失败时返回非零退出码；
- 输出目录自动创建；
- 已存在数据默认不覆盖，除非传入 `--overwrite`；
- 所有 JSON 输出可被标准 `json` 模块读取；
- 关键函数加类型标注；
- 采集过程中正确关闭 `World`；
- 测试不得依赖 GPU；
- smoke 测试不得下载外部数据。

### 18.6 自动验收项

Codex 完成后应能运行：

```bash
pytest -q tests/test_phase1_*.py
```

并依次执行：

```bash
python scripts/phase1_generate_manifest.py \
  --config configs/phase1_tworoom_speed.yaml \
  --output artifacts/phase1/manifests/tworoom_speed_v1.jsonl

python scripts/phase1_collect.py \
  --config configs/phase1_tworoom_speed.yaml \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode smoke

python scripts/phase1_validate.py \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode smoke \
  --output artifacts/phase1/reports/data_validation_smoke.json

python scripts/phase1_analytic_baselines.py \
  --manifest artifacts/phase1/manifests/tworoom_speed_v1.jsonl \
  --data-root artifacts/phase1/data \
  --mode smoke \
  --output artifacts/phase1/reports/analytic_smoke
```

批次 A 最终交付：

```text
1. 代码
2. 测试
3. smoke manifest
4. smoke dataset
5. data_validation_smoke.json
6. analytic_smoke/metrics.csv
7. analytic_smoke/summary.md
8. 执行命令和实际输出摘要
```

---

## 19. Codex 可直接使用的任务描述

```text
请在当前仓库实现 ContextWorld Phase 1 Batch A。

研究目标：
在 stable-worldmodel 的 swm/TwoRoom-v1 环境中，生成按隐藏 agent.speed 划分的 train、validation_interp、test_interp、test_extrap_low 和 test_extrap_high 数据。每个 speed 是一个 environment instance。每个实例分别生成 support.lance 和 query.lance；support 和 query 必须使用不重合的 episode seed、初始状态和动作序列。

当前只做 state-space next-state prediction 数据，不采集 pixels，不实现神经网络，不做规划。实现 SafeStartWrapper、FixedHorizonDynamicsWrapper 和 SafeExcitationPolicy，使 agent 保持在左侧无碰撞安全区域。使用 manifest 控制所有 split、speed、路径、episode 数和 seed。

实现数据解析与验证：
1. 将 episode 行构造成 (state[t], action[t], state[t+1])；
2. 检查 action alignment；
3. 检查 delta state 是否等于 speed × action；
4. 检查 support/query 隔离；
5. 检查 split speed 无重合；
6. 输出 JSON、Markdown 和图表摘要。

实现三个 training-free baseline：
- GlobalMeanSpeed
- OracleSpeed
- ContextEstimatedSpeed

ContextEstimatedSpeed 使用 support transitions 的最小二乘估计：
v_hat = sum(a_i dot delta_s_i) / (sum(||a_i||^2) + eps)

按 B=[1,2,4,8,16] 在 validation、interpolation test 和 extrapolation test 上报告 next-state MSE、delta-state MSE、speed absolute error 和 speed relative error。每个 B 重复 20 个固定随机 context sampling。

实现本文档列出的目录、CLI、YAML 配置和 pytest。先完成 smoke 模式并运行全部验收命令。不要修改 stable-worldmodel 的核心源码；优先通过独立 wrapper、policy 和 collector 实现。若当前 stable-worldmodel 的 action 存储对齐与文档假设不一致，请通过验证脚本确定正确索引，并将最终规则写入 summary.md。
```

---

# 阶段二接口预留

## 20. Reacher 数据协议

阶段二保持以下接口不变：

```text
environment instance
support episodes
query episodes
context transitions
query transition
manifest
split-level parameter isolation
matched/wrong context evaluation
```

只替换：

```text
state representation
environment parameter sampler
data policy
dynamics metrics
```

初步隐藏参数：

```text
arm_density
finger_density
```

第一轮优先单独改变 `arm_density`，确认协议后再增加第二个因素与组合测试。

Reacher 阶段同时保留：

- state lane；
- pixel lane。

state lane 先验证系统辨识，pixel lane 再接 LeWM encoder。

---

## 21. 像素 LeWM 接口预留

阶段三中的模型输入变为：

\[
z_t=E_\xi(o_t),
\]

上下文变为：

\[
C_B^z=\{(z_i,a_i,z_{i+1})\}_{i=1}^{B},
\]

预测：

\[
\hat z_{t+1}=F_\theta(z_t,a_t,C_B^z).
\]

阶段一的数据接口应避免把模型限定为 state-only。建议 dataset sample 使用统一字典：

```python
{
    "context": {
        "state": ...,
        "action": ...,
        "next_state": ...,
        "mask": ...
    },
    "query": {
        "state": ...,
        "action": ...,
        "next_state": ...
    },
    "metadata": {
        "env_id": ...,
        "split": ...
    }
}
```

像素阶段可将 `state` 字段替换为 `embedding`，保持 trainer 和评测器的整体接口。

---

## 22. 基础依赖

建议固定 Stable-WorldModel 版本或提交，确保环境和数据行为可复现。

安装示例：

```bash
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
git checkout 0ef3856875e70a1283e637fcd2ab936eae6c4e6f

uv venv --python=3.10
source .venv/bin/activate
uv sync --extra all --group dev
```

阶段一主要依赖：

```text
stable-worldmodel
gymnasium
numpy
torch
lancedb
pyarrow
pyyaml / omegaconf
pytest
matplotlib
```

阶段一无需：

```text
MuJoCo
OGBench
GPU
LeWM checkpoint
外部训练数据
```

---

## 23. 第一阶段最终产物

```text
可重复的数据生成器
固定的 train/validation/test manifest
support/query 数据集
数据质量验证工具
解析型 context baseline
神经网络 context baseline
上下文长度曲线
插值与外推结果
wrong-context 干预结果
阶段二可复用的数据接口
```

第一阶段完成后，项目将拥有一个能够从零运行的最小基准实现。第二阶段只需要替换环境参数、状态定义和采集策略，而不需要重新设计 train/test 协议。
