# TwoRoom SpeedTask 数据与训练协议

**版本**：v1.2
**日期**：2026-07-16  
**适用模型**：`H3-SpeedTask`（LeWM，history=3）

旧机器字段 `correct` 和 `wrong` 分别表示同速历史和另一档速度历史，不表示速度
正确或错误。

## 1. 目标与比较边界

`TwoRoom-SpeedTask-v1` 是 `H3-SpeedSeen` 之后的任务分布与有效 geometry
对照。此前 SpeedSeen 已修复无损像素表示并覆盖 E4 的全部速度，但其 4,096 条
train episodes 只有 128 个独立 reset-goal pairs，且 49.22% 为不需要穿门的
同房间任务。该组成与原始 `tworoom.h5` 的 100% 跨房间语义不一致。

本版本固定以下组件：

- 32 个 synthetic train speeds，包括全部八个 E4 speed values；
- `ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)`；
- lossless PNG、最大 100 个 action steps；
- original/synthetic=`50/50`、history=3 LeWM、6,420 optimizer steps；
- StableWM commit `5864b74980f6ed328fd0045e777b3865962eff43`。

相对 SpeedSeen 只改变任务组成：全部 synthetic target 位于初始状态的另一房间，
独立 train geometries 从 128 增至 512，并让每个 geometry 与全部 32 个速度
完整交叉。因此后续 `H3-SpeedTask` 相对 `H3-SpeedSeen` 的变化用于估计任务语义
和有效 geometry 覆盖的联合影响；它不是单独的 policy-quality 因果实验。

## 2. 环境约束

ContextWorld 通过声明式 reset constraints 配置 StableWM 的原生位置 variation
space，不修改 StableWM 源码，也不改变 checkpoint 或 resume 语义。约束为：

```yaml
reset_constraints:
  target_room: opposite
  exclude_wall_zone: true
  minimum_initial_distance: 40.0
```

约束谓词与速度无关。同一个 reset seed 在不同速度下产生完全相同的 start/goal，
从而保证 geometry×speed 比较不受重新采样干扰。train 的 512 个独立 reset 中，
243 个为左到右、269 个为右到左；最小初始距离为 40.883 px，距离 p10 为
85.315 px。dev 的相应数量为 30/34，最小距离为 49.573 px。

## 3. 速度与配对设计

train 使用与 SpeedSeen 相同的 32 个速度：

```text
2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.5, 3.7,
3.8, 3.9, 4.1, 4.2, 4.4, 4.5, 4.7, 5.0,
5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.2, 6.3,
6.6, 6.7, 6.8, 7.0, 7.2, 7.3, 7.8, 7.9
```

train 采用：

```text
16 seed blocks × 32 reset seeds × 32 speeds
= 512 scenarios × 32 episodes
= 16,384 train episodes
```

dev 使用 4 个独立 seed blocks、每 block 16 个 resets，并与八个 dev speeds
`2.5/3.2/4.3/4.6/4.9/5.4/5.6/5.8` 交叉，共 32 个场景、512 条 episodes。
train 与 dev 的 reset seeds 互斥。

正式检查确认 512/512 个 train geometries 均恰好包含全部 32 个速度；每个速度
均有 512 条 episodes，没有 geometry 内的重复 speed row。14 px 网格下共有
168 个 start bins、174 个 goal bins 和 500 个 start-goal pairs。

## 4. 正式生成与重放结果

| 指标 | Train | Dev | 总计 |
|---|---:|---:|---:|
| Lance scenarios | 512 | 32 | 544 |
| Episodes | 16,384 | 512 | 16,896 |
| Rows | 1,410,372 | 45,456 | 1,455,828 |

全量 validator 重放了 1,438,932 个状态转移。state transition、goal invariance、
termination flag、存储 pixel bytes 和 PNG decoded pixels 的 mismatch 均为 0，
最大状态绝对误差为 0。16,896 个 episode starts 均满足至少生成两行数据的
安全 margin，最小实测保证 margin 为 13.711 px。正式数据约占 7.3 GB。

对应机器可读产物为：

- `artifacts/synthesis/reports/tworoom_speed_task_v1.json`；
- `artifacts/synthesis/reports/tworoom_speed_task_quality_v1.json`；
- `artifacts/synthesis/catalogs/tworoom_speed_task_v1.json`；
- `artifacts/synthesis/manifests/tworoom_speed_task_v1.jsonl`。

## 5. 轨迹组成结果

| 指标 | 原始 H5 | SpeedSeen train | SpeedTask train |
|---|---:|---:|---:|
| Episodes | 10,000 | 4,096 | 16,384 |
| 独立 reset-goal pairs | 10,000 | 128 | 512 |
| 14 px start-goal pairs | 7,019 | 128 | 500 |
| 跨房间比例 | 100.00% | 50.78% | 100.00% |
| Termination success | 40.35% | 65.14% | 45.73% |
| 平均 episode rows | 92.08 | 68.44 | 86.08 |
| 平均 final distance | 50.69 px | 35.24 px | 47.53 px |

SpeedTask 的 pooled 成功率下降不是数据退化：它来自移除容易的同房间任务，使
任务组成重新接近原始 H5。固定 speed=5、cross-room 时，SpeedTask 的 512 条
episodes 成功率为 45.12%、平均 86.93 行、平均 final distance 44.47 px；原始
H5 为 40.35%、92.08 行和 50.69 px。二者使用不同 geometry samples，因此这些
数字只说明处于同一量级，不用于声称 synthetic policy 优于原始 collector。

数据不进行 success-only filtering。32 个速度中的每一个都同时包含 termination
success 与 nontermination episodes；最少分别为 34 和 102 条。这样保留真实
失败分布，避免把是否成功泄漏成数据选择规则。

仍需明确一个未消除的性质：speed 与 termination success、平均 episode rows、
平均 final distance 的 Pearson 相关分别为 `+0.994/-0.997/-0.981`。固定 100-step
horizon 下，低速更难在预算内完成跨房间任务，这是环境与收集 horizon 的联合
结果。当前版本通过完整 geometry cross、每速度等 episode 数和 scenario-balanced
loader 控制 factor exposure，但不声称 trajectory outcome 已与 speed 统计独立。

## 6. History-3 正式训练

正式 preflight 已通过全部静态和抽样门槛：

- synthetic raw train clips：1,099,271；
- train/dev scenarios：512/32；
- pixel codec：lossless PNG；
- 5 个 logical epochs 中 synthetic draws：3,287,040；
- 平均复用率：2.990 draws/raw clip，低于上限 5；
- original/synthetic 各占 50%，总 optimizer steps 为 6,420。

preflight 位于 `artifacts/training/reports/h3_speedtask_s3072_preflight.json`。
正式训练随后按相同计划完成：4 GPU、5 个 logical epochs、每 rank 12,840 个
microbatches、6,420 optimizer/scheduler steps。StableWM 原生 final checkpoint
逐张量重载通过，SHA-256 为
`4bb6edfdf9d5c3868eb0aca2fc59b0716a2890cda9508806f0ed43b4094cb3fc`；报告位于
`artifacts/training/reports/h3_speedtask_s3072.json`。

## 7. 正式评测结果

| Gate | 结果 |
|---|---:|
| E1 K=0 latent MSE | 0.05835 |
| E1 K=2 同速历史 latent MSE | 0.04471 |
| E1 K=2 同速历史收益 | 0.01364，95% CI [-0.00292, 0.03136] |
| E1 K=2 另一档−同速历史 | 0.02080，95% CI [0.01073, 0.03106] |
| 原始 ID `50×6` | 296/300（98.67%） |
| E4 同速历史 | 73/300（24.33%） |
| E4 另一档历史 | 73/300（24.33%） |
| E4 同速-only / 另一档-only | 0 / 0 |

E1 表明模型输出能够区分同速/另一档历史，但同速历史相对 K=0 的收益 CI
仍跨 0。E4 的同速与另一档历史在六个 seeds、八个速度及 300 个 evaluation
IDs 上的 success outcomes 完全相同，没有建立 planning-level ICL。

与固定 recipe 的直接 reference `H3-SpeedSeen` 配对后，SpeedTask 的二值成功
集合在两个 conditions 下均不变；同速/另一档历史 mean final distance 却分别增加
16.95/19.71 px。227 个共同失败样本的相应增加为 22.40/26.03 px。因此本实验
不支持“恢复 cross-room task semantics 并增加到 512 geometry 足以提高 E4”。

## 8. 原始 H5 与 SpeedTask 的数据差异

进一步统计模型可见的 trajectory clips，而不只统计 episode 总数：

| 指标 | 原始 H5 | SpeedTask train |
|---|---:|---:|
| Rows | 920,809 | 1,410,372 |
| History-3 clips | 730,809 | 1,099,271 |
| 独立 reset-goal geometry | 10,000 | 512 |
| 14 px reset-goal cells | 7,019 | 500 |
| 14 px state-goal cells | 23,535 | 16,143 |
| Goal-side row fraction | 30.76% | 26.86% |

原始 H5 未保存 pre-action reset，因此其 10,000 个 geometry 按第一帧模型可见
state 与 goal 计数；SpeedTask 的 512 个 geometry 使用显式 variation reset。
这一个 action step 的时刻差异不改变覆盖量级结论。

SpeedTask 的 16,384 个 episodes 来自 512 个 geometry × 32 speeds。该交叉对
factor balancing 必要，但不能替代独立任务覆盖。首 action 的组内/全局 RMS
dispersion ratio 为 0.881，未显示 action diversity 坍缩；主要差距是 reset-goal
和条件状态覆盖。

固定 speed=5、cross-room 后，原始/SpeedTask 的 mean action norm 为
1.2023/1.2016，exact action repeat 为 4.91%/4.93%，collision residual 为
7.36%/7.17%，mean goal progress 为 0.987/0.981 px，door-center reference
efficiency 为 0.3736/0.3737。因而现有证据不支持 synthetic ExpertPolicy 本身
更弱作为主要原因。

固定总训练 exposure 下，50/50 mixture 把 original draws 从 6,574,080 降为
3,287,040，并分配 3,287,040 synthetic draws；这相当于替换一半原始监督，而
非在完整原始训练上追加数据。原始 ID eval 还与训练共享同一 H5：统一
`50×6` 的 300 个 query 中，272 个 query-start clips 精确属于训练 split，299
个 query states 曾作为训练帧出现。因此高 ID 分数表示同轨迹 retention，不是
episode-held-out 泛化。

完整统计见
`artifacts/evaluation/history3/h3_speedtask_s3072/training_data_gap_v1.json`。
后续数据版本应增加独立 geometry、匹配 E4 room/template strata，并增加
episode-held-out ID；训练上应单独比较固定总预算与保留完整 original exposure
后追加 synthetic 的设置。

## 9. 执行入口

```bash
# 只编译并验证 scenario/reset 计划
bash scripts/run_tworoom_speedtask_data.sh compile

# 正式并行生成；SHARDS 可按机器资源调整
SHARDS=64 bash scripts/run_tworoom_speedtask_data.sh generate-parallel

# 中断后复用已完成场景
bash scripts/run_tworoom_speedtask_data.sh resume

# 生成轨迹级质量报告
python scripts/analyze_tworoom_speedtask_quality.py

# 正式训练前检查
bash scripts/run_h3_speedtask_train.sh preflight

# 新训练 / 自动续训 / 强制续训
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedtask_train.sh fresh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedtask_train.sh train
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedtask_train.sh resume

# 正式 E1、ID 和并行 E4；在空输出目录中启动完整评测
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedtask_eval.sh all

# 原始 H5 / SpeedTask 训练数据差异报告
python scripts/analyze_tworoom_training_data_gap.py
```

读者可通过 `STABLEWM_REPO` 指向自己的 StableWM checkout，通过
`CONTEXTWORLD_ARTIFACT_ROOT` 指定数据和 checkpoint 目录。ContextWorld 只负责
数据配置、质量约束和启动参数；模型训练、trainer checkpoint 与 resume 语义均
使用 StableWM 原生能力。
