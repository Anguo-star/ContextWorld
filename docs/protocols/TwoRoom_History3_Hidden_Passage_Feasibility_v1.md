# TwoRoom History=3 门规则 ICL v1：内部复现附录

**版本**：任务与数据协议 v1；规则切换判定 v2

**更新日期**：2026-07-25

**状态**：Validation Benchmark 已冻结，参考可学习性验证已完成。固定原始图像表示
的三个双规则模型在训练门位置和未见门位置均 3/3 通过。该结果证明合成数据可学；
Encoder 与预测器联合训练时的隐藏规则相关局部表示收缩仍是独立的待解决问题。

**文档角色**：本文件只保存历史预注册动作、数据身份和逐文件证据，供审计追溯。
任务解释、当前结果、发布配置和使用命令全部以
[门规则 ICL Benchmark v1 主文档](../TwoRoom_Door_Benchmark_Design.md)为准；普通读者
无需继续阅读本附录。

## 1. 任务定义

门洞外观始终相同，隐藏规则决定智能体能否穿过墙：

| 规则 | 相同状态下的画面 | 穿门动作的物理结果 |
|---|---|---|
| 门可通过 | 正常门洞 | 穿过墙 |
| 门不可通过 | 完全相同的门洞 | 被墙挡住 |

模型输入只有 `pixels` 和 `action`。以下字段只用于生成和审计，不得进入 LeWM：

- `passage.open`；
- 智能体状态和坐标；
- 门的位置；
- 真实下一帧和评测标签。

当前待预测帧称为 `query`。两种规则下的 query 像素、待执行动作和目标画面逐位相同，
所以 query 自身不能说明门是否可通过。规则证据只来自前面的三帧连续历史。

## 2. History=3 连续轨迹

一条样本包含三张历史画面和一个真实下一帧：

```text
历史第 1 帧
    │ 探门动作
    ▼
历史第 2 帧：穿过门，或被门挡住
    │ 恢复动作
    ▼
历史第 3 帧：两种规则回到同一个 query
    │ 待预测动作
    ▼
真实下一帧：穿过门，或停在墙边
```

每个模型动作块包含 5 个原始环境步，`agent.speed=5`。历史内部没有重置、手工移动或
画面拼接。

### 2.1 左向右

```text
探门动作：
[(+1, 0), (+1, 0), (0, 0), (0, 0), (0, 0)]

query 后动作：
[(+1, 0)] × 5
```

### 2.2 右向左

```text
探门动作：
[(-1, 0), (-1, 0), (0, 0), (0, 0), (0, 0)]

query 后动作：
[(-1, 0)] × 5
```

### 2.3 恢复动作和限制

门位于画面上半部分时使用：

```text
[(0, +1), (0, -1), (0, 0), (0, 0), (0, 0)]
```

门位于画面下半部分时使用：

```text
[(0, -1), (0, +1), (0, 0), (0, 0), (0, 0)]
```

恢复动作先离开门洞范围，使两种轨迹受同一侧墙边界约束，再回到原来的纵向位置。
可通过轨迹在这个过程中会被当前碰撞器从横坐标 108.0 推回 99.5，最大修正 8.5 px。
这是确定性的连续 `env.step()`，但不是自由运动，因此结论只适用于当前碰撞器。

## 3. 构造检查

早期可行性检查覆盖 32 对模板、64 条真实环境轨迹：

```text
4 个门位置 × 2 个方向 × 4 个偏移 = 32 对模板
32 对模板 × 2 种规则 = 64 条轨迹
```

32 对全部满足：

| 检查 | 结果 |
|---|---:|
| 两种规则的初始状态、像素和动作相同 | 32/32 |
| 历史第 2 帧状态至少相差 5 px | 全部相差 8.5 px |
| 恢复后的 query 状态和像素逐位相同 | 32/32 |
| 同一 query 动作后的真实状态至少相差 20 px | 全部相差 25 px |
| 动作签名单独猜规则的最好准确率 | 50% |

正式端到端 Lance 检查进一步确认：

- 每条 20 行 episode 恰好产生一个 History=3 clip；
- 保存、重载和独立环境重放的数组一致；
- 正式 Stable-WorldModel adapter 只把 `pixels` 和 `action` 交给模型；
- 规则、状态、坐标和模板标识不会到达模型输入边界。

## 4. 正式训练数据

训练数据由 TwoRoom 环境真实执行动作生成，不使用生成模型绘图。每个门位置包含：

```text
2 个方向 × 4 个离墙距离 × 10 个门洞偏移 = 80 个 clip / 规则
```

冻结规模：

| 数据 | 训练 shard / clip | 训练内部检查 shard / clip |
|---|---:|---:|
| 只可通过 | 96 / 7,680 | 16 / 1,280 |
| 只不可通过 | 96 / 7,680 | 16 / 1,280 |
| 双规则 | 192 / 15,360 | 32 / 2,560 |

“双规则”catalog 是两套单规则物理 shard 的完整并集，不重复生成第三份数据。

Validation 使用的 42 个门位置全部从训练和训练内部检查中排除。正式构建器还检查
query 像素哈希与 Validation 冻结清单没有交集。

### 4.1 shard 生成和恢复

- 默认串行生成；正式规模使用 4 个进程；
- 每个 worker 只生成一个独立 Lance shard；
- 主进程按固定顺序审计并写 episode 清单、catalog 和报告；
- shard 先写入临时目录，完整复制后再原子发布；
- `--resume-partial` 只复用通过完整重审的 shard，缺失 shard 会重新生成。

### 4.2 内容完整性

每个 episode 都对全部 20 行的核心列计算稳定逻辑哈希，包括：

- 画面、动作、proprio 和状态；
- 目标状态、终止标志和截断标志；
- 速度、门位置、门数量和隐藏通行规则。

Lance 自身会生成随机文件名和事务标识，所以原始目录字节哈希只用于保存现场，不用于
比较两次独立构建。训练预检会从 Lance 实际行重新计算逻辑哈希，并与 episode 清单和
shard 清单同时核对。只改数据文件中的一行、但不改清单时，预检必须失败。

训练还冻结正式的 `build_report.json`，并核对：

- 顶层状态和全部质量检查；
- 三个活动 catalog、manifest 和组报告的实际文件哈希；
- 每组精确 shard 与 clip 数；
- Validation 排除清单和 query 零重叠；
- 没有未登记的 Lance shard。

## 5. 模型与训练

模型矩阵共 10 个 checkpoint：

| 模型 | 数量 | 训练方式 |
|---|---:|---|
| 原始 H3 | 1 | 不续训，只作基线 |
| 原始 H3 续训：只用门可通过合成数据 | 3 | 从同一原始 H3 checkpoint 初始化，只抽样可通过合成数据 |
| 原始 H3 续训：只用门不可通过合成数据 | 3 | 从同一原始 H3 checkpoint 初始化，只抽样不可通过合成数据 |
| 原始 H3 续训：同时使用两种规则合成数据 | 3 | 从同一原始 H3 checkpoint 初始化，只抽样两种规则的合成数据 |

三个续训组都不混入原始 TwoRoom 样本。原始数据只提供固定的数据划分和归一化参数。
三个组使用相同初始权重、训练种子、优化器、训练步数和总逻辑抽样数。

冻结身份：

| 项目 | 值 |
|---|---|
| Stable-WorldModel commit | `5864b74980f6ed328fd0045e777b3865962eff43` |
| 原始 H3 checkpoint SHA-256 | `7d141b86cca49145444a69bff89c71ede69e8cf8252bfb933e656c3e2e962b54` |
| 归一化参数 SHA-256 | `7a5be7ea867bced446c1671b0b2c0ff6450ffc61e1a7bdbbfc5eaa0942f635db` |
| 训练种子 | 3072、4096、5120 |
| 每模型参数更新 | 1,024 |
| 每模型总逻辑抽样 | 1,048,576 |

参考可学习性验证另训练 3 个 checkpoint。它们沿用双规则组的完整 15,360 个训练
clip、三个训练种子和训练预算，只改变可更新模块：

| 项目 | 固定值 |
|---|---|
| 模型名称 | 固定原始图像表示的双规则参考模型 |
| 固定模块 | 原始 H3 的 `encoder` 与 `projector`，参数和运行统计均不更新 |
| 更新模块 | 动作编码与下一帧预测部分 |
| 训练拓扑 | 单机 8 卡；每卡 batch 128；全局 batch 1,024 |
| 训练预算 | 1,024 次参数更新；1,048,576 次逻辑抽样 |
| checkpoint 选择 | 固定最后一步，不按评测结果选择 |

每个正式训练进程的 8 个 rank 都独立重算完整 Lance 逻辑哈希，并在训练开始前重新
验证存储。8 个 rank 使用互不重叠的 CPU 集合，防止 Lance 线程池争抢。三份训练报告
均确认 `encoder` 和 `projector` 的训练前后状态哈希完全相同。

## 6. Validation v2

Validation v2 使用评测随机种子 42～47。每个种子独立、无放回选择 50 个 query，其中
左向右 25 个、右向左 25 个；6 组之间没有重复。

对每个模型，2 种真实下一帧和 3 种历史组成的每个计分条件都完整使用：

```text
50×6=300：每个种子 50 个 query，共 6 个评测随机种子
```

300 条不会分摊给其他条件。同一 query 在六个条件中复用；两种真实下一帧共享同一次
模型推理，再分别计算 loss。

三种历史为：

1. 看见智能体穿过门；
2. 看见智能体被门挡住；
3. 没有尝试穿门。

一个模型产生 900 次预测；每次预测与两张真实下一帧分别计算 loss，因此保存
1,800 条 loss 记录。评分期间不调用在线环境。

Validation v2 冻结身份：

| 文件 | SHA-256 |
|---|---|
| `catalog.json` | `d5356582f658070bb41972873720e29add04e2d6ca2e637bc0bc7bd57f204551` |
| `training_exclusion_manifest.json` | `d732ca66061b7f16436d7897e570da3626957b1bda0a94ab0e7125d98f14eee9` |
| 内容清单 | `c23f079e6e9119fd4320d6209f992ca98b1d81a889c6127ade087a447c2aea0c` |

## 7. 规则切换 v2 判定

单个模型必须同时满足：

1. 对两种真实结果，匹配规则的历史都优于相反规则历史；
2. 上述优势在 6 个评测随机种子和两个方向组成的 12 个小格中都为正；
3. 两张真实下一帧的选择准确率总体和每个小格都严格高于 50%；
4. 两种真实结果的匹配历史逐 query 胜率都严格高于 50%；
5. 4 项主要差值的 95% 置信区间下界都大于 0；
6. 每个 query 的两张真实下一帧在 latent 中可区分。

四项主要差值是两种规则各自的“匹配历史相对相反历史 loss 优势”和“正确目标相对
另一目标的距离余量”。“没有尝试穿门”的历史没有展示规则，没有唯一正确答案；
它仍完整计分，用于报告模型默认倾向，但不参与能力门槛。

置信区间使用 10,000 次成对 bootstrap，并在评测随机种子和方向组成的 12 个小格内
分别重采样；随机种子固定为 20260725。

参考训练的可学习性结论要求三个训练种子全部通过。要研究联合训练为什么失败，或把
收益进一步拆分到具体训练机制，还需要独立的模型训练对照。这些对照不改变已经冻结的
Benchmark 数据和评分标准。若要单独归因于“双规则数据”，还需要：

- 原始 H3 不通过；
- 使用相同固定表示方式训练的两个单规则组都不是 3/3 通过；
- 所有模型使用相同的 Validation、归一化参数、代码和评分入口。

规则切换 v2 只使用训练门位置诊断确定，并在任何未见门位置 checkpoint 得分产生前
冻结。旧判定要求匹配历史同时击败“相反规则历史”和“没有尝试穿门的历史”；其结果
继续保留，但不再作为规则切换能力门，因为无规则证据条件只表示默认倾向。

## 8. 正式结果与产物身份

### 8.1 固定表示参考可学习性确认

训练门位置和未见门位置均为 3/3 通过。每个模型、每种真实下一帧和每种历史条件都
独立覆盖 `50×6=300`；每个模型在每个评测范围内保存 900 次预测和 1,800 条 loss。
三个种子的两种真实下一帧选择率、匹配历史逐 query 胜率均为 100%，24 个规则、种子
与方向小格全部正确，四项 bootstrap 下界全部大于 0。

`evaluation/` 与 `training/` 路径相对于 ContextWorld artifact 根目录；
`configs/` 路径相对于仓库根目录：

| 产物 | 路径 | SHA-256 |
|---|---|---|
| 完整确认报告 | `evaluation/history3/hidden_passage_fixed_representation_confirmation_v2.json` | `4665d9af66206a38c47ec3169157ba14ed771acf4bb5c988c3621ff815484b73` |
| 训练门位置三种子汇总 | `evaluation/history3/hidden_passage_fixed_representation_train_seen_rule_switch_v2/aggregate_rule_switch_v2.json` | `4af5ba7424cdae57dde19375f5c6687247964eccf2fb98251ecab5640aaba406` |
| 未见门位置三种子汇总 | `evaluation/history3/hidden_passage_fixed_representation_validation_rule_switch_v2/aggregate_rule_switch_v2.json` | `42cb24f3cf5471a76fbceefa4ddbefdc6d1c0f84d46a85604d5549714099021c` |
| 训练配置 | `configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml` | `499ab4e47951e179929165454da1a154d32b3395661e8e2a36ec45a195bf4d50` |
| 训练门位置判定配置 | `configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_train_seen_eval_v2.yaml` | `48495a93cd2245cc0c67c897044d4cb7930181d37a59b064c1da2f7b4f1b5e7a` |
| 未见门位置判定配置 | `configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_validation_v2.yaml` | `16b5e20c43077c374cfb394a77b2fe56d45068489954021e7545f1e80833faa3` |

| 训练种子 | checkpoint SHA-256 | 训练报告 SHA-256 |
|---:|---|---|
| 3072 | `5c1174d0b0ff0716d220c82f1e681f4e8e3749c6073372427731316b584ab259` | `ed60783ba4010c287bead4cb46b58e4444c5018219c8a1b7825ba9e3bc9ffceb` |
| 4096 | `1298b9c2a1102d780bc012128d5c582fdcfe31f8be379f30d85ddd03470d26d0` | `a912018bc07ca3897d12686c4d28946b1625c50cb7f456ccf124bec0b3b85922` |
| 5120 | `6f0e0230d40325e6d40a5f3d7a1b2dc472e3aa08c4057e5ded910287649b3915` | `71592c02e343a82eaca1b4df43a53e942cdc88431c3942d23156091a78d30264` |

### 8.2 首轮端到端训练基线

首轮 10 个结果均已完成严格身份复验。原始 H3 未通过；两个单规则端到端续训组和
双规则端到端续训组都没有训练种子通过。它们用于检查普通联合训练和固定倾向，
具体指标见
[TwoRoom 门通行规则 ICL Benchmark v1](../TwoRoom_Door_Benchmark_Design.md)。

以下路径均相对于 ContextWorld artifact 根目录。哈希是当前正式文件的字节级
SHA-256。

| 产物 | 路径 | SHA-256 |
|---|---|---|
| 十模型严格汇总 | `evaluation/history3/hidden_passage_validation_v2/aggregate.json` | `7e28967e4f218c9626096888a16eee1218db83ec7a7eb0bb54d1f85a7e79494a` |
| 机器可读结果摘要 | `evaluation/history3/hidden_passage_validation_v2/results_summary.json` | `1e6f333db29d486752c488f8b1680925822881156d1d728ea94e9b7a8853bf1d` |
| 中文结果摘要 | `evaluation/history3/hidden_passage_validation_v2/results_summary.md` | `f7ba27c45a60107a6aecfa477463da61585758a9536837d50ca3b7499b2df5a2` |

`aggregate.json` 的规范化内容哈希为
`c12c112a6b58b02970797d29330e3577e4e20aa2826e1b8ea34b310f26e4e37f`，
并绑定下列 10 个评分文件：

| 评分文件（位于 `evaluation/history3/hidden_passage_validation_v2/results/`） | SHA-256 |
|---|---|
| `h3_original_lewm_s3072.json` | `79bf32049ac7e9ed7e8f0526b0aeb15f4b0cd6bd4582c82bdc9672084902a8bb` |
| `h3_original_init_plus_synth_passable_s3072.json` | `b03bbf383e534a61b0517b036e523d5b8cb1f4b9881e485becfd643302080b41` |
| `h3_original_init_plus_synth_passable_s4096.json` | `a330e5eba51f624637eb61cbec1003e6bf821451f813a112d9e6990ffe37dc63` |
| `h3_original_init_plus_synth_passable_s5120.json` | `033e8025c92bce89e646927b41c7ba9be5b818108023ff2b7ca764a3dfe1ee2e` |
| `h3_original_init_plus_synth_blocked_s3072.json` | `dada6396493813070a12a3f355f9d83dd937c05c51eec00b2763c342e48b93ba` |
| `h3_original_init_plus_synth_blocked_s4096.json` | `ce76f198ed49782b121c6fb373e90d06d4de04e91026f9c1cf197e4d6045eff1` |
| `h3_original_init_plus_synth_blocked_s5120.json` | `ba93b95986e7ae5b3d2e75eee24fc184758eaebd21f7a3269e1d7c7a269b4809` |
| `h3_original_init_plus_synth_mixed_rules_s3072.json` | `722a2647f46254df60ca3d77e4700b4168195ce04a3fbd206f122008d9397c2f` |
| `h3_original_init_plus_synth_mixed_rules_s4096.json` | `59790db4ad2780eb611ed167bc44dd1db1426d6ee619d914aa96d8318c1fd468` |
| `h3_original_init_plus_synth_mixed_rules_s5120.json` | `0dc82a9b9475425e04bbbfbb008af7d06d95076ee3137ea896ee430217e46331` |

## 9. 失败根因诊断产物

正式结果完成后，另行冻结并执行了两项诊断：

1. 用训练中出现过的门位置重跑同一 10 模型矩阵；
2. 在同一扇门的 160 个双规则训练 clip 上，对比“编码器和预测器共同更新”与
   “固定原始编码器和投影层，只更新预测部分”。

训练门位置矩阵仍为 0/10 通过。单门诊断使用 8 个精确训练 query；联合更新为
0/8 正确切换，固定表示后为 8/8。训练报告确认固定的 `encoder` 和 `projector`
在 1,024 步前后逐模块哈希完全相同。该诊断用于定位训练机制，不属于正式
`50×6` Benchmark 分数。

| 诊断产物 | 路径 | SHA-256 |
|---|---|---|
| 训练门位置十模型汇总 | `evaluation/history3/hidden_passage_train_seen_diagnostic_v1/aggregate.json` | `97b4bc7ac8371fccc48b5332b5cd09793fa09b0709e0bfe45643aab6fb4e14fc` |
| 训练门位置结果摘要 | `evaluation/history3/hidden_passage_train_seen_diagnostic_v1/results_summary.json` | `f504825bb375e2fe4c49c91cc85bedf5840e57dd57c7c7be376846296626c15e` |
| 固定表示诊断 catalog | `evaluation/history3/hidden_passage_frozen_representation_diagnostic_v1/catalog.json` | `b202d8dac970e28c0a3cf000da315b988e6ef539592fa26e30aa2f18b015d319` |
| 原始 H3 结果 | `evaluation/history3/hidden_passage_frozen_representation_diagnostic_v1/original.json` | `2c1637ff49f5f3b685588147d91117784345396cada93fbdaba6e5d8c784cf9d` |
| 联合更新结果 | `evaluation/history3/hidden_passage_frozen_representation_diagnostic_v1/joint_update.json` | `badb7d6d68ee8e20d56c5d3251a9af8fed6b6ad03f1d3ee47f73c84a2b0e5290` |
| 固定表示结果 | `evaluation/history3/hidden_passage_frozen_representation_diagnostic_v1/fixed_representation.json` | `de7abd623cf6cae9690b3887c4da87e4aecdc5c8f078d75a515833d1ad4f914e` |
| 根因机器汇总 | `evaluation/history3/hidden_passage_root_cause_v2/summary.json` | `fec1b2d83b3bcc0d75a84284502558a528b63ffac89eaf0b24a5796e9d5d6428` |
| 根因中文摘要 | `evaluation/history3/hidden_passage_root_cause_v2/summary.md` | `bbc63b7f3fb421b33400875d8baf16b5ec00f20ebb69eaada90f4702426bea64` |
| 固定表示 checkpoint | `training/runs/checkpoints/h3_passage_tiny_frozen_representation_s3072/weights_final_step_1024.pt` | `d057517a1118c0b212537f102c4612a72371add519079a91ae7a8d3bd8e3f748` |
| 固定表示训练报告 | `training/reports/h3_passage_tiny_frozen_representation_s3072.json` | `99e72305768c865e107cf069239c0f5c5e8ce8e964f4eb13cb8ca31190944609` |

## 10. 复现入口

- 可行性配置：
  [`tworoom_hidden_passage_h3_feasibility_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_feasibility_v1.yaml)
- Validation v2：
  [`tworoom_hidden_passage_h3_validation_v2.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml)
- 训练数据：
  [`tworoom_hidden_passage_h3_training_data_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_training_data_v1.yaml)
- 训练：
  [`tworoom_hidden_passage_h3_training_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_training_v1.yaml)
- 数据构建：
  [`build_tworoom_hidden_passage_h3_training_data.py`](../../scripts/build_tworoom_hidden_passage_h3_training_data.py)
- 模型评分：
  [`eval_tworoom_hidden_passage_h3_latent.py`](../../scripts/eval_tworoom_hidden_passage_h3_latent.py)
- 十模型汇总：
  [`analyze_tworoom_hidden_passage_h3.py`](../../scripts/analyze_tworoom_hidden_passage_h3.py)
- 结果摘要：
  [`render_tworoom_hidden_passage_h3_results.py`](../../scripts/render_tworoom_hidden_passage_h3_results.py)
- 训练门位置诊断：
  [`tworoom_hidden_passage_h3_train_seen_eval_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_train_seen_eval_v1.yaml)
- 固定表示训练：
  [`tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml)
- 固定表示训练门位置评测：
  [`tworoom_hidden_passage_h3_fixed_representation_train_seen_eval_v2.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_train_seen_eval_v2.yaml)
- 固定表示未见门位置评测：
  [`tworoom_hidden_passage_h3_fixed_representation_validation_v2.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_fixed_representation_validation_v2.yaml)
- 固定表示完整执行：
  [`run_tworoom_hidden_passage_h3_fixed_representation.py`](../../scripts/run_tworoom_hidden_passage_h3_fixed_representation.py)
- 单门固定表示机制评测：
  [`tworoom_hidden_passage_h3_tiny_frozen_representation_eval_v1.yaml`](../../configs/benchmark/tworoom_hidden_passage_h3_tiny_frozen_representation_eval_v1.yaml)
- 根因汇总：
  [`analyze_tworoom_hidden_passage_h3_root_cause.py`](../../scripts/analyze_tworoom_hidden_passage_h3_root_cause.py)

默认拒绝覆盖已有正式产物。任何冻结文件、checkpoint、代码版本或数据逻辑哈希不一致
都会让预检失败。
