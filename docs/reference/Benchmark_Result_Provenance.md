# ContextWorld 参考结果复现附录

本文记录主 Benchmark 文档中参考结果的来源、评测预算、门限判定和机器可读文件。普通
使用者运行 Training 或 Development 评测时不需要阅读本附录；复核表格、比较训练配方或
重新生成参考结果时再使用这些信息。

本文只记录仓库工件中可以查到的内容。文中给出的路径都相对仓库根目录。`artifacts/`
的大部分内容不随代码分发，因此在干净检出中可能缺失；本文对每一项都标明它属于哪一类
证据，缺失的测量也明确写出。

## 1. 证据分级

主文档的数值分为三类。它们可以在同一张综合表中并列展示，但不得互相替代，也不能在
缺少划分、协议和证据状态标记时当作同口径结果：

1. **冻结的 LeWM / PLDM 协议结果**：正式参考矩阵。包括原始检查点的 ICL 起点、原始
   环境 CEM，以及组件训练后的 scoreboard 行。
2. **非冻结补充证据**：DINO-WM / PreJEPA 的 ICL 诊断与 CEM，以及“完整对照记录”中
   另行运行的结果。这些工件自身标注 `official_frozen_matrix: false` 或
   `formal_scoreboard_eligible: false`，不进入正式 scoreboard。
3. **Development 报告结果**：未获 Public Test 准入的组件保留 Development 结论。历史上已经
   运行的 Test 文件仍须披露，但不能把它们当作获准的正式成绩。

这里的“非冻结”沿用历史工件对正式 scoreboard 的分类。当前研究参考 v2 已固定三模型的
数值与来源，包括 DINO-WM 补充证据；它不改变历史正式发布资格。主文档 §5.1 与 v2 是当前
比较入口，以下分阶段记录保留当时的协议与结果。

## 2. 原始 LeWM 与 PLDM 的 ICL 起点

TwoRoom、PushT、Reacher 和 Cube 各固定一枚 LeWM 与一枚 PLDM 检查点，共 8 枚。检查点
由本仓库的训练流程使用公开原环境数据、按相应 Stable-WorldModel 基线配方从头训练
10 epochs；不使用 ContextWorld 组件数据，也不继续训练已有权重。冻结矩阵的归档过程本身
没有重新训练或按评测分数选择检查点；工件中的 `training_performed: false` 与
`checkpoint_selection_performed: false` 描述的是这次归档操作，不是这些权重的训练来源。

矩阵共 18 个单元，全部标记 `formal_scoreboard_eligible: false`：它们是单检查点描述性
起点，不构成方法级结论，也不增加 scoreboard 行。18 个单元中只有速度 LeWM 通过了单
检查点门。

| 任务 | 模型 | 划分 | 主分数 | 单检查点门 | 同一单元的其他读数 |
|---|---|---|---:|---|---|
| 速度 | LeWM | Public Test | 59.33% | 通过 | 未见插值 H1 匹配/其他 loss 比 0.906 |
| 速度 | PLDM | Public Test | 51.44% | 未通过 | 四个 horizon 的 within-checkpoint 检查未全部通过 |
| 推手移动幅度 | LeWM | Public Test | 49.80% | 未通过 | 正确历史 38.67%，规律切换 38.28% |
| 推手移动幅度 | PLDM | Public Test | 43.55% | 未通过 | 正确历史 5.27%，规律切换 3.91% |
| 机械臂质量 | LeWM | Public Test | 50.00% | 未通过 | 正确历史 51.56%，上下文切换 75.78% |
| 机械臂质量 | PLDM | Public Test | 52.93% | 未通过 | 正确历史 76.37%，上下文切换 95.70% |
| 动作延迟 | LeWM | Public Test（末尾三帧） | 16.67% | 未通过 | 最弱响应组 0%，bootstrap 下界 16.67% |
| 动作延迟 | PLDM | Public Test（末尾三帧） | 16.67% | 未通过 | 最弱响应组 0%，bootstrap 下界 16.67% |
| 接触摩擦 | LeWM | Development | 49.61% | 未通过 | 正确历史 51.76%，上下文切换 49.61% |
| 接触摩擦 | PLDM | Development | 50.00% | 未通过 | 正确历史 48.83%，上下文切换 49.22% |
| 运动阻尼 | LeWM | Development | 50.00% | 未通过 | 正确历史 49.80%，上下文切换 62.11% |
| 运动阻尼 | PLDM | Development | 50.39% | 未通过 | 正确历史 51.76%，上下文切换 87.11% |
| Cube 夹爪携带规则 | LeWM | Public Test | 50.98% | 未通过 | 正确历史 75.59%，上下文切换 99.61% |
| Cube 夹爪携带规则 | PLDM | Public Test | 50.98% | 未通过 | 正确历史 55.86%，上下文切换 85.16% |
| 门通行规则 | LeWM | Public Test | 50.00% | 未通过 | 匹配历史相对相反历史胜率 51.33% |
| 门通行规则 | PLDM | Public Test | 50.00% | 未通过 | 匹配历史相对相反历史胜率 50.83% |
| 传送门出口位置 | LeWM | Public Test | 50.00% | 未通过 | 正确历史 51.37%，上下文切换 96.88% |
| 传送门出口位置 | PLDM | Public Test | 50.20% | 未通过 | 正确历史 51.56%，上下文切换 96.09% |

高上下文切换率配合约 50% 的主分数说明模型确实随历史改变预测，但改变方向与真实未来
无关，因此不能算作 ICL 证据。

接触摩擦和运动阻尼的单元记为 `development_only_public_closed`，其 Public Test 未打开、
未评分；Cube 的两个单元记为 `external_non_reference_public_descriptive`。

动作延迟要求 History=7，而原始检查点是 History=3。原始行使用 `h3_tail_projection`：只把
History=7 查询中对齐的最后三帧和最后五个动作块交给模型，不做位置编码插值，不做任何
训练或微调，权重不变。允许的表述只有“H3 检查点在该评分器下的零样本起点”。原生 H7
尝试的失败回执与恢复记录同样保留在矩阵中。

多数单元附有独立重打分证据；门通行规则等早期单元的重打分标记为
`legacy_output_omits_checkpoint_and_raw_sha`，推手移动幅度 LeWM 与传送门出口位置两枚
单元使用 float32 精确重打分恢复，接触摩擦与运动阻尼的重打分输出与原始输出逐字节相同。
这些差异记录在矩阵工件里，读表时应一并考虑。

机器可读来源：

- `artifacts/evaluation/original_baseline_matrix_v1/matrix_summary.json`
- `artifacts/evaluation/original_baseline_matrix_v1/checkpoint_identity_audit.json`
- `configs/benchmark/contextworld_original_baseline_matrix_results_freeze_v1.json`
- `python scripts/audit_contextworld_original_baseline_matrix_freeze_v1.py`

## 3. 原始环境 CEM

每个检查点完成 300 次规划评测。TwoRoom 与 PushT 使用六个评测种子（42–47）× 50 次；
Reacher 与 Cube 的这批结果使用三个评测种子（42–44）× 100 次。总预算相同，抽样结构
不同，读表时不要把两者当作同一种误差来源。

| 原始环境 | 模型 | 训练种子 3072 / 3073 / 3074 的成功数 | 平均值 ± 样本标准差 |
|---|---|---|---:|
| TwoRoom | LeWM | 276、279、277 | 92.44% ± 0.51pp |
| TwoRoom | PLDM | 278、254、283 | 90.56% ± 5.17pp |
| PushT | LeWM | 248、235、257 | 82.22% ± 3.69pp |
| PushT | PLDM | 233、219、229 | 75.67% ± 2.40pp |
| Reacher | LeWM | 164、170、169 | 55.89% ± 1.07pp |
| Reacher | PLDM | 248、240、139 | 69.67% ± 20.25pp |
| Cube | LeWM | 197、198、194 | 65.44% ± 0.69pp |
| Cube | PLDM | 158、159、164 | 53.44% ± 1.07pp |

共同规划参数：

| 参数 | 值 |
|---|---:|
| candidates | 300 |
| iterations | 30 |
| top-k | 30 |
| history | 3 |
| action block | 5 |
| goal offset | 25 |
| episode budget | 50 |

其中预注册执行的 17 个单元记录加载权重的 state-dict SHA-256 并通过一致性审计，以确认
评测过程没有改变模型权重；沿用自更早批次的 7 个单元在原始记录中不含该审计字段，工件
里按 `provenance` 区分。工件中另有两项披露：

- TwoRoom LeWM 有一枚 273/300 的历史单元，因为它来自另一条训练血统（仓库训练的
  `h3_origheldout_s3072`，与 lightning 三元组数值不同），被排除在族统计之外，只作为
  披露的血统备注保留，从不并入均值或标准差；
- `tworoom_pldm_seed3074_eval43` 因 CUDA 启动失败在产生任何分数之前中断，按独立恢复
  身份重跑一次，重跑回执与预注册一并保留。

机器可读来源：

- `artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json`
- `configs/benchmark/contextworld_original_baseline_seed_completion_results_freeze_v1.json`
- `configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json`

## 4. 组件训练后的冻结参考结果

方法级判定要求三个独立训练种子分别通过任务的主分数和全部附加条件。平均分达到门槛
不能替代逐检查点判定。

### 4.1 逐训练种子 ICL

| 组件 | 方法 | 划分 | 逐检查点主分数 | 均值 | 通过检查点 |
|---|---|---|---|---:|---|
| 速度 | LeWM | Public Test | 94.89%、95.00%、95.89% | 95.26% | 3/3 |
| 速度 | PLDM | Public Test | 97.22%、96.44%、96.44% | 96.70% | 3/3 |
| 推手移动幅度 | LeWM | Public Test | 97.07%、96.48%、96.29% | 96.61% | 3/3 |
| 推手移动幅度 | PLDM | Public Test | 94.34%、93.95%、94.53% | 94.27% | 0/3 |
| 机械臂质量 | LeWM | Public Test | 76.17%、76.56%、75.59% | 76.11% | 3/3 |
| 机械臂质量 | PLDM | Public Test | 63.48%、62.70%、63.28% | 63.15% | 0/3 |
| 动作延迟 | LeWM | Public Test | 33.00%、32.11%、32.17% | 32.43% | 0/3 |
| 动作延迟 | PLDM | Public Test | 93.06%、93.22%、93.81% | 93.36% | 3/3 |
| Cube 夹爪携带规则 | LeWM | Public Test | 77.73%、79.10%、78.52% | 78.45% | 3/3 |
| 门通行规则 | LeWM | Public Test | 100.00%、100.00%、100.00% | 100.00% | 3/3 |
| 门通行规则 | PLDM | Public Test | 99.33%、99.33%、99.33% | 99.33% | 3/3 |
| 传送门出口位置 | LeWM | Public Test | 85.55%、83.20%、83.01% | 83.92% | 0/3 |
| 传送门出口位置 | PLDM | Public Test | 59.77%、58.59%、59.57% | 59.31% | 0/3 |

速度 PLDM 的三个检查点对应训练种子 3072、4096 和 5120；推手移动幅度 PLDM 对应 13313、
13314 和 13315。其余行不单独标注训练种子，检查点顺序按冻结汇总记录。

未通过的原因分别是：推手移动幅度 PLDM 三个检查点都低于 95%；机械臂质量 PLDM 三个
检查点都低于 75%；传送门出口位置 LeWM 与 PLDM 三个检查点都低于 95%；动作延迟 LeWM
除总体 32.43% 低于 75% 外，最弱物理响应组的准确率为 0%，同时不满足 60% 的分组下限。
作为对照，动作延迟 PLDM 的最弱响应组平均 84.83%，配对 bootstrap 95% 下界平均 91.71%。

### 4.2 训练后原任务 CEM

本表是**冻结 scoreboard 对应的那一批检查点**（2026-09-03 之前）的训练后 CEM，不是
2026-09-03 批次的数值。2026-09-03 批次的逐检查点 CEM 见 5.1 节；两批的速度、门通行
规则、传送门出口位置等行数值差异明显，引用时必须先确认批次。

| 组件 | 方法 | 原任务 | 逐检查点成功率 | 判定 |
|---|---|---|---|---|
| 速度 | LeWM | TwoRoom | 96.33%、94.67%、96.67% | 保持 |
| 速度 | PLDM | TwoRoom | 94.33%、94.33%、95.67% | 保持（配对非劣性） |
| 推手移动幅度 | LeWM | PushT | 74.67%、72.00%、77.33% | 保持 |
| 机械臂质量 | LeWM | Reacher | 54.33%、54.33%、53.33% | 保持 |
| 动作延迟 | LeWM | TwoRoom | 97.33%、97.33%、97.33% | 保持 |
| 动作延迟 | PLDM | TwoRoom | 95.00%、97.33%、96.33% | 保持 |
| Cube 夹爪携带规则 | LeWM | Cube | 186、183、185 / 300 | 保持 |
| 门通行规则 | LeWM | TwoRoom | 65.67%、63.00%、62.33% | 未保持 |
| 门通行规则 | PLDM | TwoRoom | 45.00%、44.00%、48.67% | 未保持 |
| 传送门出口位置 | LeWM | TwoRoom | 92.00%、90.33%、88.67% | 保持 |

判定方式按组件预注册：Cube 使用与原始检查点比较的成功数下降上限（基线 198/300，
允许下降 15 次；三个候选分别下降 12、15、13 次）；速度 PLDM 使用配对非劣性，成功率差
的 95% 下界不得低于 −0.05，终点距离差的上界不得超过 5 px，并要求不出现可解房间关系
分层的塌缩，三个检查点全部通过。

冻结 scoreboard 中标记 `NOT_EVALUATED` 的行，表示预注册规则在 ICL 未通过后不授权训练后
CEM（推手移动幅度 PLDM、机械臂质量 PLDM、传送门出口位置 PLDM）。这类“未运行”只描述
正式流程，不表示对应的原始检查点或原始环境 CEM 缺失；在非 scoreboard 的对照记录中
另行运行的结果见第 7 节。

### 4.3 速度的两类 CEM 证据不可互换

速度 PLDM 有两组独立的 CEM 数值，主文档分别报告：

| 证据 | 逐检查点数值 | 均值 | 性质 |
|---|---|---:|---|
| 原 TwoRoom 规划保持 | 94.33%、94.33%、95.67% | 94.78% | 配对非劣性判定，PASS |
| action-planning 分析 | 70.33%、71.67%、66.67% | 69.56% | 描述性支撑指标，无预设门槛 |

action-planning 记录的语义是 `EXECUTED_VALID_DESCRIPTIVE`，`model_performance_gate: null`，
`retention_result: NOT_APPLICABLE`；速度组件的 release 配置也把 planning 标为
`supporting_utility_metrics`、`required_for_speed_icl_prediction_claim: false`。因此它既不能
当作保持判定，也不能当作 ICL 能力证据。

此外，速度 PLDM 的完成记录写明 `training_attribution.claim: false`、
`paired_training_controls_available: false`：没有预注册的同训练种子单速度对照，因此该行
只报告行为结果，不把表现归因于多速度合成数据。速度 LeWM 是唯一记为
`training_attributed_icl_demonstrated` 的行，其余通过行都是 `behavioral_icl_demonstrated`。

机器可读来源：

- `artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/public_scoreboard.json`
- `artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/public_scoreboard_spec.json`
- `configs/benchmark/contextworld_pldm_reference_completion_aggregate_results_freeze_v1.json`
- `configs/benchmark/tworoom_action_delay_icl_release_v1.yaml`（动作延迟门限与参考汇总）
- `configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml`（Cube 门限与保持判定）
- `contextworld-scoreboard --input <public_scoreboard_spec.json>` 重新渲染 scoreboard

## 5. 当前参考结果与门限失败说明

接触摩擦、运动阻尼与 Cube PLDM 的正式参考流程停在 Development，Public Test 未打开、
未读取、未评分，因此正式 scoreboard 没有对应 Public Test 成绩或训练后 CEM。接触摩擦
与运动阻尼的当前参考矩阵各自只完成一个训练种子（分别是 13313 和 14321），其余种子未
运行。另行登记的旧配方对照与 Cube PLDM 外部补充结果见第 7 节，不改写这里的正式流程。

| 组件 | 方法 | 主分数 | 正确历史 | 上下文切换 | 最弱条件 | 未通过的门 |
|---|---|---:|---:|---:|---:|---|
| 接触摩擦 | LeWM | 96.09% | 90.23% | 99.61% | 94.92% | 正确历史（门槛 95%） |
| 接触摩擦 | PLDM | 52.73% | 69.14% | 79.30% | 48.05% | 主分数、正确历史、上下文切换、最弱摩擦、响应增益 |
| 运动阻尼 | LeWM | 97.46% | 52.93% | 97.66% | 97.27% | 正确历史（门槛 95%） |
| 运动阻尼 | PLDM | 51.95% | 57.42% | 64.45% | 37.89% | 主分数、正确历史、上下文切换、最弱阻尼、响应增益 |

两枚 LeWM 检查点说明为什么主分数不能单独作为结论：它们的真实未来选择率都超过 95%，
但历史使用门未过——接触摩擦 LeWM 的正确历史为 90.23%，运动阻尼 LeWM 只有 52.93%，
后者接近随机，说明高主分数并非稳定来自历史信息。两枚 PLDM 检查点还未通过 latent 响应
增益门（接触摩擦 0.033、运动阻尼 0.160，门槛 0.50），但两者的目标 latent 可分性检查
通过，即数据本身提供了可区分的两种真实未来。

Cube PLDM 在 v4r1 Development 上三个检查点为 50.20%、50.20% 和 50.00%，均值 50.13%，
0/3 通过，未进入正式 Public Test 参考流程（`public_score: not_authorized_not_run`）。

机器可读来源：

- `artifacts/evaluation/history3/pusht_contact_friction_h3_strict_v3/development_decision.json`
- `artifacts/evaluation/history3/pusht_motion_damping_release_v1/failed_development.json`
- `artifacts/evaluation/pldm_reference_completion_v1/pusht_contact_friction_current_v3/pilot_seed13313/development_evaluation_decision_v1.json`
- `artifacts/evaluation/pldm_reference_completion_v1/pusht_motion_damping_current_v4/pilot_seed14321/development_evaluation_decision_v1.json`
- `configs/benchmark/pusht_contact_friction_icl_release_v1.yaml`、
  `configs/benchmark/pusht_motion_damping_icl_release_v1.yaml`（门限定义）
- `configs/benchmark/contextworld_public_v1_release_readiness_draft_v1.yaml`（Cube 逐种子
  Development 数值与授权状态）

### 5.1 2026-09-03 完整重训：全部九项的完整三训练种子结果

当前唯一数值来源为 `configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v2.json`。
本节逐任务运行细节保留早期执行记录；运行时迁移和 Development 载荷更新后的最终值见主文档
§5.1 与本附录 §5.2，不以早期数值覆盖最终冻结值。

LeWM 与 PLDM 均完成了标准 `joint_scratch_v1` 配方（`CW_METHOD=native`，
`CW_RESUME=reset`，10 epochs，训练种子 3072/3073/3074）下全部九项组件的从零训练
（2026-09-03 批次），并各自跑完自动 post-train 评测管线
（`eval_results/benchmark_icl/<task>/result.json` 与
`eval_results/benchmark_cem/<task>/*_metrics.json`，检查点根
`ckpt/lewm-contextworld-v1`、`ckpt/pldm-contextworld-v1`）。评测划分为 Development
（`official_scoreboard_row: false`、`formal_pass_available: false`）；已通过 Development
准入门槛的组件另有 Public Test 评测（2026-09-07），见本节下方“Public Test 评测”。
在此之前的参考状态是：接触摩擦、运动阻尼各自只有一个训练种子的结果；Cube
LeWM/PLDM、推手移动幅度、机械臂质量、门通行规则、传送门出口位置、动作延迟的
“组件训练后”数值来自另一批检查点，且**每一行的训练配方都不同**（接触摩擦
`mixed_frozen_image_paired_future_matching_1p00` 单种子 13313、运动阻尼
`mixed_frozen_image_paired_future_ranking_twin_1p00` 单种子 14321、Cube 走 `v4r1`
协议、其余见历史归档的逐行训练身份表）——都不是标准 `native` 配方的第二、第三个
训练种子。本节给出该配方下的完整三训练种子结果。

**动作延迟：正式六组口径（Public Test，2026-09-07 评测）。** 用
`python -m contextworld.benchmarks.action_delay_icl_cli eval|score`（release
`contextworld_tworoom_action_delay_icl_history7_v1`，冻结 300 查询 Public Test 资产，
6 评测种子 × 50 查询，6 个物理响应组，`online_environment_calls: 0`，
`data.full_protocol: true`）对本批 6 枚检查点评分：

| 方法 | 种子 3072 | 种子 3073 | 种子 3074 | 主分数均值 ± 样本标准差 | 最弱组均值 | bootstrap 下界均值 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 动作延迟 LeWM | 97.23% | 98.31% | 97.71% | 97.75% ± 0.54pp | 95.85% ± 0.95pp | 96.83% ± 0.64pp | 3/3 通过 |
| 动作延迟 PLDM | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00pp | 100.00% ± 0.00pp | 100.00% ± 0.00pp | 3/3 通过 |

方法级回执 `decision.passed: true`、`passed_checkpoints: 3`、
`submission_kind: three_seed_method`，输出保存在
`ckpt/{lewm,pldm}-contextworld-v1/action_delay_h7full_method_score.json`，逐检查点结果在
各 `checkpoints/<run>/eval_results/action_delay_h7full/result.json`。与第 4.1 节
32.43% / 93.36% 同 benchmark id、同 `aggregation`、同冻结资产集，可直接比较；历史 LeWM
回执的六组逐组准确率为 0%/13%/95.33%/84%/5.67%/0%（宏平均 33.00%，`gate.passed: false`），
与本批 LeWM 各组 ≥95.22% 形成同口径对照。

**自动管线的二元诊断（次要口径）。** 自动 Development 评测
管线为动作延迟构造的候选对是逐个“delay=0 对比某个非零延迟值”的配对
（`selection.rule: "d0 vs each listed contrast"`，delay_values 0–10 逐一配对，而不是
一次性的多分类）；但报告的 `correct_future_rate` 及其细分（`delay_0_correct_future_rate`、
`delayed_correct_future_rate`）只区分“delay=0”与“delay≠0（把 1–10 全部合并为一类）”
两个桶，因此下表数值是二元诊断分数。这与第 4.1 节 32.43%/93.36% 所用的“六组
（0/1/2/3/4/5–10）宏平均、最弱组 ≥60%”完整延迟识别评测是不同的评测问题；**正式结论
以六组口径为准**（见上方“动作延迟：正式六组口径”）。下表数值只能证明“历史让预测在
有无延迟之间正确切换”，不能证明模型识别了具体延迟步数对应的物理响应，也不能与
第 4.1 节的数字相减得出“提升”或“下降”的结论。

逐训练种子 ICL（Development，`correct_future_rate` 为主分数；动作延迟一行为二元
诊断分数，见上方说明；动作延迟的六组口径 Development 数值见下方“动作延迟
Development 六组口径”）：

| 任务 | 方法 | 种子 3072 | 种子 3073 | 种子 3074 | 均值 ± 样本标准差 | 正确历史范围 | 上下文切换范围 | 最弱条件范围 |
|---|---|---:|---:|---:|---:|---|---|---|
| 推手移动幅度 | LeWM | 50.00% | 50.00% | 50.00% | 50.00% ± 0.00pp | 49.02%–50.20% | n/a | 5.47%–7.03% |
| 推手移动幅度 | PLDM | 50.00% | 50.20% | 50.00% | 50.07% ± 0.12pp | 50.39%–50.78% | n/a | 38.28%–47.66% |
| 机械臂质量 | LeWM | 84.77% | 85.74% | 85.55% | 85.35% ± 0.51pp | 87.70%–90.23% | 96.48%–98.83% | 79.30%–81.64% |
| 机械臂质量 | PLDM | 52.54% | 54.69% | 50.59% | 52.61% ± 2.05pp | 54.49%–56.64% | 56.25%–59.38% | 48.83%–51.95% |
| 动作延迟 | LeWM | 99.83% | 100.00% | 99.50% | 99.78% ± 0.25pp | 99.67%–100.00% | 100.00% | 99.33%–100.00% |
| 动作延迟 | PLDM | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00pp | 100.00% | 100.00% | 100.00% |
| 接触摩擦 | LeWM | 50.00% | 50.00% | 50.20% | 50.07% ± 0.12pp | 50.59%–53.71% | 50.00%–57.03% | 46.48%–49.61% |
| 接触摩擦 | PLDM | 50.20% | 50.39% | 49.80% | 50.13% ± 0.30pp | 49.22%–52.15% | 48.44%–51.56% | 45.31%–51.17% |
| 运动阻尼 | LeWM | 50.00% | 50.20% | 50.00% | 50.07% ± 0.12pp | 49.80%–50.78% | 48.83%–57.03% | 26.17%–30.47% |
| 运动阻尼 | PLDM | 54.30% | 50.39% | 50.98% | 51.89% ± 2.11pp | 49.22%–53.13% | 55.47%–57.03% | 38.67%–50.00% |
| Cube 夹爪携带规则 | LeWM | 50.00% | 49.80% | 50.00% | 49.93% ± 0.12pp | 49.80%–49.80% | 46.09%–50.78% | 46.48%–49.22% |
| Cube 夹爪携带规则 | PLDM | 50.00% | 50.00% | 50.00% | 50.00% ± 0.00pp | 49.61%–50.98% | 53.52%–58.98% | 42.97%–50.00% |
| 门通行规则 | LeWM | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00pp | 100.00% | 100.00% | 100.00% |
| 门通行规则 | PLDM | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00pp | 100.00% | 100.00% | 100.00% |
| 传送门出口位置 | LeWM | 50.98% | 50.59% | 51.17% | 50.91% ± 0.30pp | 68.95%–74.61% | 100.00% | 33.98%–48.83%（bootstrap 下界 28.13%–35.55%） |
| 传送门出口位置 | PLDM | 67.38% | 72.66% | 68.55% | 69.53% ± 2.77pp | 75.39%–80.27% | 100.00% | 57.42%–64.84%（bootstrap 下界 51.17%–58.98%） |

逐训练种子原任务 CEM（每枚检查点 300 次，6 个评测种子 42–47 × 50）：

| 任务 | 方法 | 种子 3072 | 种子 3073 | 种子 3074 | 均值 ± 样本标准差 | 原始基线均值 |
|---|---|---:|---:|---:|---:|---:|
| 推手移动幅度 | LeWM | 235/300 | 238/300 | 207/300 | 75.55% ± 5.70pp | 82.22%（PushT） |
| 推手移动幅度 | PLDM | 180/300 | 196/300 | 178/300 | 61.55% ± 3.29pp | 75.67%（PushT） |
| 机械臂质量 | LeWM | 174/300 | 175/300 | 178/300 | 58.55% ± 0.69pp | 55.89%（Reacher） |
| 机械臂质量 | PLDM | 137/300 | 145/300 | 114/300 | 44.00% ± 5.36pp | 69.67%（Reacher） |
| 动作延迟 | LeWM | 270/300 | 259/300 | 257/300 | 87.33% ± 2.33pp | 92.44%（TwoRoom） |
| 动作延迟 | PLDM | 273/300 | 273/300 | 272/300 | 90.89% ± 0.19pp | 90.56%（TwoRoom） |
| 接触摩擦 | LeWM | 222/300 | 229/300 | 234/300 | 76.11% ± 2.01pp | 82.22%（PushT） |
| 接触摩擦 | PLDM | 194/300 | 194/300 | 211/300 | 66.56% ± 3.27pp | 75.67%（PushT） |
| 运动阻尼 | LeWM | 240/300 | 225/300 | 249/300 | 79.33% ± 4.04pp | 82.22%（PushT） |
| 运动阻尼 | PLDM | 188/300 | 183/300 | 183/300 | 61.56% ± 0.96pp | 75.67%（PushT） |
| Cube 夹爪携带规则 | LeWM | 185/300 | 181/300 | 182/300 | 60.89% ± 0.70pp | 65.44%（Cube） |
| Cube 夹爪携带规则 | PLDM | 180/300 | 172/300 | 161/300 | 57.00% ± 3.18pp | 53.44%（Cube） |
| 门通行规则 | LeWM | 292/300 | 272/300 | 292/300 | 95.11% ± 3.85pp | 92.44%（TwoRoom） |
| 门通行规则 | PLDM | 287/300 | 291/300 | 284/300 | 95.78% ± 1.17pp | 90.56%（TwoRoom） |
| 传送门出口位置 | LeWM | 295/300 | 298/300 | 295/300 | 98.66% ± 0.58pp | 92.44%（TwoRoom） |
| 传送门出口位置 | PLDM | 277/300 | 290/300 | 283/300 | 94.44% ± 2.17pp | 90.56%（TwoRoom） |

（速度、门通行规则以外任务的“正确历史/上下文切换”留空表示该任务的自动评测输出未产出
该字段，不代表未检查；门通行规则、动作延迟、速度沿用各自任务的方向/分组结构，逐评测
种子明细见对应 `result.json`。）

**保持/未保持的判定方法。** 上表 CEM 的保持判定，是新 CEM 均值与本文第 3 节
原始环境 CEM 均值的直接比较，不是第 4.2 节所用的逐评测种子配对 bootstrap 非劣性检验——
这批结果尚未运行该级统计检验。需要正式非劣性判定时，应使用与冻结 scoreboard
相同的配对协议重新计算。

**种子间方差 vs. 批次间落差。** 同一批次内，三个训练种子彼此非常接近：18 个“任务×
家族”组合中，ICL 主分数的种子间样本标准差全部 ≤2.8pp，其中 8 个 ≤0.3pp；CEM 均值的
种子间样本标准差在 0.19pp–5.70pp 之间，与原始基线 CEM 本身的种子间方差（如 Reacher
PLDM 基线 ±20.25pp）同一量级，不构成异常。相比之下，本节与
[历史归档](../archive/LeWM_PLDM_Pre_2026-09-03_Reference_Results.md)之间的
“旧结果 → 新结果”落差普遍在 30–70pp（如接触摩擦、运动阻尼、Cube、推手移动幅度），
远超种子间方差。这些落差来自训练配方差异（旧批各行使用
`mixed_frozen_image_paired_future_*` / `mixed_pldm_*` / `H3_Passage_MixedRules_*` /
`h7_action_delay_curriculum_v4` 等不同配方，训练量 1024–8192 步不等，训练种子也与
3072/3073/3074 不同）、训练协议（`v4r1` 对比 `joint_scratch_v1`）或批次间代码/数据
快照差异，而不是随机种子波动；四条差异轴的定义见主文档 §5.3，各轴的单独贡献量尚需
受控消融测定。动作延迟不计入这一比较：新旧数值分别来自二元诊断与六组宏平均两个不同
的评测问题（见上方“自动管线的二元诊断”），其差不能作为方法或批次差异的证据。在各
差异轴的单独贡献量测得之前，不应假定 `native` 配方在下一次重训中会重复同样的结果。

**动作延迟 Development 六组口径。** 自动 post-train 管线对动作延迟的 Development 评分
（`contextworld/benchmarks/bundle_development.py` 的 action_delay 路径）为六组识别，复用
`contextworld/evaluation/action_delay_h7_core.py` 的 `physical_group` 与
`summarize_action_delay_h1_physical` 内核，不读 Public Test 资产，保持 Development 不出
gate 的协议边界；其随机基线为 16.67%，与二元诊断（随机基线 50%）不是同一指标。全部
9 枚检查点的结果：

| 模型 | 种子 3072 | 种子 3073 | 种子 3074 | 均值 ± 样本标准差 | 最弱组范围 |
|---|---:|---:|---:|---:|---|
| 动作延迟 LeWM | 96.48% | 99.91% | 97.59% | 97.99% ± 1.75pp | 93.33%–99.44% |
| 动作延迟 PLDM | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00pp | 100.00% |
| 动作延迟 DINO-WM | 16.67% | 16.67% | 16.67% | 16.67% ± 0.00pp | 6.67%–10.00% |

输出在 `<run>/eval_results/benchmark_icl_sixgroup_frozenenv_v1/action_delay/result.json`，
每份输出满足：六组字段齐全、组划分为 `[0]/[1]/[2]/[3]/[4]/[5–10]`、不含二元字段、
无 `gate` 字段、`evaluation_split: development`、`bundle.manifest_sha256` 为 rc1
（`46f5cdb8…`）。执行环境按 `requirements_frozen.txt` 用 uv 复现
（`torch 2.9.1+cu128`、`transformers 4.57.1`、`numpy 1.26.4`），未安装训练/加速重型件
（vllm、sglang、transformer_engine、flash-attn、torch-tensorrt），评测链路不触达它们。

两点判读：Development 六组值与 Public Test 六组值一致（LeWM 97.99% 对 97.75%、
PLDM 100.00% 对 100.00%），两列数据划分不同、口径相同；DINO-WM 三个种子精确落在随机
基线 16.67%，逐组分数无结构，二元口径下的 50.00% 是同一事实在 50% 基线下的读数，容易
被误读为“对了一半”。二元与六组的数值**不可相减**。

**Public Test 评测（2026-09-07）。** 对本批中已通过 Development 准入门槛的
组件，按“Development-only selection, Test-only final reporting”完成 Public Test 评测。
未通过 Development 的组件按协议不授权读取 Test，仍为空。

| 组件 | 方法 | 划分 | 种子 3072 | 种子 3073 | 种子 3074 | 均值 | 判定 | 判定来源 |
|---|---|---|---:|---:|---:|---:|---|---|
| 速度（严格口径） | LeWM | Public Test | 98.33% | 98.67% | 97.56% | 98.19% | core track 3/3 | 逐检查点 |
| 速度（严格口径） | PLDM | Public Test | 97.89% | 94.44% | 97.78% | 96.70% | core track 3/3 | 逐检查点 |
| 机械臂质量 | LeWM | Public Test | 85.16% | 86.72% | 84.38% | 85.42% | 3/3 通过 | CLI 方法级 |
| 门通行规则 | LeWM | Public Test | 100.00% | 99.67% | 100.00% | 99.89% | 3/3 通过 | 逐检查点 |
| 门通行规则 | PLDM | Public Test | 100.00% | 100.00% | 100.00% | 100.00% | 3/3 通过 | 逐检查点 |
| 动作延迟（六组） | LeWM | Public Test | 97.23% | 98.31% | 97.71% | 97.75% | 3/3 通过 | CLI 方法级 |
| 动作延迟（六组） | PLDM | Public Test | 100.00% | 100.00% | 100.00% | 100.00% | 3/3 通过 | CLI 方法级 |

判定来源必须区分：动作延迟与机械臂质量由 CLI 产出三训练种子方法级回执
（`submission_kind: three_seed_method`）；速度与门通行规则的 CLI 目前只产出逐检查点
判定（其汇总文件自带 `cli_method_level_verdict_available: false` 与
`_DISCLAIMER: NOT a formal method-level verdict`），表中 “3/3” 指三枚检查点各自的
`checkpoint_passed` / `formal_within_checkpoint_pass` 全为真。在这两个方法级
emitter 提供之前，速度与门通行规则不应被写成与前两项同级的方法级结论。

速度的两条 extrapolation track（训练范围外速度）逐检查点 0/3，属协议内单独登记的
诊断轨，不影响 core track 判定。输出位置：
`checkpoints/<run>/eval_results/{speed_formal,robot_arm_mass_formal,door_formal}/`
与 `checkpoints/<run>/eval_results/benchmark_icl_publictest/<task>/result.json`
（`evaluation_split: "test"`、`official_scoreboard_row: false`、
`selection_policy: development_only_model_selection_test_final_reporting`）；三种子
汇总在 `ckpt/{lewm,pldm}-contextworld-v1/*_three_seed_*.json`。

**本批训练后原任务 CEM。** 速度的逐检查点 CEM 为 LeWM 97.33%/97.33%/97.67%（均值
97.44% ± 0.19pp）、PLDM 97.00%/96.67%/97.00%（均值 96.89% ± 0.19pp）。第 4.2 节
速度两行（LeWM 96.33%/94.67%/96.67，PLDM 94.33%/94.33%/95.67）属于**更早一批
检查点**，不是本批数值，两者不得互换引用。

机器可读来源：本节数据直接读取自 `ckpt/lewm-contextworld-v1/checkpoints/<run>/
eval_results/benchmark_icl/<task>/result.json` 与同目录下
`eval_results/benchmark_cem/<task>/*_metrics.json`（`pldm-contextworld-v1` 同构），
当前结果、独立门槛决策与源文件身份汇总在
`configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v2.json`。
早期 `contextworld_native_v1_2026-09-03_snapshot.json` 只有 60 条记录，保留为阶段快照，
不能代表当前完整的 81 个训练后单元。冻结参考结果也不自动成为官方 scoreboard 行。
旧配方 / `v4r1` / legacy 历史结果见
[《LeWM / PLDM：2026-09-03 之前的参考结果》](../archive/LeWM_PLDM_Pre_2026-09-03_Reference_Results.md)。

## 5.2 Development 逐单元数值

Development 用于实现检查、训练配方选择与准入；Public Test 用于最终报告。两者使用相同
主指标、门槛阈值与要求的分层结构，数据行保持隔离。Development 原始信封不输出 `gate`，
独立决策回执用同一内核判定；不能将 Development 数值标成 Test 成绩。

下表由当前 v2 冻结记录生成。数值为百分比，`±` 为三个训练种子的样本标准差，括号为
逐种子值。主分数与门槛总判定分别记录；主分数高不能替代全部门槛通过。

| 任务 | 模型 | 随机基线 | 训练前（逐种子） | 训练后（逐种子） |
|---|---|---:|---:|---:|
| 速度 | LeWM | 33.33% | 59.15 ± 2.37（56.44 / 60.11 / 60.89） | 98.00 ± 0.44（97.56 / 98.44 / 98.00） |
| 速度 | PLDM | 33.33% | 48.48 ± 4.36（50.89 / 43.44 / 51.11） | 97.19 ± 1.88（97.67 / 95.11 / 98.78） |
| 速度 | DINO-WM | 33.33% | — | 56.56 ± 3.84（55.22 / 53.56 / 60.89） |
| 推手移动幅度 | LeWM | 50% | 49.15 ± 0.92（48.44 / 50.20 / 48.83） | 50.00 ± 0.00（50.00 / 50.00 / 50.00） |
| 推手移动幅度 | PLDM | 50% | 47.66 ± 1.03（47.27 / 48.83 / 46.88） | 50.07 ± 0.11（50.00 / 50.20 / 50.00） |
| 推手移动幅度 | DINO-WM | 50% | — | 89.19 ± 0.81（89.84 / 89.45 / 88.28） |
| 机械臂质量 | LeWM | 50% | 50.13 ± 0.30（50.39 / 49.80 / 50.20） | 85.35 ± 0.52（84.77 / 85.74 / 85.55） |
| 机械臂质量 | PLDM | 50% | 51.43 ± 0.49（51.37 / 51.95 / 50.98） | 52.60 ± 2.05（52.54 / 54.69 / 50.59） |
| 机械臂质量 | DINO-WM | 50% | — | 51.37 ± 0.68（50.59 / 51.76 / 51.76） |
| 动作延迟 | LeWM | 16.67% | 16.67 ± 0.00（16.67 / 16.67 / 16.67） | 97.64 ± 0.38（97.21 / 97.75 / 97.95） |
| 动作延迟 | PLDM | 16.67% | 16.66 ± 0.01（16.67 / 16.65 / 16.67） | 99.98 ± 0.02（100.00 / 99.96 / 99.99） |
| 动作延迟 | DINO-WM | 16.67% | — | 16.63 ± 0.03（16.61 / 16.61 / 16.67） |
| 接触摩擦 | LeWM | 50% | 49.87 ± 0.45（49.61 / 49.61 / 50.39） | 50.07 ± 0.11（50.00 / 50.00 / 50.20） |
| 接触摩擦 | PLDM | 50% | 49.80 ± 0.20（50.00 / 49.80 / 49.61） | 50.13 ± 0.30（50.20 / 50.39 / 49.80） |
| 接触摩擦 | DINO-WM | 50% | — | 49.61 ± 0.20（49.61 / 49.41 / 49.80） |
| 运动阻尼 | LeWM | 50% | 49.93 ± 0.11（49.80 / 50.00 / 50.00） | 50.07 ± 0.11（50.00 / 50.20 / 50.00） |
| 运动阻尼 | PLDM | 50% | 50.26 ± 0.23（50.39 / 50.39 / 50.00） | 51.89 ± 2.11（54.30 / 50.39 / 50.98） |
| 运动阻尼 | DINO-WM | 50% | — | 50.13 ± 0.11（50.00 / 50.20 / 50.20） |
| Cube 夹爪携带规则 | LeWM | 50% | 51.95 ± 0.70（51.37 / 52.73 / 51.76） | 49.93 ± 0.11（50.00 / 49.80 / 50.00） |
| Cube 夹爪携带规则 | PLDM | 50% | 50.65 ± 0.45（50.39 / 50.39 / 51.17） | 50.00 ± 0.00（50.00 / 50.00 / 50.00） |
| Cube 夹爪携带规则 | DINO-WM | 50% | — | 50.33 ± 0.11（50.20 / 50.39 / 50.39） |
| 门通行规则 | LeWM | 50% | 50.00 ± 0.00（50.00 / 50.00 / 50.00） | 100.00 ± 0.00（100.00 / 100.00 / 100.00） |
| 门通行规则 | PLDM | 50% | 50.00 ± 0.17（50.00 / 49.83 / 50.17） | 100.00 ± 0.00（100.00 / 100.00 / 100.00） |
| 门通行规则 | DINO-WM | 50% | — | 100.00 ± 0.00（100.00 / 100.00 / 100.00） |
| 传送门出口位置 | LeWM | 50% | 50.00 ± 0.00（50.00 / 50.00 / 50.00） | 50.91 ± 0.30（50.98 / 50.59 / 51.17） |
| 传送门出口位置 | PLDM | 50% | 50.07 ± 0.11（50.00 / 50.20 / 50.00） | 69.53 ± 2.77（67.38 / 72.66 / 68.55） |
| 传送门出口位置 | DINO-WM | 50% | — | 99.41 ± 0.20（99.22 / 99.41 / 99.61） |

**当前共有的评分合同。** 速度按参考速度比较匹配历史与其余全部历史；门通行使用三个
历史条件、真规则与方向分层、每种子 50 query；动作延迟使用六个物理响应组主指标及完整
门槛输入，300 query 按六个评测种子组织；其余六项均为配对双目标判别。
历史 Speed history-utility、Door 288 对和 ActionDelay 30-query 结果不再作为当前比较值。

## 6. DINO-WM / PreJEPA 非冻结补充证据

### 6.1 仅用原环境数据训练的检查点

四个原环境各训练三枚 DINO-WM / PreJEPA 检查点，训练种子 3072、3073、3074，只使用原
环境数据，训练 10 epochs，History=3。汇总工件的状态为
`completed_non_frozen_v1_supplementary_evidence`。

严格的 frozen-v1 轨道包含 27 个任务×种子单元。由于原始检查点的 predictor 需要
`observation` 或 `proprio`，且动作延迟另有 H3 与 H7 的不匹配，这些单元均因输入合同
不兼容而未按正式接口计分（工件状态为 `not_compatible`）。诊断轨道把缺失输入固定为
模型归一化空间中的零
（`missing_context_policy: normalized_zero`，`privileged_state_read: false`），完成 27 个
单元，全部未通过对应门限。主文档综合表中标注（诊）的数值即来自该诊断，逐训练种子
数值保存在同一工件中。

原始环境 CEM 使用六个评测种子（42–47）× 50 次，每枚检查点 300 次，标记为
`complete_standard_post_eval_checkpoint_level_evidence` 且 `official_frozen_matrix: false`。
逐训练种子成功数为 TwoRoom 292/296/298、PushT 139/140/118、Reacher 163/162/182、
Cube 219/217/224，逐评测种子成功数同样保存在工件中。评测种子 45–47 是在不重跑既有
有效单元的前提下补齐的；失败尝试未产生有效结果，也未纳入汇总。

机器可读来源：

- `artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json`

### 6.2 使用 ContextWorld-v1 组件数据训练的检查点

以下记录保留七组件初次完成时的 Development 数值与运行恢复经过；它不是当前主分数表。
当前九组件 × 三种子均已完成，最终 Development 数值与全部身份见 §5.2 和 v2 freeze。
该阶段七个组件各完成三种子的 10-epoch 从头训练、ICL 及原任务 CEM，共 21 个单元。其中 19 份原
manifest 为 `completed`；Cube 的种子 3073 和 3074 保留原 manifest 中六个完成的 CEM
单元，并分别由新的 `completed` ICL recovery manifest 补齐。恢复过程没有改写失败记录。

| 组件 | Development 主指标（三个训练种子） | CEM 成功数（各 300 次） |
|---|---|---|
| 速度 | 41.67%、37.50%、32.99%（history utility） | 300、299、300 |
| 门通行规则 | 100.00%、100.00%、100.00% | 293、293、292 |
| 传送门出口位置 | 99.22%、99.41%、99.61% | 298、296、293 |
| 接触摩擦 | 49.61%、49.41%、49.80% | 74、82、91 |
| 运动阻尼 | 50.00%、50.20%、50.20% | 75、66、89 |
| 机械臂质量 | 50.59%、51.76%、51.76% | 163、155、154 |
| Cube 夹爪携带规则 | 50.20%、50.39%、50.39% | 216、207、227 |
| 推手移动幅度（2026-09-03 首次训练） | 89.84%、89.45%、88.28%（均值 89.19% ± 0.81pp） | 228、226、234 |
| 动作延迟（2026-09-03 首次可评分，原生 History=7） | 50.00%、50.00%、50.00% | 141、68、93 |

动作延迟的三个 `16-mixed` History=7 训练分别在 epoch 2、2、4 出现 NaN 损失，均未
生成可评分的 epoch-10 检查点；失败发生在不同批次，且失败前损失有限，合成数据中的动作
字段经全量扫描也没有非有限值，证据支持“混合精度数值不稳定”而不是“固定坏样本”。
2026-09-03 批次改用 `stable_worldmodel_prejepa_history7_v2` adapter（原生 History=7，
不使用 H3 尾部投影），产出三枚可评分的 epoch-10 检查点：三个训练
种子的 Development 主分数均为 50.00%（随机水平），数值不稳定问题不再出现，但模型
仍未学到动作延迟规律。其训练后原任务 CEM（TwoRoom，DINO-WM 原始基线均值 98.44%）
降至 141/300、68/300、93/300（均值 33.56% ± 12.36pp），是 2026-09-03 批次中相对
基线降幅最大的一项，尚无根因说明。推手移动幅度组件同批完成训练：Development
主分数 89.19% ± 0.81pp（正确历史 94.34%–96.48%，最弱强度条件 84.77%–87.50%，响应
增益 0.48–0.49），训练后 CEM 均值 76.44% ± 1.39pp，明显高于原始基线 44.11%（PushT，
139/140/118）。Cube 的三个 epoch-10 检查点均已完成 Development ICL 和全部六个 CEM
单元；种子 3073、3074 的 ICL 执行由独立 recovery manifest 补齐。

这些早期回执当时声明 `public_test_accessed=false`、`formal_pass_available=false`、
`official_scoreboard_row=false`；后续 Test 执行独立登记，这些字段不能证明此后也未读取 Test。
历史机器可读来源：

- `configs/benchmark/contextworld_dinowm_component_development_results_v1.json`
  （推手移动幅度、动作延迟两行尚未并入此工件，当前直接读取自
  `ckpt/dino-wm-contextworld-v1/checkpoints/{action_strength,action_delay}_prejepa_joint_scratch_v1_s{3072,3073,3074}/eval_results/`）

## 7. 完整对照记录（非 scoreboard）

`complete_reference_comparison_v2` 是一份独立记录：它按“全部报告”策略为每个冻结对照
运行 ICL 与 CEM，阈值只决定判定、不决定是否执行，并声明
`comparison_addendum_is_a_formal_scoreboard_rewrite: false`、
`historical_scoreboard_rows_unchanged: true`。其中的数值不是 scoreboard 行。

该记录补齐了冻结 scoreboard 标记为 `NOT_EVALUATED` 的三项原任务 CEM：

| 组件 | 方法 | 逐检查点成功数（各 300 次） | 非劣性下限 | 通过检查点 |
|---|---|---|---:|---|
| 推手移动幅度 | PLDM | 227、230、229 | 218 | 3/3 |
| 传送门出口位置 | PLDM | 284、288、286 | 263 | 3/3 |
| 机械臂质量 | PLDM | 230、226、231 | 233 | 0/3 |

这些结果不改变 ICL 判定：三个方法的 ICL 都未通过，因此在正式流程中仍然没有训练后
CEM 判定；主文档只将它们标作“补充”结果。

记录中还有两类对照，不能与当前配方的参考结果混读：

- **legacy 对照**：接触摩擦与运动阻尼的 LeWM/PLDM 检查点用更早的 2,048 对、4,096 步
  配方训练，在当前冻结 Public Test 上重打分，ICL 分别为 49.35%、50.00%、50.07% 和
  50.26%，均为 0/3；对应 PushT CEM 分别通过 2/3、1/3、1/3 和 0/3。工件明确声明它们
  “不是关于当前 8,192 对训练配方的证据”。
- **Cube PLDM 外部提交**：正式参考流程中该方法的 Public Test 仍记为
  `not_authorized_not_run`；同一批 v4r1 PLDM 检查点另以独立外部结果身份
  （`external_three_seed_method`、`formal_scoreboard_eligible: false`）在 Cube Public Test
  上得到 50.98%、50.78%、51.17%，0/3；其原 Cube CEM 为 168、164、164/300，相对基线
  159/300 通过非劣性。该提交不改写正式 scoreboard，其补充身份见
  [历史归档](../archive/LeWM_PLDM_Pre_2026-09-03_Reference_Results.md) §3。

机器可读来源：

- `artifacts/evaluation/complete_reference_comparison_v1/complete_comparison_v2.json`
- `artifacts/evaluation/complete_reference_comparison_v1/complete_comparison.json`（v1，未修改）
- `artifacts/evaluation/complete_reference_comparison_v1/cem/`、
  `artifacts/evaluation/complete_reference_comparison_v1/icl/`（逐种子聚合与逐 query 记录）

## 8. 目前不可获得的证据

现有测量与正式报告资格应分开理解：

- 原始 LeWM/PLDM 三种子 ICL 已完成：24 枚环境级检查点、54 个任务单元、两个划分共
  108 个分数，当前身份和重评记录见 v2 freeze。旧单检查点 `h3_origheldout_s3072` 的
  血统披露保留在历史归档，不混入新的原环境三种子统计。
- contact_friction/motion_damping 历史上实际产生了 30 个 Test 单元；由于未获准入，
  当前表格只报告其 Development 值。保留读取事实和排除理由，不声称 Test 从未打开。
- DINO-WM 原环境检查点的正式 ICL 输入合同仍不兼容，原任务 CEM 完整，零填充诊断另存。

以下归因或能力范围仍不由现有参考结果支持：

- `native` 配方与旧配方之间 30–70pp 落差的**各差异轴单独贡献量**：落差本身对应四条
  训练配方差异轴（配对关系是否进入训练目标、初始化是微调还是从零、`num_preds` 是 3
  还是 1、配对 clip 还是滑窗稠密 clip），见主文档 §5.3。四条同时变化，因此**不能做
  单因素归因**；要分离各轴贡献需要受控消融。已确认不是训练种子随机性（同批三个种子
  彼此高度一致，见 5.1 节“种子间方差 vs. 批次间落差”）；
- DINO-WM / PreJEPA 动作延迟组件训练后原任务 CEM 相对原始基线大幅下降（98.44%→
  33.56% ± 12.36pp）的根本原因：尚无诊断；
- 速度 PLDM 的同训练种子单速度对照：未预注册、未运行，因此不能做训练归因；
- 除速度已有的预设范围外推诊断外，其余连续参数任务的范围外外推，以及九项任务的通用
  多步闭环适应：不在现有冻结结论内；
- 仅用原环境数据训练的 DINO-WM / PreJEPA 在正式 ICL 接口下的分数：这些原始检查点不
  满足输入合同，只有诊断值；组件数据训练后的公开 Development 结果见第 6.2 节；
- 由独立实现、独立训练代码产生的第三方模型结果：仅 Cube 有试点，其余组件尚无。

## 9. 版本与完整性

复现实验应记录：代码提交、Stable-WorldModel 版本、数据 manifest SHA-256、训练配方、
训练种子、检查点 SHA-256、Adapter 版本、评测划分、评测种子和输出文件 SHA-256。

`docs/protocols/` 保存执行前确定的任务协议，`docs/archive/` 保存已经结束的实验阶段材料。
这些材料用于核对已报告结果，不是公开 Training 或 Development 工作流的运行依赖。

部分被引用的评测工件（尤其是 `artifacts/evaluation/history3/` 与
`artifacts/evaluation/history7/` 下的逐次记录）不随代码分发。它们的路径、SHA-256 和
大小固定在对应的 release 配置与冻结记录中，可据此核对；在缺少这些工件的检出中，相关
核验无法执行，也不应被当作已经通过。
