# ContextWorld 参考结果复现附录

本文记录主 Benchmark 文档中参考结果的来源、评测预算、门限判定和机器可读文件。普通
使用者运行 Training 或 Development 评测时不需要阅读本附录；复核表格、比较训练配方或
重新生成参考结果时再使用这些信息。

本文只记录仓库工件中可以查到的内容。文中给出的路径都相对仓库根目录。`artifacts/`
的大部分内容不随代码分发，因此在干净检出中可能缺失；本文对每一项都标明它属于哪一类
证据，缺失的测量也明确写出。

## 1. 证据分级

主文档的数值分为三类，彼此不可替代、不可合并：

1. **冻结的 LeWM / PLDM 协议结果**：正式参考矩阵。包括原始检查点的 ICL 起点、原始
   环境 CEM，以及组件训练后的 scoreboard 行。
2. **非冻结补充证据**：DINO-WM / PreJEPA 的 ICL 诊断与 CEM，以及“完整对照记录”中
   补跑的结果。这些工件自身标注 `official_frozen_matrix: false` 或
   `formal_scoreboard_eligible: false`，不进入正式 scoreboard。
3. **仅在 Development 上评测的结果**：对应组件尚未开放 Public Test，因此不能作为
   Public Test 成绩。

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

本次预注册新执行的 17 个单元记录加载权重的 state-dict SHA-256 并通过一致性审计，以确认
评测过程没有改变模型权重；沿用自更早批次的 7 个单元在原始记录中不含该审计字段，工件
里按 `provenance` 区分。两项披露保留在工件中：

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
13314 和 13315。其余行的检查点顺序按冻结汇总记录，不再单独标注训练种子。

未通过的原因分别是：推手移动幅度 PLDM 三个检查点都低于 95%；机械臂质量 PLDM 三个
检查点都低于 75%；传送门出口位置 LeWM 与 PLDM 三个检查点都低于 95%；动作延迟 LeWM
除总体 32.43% 低于 75% 外，最弱物理响应组的准确率为 0%，同时不满足 60% 的分组下限。
作为对照，动作延迟 PLDM 的最弱响应组平均 84.83%，配对 bootstrap 95% 下界平均 91.71%。

### 4.2 训练后原任务 CEM

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
正式流程，不表示对应的原始检查点或原始环境 CEM 缺失；后续在非 scoreboard 的对照记录中
补跑的结果见第 7 节。

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

## 5. 仅在 Development 上评测的结果与门限失败说明

接触摩擦、运动阻尼与 Cube PLDM 停在 Development，Public Test 未打开、未读取、未评分，
因此没有 Public Test 成绩，也没有训练后 CEM。接触摩擦与运动阻尼的参考矩阵各自只完成
一个训练种子（分别是 13313 和 14321），其余种子未运行。

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

## 6. DINO-WM / PreJEPA 非冻结补充证据

四个原环境各训练三枚 DINO-WM / PreJEPA 检查点，训练种子 3072、3073、3074，只使用原
环境数据，训练 10 epochs，History=3。汇总工件的状态为
`completed_non_frozen_v1_supplementary_evidence`。

严格的 frozen-v1 轨道包含 27 个任务×种子单元。由于原始检查点的 predictor 需要
`observation` 或 `proprio`，且动作延迟另有 H3 与 H7 的不匹配，这些单元均因输入合同
不兼容而未按正式接口计分（工件状态为 `not_compatible`）。诊断轨道把缺失输入固定为
模型归一化空间中的零
（`missing_context_policy: normalized_zero`，`privileged_state_read: false`），完成 27 个
单元，全部未通过对应门限。主文档 5.3 的均值即来自该诊断，逐训练种子数值保存在同一
工件中。

原始环境 CEM 使用六个评测种子（42–47）× 50 次，每枚检查点 300 次，标记为
`complete_standard_post_eval_checkpoint_level_evidence` 且 `official_frozen_matrix: false`。
逐训练种子成功数为 TwoRoom 292/296/298、PushT 139/140/118、Reacher 163/162/182、
Cube 219/217/224，逐评测种子成功数同样保存在工件中。评测种子 45–47 是在不重跑既有
有效单元的前提下补齐的；失败尝试未产生有效结果，也未纳入汇总。

机器可读来源：

- `artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json`

## 7. 完整对照记录（非 scoreboard）

`complete_reference_comparison_v2` 是一份独立记录：它按“全部报告”策略为每个冻结对照
运行 ICL 与 CEM，阈值只决定判定、不决定是否执行，并声明
`comparison_addendum_is_a_formal_scoreboard_rewrite: false`、
`historical_scoreboard_rows_unchanged: true`。其中的数值不是 scoreboard 行。

补跑了冻结 scoreboard 记为 `NOT_EVALUATED` 的三项原任务 CEM：

| 组件 | 方法 | 逐检查点成功数（各 300 次） | 非劣性下限 | 通过检查点 |
|---|---|---|---:|---|
| 推手移动幅度 | PLDM | 227、230、229 | 218 | 3/3 |
| 传送门出口位置 | PLDM | 284、288、286 | 263 | 3/3 |
| 机械臂质量 | PLDM | 230、226、231 | 233 | 0/3 |

这些结果不改变 ICL 判定：三个方法的 ICL 都未通过，因此在正式流程中仍然没有训练后
CEM 结论。

记录中还有两类对照，不能与当前配方的参考结果混读：

- **legacy 对照**：接触摩擦与运动阻尼的 LeWM/PLDM 检查点用更早的 2,048 对、4,096 步
  配方训练，在当前冻结 Public Test 上重打分，ICL 分别为 49.35%、50.00%、50.07% 和
  50.26%，均为 0/3；对应 PushT CEM 分别通过 2/3、1/3、1/3 和 0/3。工件明确声明它们
  “不是关于当前 8,192 对训练配方的证据”。
- **Cube PLDM 外部提交**：参考流程中该方法的 Public Test 仍记为
  `not_authorized_not_run`；同一批 v4r1 PLDM 检查点另以独立外部结果身份
  （`external_three_seed_method`、`formal_scoreboard_eligible: false`）在 Cube Public Test
  上得到 50.98%、50.78%、51.17%，0/3；其原 Cube CEM 为 168、164、164/300，相对基线
  159/300 通过非劣性。该提交不改写参考结果，也不改变第 5 节的 Development 结论。

机器可读来源：

- `artifacts/evaluation/complete_reference_comparison_v1/complete_comparison_v2.json`
- `artifacts/evaluation/complete_reference_comparison_v1/complete_comparison.json`（v1，未修改）
- `artifacts/evaluation/complete_reference_comparison_v1/cem/`、
  `artifacts/evaluation/complete_reference_comparison_v1/icl/`（逐种子聚合与逐 query 记录）

## 8. 目前不可获得的证据

以下内容在仓库中没有测量结果，不应从现有数字外推：

- 接触摩擦与运动阻尼的 Public Test 模型分数：Public Test 未打开、未读取、未评分；
- 接触摩擦与运动阻尼的第二、第三个训练种子：未运行；
- 接触摩擦、运动阻尼与 Cube PLDM 的训练后原任务 CEM（当前配方）：未运行；
- 速度 PLDM 的同训练种子单速度对照：未预注册、未运行，因此不能做训练归因；
- 除速度已有的预设范围外诊断外，其余连续参数任务的范围外外推，以及九项任务的通用
  多步闭环适应：不在现有冻结结论内；
- DINO-WM / PreJEPA 在正式 ICL 接口下的分数：原始检查点不满足输入合同，只有诊断值；
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
