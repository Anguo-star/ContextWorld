# 评测标准对齐历史工作记录（截至 2026-09-10 封板前）

**历史归档。** 以下状态按当时执行过程保留，不代表当前缺口。最终收尾与验收见
[Baseline_Closure_2026-09-10.md](Baseline_Closure_2026-09-10.md)。

---

## 一、标准（我的理解，待确认）

以下是我对要求的理解。**如有偏差请直接改这一节**，后续所有执行以本节为准。

### S1　评价方式必须能体现"是否真正学到该能力"

主分数不足以支撑这个判断，必须同时具备防捷径门槛。以其余六个组件的水平为基准，
一个组件的评测协议应当包含：

| 门槛 | 作用 |
|---|---|
| 主分数（配对判别正确率或分组宏平均） | 能否选对真实未来 |
| 最弱条件正确率 | 是否只在某一类隐藏条件上有效 |
| 正确历史使用率 | 是否用了与真实规律匹配的那段历史 |
| 上下文 / 规律切换率 | 换历史是否真的改变预测 |
| latent 响应增益 ≥ 0.50 | 预测响应幅度是否真由历史驱动（按真实响应归一化） |
| 归一化响应误差 < 1.00 | 是否优于"完全不使用历史"的锚点 |
| paired bootstrap 95% 下界 | 结论是否稳健 |

**判据**：缺这套检查的组件，其高分不能排除捷径，不能作为"学到"的证据。

### S2　一套标准，不并存两套

同一任务不得同时报告两个随机基线不同的主分数（例如动作延迟的二元诊断 50% 与
六组宏平均 16.67%）。诊断指标可作附加字段保留，但不得作为主分数，也不得与主分数并列比较。

### S3　Development 与 Public Test 结构完全相同，只有数据行不重叠

- **相同**：主分数定义、历史条件集合、分层维度、聚合方式、门槛项及阈值
- **不同**：只有具体数据行（场景、门位、参考速度、episode、query）不得重叠
- **理由**：dev 用于配方与检查点选择；结构相同才能与 test 对读且不产生内部结论冲突，
  数据不重叠才不泄露 test 分布

**明确排除的做法**：
- ✗ 把 Test 的门槛降级去迁就 Development（"取公共子集"）
- ✗ 以"dev 只是诊断"为由允许结构不同
- ✗ 让 dev 与 test 使用相同数据行

### S4　不得降低任何已有标准

Test 侧已有的门槛、数据、已冻结结果一律不动。对齐只能通过**补齐较弱的一侧**实现。

### S5　执行纪律

- 断言"某数据/字段/commit 不存在"之前，先读报错原文，再换一条访问路径验证一次
- 不得把明确要求按后续上下文重新解释；规格里写死判据
- 做不到就报"做不到 + 缺什么"，不要自行改判据或降低目标

---

## 二、当前状态

### 已完成且有效

| 项 | 状态 |
|---|---|
| 训练后侧 ICL：3 模型 × 9 组件 × 3 种子 × 2 划分 | 162/162 |
| 基线侧 ICL：4 环境 × 2 家族 × 3 种子 × 各环境组件 × 2 划分 | 108/108 |
| 信封校验（dev 为 rc1 且无 `gate`；test 为 rc2 且 `official_scoreboard_row: false`） | 270/270 通过 |
| 动作延迟主分数统一为六组宏平均（dev 与 test 同定义） | 完成 |
| 历史批次训练身份核实（十行原记录全部有误，已更正） | 完成 |
| 源码 pin 审计改为语义指纹三级判定 | 完成 |
| Public Test 最终报告统一入口 `contextworld-public-test-report` | 完成 |

数据身份：Development 全部 `ContextWorld-v1`（rc1，manifest `46f5cdb8…`）；
Public Test 全部 `ContextWorld-v1-full`（rc2，manifest `81eb29d8…`）。
评测环境：uv venv 按 `requirements_frozen.txt` 复现（torch 2.9.1+cu128、
transformers 4.57.1、numpy 1.26.4）。

### 未达标（本轮要解决的）

**问题 A：三个组件的 Test 门槛弱于其余六项（违反 S1）**

| 组件 | 历史使用 | 切换率 | 响应增益 | 归一响应误差 | 最弱条件 | bootstrap |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 速度 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 动作延迟 | ✗ | ✗ | ✗ | ✗ | ✗ | 有 |
| 门通行规则 | 有 | ✗ | ✗ | ✗ | ✗ | 有 |
| 其余六项 | 有 | 有 | 有 | 有 | 有 | 部分有 |

后果：这三项的高分（例如速度 LeWM Test 98.19%）**未经"是否走捷径"检验**。

**问题 B：两个组件的 dev 与 test 结构不同（违反 S3）**

| 组件 | Development | Public Test | 缺口 |
|---|---|---|---|
| 速度 | 96 成员，8 速度，group 与速度 1:1，同一 query 只有"匹配 / 一个非匹配"两条件 | 每参考速度三条件 `history_low/mid/high`，要求优于其余每一个 | dev 数据缺"同一 query 三个速度的历史" |
| 门通行规则 | 32 成员 = 16 训练相邻门位 × 2 规则，288 对，两个历史条件，无 eval_seed、无 direction 元数据、无分层 bootstrap | 42 个未见门位、1800 条，三个历史条件（含 `did_not_attempt_crossing`），按 `eval_seed × direction` 分层 + 分层 bootstrap | dev 数据缺第三条件、eval_seed、direction |

其余七项 dev/test 同结构（动作延迟已于本轮统一主分数，但门槛仍缺，属问题 A）。

---

## 三、执行计划

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| 0 | 调查：补齐三项 Test 门槛需要什么（评分器改动 / 数据字段）；生成同结构 dev 数据需要什么参数与隔离键 | — | 进行中（线程 G2） |
| 1 | 补齐速度、动作延迟、门通行的 Test 门槛到六项齐全 | 0 | 完成（全部判定 (可算)；评分器已扩展并实跑验证；新阈值在 `configs/benchmark/contextworld_test_gate_completion_v1.yaml`，待主进程确认；3 个评分器文件的字节 pin 触发 7 个身份治理测试失败，待阶段 7 identity amendment。详见 audit scratchpad `report_H2.md`） |
| 2 | 生成速度的 dev 数据（三条件，与 test 参考速度/几何不重叠） | 1 | 完成（I2；速度 Lance 导出由 J2 补完 14 表。隔离审计 reset/pixel/static-id 交集均为空） |
| 3 | 生成门通行的 dev 数据（三条件 + eval_seed + direction，与 test 的 42 门位不重叠） | 1 | 完成（I2 生成 loader_val 版本后，J2 按 S3 保守读法追加 reset 元组级去重：loader_val 池去重后仅 84/87 每方向 < 150，改用 train ∪ loader_val 全训练几何门位池并相对全部 286 个 Test 候选 reset 元组去重，得 1120 候选（560/方向），重建为 `tworoom_hidden_passage_history3_dev_reset_isolated_v1`；代价是 300 条 query 中 257 条来自 train 分裂门位（训练见过），43 条来自 loader_val——已在 report_J2.md 披露） |
| 4 | 重算 `ContextWorld-v1` 包的 manifest / sha256 / VERSION，使新 dev 载荷可被 `resolve_development_payload` 读取 | 2、3 | 完成（`scripts/register_dev_structural_parity_payloads_v1.py`：旧 dev 原样移至 `development_legacy_v1/`，新 Lance 载荷注册为唯一 split=development 载荷；manifest 46f5cdb8… → 625abb18…，VERSION 1.0.0-rc1 → 1.0.1-rc1；9 组件 resolve 全通过；`refresh_hf_clean_metadata` 从此不适用于该树——旧行为本就不能加数据行。**K2 修订（1.0.2-rc1）**：door dev 载荷换回 I2 的 loader_val 版（48 表，release `…_dev_structural_parity_v1`），推翻阶段 3 的 reset 元组去重路线——reset 去重迫使 train∪loader_val 门位池，257/300 query 落训练门位，损害 dev 选型能力，而门位级隔离已保证数据行不重叠；manifest 625abb18… → fc0fffd3…（5366→4556 行），VERSION **追加** amendment `door_dev_payload_loader_val_revert_v1`（reset 元组与 Test 重合 98 个，登记不消除）；reset-isolated 318 表移出包、字节保全于 `$OUT/door_reset_isolated_v1/`；legacy/speed/Test/full 逐 sha 断言未动；9 组件 resolve 全通过。隔离审计对包内载荷重做：门位/像素/static-id/template ∩ Test 全空、train 门位 0、98 reset 重合如实登记。详见 audit scratchpad `report_K2.md`） |
| 5 | 改 `bundle_development.py` 的速度 / 门通行路径，复用 test 侧同一指标内核 | 4 | 完成（新读者 + Test 同名主分数与全部门槛输入指标：速度 `_loss_summary`/`speed_gate_completion_metrics`、门通行 `score_validation_assets`/`summarize_validation_records`/`door_gate_completion_metrics`，全部为 Test 评分器同一函数；无任何 gate/pass 字段；旧字段保留为附加诊断；测试 87 过 + 3 个 H2 遗留 pin 失败不变；两枚真实检查点实跑冒烟通过。期间发现并修复新读者把帧按步值直接索引像素列的缺陷（等价于所有 episode 读到 episode 0 的帧），已加真实 Lance 表回归测试。详见 report_J2.md。**K2 复核**：reader 代码零改动，对回退后的 loader_val 载荷重跑结构一致性 ALL ITEMS MATCH；LeWM s3072 实跑主分数 1.0000、六门槛输入/三条件/12×25 分层/无 gate 字段全过。**更正 J2 的归因**：该检查点在冻结 Public Test 上同样 1.0000（全部 12 cell、两规则），"dev 1.0 = train 门位可识"不成立——门位饱和是检查点真实水平，train 门位的实际影响是配对边际偏离 Test 0.067-0.077（loader_val 版仅 0.026-0.045，回退后 dev 更贴 Test）。详见 report_K2.md） |
| 6 | 重跑受影响单元的 dev 与 test | 1、5 | 进行中（dev 30 枚已完成并通过信封校验；Test 45 枚中门通行 15 + 动作延迟 lewm/pldm 6 在本会话完成，速度 15 + 动作延迟 dino/基线 9 由并行会话 contextworld-1d 执行。**另发现并补齐运行时一致性缺口**：机械臂质量 6 枚（3 枚 `benchmark_icl_publictest/` + 3 枚 `robot_arm_mass_formal/`）产于 cwenv 建立之前，不在任何一方的重跑计划内，已在 cwenv 中重跑。详见下方「阶段 6 验收：主分数位移的归因」） |
| 7 | 一次性重建主表、更新文档、跑测试、提交 | 6 | 进行中。已完成：①源码 pin 治理（见下）②§5.1 加「ICL 门槛结果」结论列并更新 4 处重跑数值 ③接触摩擦/运动阻尼 6 行改标 Development 口径。余：`test_public_document_numbers_match_frozen_results.py` 36 项需改指向归档文档 |

### 阶段 6 验收：主分数位移的归因（S4 判定：通过）

重跑后 30 枚可比 Test 单元中有 4 枚主分数发生位移（最大 0.167pp，全部集中在 LeWM，
PLDM 逐位不变）。按 S4，"只新增门槛、不动主分数"是硬闸门，因此逐项查证：

**一、改动本身不可能导致**。三个 ICL 评分器**零删除行**
（`action_delay_icl_score.py` +192/−0、`door_icl_score.py` +210/−0、`speed_icl_score.py` +312/−0）。
全仓仅两个文件有被修改的行，且是同一处无害改写：`return {` 改绑为局部名 `scored`，
其后 `if return_latents: scored["latents"] = …`，再 `return scored`。
默认 `return_latents=False` 时返回内容与改动前完全相同；主分数路径上的比较、
阈值、并列处理均未触及。

**二、改动前代码在新运行时复现的是新值**。以 `git archive HEAD` 只读导出改动前源码树
（门槛补齐尚未提交，故 HEAD 即改动前状态），确认其 `return_latents` 出现 0 次后，
在同一 cwenv 中重跑两枚位移单元：

| 单元 | 旧值（cwenv 之外） | 改动前代码 + cwenv | 改动后代码 + cwenv |
|---|---|---|---|
| `action_delay\|lewm\|3072` | 0.9723148148148149 | 0.9707407407407408 | 0.9707407407407408 |
| `door\|lewm\|3073` | 0.9966666666666667 | 0.995 | 0.995 |

改动前与改动后逐位相同，两者都不复现旧值。

**三、零改动组件出现同样的位移**。机械臂质量的评分器在本次补齐中零改动
（`git status --porcelain -- '*robot_arm_mass*'` 为空），是天然对照组，
却同样出现：MSE 均值在第 7 位有效数字漂移；比率类指标按 **1/512 的整数倍**翻转，
且有升有降（s3074 `context_switch_rate` +2/512）。未被改动的组件不可能受改动影响。

**结论**：位移的根因是评测运行时迁移到冻结环境 cwenv（torch 2.9.1+cu128），
与门槛补齐无关；判别余量小的边界 query 在新运行时翻转，故只在非满分模型上可见。
S4 未被破坏。证据落盘于 audit scratchpad `test_primary_drift_v1.json`。

**由此引出的一致性要求**：旧 Test 批次本身是混合运行时的产物。以 cwenv 建立时间
（2026-09-08 04:33:39）扫描全库 Test 结果，发现 12 份产于其前，除速度/动作延迟/
门通行的 6 份由重跑覆盖外，另有机械臂质量 6 份不在计划内，已补跑。重扫后
cwenv 之前的残留为 0——Test 列首次达成单一运行时。

**遗留改进项（不在本次范围）**：结果 JSON 记录了 `adapter.device` 但不记录
torch / CUDA 版本，混合运行时只能靠文件 mtime 反推。建议后续在结果信封中
写入运行时指纹。

### 阶段 6 的第二个发现：动作延迟历史使用门槛阈值不可达（已修正）

补齐后的门槛把 `correct_history_rate ≥ 0.75` 套到动作延迟上，结果**每个检查点都不通过**，
包括主分数满分（六组宏平均 1.0000）的 PLDM。查证后确认这是**误判，不是模型缺陷**。

**成因**。`physical_future_group(delay, horizon)` 在 horizon=1 时返回 `min(delay, 5)`，
其 docstring 写明"h1 时 delays 5..10 全部静止、因而构成同一组"。主分数已据此按**六个物理组**
计算宏平均。但 chr 的来源 `matching_history_strict_win`（`action_delay_h7_score.py:390`）
判据是 `matching < min(other)`，其中 `other` 含全部 10 个非匹配延迟，**包括物理上完全相同的孪生延迟**。
对目标延迟 5..10，匹配历史不可能严格优于它的五个同组孪生，只能靠数值巧合取胜，概率约 1/6。
于是理论上限为：

    (5 个可识别 + 6 × 1/6) / 11 = 6/11 = 0.545454…

PLDM 三个种子的实测 chr 恰为 0.5454545454545454，即**满分模型被钉死在解析上限**；
LeWM 的 0.5358/0.5406/0.5388 略低于上限，与其 0.97–0.98 的主分数吻合。**没有任何模型能达到 0.75。**

同一门槛块内还存在自相矛盾：`context_switch_rate` 的 `delay_pairs` 构造**已经**用
`physical_future_group(a,1) != physical_future_group(b,1)` 排除了同组比较（故 csr 干净地等于 1.0000），
唯独 chr 没有排除。

**修正**。把门槛所用的 chr 输入改为**组感知**：匹配历史只与**物理上可区分**的延迟比较，
与 csr 采用同一条排除规则。改动位于 `action_delay_icl_score.py`（本次新增的门槛代码），
从 horizon=1 的 records 现算，**不触碰**冻结评分器及其 `matching_history_strict_win_rate` 诊断。
结果中新增 `correct_history_rate_definition` 字段自述口径。

依 108,900 条 records 离线核算的各候选口径（据此选定组感知 h1）：

| 单元 | h1 现行 | **h1 组感知** | h2 | h3 |
|---|---|---|---|---|
| pldm s3072 | 0.5455（上限） | **1.0000** | 0.9927 | 0.9700 |
| lewm s3072 | 0.5358 | **0.9873** | 0.8948 | 0.8024 |
| base-lewm s3072 | 0.0000 | **0.0000** | 0.0006 | 0.0000 |

选组感知 h1 而非 h2：门槛块其余各项与潜在响应门槛均锚定在 h1，h2 会把第二个视界带进来；
h3 则把预测难度与历史使用混为一谈（LeWM 掉到 0.80）。

**修正后的判定**（15 枚 Public Test，主分数逐位未变，S4 成立）：

| 模型 | 六组宏平均 | 历史使用 | 五项门槛 |
|---|---|---|---|
| LeWM（合成数据训练） | 0.9707 / 0.9831 / 0.9767 | 0.987–0.995 | **5/5 通过** |
| PLDM（合成数据训练） | 1.0000 ×3 | 1.000 ×3 | **5/5 通过** |
| DINO-WM（合成数据训练） | 0.1661 / 0.1667 / 0.1667 | 0.194–0.196 | 1/5 |
| 原始数据基线 | 0.1667 ×6 | 0.000–0.019 | 1/5 |

门槛依然在做反捷径的活：DINO-WM 与全部基线的主分数正好落在随机水平 16.67%，
且响应增益≈0、归一响应误差≥1.00，被如实判为未学到该能力。

### 阶段 6 的第三个发现：动作延迟 Development 缺四项门槛输入（已补齐九枚，余六枚受限）

结构对齐（阶段 2–5）只覆盖了速度与门通行，**动作延迟的 Development 被漏掉**：
其结果只有最弱条件与 bootstrap 区间，缺历史使用、切换率、响应增益、归一响应误差四项，
且仍挂在旧包 manifest `46f5cdb8`。

已在 `bundle_development.py` 的动作延迟路径补齐：用 Test 同一个内核
`action_delay_gate_completion_metrics` 现算六项输入，`losses` 的对角为匹配历史、
非对角为替代历史，并按 `physical_group` 排除同组替代（与本页上一节同一条判据）。
输出经 `_decision_free` 过滤，仍不含 `gate` / `checks` / `passed`。
`tests/test_bundle_development.py` 18 项通过。

重跑后训练后侧九枚齐全（例：LeWM s3072 —— 主分数 0.9648 未变，历史使用 0.9848、
切换率 1.0000、响应增益 0.9984、归一响应误差 0.0103、最弱条件 0.9333、
bootstrap 下界 {切换率 1.0000, 历史使用 0.9576}，manifest 已是 `fc0fffd3`）。

**残留缺口（如实登记，不粉饰）**：六枚**基线** Development 单元仍是 09-09 的旧产物，
只有主分数、没有新的六项输入。原因是基线为 H3 检查点，需经 H7→H3 尾投影才能进入
动作延迟边界，而 `external_model_cli` 有意把 `--history-adapter h3_tail_projection`
限定给 PreJEPA；仓库自带的 `H3TailProjectionActionDelayAdapter` 又按 Test 几何
（[B,9,5,A]）构造，Development 载荷是 7 个动作块，直接复用会报几何不符。
旧结果出自一个仓库外的定制适配器 `DevelopmentH3TailProjectionAdapter`（结果里有记录），
要重建需专门做几何调和，未在本轮进行。

**对结论的影响：无。** 这六枚的主分数在两个划分上都正好等于随机水平 16.67%，
且其 Public Test 侧六项门槛全数不通过（历史使用 0.000–0.019、响应增益为负、
归一响应误差 > 1.00）。补上 Development 的门槛输入只会重复同一结论。
对比表的主分数列不受影响——该列 45 个模型×组件格全部 3/3 齐全。

### 阶段 7 的发现一：源码 pin 治理没有"新增门槛"这一档（已补，未放宽）

补齐门槛改到了五个**被冻结发布 pin 住**的评分器
（`speed/action_delay/door_icl_score.py`、`action_delay_h7_score.py`、`hidden_passage_validation.py`），
三级判定（字节一致 / 已登记接受 / 语义未变）全部不适用，审计如实报 DRIFTED。
两条看似可行的路都被判据挡住：

- 改冻结 yaml 里的 sha —— 在"不得触碰"清单里；
- 登记进现有 `accepted_metadata_correction` —— 其加载器**硬性拒绝**本情形：
  `model_results_changed` 为真即报错，且 classification 必须是
  `runtime_source_fingerprint_updated_behavior_unchanged`。本次是"评分面新增字段"，
  塞进"行为未变"是撒谎。

**没有放宽既有档位，而是新增一档并写死断言。** 新增 `load_additive_scoring_extension_transitions`
（与原加载器分开，原加载器一行未动），只接受 status 为 `accepted_additive_scoring_extension`
且满足：既有输出字段未变、主分数未因源码改动而变、封存结果已全部重跑、classification 为
`scoring_surface_extended_with_additive_gate_metrics`、`additive_only` 为真、每行必须带
重跑证据指针。任何改动了既有数值的情形仍然判 DRIFTED。
登记文件 `configs/benchmark/contextworld_additive_test_gate_completion_pin_transition_v1.yaml`
收录 11 条 (config, path, pinned_sha) 组合。

**additive_only 的事实依据（实测，非声明）**：主分数改动前后逐位相同（改动前源码树在同一
cwenv 中复现新值）；speed 是唯一改动前已有 gate_completion 的组件，其 4 track × 5 项共 20 个
值零变化；action_delay 与 door 改动前载荷完全不含 gate_completion，故新增不覆盖旧值。

### 阶段 7 的发现二：接触摩擦与运动阻尼不该有 Public Test 数值（已改）

`contextworld_icl_suite_v2.yaml` 把这两个组件登记为 `failed_development`。协议是
Development 决定选型、Public Test 只用于最终报告，Development 已在随机水平的组件不应
动用留出划分。实测与登记一致：三模型在 Development 上主分数 49.6–51.9%（随机 50%），
六项门槛输入 0/3 通过。原表给了这两个组件的 Public Test 数值，属于把未授权的留出划分
结果当成正式口径发布，已改为 Development 数值并逐格标注。

### 阶段 7 的发现三：主表缺结论列，S1 信息不可见（已补）

先前按"训练前后挨着对比、去掉 CEM 变化列"重排主表时，连同把判定列一起删掉了，导致
S1 要求的"高分是否排除了捷径"在表里读不出来。已加回「ICL 门槛结果」列，并显式写出
三类高分被否决的情形——其中传送门出口位置 / DINO-WM 主分数 99.35% 但 latent 响应增益
仅 0.355–0.410（下限 0.50），是本基准里最典型的"高分不等于学到"。

### 阶段 5 的一个约束
`bundle_development.py` **不得输出 `gate` 或任何 pass 字段**
（`tests/test_bundle_development.py` 有 `assert "gate" not in result` 固定）。
"门槛相同"与"不出 gate"并存的方式：Development 产出全部门槛**输入指标**，
通过判定在外部按同一阈值核算。

### 数据合成工具链（已确认可用）
`contextworld/synthesis/`：`collector` → `compiler` → `validator` → `lance` → `manifest`。
各组件生成脚本与 config 见 `docs/Data_Generation.md` 第 84–92 行。
速度的三条件结构由 `contextworld/evaluation/speed_cube.py` 的 `_history_labels()`
与 `build_speed_cube_catalog()` 构造，入口 `scripts/build_tworoom_speed_cube_catalogs.py`；
门通行的验证集入口为 `scripts/build_tworoom_hidden_passage_h3_validation.py`。

---

## 四、不得触碰

- Public Test 的数据与已冻结结果（`artifacts/evaluation/**` 冻结工件、
  十份 `configs/benchmark/*_icl_release_v1.yaml` 内的 sha 值）
- 会话开始前就在工作区的 CoJA / rollout 改动（9 个已修改文件 + 5 个未跟踪文件），
  未获指示不提交
