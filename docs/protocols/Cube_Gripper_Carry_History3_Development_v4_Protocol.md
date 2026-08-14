# Cube 夹爪携带规则 History=3 Development v4 协议

状态：`preregistered_before_first_v4_build`；Development-only；Public Test
关闭；参考模型训练关闭。

## 1. 文档角色与声明边界

规范性定义以
`configs/benchmark/cube_gripper_carry_h3_development_prereg_v4.yaml` 为准。
本文只解释已预注册的设计，不回填正式结果。v4 只授权一次首次且正式的 Cube
Training/Development 数据构建、对应 action-support/causal 审计，以及一次冻结的
RGB-history probe。它不授权 v4 builder/Lance smoke、LeWM 或 PLDM 训练/评分、Public
Test 访问、CEM、suite 登记或 release 声明。

即使全部数据门和 RGB probe 都通过，结论也只能是 `data readiness`。这不等于模型已经
学会，也不等于 `release readiness`；模型训练和 release 各自需要后续独立冻结。

## 2. v3 的失败结论不可改写

v4 以两个不可修改的 v3 结果为起点：

- `development_decision.json` 的 SHA256 为
  `29bbb54031646fbc42ec83482c65e0ef4a5e55526360337926d1648b0ff4a24c`，结论为
  `failed_development`；
- `rgb_history_probe.json` 的 SHA256 为
  `2c58e23acb3ef6a806fdf798b2988c1d2cd02cfdd55e4fb352f985fa3e008e2b`。
  冻结 overall accuracy 是 `0.7421875`，低于 `0.75`；其余 probe 门、negative
  controls、integrity 和 leakage 检查通过，但不能抵消 overall gate 的失败。

因此 v4 不是对 v3 结果的重解释，也不是在旧 Development 上换一个统计口径。

## 3. 探索诊断只用于设计，特征改法全部否决

旧 v3 Training/Development 上曾做一次探索诊断：

`/tmp/contextworld_cube_gripper_carry_h3_v3_to_v4_exploratory_diagnostic_20260812.json`

其 SHA256 为
`f5f5064d96107e70e13ec3e623ecbe644eaf36554d7bf2075c0714064e78f306`。它只用于定位
设计风险，没有打开、读取、哈希或评分 Public，也没有训练或评分参考模型。由于它重用
了旧 v3 Development，其中的任何数字都不能验证 v4。

探索中看起来更有利的 signed delta、absolute delta、32×32 与 ROI/crop 路线均明确
否决，不得进入 v4。v4 继续使用 v3 的 16×16、
`flatten(2*x1-x0-x2)`、`StandardScaler + RidgeClassifier(alpha=1)` 配方及原门槛。
旧 v3 数据及本次探索涉及的 source/content 身份必须从 v4 正式数据中完全隔离。

## 4. 唯一科学变化：0.30 N → 0.40 N

v4 的唯一科学变化是 `can_hold` 模式下，每单位 normalized z command 的竖直 generalized
force coupling 从 `0.30 N` 提高到 `0.40 N`。其余语义和科学自由度全部冻结：

- 隐藏规则仍只在 `mj_step` 前改变 `qfrc_applied`；
- x0 后没有 qpos/qvel/reset/forward 状态安装；
- 连续轨迹、query state/RGB 配对相等、两 mode 动作 bitwise equal 均不变；
- 四个动作 anchor、nullspace 扰动、source 筛选、pair 数、camera、JPEG95、schema、
  causal/action-support 门及 RGB probe 配方和阈值均不变。

选择依据是冻结的非正式 coupling feasibility pilot，SHA256 为
`b9050fb203904bbc0dc8aec2c32e5b950567b1014cb91fb923338f1979cacad7`。pilot 在 16 个
全新 Training-source 场景上对 `0.30/0.40/0.45/0.50 N` 各运行 16 pairs，共 64 pairs，
selection seed 为 `2026081207`。唯一选定值 `0.40 N` 只依据：

1. history/future 的物理高度差幅度；
2. rendered RGB change 幅度；
3. causal invariants 继续通过；
4. 它是相对失败 v3 的最小保守增强。

pilot 没有 classifier、probe accuracy 或参考模型结果，也没有 Public 信息。`0.45/0.50`
不是因分类分数更差而排除，而是因为它们是没有必要的更大干预；`0.30` 是已失败的 v3
强度。pilot 行不得 promote 或复用为正式 v4 数据。

## 5. Split、规模和新种子

本协议只允许以下两个 split：

| split | 展示名 | pairs | episodes | rows | 每个 anchor pairs |
|---|---|---:|---:|---:|---:|
| `train` | Training | 2048 | 4096 | 16384 | 512 |
| `loader_validation` | Development | 256 | 512 | 2048 | 64 |

全套 v4 正式种子均新建且冻结：

| 用途 | seed |
|---|---:|
| candidate assignment | `2026081200` |
| Training profile + scene catalog | `2026081201` |
| Development profile + scene catalog | `2026081202` |
| pair-cluster bootstrap | `2026081203` |
| label permutation | `2026081204` |

pilot seed `2026081207` 属于非正式设计命名空间，不进入正式 catalog。任何种子都不是根据
Public、classifier accuracy 或参考模型表现选择的。

正式 catalog 的 profile index 使用四模板对齐的独立命名空间：
`catalog_index = 1,000,000 + local_index`。其中每个 split 的 `local_index` 仍从 0
连续递增；scene RNG、task 和 candidate ID 继续使用该 local index，因此场景顺序与冻结
seed 语义不变。offset 可被 4 整除，不改变 anchor assignment。这样显式隔离了预冻结期间
已经执行并归档的 real-MuJoCo `catalog_index=0,1` 两对，不允许把已观察的 profile 复用进
正式 population。

`validation` 的展示名为 Public Test，但本协议不得生成 `validation.lance`，也不得打开、
扫描、抽样、解码、统计或哈希已有 Public 数据。Development 通过不会自动打开 Public。

## 6. 连续因果轨迹与四个冻结 anchor

每个 pair 的两个隐藏条件从同一 x0 出发，执行逐 bit 相同动作：

1. `x0 -> x1`：probe；`can_hold` 仅多施加冻结的 `0.40 N` coupling；
2. `x1 -> x2`：共享 recovery，自然回到相同 query 完整 MuJoCo state；
3. `x2 -> x3`：执行相同 query，得到规则相关真实 future。

前三个 action block 的 z 轴仍为 `[p,-p,p]`，轴 0/1/3 为零。四个 family 完全继承
v3：

| anchor | `p` | gripper | 15-step NRMSE |
|---|---|---:|---:|
| endpoint4 | `[1/3,0,0,-1/3,0]` | 0.4 | 0.2867 |
| plateau | `[.25,.25,-.25,-.25,0]` | 0.4 | 0.2925 |
| ramp4 | `[.3,.1,-.1,-.3,0]` | 0.5 | 0.2914 |
| front_hold | `[.2,.2,0,-.4,0]` | 0.5 | 0.2963 |

扰动仍来自两个规范化 basis，系数独立均匀分布于 `[-0.02,0.02]`，并满足
`sum(p)=0`、`p[-1]=0`、`dot([4,3,2,1,0],p)=1`。最后一个 `[5,5]` 全零 terminal
block 仍进入 action profile 内容哈希。anchor family 可跨 split 共享，具体 float32
profile 不可共享。

## 7. 旧数据排除：basis、final receipt 与内容并集

预冻结 basis receipt 的规范路径是：

`artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/prior_episode_exclusions_basis_v1.json`

其文件 SHA256 为
`fd02914c9e3157df2a7ea9766ca9c712130def2deb371f932777535f6f0ce59f`；排除 2320 个
source episodes，对应规范 digest
`6a61cc77e5f2c769ce006a9dbbd3e7a16187ed59fbf5beec2f025713b68ac152`。组成是 v3
formal 2304 episodes 加 coupling pilot 16 episodes，二者零重叠。v3 smoke 的 source
episodes 是 formal v3 子集。

source 子集并不等于内容子集。v3 smoke 的 scene/pair/query 并非全都包含于 formal v3。
在尚未加入任何新 v4 preformal artifact 时，最终排除内容并集必须至少为：

| identity | count |
|---|---:|
| action profiles | 2304 |
| scene templates | 2312 |
| pairs | 2312 |
| query pixels | 2312 |

这些计数不得被简化成只排除 2304 个 formal 内容。正式 v4 必须对所有 v3、探索/pilot
及获授权 preformal 产物做到以下五类 overlap 均为 0：source episode、action profile、
scene template、pair content、query pixel。

basis 不能直接授权 builder。v4 prereg freeze 之后，必须运行 finalizer，生成不覆盖旧文件
的 `prior_episode_exclusions_final_vN.json`。最终 receipt 必须同时绑定当前 prereg、当前
freeze receipt、source H5、basis、v3 formal、每个 v3 smoke、旧探索诊断、coupling pilot
及全部获授权 v4 preformal 产物，并包含四个完整 content exclusion sets。builder 只能读取
最新的 final receipt。

如果 final receipt 之后新增任何获授权 preformal 产物，旧 receipt 必须保留但 supersede，
并在正式构建前再次 finalization，将新增 source/content 并入更高版本。任何 smoke 或 pilot
行都不能提升或复用到正式表。

## 8. 为什么禁止 v4 builder/Lance smoke

本协议的 preformal 验证已经限定为：

- 16 场景 × 4 coupling = 64 pairs 的 coupling pilot；
- 2 个真实 MuJoCo v4 pairs；
- 不产生 Lance 的 fixture 回归测试。

不得运行 v4 builder/Lance smoke。原因是 smoke 和 formal builder 使用同一个确定性 profile
factory；smoke 会先消费正式 profile IDs，随后又必须把这些 IDs 加入 preformal exclusions，
从而与已冻结正式 population 冲突。正式 2048/256 构建必须是第一次 v4 Lance build。
coupling pilot 已满足 final receipt 中 `v4_preformal_smokes_and_pilots` 的覆盖角色；该角色名
不构成运行 builder smoke 的授权。

上述 coupling pilot 与 2 个 real-MuJoCo pairs 必须先形成独立、不可覆盖的 preformal
content receipt。该 receipt 必须确定性重放既有轨迹并归档 source episode 以及 action
profile、scene template、pair content、query pixel 四类 hash；finalizer 必须把这些集合
并入最终 prior exclusion。它是对既有 preformal 执行的身份补充审计，不是新增 coupling
选择、builder smoke 或正式数据尝试。

已冻结的 `preformal_content_receipt_v1.json` SHA256 为
`972f95fc9c2f46b1ba1c239e5a45f0ccaeb2c88345c925efc1f8b571b6ce31f6`。它重放并绑定
17 个 source episodes（16 个 pilot 场景及 real-pair 的 episode 0），四类内容集合各 18
个；四种 coupling 共享相同 scene/action/query identity，所以不会虚报成 64 个不同内容。
与 v3 formal+smoke union 合并后，最终 receipt 的预期计数为 2321 source episodes、2322
action profiles，以及各 2330 个 scene templates、pair contents 和 query pixels。

## 9. 内容身份与严格隔离

split 或版本前缀不能伪造隔离：

| 字段 | 语义 | Training/Development overlap |
|---|---|---:|
| `action_anchor_id` | 共享 support stratum | 4 |
| `action_profile_id` | 实际 `[4,5,5]` float32 blocks 内容 | 0 |
| `scene_template_content_hash` | source/state/task/seed/color/target 内容 | 0 |
| `pair_content_hash` | scene hash + profile hash | 0 |
| query pixel hash | 实际 query RGB 内容 | 0 |
| source episode | 上游 episode identity | 0 |

上述零重叠不仅适用于 v4 Training/Development 之间，还适用于正式 v4 对全部 v3、pilot
和 preformal 内容。`pair_id` 只是分组 ID，不能作为内容隔离证据。Development 只能描述为
四个 anchor 邻域内的 unseen-profile 测试，不能描述成 unseen-family 或任意动作泛化。

## 10. 硬 causal 与 action-support 门

v4 不改变任何 v3 门槛。每个保存 pair 必须满足：

- x0 pixels bitwise equal，x0 complete-state gap 为 0；
- history 与 future cube-z gap 均至少 `0.008 m`；
- history 与 future changed RGB values 均至少 100；
- pre-query object residual、query physical gap、query complete-state gap 均不超过
  `1e-12`；
- query RGB bitwise equal，两 mode 所有动作 bitwise equal；
- x0 后状态安装数为 0，query simulator 不重建；
- deterministic replay、solver-cache 与独立 fresh-simulator replay 均通过；
- 不保存任何失败 pair。

全部 2304 个具体 profiles 都必须 finite、满足三项 profile constraints、绝对动作不超过
1、gripper 不超过原始 H5 最大值 `0.9075843095779419`，且相对原 H5 的保守最近
15-step joint NRMSE 不超过 `0.5`。action audit 必须报告 Training 2048、Development 256，
失败 profile 数为 0。

## 11. 冻结 RGB-history probe

probe recipe 与所有阈值逐项继承 v3。只在 Training fit 一次，在 Development evaluate
一次。pixels 只允许读取 x0/x1/x2；x3 pixels 不得读取或解码，x3 metadata/actions 仅可
用于内容审计。

精确配方是 Pillow decode + `convert("RGB")`，BILINEAR resize 到 16×16，转 float64，
C-order 展平 `2*x1-x0-x2`，Training-only `StandardScaler`，再用
`RidgeClassifier(alpha=1)`。禁止 x3、action、物理状态、隐藏字段、IDs 和行序进入特征。

冻结门槛：overall ≥ `.75`、worst mode ≥ `.70`、worst anchor ≥ `.70`、按 pair cluster
并按 anchor 分层的 10000 次 bootstrap 95% lower bound ≥ `.70`。bootstrap seed 是
`2026081203`。同时必须通过 16 次 label permutation（seed `2026081204`，mean ≤ `.60`）
及 x0-only、query-only、action-only 三个 ≤ `.51` controls。

## 12. 一次正式 build、一次 probe 的 stop rule

v4 builder/Lance smoke 次数为 0；正式 build 次数为 1，且它是第一次 v4 Lance build；
RGB probe 次数为 1。不得看完科学结果后修改数据、feature、阈值或重新运行。

- data contract、prior exclusion、support 或 causal 任一失败：保存精确失败阶段，写
  `failed_development`，无有效 probe 输入时不运行 probe，不重建、不训练，Public 保持关闭；
- RGB probe 任一门失败：完整保存失败报告，写 `failed_development/rgb_probe`，不重跑、
  不改配方、不训练，Public 保持关闭；
- 全部门通过：只写 `passed_development`，范围明确为 data readiness，然后另行冻结
  LeWM/PLDM 训练预注册。

基础设施在产生可科学检查结果前失败，也必须留下不可覆盖的 attempt 记录。任何 retry
都需要新的冻结预注册，不能把已发生的 attempt 静默从预算中删除。

## 13. 身份冻结与继承依赖

v4 physics 直接继承/import v3 simulator 和 private fresh-replay helper。因此 freeze
receipt 必须显式绑定：

`contextworld/evaluation/cube_grasp_rule_h3_v3.py`

其冻结 SHA256 为
`45860ac696499458ca2e950735aa1fea67e1670cd36071968c386fba27beb0b9`。不能只绑定
v4 wrapper 而遗漏这项运行时依赖。

首次正式 build 前，身份 DAG 还必须逐文件冻结 v2 base physics、v4 physics、builder、
physics/builder tests、action audit/tests、RGB probe/tests、basis freezer/tests、finalizer/tests、
v4 prereg freezer/tests、公共 `causal_data_contract.py` 和本文。v4 builder 对公共 causal
contract 的依赖必须显式记录。YAML 中的
`TO_BE_FROZEN_BEFORE_FIRST_V4_BUILD` 只能在 draft 阶段存在；正式 freeze receipt 生成前
必须替换为真实逐文件 SHA256。prereg 不内嵌自身 hash，由外部 freeze receipt 记录，以避免
递归。

## 14. 参考模型和 Public 仍关闭

本协议下 reference-model training/scoring 明确为 false。不得启动 optimizer、生成或选择
checkpoint，也不得用 LeWM/PLDM 结果决定数据。数据通过后，下一步只能先冻结一份完整
的 reference-training prereg，其中列出训练 seeds、runner、optimizer、loss、gradient
clip、checkpoint、selection 和 scorer 身份。

Public Test 在失败和通过两种情况下都保持关闭。只有未来 Development 参考模型全部满足
另行冻结的标准后，才可提出独立 release freeze；该动作不属于 v4 data prereg。

## 15. 计划产物与禁止声明

正式数据根为
`artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4/`，审计根为
`artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/`。归档至少包含当前
freeze receipt、最新 prior-exclusion final receipt、request、manifest、build report、
train/dev table hashes、support/causal/fresh-replay audit、一次 RGB probe 和最终
`development_decision.json`。

本阶段禁止声称：参考模型已学会隐藏规则、Public 泛化、三种子稳定性、Cube CEM 保持、
release candidate、suite membership、四个 anchor 邻域外泛化、连续 coupling 估计、闭环
规划、真实机器人抓取能力，或“physical/RGB probe 通过即模型/release 通过”。
