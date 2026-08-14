# Cube 夹爪携带规则 History=3 Development v3 协议

状态：Development-only 预注册；Public Test 关闭。

2026-08-11 source-CLI enforcement amendment：首轮 32/8 smoke 运行期间，只读审计
发现 builder 虽在命令中收到显式 source，CLI 本身仍保留本机绝对路径默认值，且 source
symbol 与规范名不一致。进程在取消到达前自然完成，但该输出整棵 superseded，不作证据、
不 promote、不复用，也没有被用于改变任何科学设计、阈值或模型配方。v2 receipt 只修改
source 参数强制执行、source symbol 和 receipt 版本；动作、split、因果、support、probe
及 Public 封存全部不变。

## 1. 文档角色

规范性定义以
`configs/benchmark/cube_gripper_carry_h3_development_prereg_v3.yaml`
为准。本文解释设计理由，不回填实验结果。数据、probe 和模型结果必须写入
独立的审计文件；本阶段的 `development_decision.json` 只表示 data readiness。
未来模型结果属于另一份训练预注册。在另行冻结公开 release 之前，统一基准文档和
suite 不登记 Cube v3。

## 2. 研究问题与声明边界

模型看到最近三帧 RGB 和对应五维动作。在当前画面与 query 动作相同的条件下，
它是否能从历史中的“夹爪竖直动作—方块响应”判断隐藏携带规则，并预测真实模拟器
下一帧？v3 进一步要求 Training 与 Development 不复用完全相同的动作 profile。

本阶段只允许证明数据的因果有效性、动作支持改善和 RGB-history 可解码性。它不授权
任何参考模型训练或评分，也不允许声称 Public 泛化、三种子稳定性、Cube CEM 保持、
公开 release、真实抓取能力或四个 anchor 邻域之外的动作泛化。

## 3. 冻结 v2 基线与变更原因

v2 是不可修改的失败 Development 基线。它只有一条固定动作序列，15-step 联合动作
相对原始 Cube H5 的最近 population-standardized NRMSE 为 `0.9893`，阈值
`0.5` 内自然窗口数为 0；夹爪常量 `1.0` 还超过原始数据全局最大值
`0.907584`。LeWM 可记忆固定 32 对但不泛化；严格 replay A/B 没有证明 original
replay 导致 Development 变差。PLDM 另有 pair-conditional response collapse。

v3 不修改 v2 physics、builder、数据或 decision receipt，而是使用新的模块、builder、
数据根和预注册。

## 4. Split 与 Public Test 封存

本协议只授权：

- `train` = Training，2048 pairs；
- `loader_validation` = Development，256 pairs。

`validation` 的展示名是 Public Test，但它不属于本协议的数据构建范围。
`validation.lance` 不得被打开、扫描、抽样、解码、统计、哈希或用于动作、阈值、
配方、checkpoint 与声明设计。v3 builder 必须在代码层只生成前两个 split，且产物中
不得出现 `validation.lance`。Development 通过也不会自动授权 Public。

## 5. 因果轨迹

每个 pair 的两个隐藏条件从相同 x0 开始，执行逐 bit 相同的动作：

1. `x0 -> x1`：probe，`can_hold` 的额外竖直 generalized force 使方块上移；
2. `x1 -> x2`：共享 recovery，自然恢复到相同 query 完整 MuJoCo 状态；
3. `x2 -> x3`：重复 query probe，真实模拟器产生规则相关 future。

x0 后禁止 qpos/qvel/reset/forward 等状态安装。隐藏规则只能改变下一次 `mj_step`
使用的 `qfrc_applied` transition input。x2 不要求等于 x0；它只要求两个隐藏条件的
完整 query 状态和 RGB 相同。

## 6. 四个动作 anchor 与约束扰动

前三个 action block 的 z 轴结构为 `[p, -p, p]`，轴 0/1/3 为 0。四个 family：

| anchor | `p` | gripper | anchor 最近 15-step NRMSE |
|---|---|---:|---:|
| endpoint4 | `[1/3, 0, 0, -1/3, 0]` | 0.4 | 0.2867 |
| plateau | `[.25, .25, -.25, -.25, 0]` | 0.4 | 0.2925 |
| ramp4 | `[.3, .1, -.1, -.3, 0]` | 0.5 | 0.2914 |
| front_hold | `[.2, .2, 0, -.4, 0]` | 0.5 | 0.2963 |

每个 concrete profile 在 anchor 周围加入 split-specific、确定性的 nullspace 小扰动，
并满足：

- `sum(p) = 0`；
- `p[-1] = 0`；
- `dot([4,3,2,1,0], p) = 1`。

最后一项为位移矩；`p[-1]=0` 还确保 query 记录时不遗留 mode-dependent
`qfrc_applied`。每个 split 四个 family 精确均衡，每个 pair 的两个隐藏条件共享同一
concrete profile。冻结策略名为 `shared_families_disjoint_profiles`：Training 每个
family 512 pairs，Development 每个 family 64 pairs；family 是共享支持层，但实际采样
profile 必须在 split 内唯一且跨 split 零重叠。

动作内容按 contiguous C-order float32 `[4,5,5]` bytes 哈希。第四个 `[5,5]` block 是
格式用全零 terminal padding，也必须进入内容哈希；split、candidate 和 metadata 都不
进入该哈希。

## 7. 数据 schema 与构造

每个 pair 有 `cannot_hold`、`can_hold` 两个 episode，每个 episode 含 x0:x3 四行。
model-visible 字段只有 JPEG RGB 与 5×5 raw action block。`hidden_mode`、物理状态、
source、pair、anchor 与 profile 标识均为 audit-only。

候选来自原始 Cube H5 的互不重叠 source episodes。Training 与 Development 使用不同
catalog seeds；pair counts 必须可被 4 整除。每个 split 必须报告 pair/episode/row 数、
四 family 计数、source receipts、JPEG 参数与所有表哈希。

source H5 以 `2,010,000` rows、`10,000` episodes、`101,942,558,720` bytes 和冻结
SHA256 定义，绝对路径不是身份。candidate assignment seed 固定为 `2026081100`，候选
pool multiplier 为 2；scene 与 profile 的 split seeds 均分别为 `2026081101` 和
`2026081102`。候选过滤阈值和 one-best-row 排序完整记录在 YAML。

场景隔离不能由带 split 前缀的 `pair_id` 证明。必须另外计算
`scene_template_content_hash`，内容覆盖 source row/episode/step、simulator seed、task、
qpos/control、cube color 与 target position，且排除 split、人工 ID 和 action 字段；再以
scene hash 与 `action_profile_id` 计算 `pair_content_hash`。二者跨 split overlap 都必须为 0。

## 8. Support 与 identity 是两个命名空间

不能用 split 前缀伪造内容隔离：

| 字段 | 语义 | Training/Development 预期 overlap |
|---|---|---:|
| `action_anchor_id` | 共享 support stratum | 4 |
| `action_profile_id` | 实际 float32 action blocks 的内容 SHA256 | 0 |
| `scene_template_content_hash` | 实际场景生成输入的内容 SHA256 | 0 |
| `pair_content_hash` | scene hash 与 action profile hash 的组合内容 SHA256 | 0 |
| `pair_id` | 仅用于 split 内行分组，不是隔离证据 | 0 |
| query pixel hash | 实际 query RGB 内容 | 0 |
| source episode | 原始 episode identity | 0 |

`action_profile_id` 不包含 split、candidate 或任意人为 ID。四个 anchor family 的共享是
有意分层，不得误报为 template 泄漏；完全相同的 concrete action 内容则禁止跨 split
复用。因此 Development 只可描述为四个冻结 family 邻域内的 unseen-profile 评估，
不能描述为 unseen-family 或任意动作分布泛化。

## 9. 硬因果门

保存的每个 pair 都必须通过：

- x0 pixels 与完整 simulator state 相同；
- history/future cube-z gap 均至少 `0.008 m`；
- history/future changed RGB values 均至少 100；
- pre-query object residual、query physical gap、query complete-state gap 均不超过
  `1e-12`；
- query RGB bitwise equal；
- 两条件所有动作 bitwise equal；
- x0 后状态安装数为 0，query simulator 不重建；
- deterministic replay 与 solver-cache 审计通过；
- 在独立 fresh simulator 中从同一 candidate/mode/actions 重放，完整 state 与 RGB 对连续
  rollout 一致；query-state equality 不能冒充该 clean-replay 门；
- 不保存任何失败 pair。

## 10. RGB-history 可学性门

在 Training 拟合一次固定 probe，在 Development 评估一次。输入只能是单条 condition
的 x0/x1/x2 RGB：各帧 resize 到 16×16，特征固定为
`flatten(2*x1 - x0 - x2)`，估计器固定为 `StandardScaler + RidgeClassifier(alpha=1)`。

JPEG 固定由 Pillow 解码并 `convert("RGB")`，resize 使用
`Image.Resampling.BILINEAR`；转为 float64 后按 C-order 展平。StandardScaler 只能在
Training fit。标签固定 `cannot_hold=0`、`can_hold=1`。bootstrap 以 pair 为 cluster，
并按 anchor family 分层；permuted-label control 固定 16 次和独立 seed。
probe 读取四行 metadata/actions 以重算动作内容哈希，但 `pixels` 列只允许通过
`model_step_idx <= 2` 的 Lance filter 投影；x3 pixel bytes 不得读取，更不得解码。

门槛：overall ≥ .75、worst mode ≥ .70、worst anchor family ≥ .70、按 pair cluster
bootstrap 的 overall 95% lower bound ≥ .70。必须同时报告 permuted-y、x0-only、
query-only 与 action-only chance controls。x3、物理状态、隐藏字段、ID 和行顺序均禁止
进入特征。

## 11. 参考模型阶段未授权

即使数据、causal、support 和 RGB probe 全部通过，本协议也不会启动 LeWM 或 PLDM。
下一步必须先冻结独立的 reference-training 预注册，其中完整列出 optimizer、学习率、
batch/replay mix、loss、gradient clip、初始化 checkpoint、runner、scorer、seed、固定
endpoint 和选择规则。任何 optimizer step 或模型评分都不得先于该新预注册。

## 12. 状态机与 stop rule

- 数据、support、causal 或 RGB probe 任一失败：写 `failed_development` 及明确的
  `failure_stage`，不训练，Public 保持关闭；
- 全部通过：只写 data-readiness 范围的 `passed_development`，然后另冻模型训练协议；
- Public 开启需要未来参考模型 Development 全方法通过后的另一份 release freeze 和
  明确授权。

结果只写入独立 `development_decision.json`，允许的结果状态为
`passed_development` 或 `failed_development`，不得改写本预注册状态。

## 13. Hash DAG 与归档

第一次 v3 数据构建前，freeze receipt 必须记录本 YAML、本文、v3 physics、builder、
tests、公共 causal contract 的逐文件 SHA256。工作树不是 clean 时，Git HEAD 只作背景，
逐文件 hash 才是权威。预注册不能内嵌自身 hash；receipt 单独保存它以避免递归。

正式 Development 归档至少包含 request、manifest、build report、train/dev table hashes、
action/profile/scene receipts、causal clean-replay audit、support audit、RGB probe 与本阶段
decision receipt。训练 config、provenance、checkpoint 和模型 eval 明确不属于本次冻结；
hash scope 明确排除 Public。

## 14. 禁止的正向声明

physical oracle 或 RGB probe 通过不能替代参考模型通过。Development-only 结果不能写成
learned hidden rule、Public generalization、release candidate、suite membership、三种子
稳定性、CEM retention、连续耦合估计或真实抓取能力。

## 15. 计划产物

- 数据：`artifacts/synthesis/cube_gripper_carry_rule_h3_development_v3/`；
- 审计：`artifacts/evaluation/history3/cube_gripper_carry_h3_development_v3/`；
- freeze：`development_prereg_freeze_receipt_v2.json`；旧版 receipt 和首轮 smoke
  只作 superseded 审计痕迹，不进入证据链；
- 最终 Development 决策：`development_decision.json`。

任何命令在运行前都必须确认输入/输出参数没有 `validation.lance` 或 Public Test 路径。
