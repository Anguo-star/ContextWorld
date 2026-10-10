# ContextWorld 数据生成方法

本文说明九项能力任务的 Training、Development 和 Test 数据如何产生，以及
这些数据如何进入分发包。目标是让读者能够判断样本是否真正来自连续物理过程、隐藏
规律是否可能由无关线索泄漏，并找到每项任务对应的实现入口。任务定义、评分与参考结果见
[Benchmark 规范](ContextWorld_ICL_Benchmark.md)，目录和加载方式见
[ContextWorld-v1 数据集指南](HF_Dataset_Export.md)。

当前训练比较使用扩量后的 Training 与固定的 Development / Test。多步预测从九任务主分布的注册 Development 查询扩展未来候选，原评测数据字节不变。各任务规模见[技术报告 §2](ContextWorld_ICL_Benchmark.md#2-数据与划分)。

Training、Development 和 Test 已在本地组装为发布候选，但稳定公共下载地址与固定 revision 尚未公布；因此公开汇总可检查，外部端到端复现暂不可用。Test 与其他划分隔离，仅用于离线最终报告。配置与 manifest 记录各数据版本身份，打包进度见[发布说明](Expanded_Training_Release.md)。

## 从隐藏规律到本地发布候选

九项任务共享同一条构造链：

```text
隐藏动力学或结构规则
        ↓
连续执行环境模拟器
        ↓
匹配反事实或任务专用样本构造
        ↓
互不重叠的 Training / Development / Test
        ↓
因果连续性、可辨识性与完整性审计
        ↓
冻结的组件工件
        ↓
冻结组件的 clean export 候选
```

生成器改变速度、延迟、质量、接触属性或结构转移规则，然后让真实环境模拟器执行动作。
保存的图像和真实未来均由模拟器渲染，不由图像生成模型合成、补帧或编辑。clean exporter
只把已经审计的文件复制到候选目录并生成 manifest；它不会重新仿真，也不会改变样本语义。
当前候选尚未发布为带固定 revision 的公共下载包。

## 连续因果轨迹

History=3 的一步任务可写成同一条连续轨迹：

```text
x0 --u0--> x1 --u1--> x2（查询状态）--u2--> x3（真实未来）
```

上下文与未来由同一个模拟器实例连续向前执行。在上下文和未来之间没有 reset 或重新初始
化，也不写入、覆盖或手工修改模拟器状态。History=7 的动作延迟任务延长同一因果链，但
保持相同原则。重放审计会从允许的初始状态重新执行完整轨迹，检查求解器缓存或序列化没有
改变结果；重放不是用来替换保存帧的第二条生成路径。

这种约束排除了两类容易产生虚假 ICL 信号的数据：把来自不同轨迹的历史和未来拼在一起，
以及在 query 前后直接安装目标状态。模型必须从已经观察到的物理响应推断隐藏规律。

## 匹配、起点交换与划分隔离

多数任务使用 matched counterfactual（匹配反事实）对。一个 pair 在两种隐藏规律下保持
query 画面、query 动作和允许比较的可观测状态一致，只让历史响应和对应的模拟器真实未来
随隐藏规律变化。生成器同时检查两种未来具有足够的物理或视觉差异，否则该 pair 不会进入
发布数据。

有些环境不能让两种规律自然共享完全相同的 `x0`。这时构造器在两种标签之间交换起点，
并验证起点图像或几何本身不能预测标签。Training、Development 与 Test 使用不同的源 episode、
生成 seed、场景或动作 profile；各任务还检查 query 图像、pair 内容和任务相关模板的交集
为零。具体隔离键因环境而异，但不会只依赖目录名来声明拆分独立。

旧一步协议的 Speed Development 使用结构配对：不同速度共享当前状态和查询动作，评分要求正确速度的
历史预测优于该组其余全部历史。四种速度分布分别评测，每个参考速度有 300 个查询。
早期的 288 个 history-utility case 不是当前主表的数据来源；完整历史与单帧输入的差值
仅作为补充诊断，不代替严格历史比较。Development 成绩与 Test 成绩始终分别报告。

## 模型可见字段与审计字段

在 Development/Test ICL 评分中，模型可见字段是图像 `pixels` 和动作 `action`；真实未来图像
只交给同一检查点的冻结目标编码器。分发的 Training 表可以保留任务注册表声明的标准模型
输入列，但隐藏因子、模拟器完整状态、生成 seed、pair 身份和完整性收据只用于审计或数据
选择，不进入模型输入，也不向参评 Adapter 提供。原始表的物理状态因此可以用于证明两个
query 匹配，却不能成为模型识别标签的捷径。

## 九项组件的生成入口

下表记录各任务的基础训练构造器和当前 Development 来源，不表示存在
一个适用于所有环境的一键重建命令；不同模拟器需要各自的上游环境数据与依赖。数据包格式与加载方式见数据集指南；复核或重新生成数据时使用这些入口。
表内六项配对任务列出基础 Training 数量，当前扩量规模另见技术报告 §2，不能将两者混用。

| 任务 | 环境与隐藏变量 | H | Training / Development 构造 | 构造器与配置 | `ContextWorld-v1` 输出 | 主要泄漏控制 |
|---|---|---:|---|---|---|---|
| 速度 | TwoRoom；移动速度 | 3 | Training 覆盖 32 档速度；Development 为四种分布，每个参考速度 300 个查询 | `scripts/collect_tworoom_synthesis_shard.py`；`configs/synthesis/tworoom_speed_full_v1.yaml`；`configs/benchmark/tworoom_speed_dev_structural_parity_v1.yaml` | `components/tworoom-speed/v1/{training,development}/data` | 同一查询下匹配当前状态与动作；划分间几何与查询隔离，已见速度组允许参数值重合 |
| 门通行规则 | TwoRoom；门可通过或被阻挡 | 3 | 连续轨迹配对；Training 96 个门位置，Development 16 个位置、300 个查询场景 | `scripts/build_tworoom_hidden_passage_h3_training_data.py`；`configs/benchmark/tworoom_hidden_passage_h3_dev_structural_parity_v1.yaml` | `components/tworoom-door/v1/{training,development}/data` | 门位置与 episode 跨 split 分离；两个方向分别审计 |
| 动作延迟 | TwoRoom；动作生效延迟 0–10 | 7 | `coarse` 提供差异明显的配对条件；`full` 覆盖 0–10 并作为登记的 Development payload | `scripts/build_tworoom_action_delay_h7_paired_training_data.py`；`scripts/build_tworoom_action_delay_h7_training_data.py`；`configs/benchmark/tworoom_action_delay_h7_core_training_data_v3.yaml` | `components/tworoom-action-delay/v1/{training,development}/{coarse,full}` | H=7 保留延迟响应历史；profile、query 与 split 独立 |
| 推手移动幅度 | PushT；动作增益 60 / 140 | 3 | 从原始 replay 状态和动作构造低/高增益 matched pair；2,048 / 256 对 | `scripts/build_pusht_replay_matched_hidden_actuation_h3.py`；`configs/benchmark/pusht_action_strength_icl_release_v1.yaml` | `components/pusht-action-strength/v1/{training,development}/data.lance` | pair 的 query 状态、画面和动作一致；源 episode 分区独立 |
| 接触摩擦 | PushT；摩擦系数 0.05 / 0.80 | 3 | 接触响应 matched pair；8,192 / 256 对 | `scripts/build_pusht_contact_friction_h3_data.py`；`configs/benchmark/pusht_contact_friction_icl_release_v1.yaml` | `components/pusht-contact-friction/v1/{training,development}/data.lance` | query 完整状态容差、真实未来差异与 RGB 可辨识性审计 |
| 运动阻尼 | PushT；阻尼 0.2 / 1.0 | 3 | 接触结束后的衰减响应 matched pair；8,192 / 256 对 | `scripts/build_pusht_motion_damping_h3_data.py`；`configs/benchmark/pusht_motion_damping_icl_release_v1.yaml` | `components/pusht-motion-damping/v1/{training,development}/data.lance` | 起点交换；源 episode、query、模板和 pair 内容跨 split 不相交 |
| 机械臂质量 | Reacher；主体与末端密度 500 / 1500 | 3 | 相同控制下的轻/重机械臂 matched pair；2,048 / 256 对 | `scripts/build_reacher_arm_mass_h3_data.py`；`configs/benchmark/reacher_arm_mass_icl_release_v1.yaml` | `components/reacher-arm-mass/v1/{training,development}/data.lance` | query 状态与动作匹配；源 episode 和场景跨 split 分离 |
| 传送门出口位置 | TwoRoom；靠近或远离边界的出口 | 3 | 两条真实轨迹在 query 前汇合到共享入口状态；2,048 / 256 对 | `scripts/build_tworoom_portal_exit_h3_data.py`；`configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml` | `components/tworoom-portal-exit/v1/{training,development}/data.lance` | 当前画面与动作相同；出口模板、query 和 pair 内容跨 split 分离 |
| Cube 夹爪携带规则 | Cube；闭合夹爪能否携带方块 | 3 | can-hold / cannot-hold matched pair；四种动作模板在每个 split 内 pair-balanced，2,048 / 256 对 | `scripts/build_cube_grasp_rule_h3_v4_data.py`；`configs/benchmark/cube_gripper_carry_h3_development_recovery_prereg_v4r1.yaml`；`scripts/package_cube_grasp_rule_h3_v4r1_icl_release.py` | `components/cube-gripper-carry/v1/{training,development}/data.lance` | 源 episode、动作 profile、场景模板、pair 内容和 query 图像全部 split-disjoint |

### 动作延迟的 coarse / full 关系

动作延迟是唯一使用 History=7 的组件。`coarse` 数据先提供容易区分的延迟条件，用于稳定
建立长历史响应；`full` 数据覆盖 0–10，并按真实下一步响应合并为 0、1、2、3、4、5–10
六个物理组。clean bundle 同时保留两种 Training/Development payload，注册表把 `full`
指定为公开 Development 读取对象。把 H3 检查点的末尾三帧投影到 H7 评分器只能作为明确
标注的诊断，不能当作原生 H7 结果。

### Cube 的四模板约束

Cube v4r1 使用 `endpoint4`、`plateau`、`ramp4` 和 `front_hold` 四个 anchor。每个 anchor
在 Training 和 Development 内分别等量出现，具体扰动由 split 专用 seed 生成，因此相同
profile 不跨 split 复用。对长度为 5 的 probe `p`，每个扰动必须精确满足
`sum(p)=0` 和 `p[-1]=0`；动作块按 `[p, -p, p, 0]` 排列。构造器还固定加权位移矩并限制
扰动系数，避免“净漂移”或最后一步动作直接泄漏携带标签。完整约束见
[Cube v4r1 数据恢复协议](protocols/Cube_Gripper_Carry_History3_Development_v4r1_Recovery_Protocol.md)。

## 审计、冻结与 clean export

组件构造完成后，审计至少覆盖以下层次：

1. **因果连续性**：动作、图像和状态沿同一模拟器轨迹连续，重放结果一致；
2. **匹配有效性**：pair 的 query 条件在协议容差内相同，两种真实未来确实可区分；
3. **历史可辨识性**：隐藏规律在历史中产生可测响应，但静态起点不能直接预测标签；
4. **划分隔离**：源 episode、场景、动作 profile、query 哈希和 pair 内容按任务要求无交集；
5. **文件完整性**：发布配置与 manifest 固定文件路径、大小和 SHA-256。

通过组件级检查后，`configs/benchmark/contextworld_hf_clean_export_v1.yaml` 把明确登记的
Training、Development 和 Test 工件映射到 clean export 候选目录。`scripts/export_contextworld_hf_clean.py`
拒绝符号链接、凭据样文本、未登记目录和已有目标，复制后重新校验每个文件，并生成
`task_registry.json`、`manifest.jsonl` 与 `manifest.sha256`。原始环境数据、模型检查点、
训练日志、上游源码、历史模型输出和内部 Test `score_receipts` 均不进入 clean export。

因此，数据生成与数据分发是两件事：前者决定物理轨迹和因果对照，后者只收录已冻结的
Training/Development/Test 字节。重新运行 exporter 不能替代生成审计，也不会生成新的
测试数据。

## 诊断数据的补充复现

多步主分覆盖九项任务共 2,436 个 Development 场景，Speed 仅含未见速度插值。下方展开项记录候选动作、模型输入校验及逐查询复算；这些诊断不改变主分。完整面板与全部权重尚无稳定公开下载，当前汇总不能替代端到端复现。

<details>
<summary>动作选择、模型输入校验与预测误差复算（展开）</summary>

<a id="speed-action-selection"></a>

## 九任务的动作选择与多步预测诊断

多步数据从现有 Development 查询扩展未来候选，不增加训练样本，不访问 Test，也不改变隐藏参数。当前版本覆盖九任务主分布的全部注册源查询，共 2,436 个；Speed 仅覆盖未见速度插值。它与原一步评测分别报告，不覆盖原数据。

| 任务 | 源场景 | 隐藏条件 | 区间重采样单位 |
|---|---:|---:|---|
| 速度（未见插值） | 300 | 3 档 | 源查询 |
| 推手移动幅度 | 256 | 2 档 | 243 个来源 episode |
| 机械臂质量 | 256 | 2 档 | 源查询 |
| 动作延迟 | 300 | 11 档 | 源查询，保留全部延迟 |
| 接触摩擦 | 256 | 2 档 | 源查询 |
| 运动阻尼 | 256 | 2 档 | 128 组前向／镜像来源 |
| Cube 夹爪携带 | 256 | 2 档 | 256 个来源 episode |
| 门通行规则 | 300 | 2 档 | 源查询，保留两种规则 |
| 传送门出口 | 256 | 2 档 | 源查询 |

所有源场景保留，不按模型成绩或目标可达性筛选。当前完整覆盖结果使用 `multistep_development_coverage_v2` 数据版本。最初 134 场景子集的公式与候选规则不变，原结果保存在 `multistep_prediction_v1.json`；扩展后逐项检查这 134 个场景的历史、动作与真实未来未变。新的置信区间将已知同源窗口和镜像样本成组抽样，不更改点估计的等权规则。

### 候选、目标与真实代价

所有隐藏条件共用候选集合与目标。生成器从源初始状态重放完整真实历史，然后连续执行每个候选，保持接触状态、速度和待生效动作队列。每个候选长五个动作块，每块五个物理步；在第 5、10、15、20、25 步保存图像、物理状态与代价。

基础候选包括原查询动作的幅值、持续时间、停止和反向变体。Speed 另含指向共同目标的控制；Portal 加入出口方向的横向修正；Door 加入对齐与穿越动作；Damping 加入接触干预；Cube 加入夹爪控制。候选按模拟器裁剪规则限制在合法范围内。目标通常取条件 0 原查询五步后的未来；Speed 使用距离 32 px 的目标，Delay 使用延迟 0 的十五步未来，Door 使用可通行规则的十五步未来。各条件仍共享同一目标。

| 环境 / 任务 | 真实代价 | 汇报单位 |
|---|---|---|
| TwoRoom 的 Speed、Delay、Door、Portal | 智能体到目标的欧氏距离 | px |
| PushT 的 Strength、Friction、Damping | 推手与方块的位置误差，加上乘以 40 的环绕角度误差，共同取欧氏范数 | px 等效 |
| Reacher Mass | 末端位置到目标的欧氏距离 | 原始结果为 m，报告为 mm |
| Cube Carry | 方块位置距离与末端位置距离之和 | 原始结果为 m，报告为 mm |

固定时刻代价使“提前经过、随后越过目标”的动作仍受到惩罚。记录只包含五个动作块末端，不能据此声称逐物理步无碰撞、无越界或持续稳定。模拟器最优候选与最佳共用候选的差距仅证明这个有限集合内的决策区分度；不能代替全局可达性或全动作空间规划。接触、绕行或制动覆盖不足时，应先完善评测数据，再解释模型差异。

### 图像、状态与模型输入校验

面板校验要求各条件当前图像完全相同、历史动作一致、候选与目标共用、代价有限非负。原查询候选的未来与源轨迹核对；Delay 核对全部 11 个延迟、三个已有未来时刻及待执行队列。Cube 的诊断历史、目标和候选未来全部由同一模拟器与 JPEG95 编码重新生成：受控物体位置与源轨迹在 $10^{-6}$ m 内一致，新面板当前图像与原查询候选的重放结果精确一致。这是单独的诊断数据，不覆盖冻结的 Cube 图像。

每个场景保存一个 NPZ，不允许 pickle 对象：

| 字段 | 形状 / 含义 |
|---|---|
| `history_pixels` | `[K,H,224,224,3]`；各条件的连续真实图像，uint8 |
| `context_actions` | `[K,H-1,5,A]`；真实历史动作 |
| `candidate_actions` | `[C,5,5,A]`；共用未来候选 |
| `future_pixels` / `future_states` | `[K,C,5,...]`；模拟器真实未来 |
| `goal_pixels` / `goal_state` | 各条件共用的目标 |
| `physical_cost` | `[K,C,5]`；真实物理代价 |
| `conditions` / `physical_steps` | 审计标签与五个评分时刻，不作为模型输入 |

`manifest.json` 给出任务、固定动作归一化、逐场景路径与 SHA-256。`H=7` 用于 Delay，其余为 3；`A` 为任务动作维度。物理状态、隐藏参数和真实未来仅供模拟器与评分器使用，原生预测只接收图像和动作。

### 冻结权重评测与误差定位

`models.json` 为每个检查点登记 `id`、`task`、`family`、`regime`、`training_seed`、`checkpoint`、`checkpoint_sha256`、`stable_repo`、`stable_ref`。`family` 使用 `lewm`、`pldm` 或 `dinowm`；路径由使用者指定。评测入口：

```bash
python scripts/diagnose_cross_task_decisions.py \
  --models /path/to/models.json --id action_strength/lewm/scratch \
  --panel /path/to/panels/action_strength \
  --output /path/to/results/action_strength/lewm/scratch --device cuda --modes free full
```

主分仅使用 `free` 自由推演，初始真实历史之后不补充真实观测。对登记的每个检查点运行推理后，执行：

```bash
python scripts/multistep_prediction_score.py \
  --models /path/to/models.json --results-root /path/to/results \
  --panels-root /path/to/panels --output /path/to/multistep.json \
  --split-query-records
```

评分器先在源场景内平均条件、候选和五个时刻的完整误差，以及真实未来相对条件均值的误差；再分别跨场景求和，计算 `100 × (1 − 完整误差 / 均值参照误差)`。它不平均逐场景的比值。各检查点先独立归一化，再等权平均同配置训练重复。bootstrap 按上表中的来源单位整组重采样 4,000 次，种子为 20260930，在全部训练重复间使用同一组索引。区间表示给定检查点的来源抽样不确定性，训练标准差另外记录。

分数可以为负，没有有限下界；参照为零则不能定义分数，不填零或删除困难样本。[结果 JSON](research/data/multistep_prediction_coverage_v2.json) 和 [CSV](research/data/multistep_prediction_coverage_v2.csv) 对应技术报告主表，包含 136 个检查点×任务单元与 86 个汇总组合。逐查询能量保存于[压缩 JSON](research/data/multistep_prediction_coverage_v2_queries.json.gz)，按运行 ID 索引，主 JSON 记录其 SHA-256。每条查询保存完整误差、参照误差、五时刻能量和来源组；检查点及面板身份保留在运行记录中。

**误差随深度的对照。** `--modes free full` 同时运行自由推演和真实观测输入；`full` 只影响诊断，主分始终只读 `free`。首个预测时刻必须一致，随后用模拟器真实历史替换预测窗口，不提供待预测目标帧。两分支在全部五个时刻共用同一个整段未来参照分母，因此曲线变化反映误差变化，不会混入分母随深度的变化。逐时刻曲线和自由误差减真实输入误差的配对区间记录在 `error_diagnostics`。该差值同时受到状态修正与动力学证据更新的影响，不作可相加的因果归因。

**其他观测替换诊断。** 需要进一步区分最新与过去观测时，使用单独输出目录并选择 `--modes free full current past`。这些分支不会增加新的主指标。

程序比较自由推演、全部窗口使用真实观测、仅替换最新观测、仅替换更早观测四种输入。替换只作用于本次调用，后续仍从各分支自己的预测构造窗口；部分替换不再是连续真实轨迹，只作敏感性分析。输入不会包含本次待预测的目标帧，但替换观测属于开放环规划时未知的未来信息，不能计入规划成绩。直接编码真实未来的动作选择另列，用于检验 latent 目标代价与物理代价是否一致。

评测检查参数未改变、缓存预测与原生 Adapter 的一致性；启用观测替换时，另检查首步各分支完全一致。为避免精度模式制造差异，使用 float32、关闭 TF32，并同时检查最大绝对误差与相对 L2 误差。Delay 五步诊断不改变其原生三步评测合同：原生支持的前三步做自由推演一致性检查，第五步的真实输入对照另与原生单步调用核对。

将含四种分支的 `models.json`、`panels/<task>`、`results/<task>/<family>/<regime>` 放在同一根目录后，可另行汇总机制诊断：

```bash
python scripts/summarize_cross_task_decisions.py \
  --root /path/to/diagnosis --output /path/to/summary.json
```

汇总验证面板、逐查询结果和数组哈希。物理代价先在源场景内等权平均条件，再平均场景；预测误差比先累计分子、分母再相除。95% 区间整簇重采样源场景 4,000 次，种子为 20260930，不将候选或隐藏条件当作独立重复。完整结果与限制见[技术报告](ContextWorld_ICL_Benchmark.md#cross-task-decisions)及[JSON](research/data/cross_task_decision_v1.json)、[CSV](research/data/cross_task_decision_v1.csv)。这些汇总与文件身份已公开；原始面板和全部权重尚未稳定分发，不宣称仅凭汇总即可完整复现。



</details>

<a id="true-target-geometry"></a>

## 真实轨迹的表示距离校准

这项诊断检查：两条真实未来在所选物理几何上偏得更远时，它们的 latent 距离是否也更大。它复用 Development 真实轨迹和已缓存的目标编码，不运行预测器、不拟合坐标读出器，也不重新生成数据。

每个来源场景包含两种隐藏条件、各 11 条候选动作，以及第 5、10、15、20、25 步的真实图像与状态。依次以一条真实未来为参照，其他 21 条作为具有已知误差的替代输出。排除参照自身，避免完美匹配让排序表现虚高；这里不评价候选动作优劣，也不将替代输出当作模型实际预测。

对每条替代轨迹，分别计算五个时刻等权平均的物理平方距离和原生 latent 平方距离。比较两种替代输出时，如果物理距离与 latent 距离的大小关系相反，记作一次排序反转。分母只包含两种距离都能明确排序的比较；物理平局数量和物理可排序时的 latent 平局率单独报告。数值平局容差为 `1e-8 + 1e-7 × max(|距离一|, |距离二|)`，只是浮点容差，不是能力门槛。

每场景先统计反序比例，再对场景等权平均。95% 区间按来源组 bootstrap 4,000 次，随机种子 20261008；Damping 的前向与镜像样本始终一起抽取。条件、候选和替代输出之间的相关比较不当作独立样本。另报告物理 RMS 误差至少相差两倍、且较大误差至少为一个任务单位的子集，检查反序是否仅来自细微误差差别。没有可排序比较的场景保留并报告覆盖，不计为零反序。

| 任务 | 全几何距离使用的坐标 | 任务相关几何补充 | 单位 |
|---|---|---|---|
| 推手移动幅度 | 推手 xy、方块 xy、40 sinθ、40 cosθ | 推手 xy | px 等效 |
| 运动阻尼 | 推手 xy、方块 xy、40 sinθ、40 cosθ | 方块 xy、40 sinθ、40 cosθ | px 等效 |
| 机械臂质量 | 末端 xy | 与全几何相同 | mm |

这些坐标是指定的任务几何量，不是完整物理状态。全几何结果保持不变，任务相关几何和同条件不同动作结果作为补充。不能将反序全部归因于编码器：图像可能还包含其他物理坐标没有度量的变化，也可能无法呈现离屏物体的位置。

像素检查进一步比较同一时刻的完整图像，以及包含全部五个时刻的图像序列。先按哈希分组，再逐字节确认一致，记录同图但几何不同的反例及最大距离。完全相同的目标图像无法通过确定性图像编码区分；未找到精确反例也不证明充分可观测。所有场景均保留，不根据模型成绩删样。

本次覆盖 LeWM、PLDM 的 T0–T3 各一个已登记检查点，每项任务各 256 个场景。DINO-WM 现有辅助特征缓存经过 patch 池化，不能替代主分所用的完整表示，故不纳入这项原生距离诊断。低反序率只说明这些真实目标之间的排序较一致，不能证明模型输出的预测 latent 已具有可靠物理精度。结果见[统一报告](ContextWorld_ICL_Benchmark.md#true-target-geometry-calibration)。

复算需要已生成的真实轨迹面板及原生目标特征缓存；公开结果文件提供逐场景统计，尚不能替代这些原始输入。

```bash
python scripts/calibrate_native_geometry.py \
  --cache-root /path/to/validation/results \
  --panels-root /path/to/multistep/panels \
  --output /path/to/output/geometry.json

python scripts/check_trajectory_pixel_aliases.py \
  --panel-root /path/to/multistep/panels \
  --output /path/to/output/pixel_aliases.json

python scripts/render_target_geometry_calibration.py --check
```

<a id="visible-target-geometry"></a>

### 画布内对照

画布内版本复用 Strength 的 256 个来源场景、243 个来源组，保留历史图像、历史动作、隐藏条件和评分时刻。未来动作按已有生成规则缩放，使推手与方块轮廓在两种条件的全部 25 步内均位于 `[2, 510]`；实际候选去重后每场景为 8–11 条，不补齐重复候选。它改变了未来动作分布，所以新旧差值不能单独归因于可见性。

对照使用 LeWM 的 T0 与 T3 各一份检查点，两套数据内逐一匹配相同 SHA256。每条真实轨迹依次作为参照，其他轨迹之间仍按上述规则计算反序率；推手 xy 为任务相关几何，全六维几何作为补充。两套数据共享来源组 bootstrap 抽样，输出配对差值区间。另保留动作完全不变、每场景至少三个唯一候选的子集，仅作一致性检查，不替代完整数据结果。

候选数量以实际轨迹数组为准，同时核对目标特征的候选轴。所有场景均保留，不按清单计数补齐或截断样本；数组身份及清单差异在结果文件中单独记录。公开 [JSON](research/data/visible_target_geometry_v1.json) 与[逐场景统计](research/data/visible_target_geometry_v1_queries.jsonl.gz) 保留数据和检查点身份。[计算定义](research/data/visible_target_geometry_protocol_v1.json) 固定比较规则；原始轨迹与缓存仍需自行提供。

```bash
python scripts/calibrate_visible_geometry.py \
  --old-cache-root /path/to/original/results/action_strength/lewm \
  --visible-cache-root /path/to/visible/results/action_strength/lewm \
  --old-panels-root /path/to/original/panels/action_strength \
  --visible-panels-root /path/to/visible/panels/action_strength \
  --output-dir /path/to/output

python scripts/render_visible_target_geometry.py --check
```

## 历史对照与物理校准

这两项验证直接复用上述多步面板与冻结检查点，不生成新轨迹、不访问 Test，也不修改已有主分。

`validate_history_conditioning.py` 对每种真实历史各推演一次，计算“输入历史 × 真实未来条件”的完整误差矩阵。匹配项组成正确历史误差，所有非匹配项等权组成错误历史误差；多条件任务不只选择一组任意交换。每个源场景保留全部动作和时刻。误差先跨场景累计再除以主分参照，训练重复等权平均，来源组配对 bootstrap 4,000 次（种子 20260930）。同条件未来无法区分的动作仍保留。

```bash
python scripts/validate_history_conditioning.py \
  --models /path/to/models.json --id action_strength/lewm/scratch \
  --panel /path/to/panels/action_strength \
  --output /path/to/validation/results/action_strength/lewm/scratch \
  --device cuda

python scripts/validate_physical_readout.py \
  --features-dir /path/to/validation/results/action_strength/lewm/scratch \
  --panels-dir /path/to/panels/action_strength --task action_strength \
  --output /path/to/validation/physical/action_strength/lewm/scratch.json \
  --seed 20261007 --bootstrap-reps 1000
```

物理读出使用与区间统计相同的来源组，按种子 20261007 确定性地分为三折。每折只在其他两折的真实未来 latent 上拟合带截距的 ridge 回归；特征标准化也只使用拟合折。每来源最多均匀取 32 帧，正则系数为 0.001，按平均平方损失定义。DINO-WM 先固定池化为 4×4 个 patch 区域；超过 512 维的特征再用固定稀疏随机投影降至 512 维，种子 20261007。LeWM / PLDM 保留原生向量。这些变换仅供辅助读出，不改变主分所用的完整 latent。

| 任务环境 | 校准目标 | RMSE 单位 |
|---|---|---|
| TwoRoom | 智能体 x、y 坐标 | px |
| PushT | 推手及方块的 x、y，另加 $40\sin\theta$、$40\cos\theta$ | px 等效 |
| Reacher Mass | 末端 x、y 坐标 | mm |
| Cube | 末端与方块的 x、y、z 坐标 | mm |

RMSE 是全部指定坐标分量的均方误差再开方，不是物体欧氏距离的均值。每场景先平均条件、候选、时刻和坐标，再跨场景平均；来源组只影响折分及区间，不改变场景等权的点估计。归一化 RMSE 先累计场景 MSE，再除以同一物理坐标的条件方差之和、最后开方，不能平均逐场景比值。零条件方差的场景仍保留误差分子；总方差为零则该比值无定义。

物理坐标来自模拟器，并非都能从单帧图像恢复。PushT 扩展轨迹存在离屏推手，已找到图像完全相同而坐标不同的反例；因此相关物理读出不能直接作为准确度排名。可见性检查入口为 `scripts/check_physical_visibility.py --panels-root /path/to/panels --output /path/to/visibility.json`，结果见[物理可见性检查](research/data/icl_physical_visibility_v1.json)。检查不删除任何原查询，也不更改主分。

每折分别评价未参与拟合的真实 latent 和预测 latent，并保存逐场景误差。真实 latent 的校准误差是对读出质量的检查，不是不可约误差下界；不能把预测误差减去它后称为纯模型误差。该校准使用 Development 的物理标注，不等于只输入图像和动作的世界模型获得了这些标注，也不构成无需校准的公共物理评分。

全部检查点完成后，将登记文件放到 `validation/models.json`，使用相同 ID 的结果目录汇总：

```bash
python scripts/summarize_icl_measurement_validation.py \
  --root /path/to/validation --panels-root /path/to/panels \
  --output /path/to/measurement_validation.json
```

汇总对正确／错误历史及物理读出使用同一来源重采样，并检查主分与已有结果一致。可下载的 [JSON](research/data/icl_measurement_validation_v1.json) 与 [CSV](research/data/icl_measurement_validation_v1.csv) 保存全部训练方案、配对区间和五时刻历史收益。数据构造的独立检查见[面板验证结果](research/data/icl_measurement_panel_v1.json)；历史图像不同只证明输入存在差异，不作为充分辨识的证明。


## 物理目标分解与画布范围诊断

该诊断复用冻结的逐场景 latent 缓存和原三折读出，不运行世界模型推理，也不改变主分。固定范围为九任务的 DINO-WM T1 检查点，以及三项 PushT 的 LeWM / PLDM T3 检查点，共 15 项。具体检查点身份随[协议与结果](research/data/icl_observable_targets_v1.json)提供。

PushT 用相同折分、每来源 32 帧的采样、正则及特征变换重建原读出；全量结果必须与原记录一致后才能分解。其他任务直接复用原记录的逐坐标误差。推手单独报告 x、y；方块联合报告 x、y、$40\sin\theta$、$40\cos\theta$；Cube 的末端与方块分开。各物体的 MSE 按坐标数加权后必须恢复原总 MSE。

画布判定仅适用于 PushT：坐标中心必须在闭区间 [0,512] 内。固定场景、候选和时刻后，只有全部隐藏条件的两个物体都满足条件，才保留整个配对单元。它是几何范围检查，不证明无遮挡或单帧可辨识。报告全部、画布内和其余配对单元的覆盖与误差；不删除原查询。子集 RMSE 先在每场景的选中单元内平均，再在有选中单元的场景间等权平均，空子集显式计数。子集误差贡献则使用原全部单元作分母，两部分之和必须恢复原全量 MSE；不能用不同分母的子集 RMSE 直接作根因占比分解。

条件差异能量逐坐标计算 $\sum_{q,k,c,t}(y_{qkct}-\bar y_{qct})^2$，报告子集占原总能量的比例。条件方差为零的查询仍计入误差分子；只有总条件方差为零时，归一化读数才为空。区间按来源组重采样 1,000 次，种子 20261108；不同物理单位不合并。

```bash
python -m scripts.diagnose_observable_targets \
  --task contact_friction \
  --reference-result /path/to/physical/contact_friction/dinowm/scratch/s3072.json \
  --features-dir /path/to/features/contact_friction/dinowm/scratch/s3072 \
  --panels-dir /path/to/panels/contact_friction \
  --seed 20261007 --bootstrap-reps 1000 \
  --output /path/to/diagnostic/results/contact_friction/dinowm/scratch/s3072.json

# 非 PushT 任务只需 --task、--reference-result 和 --output；
# 全部结果齐备后，使用同目录下的 models.json 与 protocol.json 汇总。
python -m scripts.summarize_observable_targets \
  --root /path/to/diagnostic \
  --output-prefix /path/to/icl_observable_targets_v1
```

诊断中的校准误差与预测读出误差并列报告，不相减后称为纯模型误差；按画布条件分组也不代表新模型成绩。现有数据与主分保持原样，新生成规则需另设版本并重新验证历史证据、可见后果及完整配对覆盖。

## PushT 画布内多步轨迹

`visible_future_development_v1` 用于检验离屏问题修正后的预测测量方法，覆盖推手移动幅度、接触摩擦和运动阻尼。它保留每项任务原有的 256 个 Development 来源、查询前历史、当前状态、隐藏参数和来源组，仅重新生成查询后的动作与真实未来。它改变了未来动作分布，因此与原多步面板分版本报告，不替换九任务主表。

**动作构造。** 从原有 11 个候选动作出发，分别按 `1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0` 缩放，选取能够满足画布条件的最大系数。同一候选在全部隐藏条件下使用完全相同的动作。连续重放历史后再执行未来，查询点不重置；25 个原始仿真步均检查推手和方块完整碰撞形状的外接边界，要求位于 512×512 画布的 `[2,510]` 范围内。每 5 步保存一帧，共五个未来时刻。

缩放后按完整 float32 动作字节去重。模型输入使用唯一候选轴，11 个原候选与缩放系数、唯一候选的映射另行保存；条件、唯一候选和时刻先在来源内等权平均，再汇总来源。无可行缩放时保留来源和失败记录，不将其静默剔除，也不把失败集合报告为已通过的画布内数据。零条件差异的动作仍保留，接触与自由衰减用作数据诊断，不按模型成绩筛选。

**校验范围。** `check_visible_future_panel.py` 检查来源覆盖、历史与目标身份、动作映射、逐步状态、完整形状边界和生成器的连续重放记录；它不另行重跑全部物理仿真。条件信号用两条件真实坐标的距离衡量：先对来源内的唯一候选与五个时刻平均，再对来源等权平均，以免候选数变化造成虚假差异。推手使用平面位置距离，方块使用位置及半径 40 px 的角度弦长合成距离。另搜索精确同图像但物理坐标相差超过两个渲染像素的反例；未发现反例不等于证明无遮挡或充分可辨识。

三个任务使用与原轨迹匹配的环境版本：Strength 与 Friction 为 `6ab823fdc6921c95089992ed49c39e431e21ca4a`，Damping 为 `875e607fc08aa72eacb94d5d178127804134cc06`。Damping 的方块几何必须保持原版本，不能用更改后的环境代替。生成还需原 Development 数据、来源合成记录与旧多步面板；这些输入尚无稳定公共下载，命令用于说明本地复现接口。

```bash
python scripts/builder_visible_strength.py \
  --contextworld-root /path/to/ContextWorld \
  --stable-worldmodel-root /path/to/stable-worldmodel-at-source-ref \
  --payload-root /path/to/ContextWorld-v1 \
  --legacy-panel-manifest /path/to/old-panels/action_strength/manifest.json \
  --output /path/to/new-panels/action_strength --workers 2

python scripts/builder_visible_contact_damping.py \
  --task contact_friction \
  --stable-repo /path/to/stable-worldmodel \
  --stable-ref 6ab823fdc6921c95089992ed49c39e431e21ca4a \
  --bundle /path/to/ContextWorld-v1 \
  --artifacts-root /path/to/ContextWorld/artifacts \
  --legacy-panel-root /path/to/old-panels/contact_friction \
  --output /path/to/new-panels/contact_friction --workers 16

python scripts/builder_visible_contact_damping.py \
  --task motion_damping \
  --stable-repo /path/to/stable-worldmodel \
  --stable-ref 875e607fc08aa72eacb94d5d178127804134cc06 \
  --bundle /path/to/ContextWorld-v1 \
  --artifacts-root /path/to/ContextWorld/artifacts \
  --legacy-panel-root /path/to/old-panels/motion_damping \
  --output /path/to/new-panels/motion_damping --workers 16

python scripts/check_visible_future_panel.py \
  --panel-root /path/to/new-panels --legacy-root /path/to/old-panels \
  --output /path/to/visible_validation.json
```

已有检查点使用同一套完整预测误差和正确／错误历史对照，参数保持冻结。辅助物理读出在新真实未来上重新按来源组交叉拟合，配置沿用前述设置；真实 latent 的校准误差与预测 latent 的读出误差分别报告，二者不能相减解释为纯模型误差。新旧面板的得分变化也不能当作模型进步。结果见技术报告的[测量范围与验证](ContextWorld_ICL_Benchmark.md#main-score-interpretation-and-auxiliary-physical-readout)。

## Speed 单任务诊断复算

<details>
<summary>实验生成与复算细节（展开）</summary>

## 速度任务的候选动作评测数据

这套 Development 诊断复用速度任务的全部场景和连续真实历史，为每个查询新增固定候选及其仿真后果。四种速度分布各包含 300 个场景，分别报告；它不改变原任务的训练数据或评分。

每个场景内，不同速度共享当前状态、目标与 31 个候选。候选将原查询的 5 步动作乘以 0–1.5、间隔 0.05 的幅度；目标固定为该分布中最慢速度执行原动作后的终点。候选后果由相同 TwoRoom 模拟器逐步执行，实际代价是终点到目标的欧氏距离。模型只接收图像和动作；状态与速度仅用于生成、校验和物理评分。

构造器保留全部场景，不按模型得分筛选。它检查原始 15 步轨迹的状态和图像能否完整重放、不同速度的当前状态及已执行动作是否相同、零动作是否保持当前画面、原动作候选是否复现原目标。生成完成后记录逐文件哈希、候选最优代价，以及所有速度必须共用一个候选时的 regret 下界。

```bash
python scripts/build_speed_action_selection.py \
  --benchmark-root /path/to/ContextWorld-v1 \
  --stable-repo /path/to/stable-worldmodel \
  --stable-ref 6ab823fdc6921c95089992ed49c39e431e21ca4a \
  --output /path/to/speed-action-selection-v1
```

模型评分入口为 `scripts/eval_speed_action_selection.py`，汇总入口为 `scripts/summarize_speed_action_selection.py`；各入口的 `--help` 列出检查点、数据位置与并行参数。评分保存所有候选成本，检查权重未变，并将缓存历史编码的计算与原生 Adapter 对照。汇总对同一场景的速度条件等权平均，以场景为 bootstrap 单位；不同速度分布不合并。完整协议与结果见[速度动作选择结果](research/data/speed_action_selection_v1.json)，解释见[技术报告 §5.5](ContextWorld_ICL_Benchmark.md#55-预测误差与动作选择)。

<a id="complete-prediction"></a>

## 完整预测误差的复算

这项诊断复用已有模型输出，不生成新的评测场景。Strength 使用原查询动作的 256 对逐样本误差，Speed 使用四类分布各 300 个原查询动作预测；多步诊断使用已有六个场景、四种动作条件及 5–25 步的全部预测。数据划分均为 Development。缺少逐样本记录的原始 LeWM Strength 使用 `scripts/eval_strength_native_panel.py` 补充一次冻结权重推理，只评价原查询动作，不搜索候选。

`scripts/complete_prediction_metrics.py` 将正确历史的完整误差拆成条件响应误差与共同偏差，并计算错配历史误差。每个场景的条件和候选等权；先累计各场景的误差与条件均值参照能量，再求比。这种汇总按目标分离能量隐式加权，不等于逐场景误差比的平均。零分离候选仍保留在误差总量中并报告数量，分母整体为零时不定义比值。历史误差下降在 K 个等权条件下等于 `2K/(K−1) × Gain`，因此不是新增独立的历史利用证据。

汇总入口为 `scripts/summarize_complete_prediction.py`，参数指定 Speed 单步目录、多步目录、Strength 记录清单和诊断协议；各入口的 `--help` 提供完整接口。结果包含逐场景能量及输入哈希。Strength 清单每行指定方案 ID、记录文件、权重身份和推理凭据，避免凭目录名称推断模型。bootstrap 每次整簇重采样场景，保留其全部条件、候选和时域；比较训练方案或时域时使用相同的重采样索引。

随时域比较时，同时报告预测平方误差和参照能量的增长倍数，避免把分母增长解释为预测变准。各时域的真实未来不同，误差增长本身也不能单独归因于自回归误差累积。定义及结果见[完整预测诊断](ContextWorld_ICL_Benchmark.md#complete-prediction)，数值见[结果 JSON](research/data/icl_complete_prediction_v1.json)。原任务成绩保持独立。

<a id="speed-timed-arrival"></a>

## 速度任务的定时精确到达数据

这项诊断检验历史中的速度信息能否帮助 CEM 完成精确到达。它使用六个固定 Development 场景（评测种子 42–47 各取索引 0）、三档原有速度 3.4、4.8、6.9，以及原有连续真实历史。不同速度共享当前状态、目标图像和任务要求。目标沿原场景目标方向设在当前状态外 32 px；全部场景保留，不按模型得分筛选。

模型一次规划并执行 25 个物理步。成功要求第 25 步的目标距离不超过 2 px，且执行期间不碰墙或边界。进入目标附近后仍执行到截止时刻，不采用原闭环评测的“首次进入 16 px 区域即结束”。接触通过真实下一位置与自由运动 `位置 + 速度 × 裁剪后动作` 的差异检测，数值容差为 `1e-4`。

**先验证任务，再评测模型。** 构造器从原初始状态逐步重放历史，核对每张历史图像与查询状态；随后验证每种速度均有无碰撞精确到达的控制。对不使用历史、所有速度共用的任意无反馈动作序列，无碰撞时终点必为 `查询位置 + 速度 × 累计动作`。由此可计算最佳共用控制的平均误差及成功率上限。目标容差区间在三档速度下互不重叠，因此同一计划最多让一档速度成功。不碰撞要求排除了借助边界饱和把不同速度的轨迹压到同一状态的捷径。该保证依赖固定截止时间、无反馈和不碰撞要求，不适用于原闭环任务。

```bash
python scripts/build_speed_timed_arrival.py \
  --source-panel /path/to/speed-cem-v1 \
  --stable-repo /path/to/stable-worldmodel \
  --stable-ref 6ab823fdc6921c95089992ed49c39e431e21ca4a \
  --output /path/to/speed-timed-arrival-v1
```

模型入口为 `scripts/eval_speed_timed_arrival.py`，使用已有检查点和查询 ID。CEM 保留原搜索预算与完整动作序列：300 个候选、30 轮、30 个精英、5 个动作块、每块 5 步。预测器接收与环境一致的裁剪后未来动作，再使用原归一化；优化目标仍为原生 latent 目标距离，碰撞在真实执行后计入评分。模拟器状态、速度值和验证控制均不进入预测器，也不用于初始化 CEM。

同一检查点、场景和随机种子分别根据三种真实历史生成计划，再将每条计划在三档真实速度下执行。这得到每模型 18 次匹配历史、36 次错配历史的结果；只需 18 次 CEM 搜索，因为同一历史下的计划不读取真实速度。汇总入口 `scripts/summarize_speed_timed_arrival.py` 先在场景内平均条件，再平均六个场景，以场景作配对 bootstrap 单位。结果、检查点和数据身份见[定时到达结果](research/data/speed_timed_arrival_v1.json)。

**区分搜索与预测误差。** `scripts/diagnose_speed_timed_arrival.py` 读取同一数据面板、检查点和已保存的 CEM 结果，只重放与评分，不训练或重新搜索。对每个正确历史条件，比较实际执行动作与使用真实速度构造的可达恒定控制；分别保存原生预测代价、真实未来编码代价、物理终点误差和碰撞。真实速度仅用于诊断控制，不进入模型输入。这里重新计算最终执行动作的代价，不使用 CEM 日志中的精英平均代价。

逐条件结果由 `scripts/summarize_speed_search_diagnosis.py` 汇总；输入按 `T0/<query_id>.json`、`T1/<query_id>.json` 保存，`protocol.json` 固定候选规则及数值平局容差。汇总核对原计划、数据与权重身份，并分别统计搜索差距和动作错排；同场景的三个条件仅作描述性计数，不当作独立显著性证据。[动作评分诊断结果](research/data/speed_timed_arrival_search_v1.json)包含协议与逐条件测量。该诊断不改变原 CEM 成绩，也不代表可部署的速度未知控制器。

<a id="speed-rollout-refresh"></a>

## 真实观测与自回归输入的对照

这项诊断不生成新场景，也不重新训练或搜索。它复用定时到达面板、两模型已保存的 CEM 计划及可达恒定控制，在每个 5 步动作块末保存真实图像。每个模型覆盖六个场景、三种速度、两条动作序列。

`scripts/diagnose_speed_rollout_refresh.py` 比较自由推演、仅替换当前 latent、仅替换过去两帧 latent、全部三帧使用真实编码四种输入。第 t 次预测使用真实帧序列的 `[t:t+3]` 和从这些帧出发的动作块，不使用待预测的下一帧。部分替换仅作用于当次调用，下一次仍从该分支自己的预测序列构造窗口。完整替换是连续真实历史下的一步预测；部分替换不是连续真实轨迹，只作敏感性分析。所有替换均读取开放环规划时未知的未来图像，不计入模型规划成绩。

运行时，先将[结果 JSON](research/data/speed_rollout_refresh_v1.json)的 `protocol` 字段保存为 `protocol.json`，再对每个检查点与场景调用：

```bash
python scripts/diagnose_speed_rollout_refresh.py \
  --panel /path/to/speed-timed-arrival/panel \
  --saved-plan /path/to/speed-timed-arrival/T1/QUERY_ID.json \
  --previous-diagnosis /path/to/search-diagnosis/T1/QUERY_ID.json \
  --checkpoint /path/to/checkpoint.pt --expected-sha256 CHECKPOINT_SHA256 \
  --stable-repo /path/to/stable-worldmodel \
  --stable-ref 6ab823fdc6921c95089992ed49c39e431e21ca4a \
  --protocol /path/to/refresh/protocol.json --output /path/to/refresh/T1
```

输出保存逐条件、逐候选、逐时域预测数组与凭据。程序核对仿真轨迹和既有记录一致、自由推演与原生 Adapter 一致、所有分支首步相同，并检查权重未变。像素速度估计器按已知渲染规则定位红色智能体，以两段实际位移对动作累计量做最小二乘估计；估计时不读取隐藏速度或物理状态，只在评分时比较真值。它检验该面板的图像历史是否含可辨识信息，不评价原生 encoder 是否已提取该信息。

`scripts/summarize_speed_rollout_refresh.py` 接收 `--root`（含 T0、T1 输出及协议）、`--source-root`（定时到达）、`--previous-root`（候选评分对照）和 `--output`。汇总先在场景内等权平均速度和候选，再累计误差求比；不平均逐条件比值。配对 bootstrap 整簇抽取六个场景，10,000 次、种子 20260930。误差比使用同一检查点、同一时域的自由推演作分母，不混同于条件均值参照或静态参照；另外报告两候选各自的误差比和无碰撞子集的排序。解释见[技术报告](ContextWorld_ICL_Benchmark.md#speed-rollout-refresh)。

<a id="speed-cem"></a>

## 速度任务的闭环规划数据

闭环数据复用上节未见速度插值的全部 300 个 Development 场景和三种速度（3.4、4.8、6.9），共 900 个物理条件。目标采用场景原有的远距离目标，与当前状态相距 72–112 px；各速度共享目标。它不同于单动作块实验中的近距离目标，也不是从原始 H5 按 25 步偏移抽出的目标。

构造器逐步重放查询前的 10 个真实动作，验证三帧历史及当前状态。它不在查询处重置模拟器。所有场景都保留，不按模型分数筛选；状态可见的控制器仅用于验证可达性，不参与模型评分。当前 900 个条件的状态与图像均精确重放，全部可在 50 步内达到原环境的 16 px 成功半径，所需步数为 7–29。

```bash
python scripts/build_speed_cem_panel.py \
  --source-panel /path/to/speed-action-selection-v1 \
  --stable-repo /path/to/stable-worldmodel \
  --stable-ref 6ab823fdc6921c95089992ed49c39e431e21ca4a \
  --output /path/to/speed-cem-v1
```

`scripts/eval_speed_cem_panel.py` 接收已有检查点、查询 ID、真实速度索引与初始历史索引。正确与错误历史仅在首次规划时不同；后续规划使用各自实际执行产生的连续历史。入口从固定上游版本读取原 CEM 配置：300 个候选、30 轮、30 个精英，规划与执行均为 5 个动作块，每块 5 步，总预算 50 步。动作归一化与环境裁剪沿用原实现，权重在评测前后保持不变。

`scripts/summarize_speed_cem_panel.py` 对同一场景的两种错误初始历史等权平均，再跨场景汇总。完整数据集与小规模模型试验的覆盖范围分别报告；试验不能代替全部 300 场景的正式成绩。

<a id="speed-rollout-diagnostic"></a>

## 速度任务的多步预测诊断

这项诊断复用闭环试验中的六个场景、三种速度、真实历史和远距离目标，比较预测到第 5、10、15、20、25 个物理步时的误差与动作选择。它使用冻结检查点，不改变正式 CEM 的设置，也不产生新的训练数据。

| 动作条件 | 候选构造 | 对照含义 |
|---|---|---|
| 固定方向 | 将原单动作块评测的 31 个候选各重复五次 | 检查原查询方向上的响应能否保持到更长时域 |
| 原始命令 | 同种子 CEM 首轮的 300 个样本，加上两个模型、三种速度、三种初始历史产生的全部 18 条首轮规划 | 在同一个候选集上比较模型，不按动作效果筛选 |
| 裁剪输入 | 与原始命令使用相同真实未来；仅将模型接收的未来动作裁剪到 `[-1,1]` 后再归一化 | 环境本身已执行相同裁剪，因此可隔离模型动作输入的影响 |
| 缩小动作 | 裁剪后的动作再乘以 0.525，同时用于模型和模拟器 | 检查较小动作范围；真实轨迹及候选覆盖也改变，不能作同未来的因果对照 |

每个候选都从原始初始状态重放真实历史，然后连续模拟五个动作块；中途不重置。构造器检查固定方向候选的首块未来精确复现原数据，并独立重放指定候选以核对环境裁剪的等价性。模型只读取图像和动作，物理状态仅供模拟器与评分器使用。所有模型和历史条件共用已保存的候选与目标。

生成入口为 `scripts/probe_speed_planning.py build`：`--cem-panel` 指向闭环数据，`--candidate-panel` 指向单动作块数据，`--pilot-root` 指向归档闭环结果；`--query-id` 选择每个评测种子中索引为 0 的场景。另需提供 `--stable-repo`、`--stable-ref` 与 `--output`。上游固定版本与上述闭环数据相同。

评分使用同脚本的 `evaluate` 子命令，传入生成目录 `--panel`、查询 ID、已有检查点及其 SHA-256；汇总入口为 `scripts/summarize_speed_planning_probe.py --root <结果目录> --output <JSON>`。目录中 `panel/` 保存场景，`T0/`、`T1/` 保存模型输出。评分抽查缓存预测与原生五个动作块的 rollout 的一致性，并检查权重未变；汇总再次检查原始命令与裁剪输入的真实未来完全相同。物理 regret 先在场景内平均速度条件，再跨场景平均；裁剪差值的区间以场景为配对重采样单位。汇总还比较“按速度分别选最优动作”与“所有速度共用一个最优动作”，检查真实未来有差异时，决策是否也需要区分速度。这里的代价取固定时刻终点，不能直接作为执行途中首次进入目标区的闭环成功率。定义、逐场景统计与文件身份见[诊断结果](research/data/speed_planning_horizon_probe_v1.json)。

</details>


<a id="prediction-accuracy-validation"></a>

## 多容差预测准确率验证

该验证使用已生成的 Development 轨迹和冻结检查点缓存，检查百分制准确率是否正确反映几何误差。它不生成新训练数据，也不更新世界模型。评分定义及结论见[基准报告 §4.6、§5.7](ContextWorld_ICL_Benchmark.md#main-score-interpretation-and-auxiliary-physical-readout)。

数据对照覆盖三个 PushT 画布内任务各 256 个场景，以及原 Speed、Door 多步面板各 300 个场景。两种数据分布不混合；真值、固定位置偏差、位移幅值错误、时间延迟、共同未来均值和错误规律未来均使用同一套容差。共同未来均值使用全部条件的真实未来，仅供检验“不区分规律也能得多少分”，不是可部署模型。

模型对照复用 12 组检查点的缓存预测。原来源组折分、读出参数和拟合样本保持固定；从缓存重建辅助读出器，只是为了导出逐场景坐标。所有校准及预测的逐维 RMSE 与原记录一致。阻尼的 256 场景按 128 个来源组重采样，不能将镜像场景当成独立样本。

**复现。** 将下例变量设置为对应数据和输出目录。清单中的缓存路径应按本地位置调整，保留相应的检查点、面板和来源身份。`RUN` 仅存本次验证产物。

```bash
# 在仓库根目录执行；复用既有缓存，不运行世界模型。
mkdir -p "$RUN"
cp docs/research/data/prediction_accuracy_protocol_v1.json "$RUN/protocol.json"
python scripts/export_prediction_accuracy_readouts.py \
  --manifest "$READOUT_MANIFEST" --output-root "$RUN/readout" --workers 4

python scripts/validate_prediction_accuracy_controls.py \
  --pusht-panels "$VISIBLE_PUSHT_PANELS" \
  --tworoom-panels "$MULTISTEP_PANELS" \
  --strength-query-states docs/research/data/prediction_accuracy_strength_query_states_v1.json \
  --output-dir "$RUN/controls"

python scripts/summarize_prediction_accuracy.py \
  --root "$RUN" --output "$RUN/prediction_accuracy_validation_v1"
```

批量入口接受包含 `settings` 和 `units` 的 [读出清单](research/data/prediction_accuracy_readout_manifest_v1.json)。每项列明任务、特征缓存、原校准结果及面板路径。Strength 的当前状态旁文件来自原 Development 的历史末帧；全部 256 个来源的历史图像与动作已逐项核对，未从未来状态反推当前状态。旁文件与指定面板绑定，不能移用于其他划分。

[逐容差对照](research/data/prediction_accuracy_controls_v1.csv)、[固定容差子集敏感性](research/data/prediction_accuracy_threshold_sensitivity_v1.csv)和[逐场景模型分数](research/data/prediction_accuracy_validation_v1_queries.json.gz)支持复算。公开文件包含验证统计与来源身份；面板、权重和特征缓存尚无稳定公共下载地址，完整端到端复现仍需这些输入。该候选分尚未通过统一物理准确率的测量验证，不替换冻结结果。


<a id="paired-scoring-controls"></a>

## 配对评分函数的已知输出验证

这项验证直接复用 Strength 画布内面板及两套 native latent 目标缓存，不重新合成数据或运行模型。每个来源场景固定候选与第 5 个物理步，将两条件目标构造成已知正确、忽略历史、错配和微小正确响应的输出，调用实际配对评分函数并独立核算。候选 0 的零动作单独用于检查不可定义情况；候选 1 用于分离目标上的控制。全部 256 个场景保留，不按模型分数筛选。

```bash
# 在仓库根目录执行；变量指向现有画布内面板、LeWM 缓存和独立输出目录。
# 缓存目录包含 original/s3073 与 frozen/s3072 两套结果。
python scripts/validate_paired_scoring_controls.py \
  --panel-dir "$VISIBLE_PUSHT_PANELS" \
  --results-root "$LEWM_NATIVE_CACHE" \
  --candidate-index 1 --output-dir "$CONTROL_OUTPUT"
python scripts/render_paired_scoring_controls.py --check
```

缓存保存的 float32 目标转换为 float64 进行受控构造与严格比较；不使用近似平局阈值。记录精确零分离与平局数量、不可定义状态，以及评分实现和缓存身份。控制实验不是训练模型评测，也未覆盖其他任务的独立评分逻辑。结果及限制见[指标研究](ICL_Metric_Study.md#paired-score-controls)；完整复跑需要对应缓存，当前没有稳定公共下载地址。


<a id="same-output-metric-comparison"></a>

## 同一输出的指标比较

该比较复用画布内面板及 `models.json` 列出的八组缓存。读取原生 `[history, truth, candidate, time]` 距离和条件均值参照能量，在同一查询上计算严格未来选择率、完整误差和历史收益；不使用池化特征计算距离。相同目标的数值平局通过原生向量或相同未来图像确认，保留处理前后的记录。所有场景、候选和时刻均参与统计。

```bash
python scripts/compare_cached_icl_metrics.py \
  --root "$VISIBLE_FUTURE_RUN" \
  --output-dir "$COMPARISON_OUTPUT"
python scripts/complete_metric_comparison.py \
  --root "$VISIBLE_FUTURE_RUN" \
  --comparison docs/research/data/same_output_metric_comparison_v1.json \
  --queries docs/research/data/same_output_metric_comparison_v1_queries.csv.gz \
  --output-dir "$COMPARISON_COMPLETION_OUTPUT"
python scripts/render_same_output_metric_comparison.py --check
```

来源分组沿用面板清单。每模型的区间与 DINO-WM T1−T0 对照使用 1,000 次来源组重采样；差值对照在两方案间共享抽样。独立训练重复的不确定性不在此区间内。结果与测量边界见[指标研究](ICL_Metric_Study.md#real-output-metric-comparison)。脚本要求对应缓存，当前无稳定公共下载地址。

补充程序从原生误差缓存读取条件响应，使用与完整误差相同的场景权重和分母，得到响应 NRE 与共同预测偏差。LeWM 使用完整特征独立复算；DINO-WM 使用原生评分时保存的误差项，不使用池化特征计算距离。原 v1 选择率、完整分与历史收益保留，补充分解保存为 v2；公开表由 v2 JSON 生成。生成的输出目录独立于源结果，区间按来源组重采样，逐场景能量可复核分解恒等式。


<a id="nine-task-root-cause-inventory"></a>

## 九任务结果统一索引

索引将原协议成绩、现有多步完整误差、历史对照和缓存中的条件响应误差按任务、模型、训练方案对应。它不改变原评分，不运行模型，也不合并不同面板为一个新分数。逐场景响应能量与已有完整误差使用相同的条件、候选和时间权重；先对每个检查点计算能量比，再对训练重复等权平均。

公开数据已包含复算所需的响应统计和原始汇总，常规校验无需模型权重或预测数组：

```bash
OPENBLAS_NUM_THREADS=1 python scripts/build_nine_task_inventory.py --check
```

如需从原生结果缓存重新导出响应统计，使用原多步面板运行目录：

```bash
OPENBLAS_NUM_THREADS=1 python scripts/build_nine_task_inventory.py \
  --cache-root "$MULTISTEP_RUN"
```

该目录需包含 `models.json` 及其指向的逐场景结果和 `receipt.json`。导出时核对检查点、面板和结果摘要，并逐场景验证完整误差与已有公开记录一致；不读取池化特征。无 `--cache-root` 时从公开统计重建 JSON、CSV 和研究文档表格。Bootstrap 使用 1,000 次来源组抽样，跨训练重复共享抽样，不将条件、候选动作或镜像场景视为独立来源。结果、范围和缺项见[九任务根因分析入口](ICL_Metric_Study.md#root-cause-entry)。


<a id="delay-first-step-mechanism"></a>

## Delay 首端点机制对照

本对照使用原 H7 Development 面板和已完成的原生历史对照，不更改模型训练。冻结编码器检查仅编码真实历史，逐场景比较完整表示和时间差分是否仍区分延迟，不拟合读出器。

```bash
python scripts/diagnose_delay_history_encoding.py \
  --run-root "$MULTISTEP_RUN" --family dinowm --seed 3072 \
  --device cuda:0 --output "$HISTORY_ENCODING_OUTPUT"
python scripts/analyze_delay_first_step.py \
  --export-root "$HISTORY_VALIDATION_RUN" \
  --panel-root "$MULTISTEP_RUN/panels/action_delay" \
  --history-encoding "$HISTORY_ENCODING_OUTPUT/history_encoding.json"
```

冻结编码器推理应使用原评测的依赖环境；不更换 Backbone 或特征池化方式。首端点统计读取原生误差能量，物理目标组可区分性另用已保存的目标特征验证。队列检查直接读取面板中的真实待执行动作。脚本保存的充分统计可独立复算，无需检查点或 GPU：

```bash
OPENBLAS_NUM_THREADS=1 python scripts/analyze_delay_first_step.py --check
```

对 $K=11$ 个等权历史条件，设 $R$ 为中心化响应误差比，$G=(E_- - E_+)/B$ 为其他历史平均误差与匹配误差之差，则沿真实响应方向的增益 $g=(K-1)G/(2K)$；预测条件响应的相对能量为 $R-1+2g$，响应幅度比为其平方根。这里 $B$ 仅来自第 5 个物理步。每个检查点先按所有场景的能量和计算，再对三个训练重复平均；不能把不同检查点的能量混在一起。该恒等式还用随机已知预测与直接向量计算核对。解释范围及结果见[Delay 机制分析](ICL_Metric_Study.md#delay-first-step-mechanism)。


<a id="delay-train-development"></a>

## Delay 训练集与 Development 对照

本对照从既有 Lance 数据提取原生 H7 窗口，不生成新轨迹，不使用 Test，不重新训练。两个 split 各 128 个来源查询，按左右房间与上下动作方向分成四层，每层 32 个；查询按来源身份的 SHA-256 顺序选择，和预测结果无关。

训练集入口必须与检查点登记的 `action_delay / training / full` 成员一致。脚本读取每个同来源 episode 的 11 个延迟条件，使用第 0、5、10、15、20、25、30 行作为历史，第 35 行作为目标，第 0–34 行的动作作为模型输入。当前帧、动作与来源关系逐组核对；物理状态和延迟标签仅供检查，不输入预测器。

还需复算训练加载器的内部切分。检查点绑定的运行时先对六个物理延迟组等权采样，再按 50/50 混入原始数据，最终用各训练种子的 90/10 随机索引切分。所选训练查询的每个延迟条件，在三个种子的训练子集中都必须至少有一个对应起点窗口。它证明训练索引资格，不证明每个窗口实际进入过多少个优化批次。加载器实际允许每个 episode 的 11 个滑动起点；本对照只选保持当前帧相同的起点 0。

```bash
python scripts/prepare_delay_train_development.py \
  --bundle-root "$CONTEXTWORLD_BENCHMARK_ROOT" \
  --original-h5 "$TWOROOM_ORIGINAL_H5" \
  --models "$MODELS_JSON" --output "$DELAY_SPLIT_PANEL" \
  --scenes-per-split 128
python scripts/evaluate_delay_train_development.py \
  --models "$MODELS_JSON" --id action_delay/dinowm/scratch/s3072 \
  --panel "$DELAY_SPLIT_PANEL" \
  --output "$DELAY_SPLIT_PANEL/results/dinowm/3072" --device cuda:0
```

`MODELS_JSON` 使用 [九任务冻结推理清单](#nine-task-root-cause-inventory)的模型条目；需提供三模型、三个训练种子的全部九项 Delay T1 条目。依次替换上述模型 ID 与输出目录，保持每个检查点独立进程和原评测依赖环境。评测核对原生 Adapter 单步预测一致性及前后参数摘要，不拟合读出器，不执行优化器更新。

汇总时，每个查询的 11 个条件等权，先累加能量再归一化，最后等权平均三个检查点；六物理组选择率单独保留。区间按来源查询在四层中重采样，不把同查询的延迟条件当作独立样本，也不把不同编码器的能量混在一起。零响应、完美预测、微弱正确响应和共同偏差反例已与直接向量计算核对。

```bash
OPENBLAS_NUM_THREADS=1 python scripts/analyze_delay_train_development.py \
  --run-root "$DELAY_SPLIT_PANEL"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_delay_train_development.py --check
```

第一条命令汇总已有九份推理结果，第二条仅从公开充分统计量复算结果与文档表格，不需 GPU。来源记录同时保留检查点的历史身份与当前镜像身份：成员名单和适配器字节匹配，整体发布哈希不同，因此不能据此宣称所有数据字节与历史训练时完全一致。结果与解释见[训练域与迁移对照](ICL_Metric_Study.md#delay-first-step-mechanism)。


<a id="training-curve-reproduction"></a>

### 训练过程诊断的样本与数值复现

训练样本在推理前抽取，每对的当前图像与查询动作匹配。训练集成员资格按原生划分规则重建，未按逐批日志确认曝光次数。第 10 epoch 的诊断复测与归档主分最多相差 512 次判断中的 2 次，NRE 差异小于 0.002；差异原因尚未隔离。九任务主表引用归档成绩，本节使用同一次诊断评测所得的第 5、10 epoch 结果。


<a id="delay-native-objective"></a>

## Delay 原生视觉预测目标诊断

使用 [训练集与 Development 对照](#delay-train-development)已经提取的 Training 查询，以及同一模型清单。三个模型分别运行独立进程；各自使用 T1 种子 3072。四个房间／方向层各取清单中最先出现的四个来源查询，不按预测结果选择。每个查询包含全部 11 种延迟、七个真实历史帧及首个真实未来帧。

```bash
python scripts/diagnose_delay_native_objective.py \
  --models "$MODELS_JSON" --id action_delay/dinowm/scratch/s3072 \
  --panel "$DELAY_SPLIT_PANEL" --output "$DELAY_OBJECTIVE_RUN/dinowm" \
  --device cuda:0 --scenes-per-stratum 4
```

将模型 ID 与输出目录分别替换为 LeWM、PLDM 对应条目。脚本缓存原生视觉表示，使用原动作标准化与完整预测路径计算七个位置的视觉 MSE。编码器、视觉投影及动作编码器不参与求导；只计算主预测器梯度，不执行优化器。最终位置必须与原生 Adapter 的首端点预测一致。

损失在 float64 中中心化并汇总，模型前向及反向为 float32，关闭 dropout 与 TF32。梯度分解残差同时按整体梯度和组件范数和报告，防止近抵消造成的比值误读；验算要求组件归一化残差不超过 1e-3、逐参数最大绝对残差不超过 1e-5。它们是计算一致性容差，不是模型能力门槛。源 JSON 保存逐查询损失、梯度统计、切分与检查点身份；输入面板和全部权重尚无稳定公共下载。

从公开三份源 JSON 重建汇总与文档表，无需 GPU：

```bash
python scripts/analyze_delay_native_objective.py
python scripts/analyze_delay_native_objective.py --check
```

<a id="expanded-history-prerequisites"></a>

## 扩量任务的图像历史参考验证

这项验证检验模型允许读取的历史是否包含隐藏规律线索，并核对原生训练目标是否覆盖查询未来。它不训练世界模型，也不将规则分类准确率作为未来预测分数。所有参数仅在 Training 拟合，使用完整 Development 评价；不读取 Test。

| `--task` | 数据包 | `--bundle-root` 对应组件目录 |
|---|---|---|
| `friction` | `ContextWorld-contact-friction-32k-v1` | `components/pusht-contact-friction/v1` |
| `damping` | `ContextWorld-motion-damping-32k-v1` | `components/pusht-motion-damping/v1` |
| `cube` | `ContextWorld-cube-gripper-carry-10k-independent-v2` | `components/cube-gripper-carry/v1` |

[`validate_expanded_history_reference.py`](../scripts/validate_expanded_history_reference.py) 从 Training 选择目标为 4,096 对的完整来源组，排序只依赖来源组 SHA256。Friction 按去掉动作锚点后缀的基础场景分组；Damping 按 `catalog_index // 2` 合并正反向镜像；Cube 按来源 episode 分组。来源字段只用于选择、划分检查和重采样，不进入分类特征。Development 的全部 256 对保留，两条件均评分，bootstrap 按来源组进行 4,000 次重采样。

固定参考方法如下：

- Friction：三帧 RGB 中物体的可见几何、两段位移及三个动作块；Training 拟合标准化与 256 棵 ExtraTrees，随机种子 20261009。
- Damping：三帧 RGB 中方块两段位移范数的比值；只在 Training 选择最小分类误差的单阈值及方向。
- Cube：将三帧 RGB 以 bilinear 缩为 16×16，使用 `2*x1-x0-x2` 的展平特征；Training 拟合标准化与 RidgeClassifier，`alpha=1`。

Friction／Damping 读取物理步 0、5、10 的图像与 0–14 的动作，共 30 个动作标量；Cube 读取前三个模型帧和动作块，每块含五个原始动作，共 75 个标量。未来帧只检查配对是否视觉不同，不参与分类。控制上限通过成对当前图像与动作的解码后相等性计算，不拟合另一个模型。任何提取失败均记录完整分母并停止分类汇总，不能删除后冒充全量结果，也不能由此宣称历史不可辨识。

以下命令以 Friction 为例；另两项替换上表的任务与组件。输出路径必须尚不存在，以保留已发布参考结果。

```bash
python scripts/validate_expanded_history_reference.py \
  --task friction \
  --bundle-root "$DATA_ROOT/ContextWorld-contact-friction-32k-v1/components/pusht-contact-friction/v1" \
  --train-pairs 4096 --workers 4 \
  --output "$OUTPUT_DIR/friction.json"

python scripts/analyze_expanded_history_reference.py --check
```

`DATA_ROOT` 是数据包父目录，`OUTPUT_DIR` 是新的结果目录。第二条命令直接复核仓库中的公开结果：完整分母、预测标签、来源组区间、数据身份和生成表格，均不需要 GPU 或模型权重。代码依赖 Lance、Pillow、NumPy 与 scikit-learn；运行版本随参考结果保存。

[`expanded_supervision_coverage_v1.json`](research/data/expanded_supervision_coverage_v1.json) 另记录数据校验记录、加载器摘要与固定版本训练源码，区分可采样窗口、目标位置和实际批次曝光。输入侧结果与解释边界见[研究说明](ICL_Metric_Study.md#expanded-history-prerequisites)；冻结的旧 Cube 探针与评分身份保留不变。


<a id="strength-representation-objective"></a>

## Strength 的表示与预测目标对照

该诊断比较当前 32k Strength 数据上 LeWM T1、T2、T3 的既有检查点，结果见[指标研究](ICL_Metric_Study.md#strength-representation-objective)。参考分类器只在 Training 拟合，世界模型始终固定；梯度计算不执行优化器更新。

来源组由原始 PushT 文件 SHA256 与 `source_episode_index` 共同确定。按种子 `20261010` 打乱 Training 来源组，完整保留组内查询，取足 512 对；本面板有 188 个来源。Development 使用全部 256 对、243 个来源。读取原轨迹的第 0、5、10 帧作为历史，第 15 帧作为查询目标，动作使用第 0–14 步。输入标签仅用于参考分类器监督，不传入世界模型；不读取 Test。

原生 Encoder 的每帧 192 维输出组成 `[z0, z1-z0, z2-z1]`，保留完整三帧信息。固定 `StandardScaler + RidgeClassifier(alpha=1)` 与 256 棵 `ExtraTrees` 在 Training 拟合，不使用 Development 选择参数或分类器。区间对 243 个 Development 来源进行 2,000 次成组重采样；三方案差值共用重采样索引。Training 分数是参考分类器的拟合集表现。

梯度面板取选定 Training 查询顺序中的前 16 个不同来源，每来源取第一对。H3 原生监督的三个目标为第 5、10、15 帧，视觉 MSE 对全部条件、位置与 latent 维度平均。最后位置的条件中心化误差与共同偏差各保留 1/3 权重，前两个位置保留其原权重。仅对主预测器求导；Encoder、动作编码器和其他目标项不参与本次梯度测量。响应 NRE 按目标能量汇总，不平均逐对误差比。关闭 TF32 后，在每个检查点的首个查询上核对公开 RGB Adapter，直接预测的端点差不超过 `1.1e-6`；损失与梯度分解均核对闭合，模型状态保持不变。

运行脚本为 `scripts/diagnose_strength_history_readout.py` 与 `scripts/diagnose_strength_native_objective.py`，汇总脚本为 `scripts/analyze_strength_mechanism.py`。面板 JSON 记录选定查询、来源、数据与检查点哈希；原始 RGB 与 latent 缓存保存在运行输出中，不包含在公开汇总记录内。

```bash
python scripts/diagnose_strength_history_readout.py --build-panel --panel /path/to/panel
python scripts/diagnose_strength_history_readout.py --panel /path/to/panel --models /path/to/models.json --id action_strength/lewm/scratch/s3072 --device cuda:0 --output /path/to/history_result
python scripts/diagnose_strength_native_objective.py --cache /path/to/panel --models /path/to/models.json --id action_strength/lewm/scratch/s3072 --device cuda:3 --output /path/to/objective_result
python scripts/analyze_strength_mechanism.py --check
```

T2、T3 分别使用模型清单中的 `joint`、`frozen` ID；输出目录分别设置。当前模型清单、面板和权重仍需本地准备，公开 JSON 支持结果复算，不等于完整实验资产已公开分发。

<a id="cross-task-mechanism"></a>

## 九任务的历史表示与原生预测目标诊断

本诊断使用每任务、每模型的 T1 seed 3072 固定检查点。研究结果见[九任务机制对照](ICL_Metric_Study.md#cross-task-mechanism)，不改变已有主分。训练包分别为五项 32k 配对数据、Cube 的 10k 独立来源数据，以及 Speed、Delay、Door 的基础数据；Development 保持原定义。

参考读出的拟合预算为：六项扩量任务各 512 对 Training 查询；Delay 为 128 个查询、每查询 11 个延迟；Speed 和 Door 各 512 条自然 Training 历史。Strength 的 512 对来自 188 个来源，Damping 的镜像配对归并为 256 个来源，其余四项为 512 个来源。Development 使用全部查询：六项配对任务各 256 个，Speed、Delay、Door 各 300 个，共 2,436 个。当前 RGB 与原始动作在各 Development 查询的条件间完全相同。标签按物理规律定义，不能使用条件在文件中的位置代替规律值。

对历史表示使用可逆坐标 `[z0, z1−z0, …]`，保留全部原生维度；DINO-WM 不进行 patch 池化。固定 `StandardScaler + Ridge(alpha=1)`，标准化及拟合只使用 Training。分类采用 RidgeClassifier 的标签编码；Speed 回归连续物理速度，参照是固定的 Training 速度均值。实现用分块的样本空间矩阵求解，与 sklearn 的固定配方进行数值核对。区间按 Development 来源组重采样 2,000 次，种子 `20261010`；分类重算各类别准确率再宏平均，不将 Delay 的六种静止延迟当作六个等权物理组。隐藏标签仅用于参考读出，不输入世界模型。不同模型的特征维度不同，固定读出配方的准确率不等于可用信息量的统一度量。

局部梯度从每任务的 Development 来源组按 `SHA256(20261010:source_group)` 排序选取 16 个来源，每来源选清单中的首个查询。监督目标沿用原生移位序列：H3 的目标在第 5、10、15 步，Delay H7 延伸到第 35 步。Cube 按五物理步一块的模型帧解释。各查询包含全部条件；Delay 保留 11 条延迟轨迹，未重新按六个物理组加权。

设预测与真实目标在条件维度上的均值分别为 $\bar{\hat z}_t,\bar z_t$，原生视觉损失按条件、时间、坐标平均。逐时刻有精确分解：

$$
L_t = \underbrace{\mathbb E_k\| (\hat z_{kt}-\bar{\hat z}_t)-(z_{kt}-\bar z_t)\|^2}_{L_{\mathrm{response},t}}
+ \underbrace{\|\bar{\hat z}_t-\bar z_t\|^2}_{L_{\mathrm{common},t}}.
$$

范数对 latent 坐标取均方。查询端点的响应与共同误差保留 $1/H$ 权重，其余位置保持原权重。仅对主预测器求导，固定视觉及动作编码器；不计入其他正则项，DINO-WM 也不计入动作流预测损失。梯度子集的响应 NRE 将 16 个查询的中心化误差与目标能量分别相加后相除，单独保存在下载数据；正文表格引用同一检查点的全量原协议响应 NRE，避免将局部样本当作完整能力表现。先平均梯度向量，再计算范数比及余弦；这不同于平均各查询的范数比或余弦。检查每个查询的损失及梯度分解，在每个检查点的首个查询上核对公开 RGB Adapter 的端点，并确认计算前后模型状态相同。计算不执行优化器更新。

脚本入口如下。模型清单须提供实际权重、SHA256 及对应的源码目录；数据面板与 latent 缓存仍需本地准备。

```bash
python scripts/build_task_mechanism_panels.py --tasks contact_friction --output /path/to/panels
python scripts/diagnose_task_history_readout.py --panel /path/to/panels/contact_friction --models /path/to/models.json --id contact_friction/lewm/scratch/s3072 --device cuda:0 --output /path/to/readout
python scripts/diagnose_task_native_objective.py --cache /path/to/panels/contact_friction --models /path/to/models.json --id contact_friction/lewm/scratch/s3072 --split development --device cuda:1 --output /path/to/gradient
python scripts/analyze_cross_task_mechanism.py --check
```

[汇总 JSON](research/data/cross_task_mechanism_v1.json)与 [CSV](research/data/cross_task_mechanism_v1.csv)链接所用来源记录和哈希；公开记录可用于复算汇总。参考读出只检验固定方法能否识别规律，局部梯度只描述所选检查点的视觉损失。它们不能替代世界模型的多步成绩，也不单独证明某种训练策略造成了失败。
