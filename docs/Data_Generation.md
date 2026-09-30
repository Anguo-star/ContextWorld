# ContextWorld 数据生成方法

本文说明九项能力任务的 Training、Development 和 Test 数据如何产生，以及
这些数据如何进入分发包。目标是让读者能够判断样本是否真正来自连续物理过程、隐藏
规律是否可能由无关线索泄漏，并找到每项任务对应的实现入口。任务定义、评分与参考结果见
[Benchmark 规范](ContextWorld_ICL_Benchmark.md)，目录和加载方式见
[ContextWorld-v1 数据集指南](HF_Dataset_Export.md)。

当前结果使用扩量后的 Training 与固定的 Development / Test；扩量没有重新生成评测数据。
各任务现用规模见[技术报告 §2](ContextWorld_ICL_Benchmark.md#2-数据与划分)，打包状态见
[发布说明](Expanded_Training_Release.md)。Test 与 Training / Development 隔离，用于离线最终报告；
稳定公共下载版本尚未公布。配置与 manifest 记录各数据版本的精确身份。

## 从隐藏规律到公开数据包

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
ContextWorld-v1 clean export
```

生成器改变速度、延迟、质量、接触属性或结构转移规则，然后让真实环境模拟器执行动作。
保存的图像和真实未来均由模拟器渲染，不由图像生成模型合成、补帧或编辑。clean exporter
只把已经审计的文件复制到公共目录并生成 manifest；它不会重新仿真，也不会改变样本语义。

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

当前 Speed Development 使用结构配对：不同速度共享当前状态和查询动作，评分要求正确速度的
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
一个适用于所有环境的一键重建命令；不同模拟器需要各自的上游环境数据与依赖。公共用户
可用数据包的加载方式见数据集指南；复核或重新生成数据时使用这些入口。
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
Training、Development 和 Test 工件映射到公共目录。`scripts/export_contextworld_hf_clean.py`
拒绝符号链接、凭据样文本、未登记目录和已有目标，复制后重新校验每个文件，并生成
`task_registry.json`、`manifest.jsonl` 与 `manifest.sha256`。原始环境数据、模型检查点、
训练日志、上游源码、历史模型输出和内部 Test `score_receipts` 均不进入 clean export。

因此，数据生成与数据分发是两件事：前者决定物理轨迹和因果对照，后者只发布已冻结的
Training/Development/Test 字节。重新运行 exporter 不能替代生成审计，也不会生成新的
测试数据。

<a id="speed-action-selection"></a>

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

模型评分入口为 `scripts/eval_speed_action_selection.py`，汇总入口为 `scripts/summarize_speed_action_selection.py`；各入口的 `--help` 列出检查点、数据位置与并行参数。评分保存所有候选成本，检查权重未变，并将缓存历史编码的计算与原生 Adapter 对照。汇总对同一场景的速度条件等权平均，以场景为 bootstrap 单位；不同速度分布不合并。完整协议与结果见[速度动作选择结果](research/data/speed_action_selection_v1.json)，解释见[技术报告 §5.5](ContextWorld_ICL_Benchmark.md#55-条件预测与动作选择)。

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
