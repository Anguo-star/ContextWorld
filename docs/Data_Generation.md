# ContextWorld 数据生成方法

本文说明 `ContextWorld-v1` 中九项能力任务的 Training 和 Development 数据如何产生，以及
这些数据如何进入公开分发包。目标是让读者能够判断样本是否真正来自连续物理过程、隐藏
规律是否可能由无关线索泄漏，并找到每项任务对应的实现入口。任务定义、评分与参考结果见
[Benchmark 规范](ContextWorld_ICL_Benchmark.md)，目录和加载方式见
[ContextWorld-v1 数据集指南](HF_Dataset_Export.md)。

本文不是 Public Test 的生成说明。Public Test 保持封存；阅读本文不授权访问、生成、重新
生成或重跑 Public Test。

## 从隐藏规律到公开数据包

九项任务共享同一条构造链：

```text
隐藏动力学或结构规则
        ↓
连续执行环境模拟器
        ↓
匹配反事实或任务专用样本构造
        ↓
互不重叠的 Training / Development
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
并验证起点图像或几何本身不能预测标签。Training 与 Development 使用不同的源 episode、
生成 seed、场景或动作 profile；各任务还检查 query 图像、pair 内容和任务相关模板的交集
为零。具体隔离键因环境而异，但不会只依赖目录名来声明拆分独立。

速度任务是公开 Development 的例外。Speed Development 比较完整 H3 历史与
current-frame-only 消融，是 history-utility 诊断；它不是多数任务采用的 matched formal
scoring 构造，不产生正式通过判定。速度的正式能力判定来自封存协议中的严格历史比较。

## 模型可见字段与审计字段

在 Development ICL 评分中，模型可见字段是图像 `pixels` 和动作 `action`；真实未来图像
只交给同一检查点的冻结目标编码器。分发的 Training 表可以保留任务注册表声明的标准模型
输入列，但隐藏因子、模拟器完整状态、生成 seed、pair 身份和完整性收据只用于审计或数据
选择，不进入模型输入，也不向参评 Adapter 提供。原始表的物理状态因此可以用于证明两个
query 匹配，却不能成为模型识别标签的捷径。

## 九项组件的生成入口

下表中的路径是当前仓库真实存在的构造器或冻结配置。它们记录每项组件的来源，不表示存在
一个适用于所有环境的一键重建命令；不同模拟器需要各自的上游环境数据与依赖。公共用户
通常直接下载 `ContextWorld-v1`，只有复核数据来源时才需要这些入口。

| 任务 | 环境与隐藏变量 | H | Training / Development 构造 | 构造器与配置 | `ContextWorld-v1` 输出 | 主要泄漏控制 |
|---|---|---:|---|---|---|---|
| 速度 | TwoRoom；移动速度 | 3 | Training 覆盖 32 档速度；Development 为 288 个 history-utility case | `scripts/collect_tworoom_synthesis_shard.py`；`configs/synthesis/tworoom_speed_full_v1.yaml` | `components/tworoom-speed/v1/{training,development}/data` | 训练与开发速度、seed group 和 reset 几何分离；Development 不冒充 matched 正式分数 |
| 门通行规则 | TwoRoom；门可通过或被阻挡 | 3 | passable / blocked 连续轨迹配对；Training 96 个门位置，Development 16 个位置、288 对 | `scripts/build_tworoom_hidden_passage_h3_training_data.py`；`configs/benchmark/tworoom_hidden_passage_h3_training_data_v1.yaml` | `components/tworoom-door/v1/{training,development}/data` | 门位置与 episode 跨 split 分离；两个方向分别审计 |
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
Training 和 Development 工件映射到公共目录。`scripts/export_contextworld_hf_clean.py`
拒绝符号链接、凭据样文本、未登记目录和已有目标，复制后重新校验每个文件，并生成
`task_registry.json`、`manifest.jsonl` 与 `manifest.sha256`。原始环境数据、模型检查点、
训练日志、上游源码和 Public Test 均不进入 clean export。

因此，数据生成与数据分发是两件事：前者决定物理轨迹和因果对照，后者只发布已冻结的
Training/Development 字节。重新运行 exporter 不能替代生成审计，也不会授权生成新的
测试数据。
