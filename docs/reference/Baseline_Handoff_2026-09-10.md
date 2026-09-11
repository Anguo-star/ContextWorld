# ICL baseline 接手审计（2026-09-10）

> 本文是接手时的缺口审计。后续执行结果与当前固定参考见 [封板报告](Baseline_Closure_2026-09-10.md)。

结论：可以接续现有工作。训练与分数矩阵已基本补齐，但当前状态还不能作为“不再变化”的最终基准。
优先缺口是动作延迟 Development 的收尾、统一门槛判定、结果身份绑定和文档同步。
无需为了封板把所有模型训练到通过；诚实、完整、可复现的负结果也是 baseline。

审计起点为 `e8f5f1a362c8c89e5f9f6984a3f75d828f175ded` 加已有未提交工作区。
读取了 Claude 的 `docs/WIP_eval_standard_alignment.md`、近期提交、当前源码、数据包元数据和实际结果文件。
本次没有训练、模型推理、改动评分器、修改数据包或覆盖原始结果；只新增本报告及逐单元审计 JSON。

## 已完成且本次能核对的部分

| 项目 | 核对结果 |
|---|---|
| 当前参考训练后矩阵 | 3 模型 × 9 任务 × 3 训练种子 = 81 个单元；两个划分各有结果 |
| 原环境 LeWM/PLDM ICL | 24 个环境级检查点映射成 54 个任务单元；两个划分各有结果，共 108 个分数 |
| ICL 汇总与源文件 | 270/270 找到；263 个主分数与当前冻结汇总相同，7 个不同，全部是动作延迟 Development |
| Public Test 主分数 | 135/135 与当前冻结汇总相同；这不代表全部获准纳入正式结论 |
| 原环境 CEM | 三种子补齐记录含 8 个环境×模型组合、24 个成员，每成员 300 次 |
| 训练后 CEM | 当前参考汇总已记录 81 个训练后单元的逐评测种子成功率；本次未重新逐 episode 审计 |
| 三项新增反捷径检查 | speed/action_delay/door 已有指标内核、阈值配置和已登记的源码 pin 变更 |
| 原始历史 baseline 归档 | `python -m scripts.audit_contextworld_original_baseline_matrix_freeze_v1` 通过；它审计历史归档，不覆盖当前 270 单元快照 |
| 相关测试 | 171 passed、6 skipped；跳过项是 contact_friction/motion_damping 三模型的 Development 判定缺值 |
| 动作延迟新 Development 数据 | 独立校验 38 passed、0 failed；覆盖目录隔离、300 个 NPZ 身份、66 张 Lance 表结构；像素对照抽查 2 个 episode |

逐单元路径、结果文件 SHA-256、分数与差值见
[`baseline_handoff_audit_2026-09-10.json`](baseline_handoff_audit_2026-09-10.json)。
此处的 135 是“检查点×任务”单元，原环境检查点跨任务复用，不是 135 个不同权重文件。
本次未重新计算全部模型权重哈希，也未重跑完整仓库测试或干净环境安装。

当前快照中的 ICL 通过组合为：门通行三模型、动作延迟 LeWM/PLDM、机械臂质量 LeWM，均 3/3。
其余是已记录的负结果或局部能力，不是因此必须补做训练。DINO-WM 原环境 ICL 的 `not_compatible`
也应作为明确边界保留；是否另建仅 RGB+action 的 DINO 基线属于后续版本的选择。

## 封板前必须收尾

### 1. 动作延迟的新数据已落地，六个原始基线尚未跟上

本地 `ContextWorld-v1` 实际版本是 **1.0.3-rc1**，manifest 为
`4c5b9cdc84006caeac2a9770501f943affe2b81e32d70c27ab504770b28a93c4`。
新动作延迟 Development 是 300 个 query、11 个延迟、6 个评测种子，每种子 50 query。
数据与 Test 的 query、模板、像素和起点隔离均通过独立核验。

- 训练后九个单元已经重评：新 manifest、300 query、含 `gate_completion_inputs`。
- 原始 LeWM/PLDM 六个单元仍是旧 manifest `46f5cdb8…`、30 query，缺新增门槛输入。
- 这些旧基线使用 `__main__.DevelopmentH3TailProjectionAdapter`；仓库通用 CLI 的
  `--history-adapter h3_tail_projection` 仍只接受 PreJEPA，无法直接复现这六个 LeWM/PLDM 单元。

接手工作：把已有 H3 尾投影适配器以明确的 Development/Test 动作几何合同纳入仓库，补跑六枚基线。
不能因为原始分数恰好是随机水平，就把不同 query 集视为可直接比较。

### 2. 动作延迟 reader 仍丢失评测种子

`contextworld/benchmarks/bundle_development.py` 的 `_action_delay_physical_group_metrics`
仍硬编码 `eval_seed: 0`，注释声称 Development 没有 seed。
九个新结果实际都输出 `eval_seed_query_counts: {"0": 300}`；新目录实际有 52–57 六个种子。
因此数据层已对齐，结果层的分层信息尚未对齐。Development 当前也只执行 h1；Test 保留 h1/h2/h3。
正式同结构合同需要明确辅助 horizon 是否属于对齐范围，不能只凭数据导出校验宣告评分结构全部一致。

接手工作：保留 query 的真实 seed/方向/房间元数据，核对共同主指标与门槛的分层范围，再重评受影响单元。
不改变冻结 Test 数据或降低已有阈值。

### 3. 新门槛尚未成为所有入口的统一最终判定

新增 `gate_completion.passed` 已写入结果，但旧入口仍有自己的通过字段：

- 动作延迟 `score_action_delay_icl_results` 的逐检查点判定仍读 `result["gate"]["passed"]`；
- 门通行 `formal_checkpoint_passed` 和 `_reader_metrics` 仍读旧 `summary.decision.passed`；
- 速度 `aggregate_speed_icl_method` 仍按原有 track/horizon 判定计算结论，没有引用新增 `gate_completion`。

这不证明当前表里的六个通过组合有误，但意味着新方法可能从不同入口得到不同的“通过”语义。
Development 无需在原始结果中添加 `gate` 字段；可由共同判定器读取相同输入指标与阈值，产出独立决策回执。

接手工作：保留历史字段语义，新增明确版本的统一总判定，并让当前参考汇总、方法聚合、准入与文档使用同一判定。
增加“主分数通过但新增门槛失败”必须被拒绝的集成检查。阈值配置内的“待 coordinating process 确认”
与已接受的 pin-transition 状态也应统一登记。

### 4. 当前 freeze 是数值摘录，尚不是完整可验证的最终身份清单

`configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v1.json` 有 135 行，
但每行没有源结果路径/SHA、checkpoint SHA、数据身份或评分代码/运行时身份。
目前文档测试主要比较“表格与这个 JSON”，不能证明 JSON 仍与源结果一致。
本次源文件对照发现 7 个动作延迟 Development 值已经变化：例如 LeWM s3073
从汇总里的 `0.999074074074074` 变成新结果的 `0.9775`。

引用的训练快照 `contextworld_native_v1_2026-09-03_snapshot.json` 只有 60 条
（LeWM 27、PLDM 27、PreJEPA 6），并不覆盖当前 81 个训练后单元。
135 个 Development 的 `all_gates_passed` 全为 null，六个负结果表格判定因此在现有测试中跳过。

当前 Development 结果身份分布为：旧 `46f5cdb8…` 96 个、speed/door 新版 `fc0fffd3…` 30 个、
动作延迟最终新版 `4c5b9cdc…` 9 个。全包 manifest 不同不必自动触发全部重评：未修改组件可以通过
成员文件及选择合同的等价回执保留旧结果；动作延迟实际换了数据，不能这样处理。

接手工作：在评分收尾后生成新的、带逐单元证据的 freeze 版本，保留旧版；提供确定性的重建和审计入口。
绑定 checkpoint、组件载荷/选择合同、scorer+adapter、全部门槛配置、运行时、结果文件与最终判定。
以后新方法只追加结果；任何数据或评分语义变化升版本，不能覆盖这版参考。

### 5. 文档和交接记录需要一次性同步

当前规范中仍同时存在以下互相冲突的叙述：

- §4.1、§5.1、§5.1.1 仍称 Speed Development 是 history-utility，且 Dev/Test 刻意不同结构；
  实际 reader 和新载荷已经使用 Test 的指标内核。
- 数据身份表仍称 270 个单元都用旧 rc1/rc2；当前 Development 已有三种 manifest。
- “本表不含通过判定”与实际“ICL 门槛结果”列并存。
- 复现附录仍称原始三种子 ICL 未补跑，但当前 108 个基线分数已经存在。
- 主文档称 contact_friction/motion_damping 的 Public Test “没有打开”，但实际存在对应 30 个
  Test 单元，且本次全部与汇总匹配。改用 Development 展示不能消除已经读取 Test 的事实；
  应保留历史披露并明确这些结果不获正式准入，不能把“排除报告”写成“从未评测”。
- WIP 仍记载文档测试未完成、部分重评进行中；其中一些已被最新提交或本地数据改动取代。
- 工作区未提交的文档还把 H3 基线实测 16.67% 写成“表达上限”；现有审计不足以证明该理论结论。

发布准备文档也有过时状态，例如仍称源码/数据许可证未声明，而根目录已有 `LICENSE`、`DATA_LICENSE`
和 `NOTICE`。稳定公共下载、干净环境复现与外部模型验证应独立核对；它们与先固定内部研究参考不是同一验收层。

## 接手执行顺序

1. 固定本轮验收合同：Dev/Test 共用主分数、门槛与要求的分层，数据行隔离；保留真实负结果。
2. 修复动作延迟 reader 和原始 H3 adapter，完成最终数据上的受影响重评。
3. 统一当前版本的总判定与 Development 准入回执，打通集成测试。
4. 从源结果生成完整身份 freeze；未变化组件走等价核验，避免无依据的全量重训/重评。
5. 同步规范、复现附录、状态记录，单独提交 baseline 收尾，形成不可变参考版本。

已有 CoJA、motion-damping rollout 等训练改动仍留在工作区，本次未修改或提交。
这些新方法的探索可以接着做，但其比较对象应引用最终冻结的 baseline ID；新方法好坏不应再反向改变基准。

## 本次核验命令

```bash
python -m pytest \
  tests/test_public_document_numbers_match_frozen_results.py \
  tests/test_test_gate_completion.py tests/test_source_pin_grading.py \
  tests/test_bundle_development.py tests/test_action_delay_h3_tail_projection.py \
  tests/test_public_test_report_cli.py -q --tb=short

python -m scripts.audit_contextworld_original_baseline_matrix_freeze_v1

python scripts/verify_action_delay_dev_structural_parity_v1.py \
  --lance-root /opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1/components/tworoom-action-delay/v1/development/full \
  --sample-episodes 2
```

分数源文件对照逐项记录在伴随 JSON；它是本次读文件审计，不替代待补的正式 freeze 生成器。
