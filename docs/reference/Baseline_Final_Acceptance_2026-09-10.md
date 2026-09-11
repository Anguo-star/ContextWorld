# 固定研究基准最终验收

**验收通过：在既定任务与报告范围内，v3 可作为后续研究的固定内部对照基准。**
本轮发现的必要判定与数据绑定缺口已修复，没有待补的已知阻塞项。

本次验收目标是为后续问题探索和新方法提供固定的对照数据、评分合同与结果，
不要求每个 baseline 都通过，也不把限定任务的成功解释为通用物理能力。
历史发现见 [v2 有效性校核](Baseline_Validity_Review_2026-09-10.md)。

## 修复内容

- Door 的 Development/Test 使用同一数值判定：两条规则、六种子 × 两方向的完整格子、
  正历史收益、目标选择率、历史胜率、bootstrap 下界与目标分离必须齐全并达标。
  旧 `passed/checks` 仅作一致性诊断；不能覆盖数值失败或缺失。
- 判定合同升级为 `contextworld_reference_decision_v2`；各任务阈值保持不变。
- 新冻结版本为 `contextworld_joint_scratch_v1_reference_results_freeze_v3`；
  v1/v2 保留，已有原始结果快照复用，新增来源与数据选择的可验证绑定。
- 固定 CPU 正/负控制覆盖速度、延迟与六项配对任务的共同响应门槛，纳入 CI。
  同步更正摩擦/阻尼的历史 Test 披露、当前参考入口及能力声明。

## 已核对的结果

- 135 个检查点×任务单元、270 份原始结果逐份对比：主分数、通过结论、准入及 CEM 不变。
  Development/Test 各 18 个单元通过；Door 各 9 个通过，原来的两个绕过反例均被拒绝。
- v3 绑定 48 个评测源码文件；六项 paired Test 的 90 个结果绑定到 6 张实际表，
  18 个文件共 275,529,976 bytes。已有 Development 成员也验证实际文件字节，
  不能以未变化的 manifest 掩盖数据变化；相同文件在一次验证中复用哈希以控制耗时。
- 部分早期 native manifest/provenance 没有随包保留，仍明确记为历史声明；当前表以
  实际字节、发布表 SHA 和结果所绑定的外层包 manifest 成员核对，不伪造缺失历史文件。
- v1/v2 文件与原始结果快照保持原样；本次没有重新训练、GPU 推理或调整阈值。

冻结文件：[v3](../../configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v3.json)；
SHA-256：`01298ca407c4a72bdd9a1238879ae7109b1141a7cfdd2b25065cb0bdec87fef0`。
机器可读结果见 [验收证据](baseline_final_acceptance_2026-09-10.json)。

304 项目标回归全部通过、零跳过；整套冻结验证及主表/Development 表格核对通过。
验证日志及固定控制输出保存在 `artifacts/evaluation/baseline_completion_v3/validation/`。

## 什么可以作为后续基准

主表包含九项任务：七项按现有 Test 口径展示，摩擦/阻尼明确使用 Development。
历史 Test 的存在不等于通过当前准入；这两项历史 Test 排除正式报告。
DINO-WM 原环境 ICL 保持 `not_compatible`，其 CEM 与组件结果保持既有补充证据身份。
不通过的固定参考仍可用于测量新方法是否提高了相应指标、是否跨过全部门槛。
样本身份按 `(task, split, pair_id)` 解释；Action Strength 跨 split 的旧 ID 重名
不作为样本重叠证据，也不为修饰标识而改写冻结数据。

“通过”只支持固定任务分布内，模型利用历史产生方向正确、幅度达标的条件响应。
速度主分数表示内插，总门槛还要求外推；延迟 h1 区分 0、1、2、3、4、5–10 六组，
不能精确识别组内延迟；Door 饱和高分不证明抽象规则迁移。
差分响应允许共同偏置，因此绝对预测、隐藏规律下的闭环规划、长时 rollout 需单独评测。
正/负控制验证已知辨别机制，没有估计真实模型总体的假阳性率。

## 如何保持地基不变

```bash
python scripts/freeze_current_reference_baseline.py verify
python scripts/render_current_reference_tables.py --check
python scripts/verify_reference_capability_controls.py
```

新方法保存独立结果，引用 v3 freeze ID，使用同一 `reference_decision`，并保持三个训练
种子、预算和比较问题可对应。更换数据、指标或门槛语义时建立新版本，保留此基准，
不通过覆盖 native 数值或放宽门槛把新方法变成“通过”。

## 尚未覆盖的目标

- 干净环境从头复现整套历史运行：255 份旧结果没有自动运行时指纹，不能追写成已验证；
  权重身份保留评测时声明的 SHA，本次不重读全部权重或重新训练。
- 完整公共发布：稳定下载与分发修订、独立外部实现验证仍是单独的发布工作。
- 更强的能力主张：九项任务均有通过模型、精确延迟、参数外推、复杂规则迁移与规划可靠性，
  需要相应实验，不能由当前分数自动推出。

这些边界随当前基准固定，不作为后续无休止修改 baseline 的理由。原始大结果及数据按仓库
约定保存在本地 artifacts/数据目录；跨机器使用时需同时保留这些受哈希绑定的材料。
