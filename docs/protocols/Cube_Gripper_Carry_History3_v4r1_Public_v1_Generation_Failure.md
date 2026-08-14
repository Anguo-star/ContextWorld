# Cube Gripper Carry History=3 v4r1 Public v1 生成失败与恢复边界

状态日期：2026-08-14。状态：
`generation_failed_before_publication_not_model_read_not_scored`。这是一次基础设施元数据
封装失败，不是科学数据门失败，也不是 LeWM Public 结果。

## 已发生的唯一尝试

Public v1 在有效预注册和前访问冻结下执行了一次、且仅一次生成。builder 完整复核了
101,942,558,720-byte source H5，并在本地 `/tmp` staging 中接受 256/256 个配对；
`endpoint4`、`front_hold`、`plateau`、`ramp4` 各 64 个。随后在构造发布成功回执时，代码
读取不存在的 `request["preregistration"]`，触发 `KeyError` 并退出 1。

正式 root
`artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1` 只保留两份不可覆盖回执：

- `_GENERATION_STARTED.json`：SHA256
  `a8c985f2f13fff93a0ac3629ffb5feee19803848ec15b6b2ac128ca7fb0e1965`，908 bytes；
- `_GENERATION_FAILURE.json`：SHA256
  `fc5e6e21b43af548102c105ec21e75bdd7542808f3ede818d65c683063907fcc`，979 bytes。

没有 `validation.lance`、`request.json`、`build_report.json`、`manifest.json` 或
`_SUCCESS.json` 发布到该 root。Public 模型读取和评分均为 false，score root 与最终
decision 均未创建。

## 根因与修复

冻结 builder 的 `request` 只记录了生成参数、catalog、排除集合和开始回执，却没有写入
preregistration 与 freeze-receipt 身份；同一函数的成功回执又无条件索引这两个字段。
`preregistration` 是实际首个异常，`freeze_receipt` 是紧随其后的同类潜在异常。

修复只把这两份已冻结身份写入 staged `request.json`，没有改变候选分配、数据、动作、
checkpoint、模型、评分门槛或 Public 可见字段。新增的无 Lance 成功路径回归测试要求
staged request 和传给 publisher 的 success payload 同时精确绑定二者。

原 freeze 绑定的 builder SHA256 为
`df134e8688a6018e52ed30076f814684ff7b84508bd627ea1d82f3ef410f9c8b`（27,703 bytes）。
修复后实现身份已改变，因此原 freeze 即使没有失败 root 也不能用于新执行。

## 不可变边界

- 原生成尝试预算已消耗；不得删除、覆盖、修复或重跑原 root。
- 不得把 256/256 staging 完成解释成已发布 Public 数据或模型通过。
- 不得启动原 score/finalizer，也不得修改 checkpoint、训练配方、设备、batch size、
  bootstrap 或门槛。
- Cube 仍是未发布研发候选，不属于当前八项 Suite。

## 恢复草案（尚未授权执行）

恢复必须维持科学协议不变，并使用完全不同的身份与路径：

- preregistration ID：`contextworld_cube_gripper_carry_h3_v4r1_public_recovery_v1`；
- recovery authorization：`cube_gripper_carry_rule_history3_v4r1_public_recovery_v1`；
- scientific protocol：仍为
  `cube_gripper_carry_rule_history3_v4r1_public_release_v1`；
- prereg：
  `configs/benchmark/cube_gripper_carry_h3_v4r1_public_recovery_prereg_v1.yaml`；
- freeze：
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/public_recovery_freeze_receipt_v1.json`；
- data：`artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1_recovery_v1`；
- score：
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/public_score_v1`；
- decision：
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/public_recovery_decision_v1.json`。

新的 freeze 必须逐字节绑定原 prereg、原 freeze、两份原 marker、修复后的实现与回归测试；
科学参数必须与 Public v1 完全相同。本文只定义恢复边界，不授权生成、Public 数据访问、
评分、finalizer、发布或 Suite 登记；这些不可逆动作需要新的明确授权。
