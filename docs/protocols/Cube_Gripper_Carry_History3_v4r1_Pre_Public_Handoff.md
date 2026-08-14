# Cube Gripper-Carry History=3 v4r1：Public 前冻结交接状态

状态日期：2026-08-14。本文是研发状态索引，不是预注册、冻结回执或发布声明。机器判定
以文末列出的 JSON 及其 SHA256 为准。

状态更新：本文记录的 Public 前状态已被 2026-08-14 的唯一生成尝试取代。该尝试在临时
staging 中完成 256/256 配对后，因发布回执缺少 preregistration/freeze-receipt 身份而
失败；正式 Public root 未发布 Lance 数据，模型未读、未评分。原命名空间已消耗，后续
只能走新的恢复预注册。详见
[Public v1 失败与恢复边界](Cube_Gripper_Carry_History3_v4r1_Public_v1_Generation_Failure.md)。

## 任务目标

模型只看到最近三帧画面和动作，需要根据历史中的方块响应判断当前夹爪规则是
`cannot_hold` 还是 `can_hold`，再预测共享 query 动作对应的真实下一帧。当前轨道只验证
二值夹爪携带规则的一步 latent 预测。

## 数据构成

v4r1 Development 数据包含 2,048 个 Training 配对和 256 个 Loader Validation 配对。
每个 split 在 `endpoint4`、`plateau`、`ramp4`、`front_hold` 四种动作模板间严格均衡；
Training 与 Development 的 source episode、动作 profile、场景模板、配对内容和 query
像素均不相交。

每个扰动向量 `p` 满足 `sum(p)=0`、`p[-1]=0`，以及
`dot([4,3,2,1,0], p)=1`（绝对容差 `1e-6`）。配对条件共享 query 状态、像素和五维动作，
历史及真实未来由模拟器因隐藏规则不同而产生可辨识差异。Public v1 的临时 staging 数据
已生成并完成内部完整性构建，但没有发布为正式 Public split；模型未读取、未评分。

## 评测方法

LeWM 和 PLDM 各使用训练种子 17321、17322、17323，在固定 4,096 optimizer step 的
checkpoint 上一次性评估 Loader Validation。每个 checkpoint 必须同时通过真实未来、
正确历史、上下文切换、最弱规则、paired bootstrap、target latent separation、response
gain 和 normalized response error 门；方法级要求三个独立 checkpoint 全部通过。

通过 Development 的模型族还必须在相同的 300 个原 Cube CEM query 上与原始 checkpoint
配对比较。每个候选最多允许比 baseline 少 15 次成功，三个训练种子必须全部通过。

## 参考结果

LeWM 三个 checkpoint 全部通过 Development：

| seed | 真实未来 | 正确历史 | 上下文切换 | 最弱规则 |
|---:|---:|---:|---:|---:|
| 17321 | 77.93% | 78.52% | 99.61% | 76.56% |
| 17322 | 77.34% | 77.93% | 99.61% | 76.17% |
| 17323 | 77.15% | 78.52% | 99.61% | 74.22% |

PLDM 的真实未来正确率为 50.00%–50.20%，三个 checkpoint 均未通过 Development，因此
没有进入原任务留存阶段。

原始 LeWM baseline 的标准 Cube CEM 为 198/300。训练后 LeWM 三个 checkpoint 分别为
186/300、183/300、185/300，相对差值为 -12、-15、-13；第三个数值中的 -15 正好位于
预注册非劣效边界，三者均通过。

## 适用范围与发布边界

当前可以声明：v4r1 数据已通过 Development 数据门；LeWM 在三个训练种子上通过参考
Development，并保持原 Cube CEM 能力；Public v1 生成因发布元数据缺陷在发布前失败。
不能声明 Public Test 分数、正式 release、统一 Suite 成员或公开 CLI。结果也不
验证连续夹持强度、范围外规则、多步闭环适应或 PLDM 已经解决该任务。

冻结授权只包含 LeWM 的 17321、17322、17323 三个固定 step-4096 checkpoint；PLDM 被
排除。当前 Public 状态为
`generation_failed_before_publication_not_model_read_not_scored`：原 root 只含开始与失败
回执，score 和 decision 路径不存在。同一命名空间不得重跑；新的生成或评分必须先完成
独立恢复预注册、全新 freeze 和明确授权。

## 冻结证据入口

- 参考 Development 判定：
  `artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/reference_development_decision_v3.json`，
  SHA256 `797e5a9722435257fae55e1f9d97424cc77d2d3779576833322b84160375954f`；
- 原任务留存判定：
  `artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/original_task_retention_decision_v2.json`，
  SHA256 `12dbe11eb4cf025359987962dfd869e73e0deb0ecb0eca007fad727889a07ef0`；
- Public release v1 预注册：
  `configs/benchmark/cube_gripper_carry_h3_v4r1_public_release_prereg_v1.yaml`，
  SHA256 `633589015d23279a859b20d3ce02d6804fb25ced95858e4a419f733d8794903c`；
- Public 前冻结回执：
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_release_v1/public_release_freeze_receipt_v1.json`，
  SHA256 `9215a8fbb74677717d088df536f26f62922667f501a1b168481df1c8007066c7`；
- 留存 v1 的零 episode EGL 基础设施失败已独立归档；v2 只把 render backend 改为
  OSMesa，科学 query、checkpoint、CEM 参数和门槛均未改变。
