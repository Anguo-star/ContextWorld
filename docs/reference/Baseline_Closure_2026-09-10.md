# ICL baseline 封板报告（2026-09-10）

> 后续[测量有效性校核](Baseline_Validity_Review_2026-09-10.md)发现判定与关键依赖绑定缺口。
> 本文保留工程封板记录；当前尚未通过最终研究基准验收，不能据此直接宣告地基完成。

本轮内部研究参考的五项收尾已完成：动作延迟 Development、统一判定、逐单元身份冻结、
完整 CEM 与训练身份绑定、文档及交接记录同步。后续新方法引用此版本，数据或评分语义变动另建版本。

固定参考：`contextworld_joint_scratch_v1_reference_results_freeze_v2`。

- 文件：[reference results freeze v2](../../configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v2.json)
- SHA-256：`dfc670ca4e7ea19490a64152eac935c89bc050d18beea2ca0c3e1f40238b97f2`
- 主表：[Benchmark §5.1](../ContextWorld_ICL_Benchmark.md)
- 原始结果快照：`artifacts/evaluation/baseline_completion_v2/frozen_results/`，270 个文件，668,792,209 bytes。
- 补评清单：`artifacts/evaluation/baseline_completion_v2/action_delay/action_delay_overlay.json`。

## 完成范围

| 项目 | 验收 |
|---|---|
| 训练后参考 | 3 模型 × 9 任务 × 3 种子 = 81 单元；81 份训练身份收据全部绑定 |
| 原环境 ICL | 54 个任务单元，来自 24 枚 LeWM/PLDM 环境级检查点 |
| 两划分源结果 | 270/270 有独立快照、SHA、分数、布尔门槛结果和来源身份 |
| 训练后 CEM | 81 单元 × 6 评测种子，逐种子原始值与 v1 一致 |
| 原环境 CEM | 绑定完整三种子 family summary；DINO-WM 补充 CEM 同时绑定来源 |
| Public Test | 135/135 主分数与 v1 逐位一致；18 枚通过全部门槛 |
| Development | 135/135 有统一判定；18 枚通过；旧测试中的 6 个缺判定跳过项已消除 |
| 历史 Test 边界 | contact_friction/motion_damping 的 30 份 Test 保留来源，并明确排除正式报告 |

三个 Luna / max 独立线程分别完成动作延迟、统一判定和冻结工具；主进程完成交叉检查、
真实准入正例、最终冻结与文档生成。没有为提高分数重新训练，也没有降低门槛或改写原冻结 Test。

## 动作延迟补评

15 个 Development 单元全部重评：九枚训练后模型、六枚原环境基线。每枚均使用最终
1.0.3-rc1 数据：300 query、11 个延迟、52–57 六个 seed 各 50 query；结果保留真实
query/seed/room/direction，补齐 h2/h3 辅助评分，使用与 Test 一致的三步预测请求。

H3 基线通过仓库内显式适配器进入 H7 输入边界。原 pinned H3 adapter 字节保持不变，
新适配器同时支持 Dev 七动作块及 Test 九动作块的调用。实际 torch 为 2.9.1+cu128、
transformers 4.57.1、numpy 1.26.4，运行时指纹直接由评测进程记录。

| 模型 | 原环境 Development | 训练后 Development | 训练后门槛 |
|---|---:|---:|---|
| LeWM | 16.67% ± 0.00pp | 97.64% ± 0.38pp | 3/3 |
| PLDM | 16.66% ± 0.01pp | 99.98% ± 0.02pp | 3/3 |
| DINO-WM | 不兼容 RGB+action 正式 ICL 合同 | 16.63% ± 0.03pp | 0/3 |

相对旧 v1，所有 Development 分数变动都在动作延迟：14 个值逐位不同，其中 6 个仅为
浮点表示微差。实际数据与预测请求结构均已更新，因此完整保留新旧来源；Test 数值未变化。

批执行曾因解释器软链接被解析到系统 Python 而中断，未产生有效结果；修正后在 cwenv
完成。旧失败 sidecar 的写入冲突仅恢复收据，没有重跑已生成的成功结果。执行代码指纹与
收据恢复记录见 batch receipts；原始结果保持不变。

## 固定比较入口

统一入口为 `contextworld-reference-decision`，或：

```bash
python -m contextworld.benchmarks.reference_decision single \
  --component action_delay --split development --input result.json
```

`method` 子命令要求三个独立训练种子、三个不同 checkpoint SHA、一致配方及 adapter，
且每枚各自通过；主分数达线不能覆盖缺失或失败的附加门槛。Public Test 清单 schema v2
必须绑定同 checkpoint 的真实 Development 结果和 SHA，再使用同一判定器准入。
旧 scorer 字段及 schema v1 只保留历史语义。

```bash
python scripts/run_baseline_action_delay_completion_v2.py verify
python scripts/freeze_current_reference_baseline.py verify
python scripts/render_current_reference_tables.py --check
```

冻结工具拒绝覆盖不同内容的已有版本，校验源快照、分数、门槛、准入、CEM、训练身份与
评分源码。主文档 27 行和附录 27 行 Development 表由冻结记录自动生成。

## 验证与边界

- 15/15 补评结果通过完整性核对；真实训练后 LeWM 的 schema v2 准入正例通过，仅运行 plan。
- 冻结校验通过：135 单元、270 份结果；文档生成器 `--check` 通过。
- 274 项目标回归全部通过、零跳过；[紧凑验收记录](baseline_completion_validation_2026-09-10.json)。详细结果记录在 `artifacts/evaluation/baseline_completion_v2/validation/`。
  首轮 271 passed、1 failed；失败是过时的“Test 没有打开”文档断言，修正后该文件 5/5 通过。
  额外覆盖篡改冻结通过标志及准入标志时必须拒绝。

历史 255 份结果没有自动采集运行时指纹，保留 `legacy_runtime_unrecorded`，不追写成已验证。
检查点保留源结果声明的 SHA，本次未重新遍历全部权重计算哈希。旧包上的未变化组件保留
历史 manifest 声明及当前成员 manifest 哈希，不声称已经在新包上重跑或逐字节重验全部数据。
这些已知证据范围不妨碍固定本轮参考数值，但不能替代以后从干净环境复现的验证。

原始结果快照和大数据按仓库既有约定保存在本地 artifacts，不纳入 Git；代码、冻结清单、
紧凑验证记录和规范可以版本化。稳定公共下载、独立外部模型验证属于公共发布工作，未混入
本次内部参考封板。已有 CoJA / rollout 训练改动不属于本次提交。

接手时发现的问题见 [原始审计](Baseline_Handoff_2026-09-10.md)，Claude 阶段过程见
[历史工作记录](Evaluation_Alignment_Worklog_Archive_2026-09-10.md)。
