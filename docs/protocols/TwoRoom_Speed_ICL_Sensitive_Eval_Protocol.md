# TwoRoom 上下文敏感距离校准协议与结果

> **文档角色**：支持性实验，仅用于复现距离敏感区诊断。当前数据比较和正式结论
> 统一见[Benchmark 主文档中的速度章节](../ContextWorld_ICL_Benchmark.md#611-速度)。

**协议版本**：v1.2
**日期**：2026-07-18  
**执行状态**：calibration 已完成；正式 heldout 按预注册停止  
**用途**：检验“只要把 Eval 目标距离调到 CEM 的敏感区，同速历史收益就会出现”

> 当前统一结论见
> [ContextWorld ICL Benchmark：速度](../ContextWorld_ICL_Benchmark.md#611-速度)；
> 本文只保留 v1 距离校准和停止规则。

同速历史表示历史速度与查询环境速度相同；对照历史是冻结 catalog 中的另一档
速度。旧机器字段 `correct` 和 `wrong` 仅用于复现，不表示速度正确或错误。

## 1. 为什么要做距离校准

旧 E4 只有四个固定模板。在固定 speed=5 的结果中：

- s0/s1/s2 全失败；
- s3 全成功。

这使二值成功率主要由模板决定。即使上下文改变了模型预测，只要没有让失败
模板跨过 16 px 成功半径，同速历史和对照历史的成功数仍会相同。

v1 保持模型和 CEM 不变，只扫描起点到目标的欧氏距离，目的是找出：

1. 任务不是几乎全成功或全失败；
2. 同速历史比对照历史至少高 5 个百分点。

## 2. 难度校准和正式留出数据必须分开

在任何分数出现前同时生成两套 bank：

| Bank | 每距离 geometry 数 | 用途 |
|---|---:|---|
| Calibration | 4 | 选择是否存在合格距离 |
| Heldout | 3 | 只有 calibration 过门后才能正式评分 |

两套 bank 使用不同 geometry seed，并要求：

- reset-goal pair 零重叠；
- query ID 零重叠；
- 全部 context rollout simulator replay；
- 全部 query 用 direct policy 验证 50 steps 内可达。

实际 catalog 共 504 bundles：

- calibration 288 bundles；
- heldout 216 bundles；
- 3,024 条 context rollouts 全量 replay；
- 0 failure；
- geometry/query overlap 均为 0。

Heldout bank 在 v1 主判据中从未评分。

## 3. 冻结任务和 CEM 规划配置

| 项目 | 数值 |
|---|---|
| Room relation | same-room |
| Door | 49 |
| 速度 | 3.1/3.3/3.5/4.1/5.0/5.1/5.9/7.0 |
| 距离 | 48/56/64/72/80/88/96/104/112 px |
| Context budget | 2 transitions |
| Eval budget | 50 raw steps |
| Horizon | 5 action blocks × 5 raw steps |
| CEM | 300 samples × 30 iterations，top-k 30 |
| Calibration seeds | 2401/2402 |

每个 calibration result 对一个速度和一个 seed 执行 72 个同速/对照配对。
合计 1,152 pairs，每距离 128 pairs。

## 4. 出分前冻结的距离门

一个距离只有同时满足以下条件才可进入正式 heldout：

1. 同速/对照 pooled success 在 10%–90%；
2. 同速 − 对照至少 5 个百分点；
3. 同速-only > 对照-only。

最多选择两个相邻距离。如果没有距离过门：

- 写入 `formal_eval_authorized=false`；
- 不评分 heldout；
- 不运行四模型正式矩阵；
- 不允许看到结果后降低 5 pp 门槛。

## 5. 难度校准正式结果

| 距离 | 同速历史 | 对照历史 | 同速−对照 | Pooled | 判定 |
|---:|---:|---:|---:|---:|---|
| 48 | 98.44% | 96.09% | +2.34 pp | 97.27% | 天花板 |
| 56 | 92.97% | 89.84% | +3.13 pp | 91.41% | 天花板 |
| 64 | 91.41% | 89.84% | +1.56 pp | 90.63% | 天花板 |
| 72 | 66.41% | 64.06% | +2.34 pp | 65.23% | effect 不足 |
| 80 | 24.22% | 21.09% | +3.13 pp | 22.66% | effect 不足 |
| 88 | 72.66% | 77.34% | -4.69 pp | 75.00% | 方向相反 |
| 96 | 38.28% | 39.84% | -1.56 pp | 39.06% | 方向相反 |
| 104 | 34.38% | 42.19% | -7.81 pp | 38.28% | 方向相反 |
| 112 | 20.31% | 19.53% | +0.78 pp | 19.92% | effect 不足 |

没有距离同时满足三个门。正式结果为：

```text
formal_eval_authorized = false
selected_distance_bins = []
heldout_scored = false
```

## 6. 这个结果说明什么

已确认：

- 48–64 px 仍在天花板；
- 72–112 px 已出现多个非饱和区；
- 离开地板/天花板没有自动产生同速历史收益；
- “只要调整距离，同速−对照就会打开”没有得到支持。

但不能因此说距离不重要。距离决定总体任务尺度，只是它不是充分的难度变量。
同一距离的不同 geometry 可以分别全成功、全失败或对 context 敏感。正式评测
必须按 geometry 和 baseline difficulty 分层，不能只按欧氏距离汇总。

## 7. 主结果后的探索性诊断

以下分析在 v1 停止后进行，只用于生成下一假设：

| 查询环境速度组 | 对照历史速度 | 同速历史 | 对照历史 | 同速−对照 |
|---|---:|---:|---:|---:|
| 3.1–5.0 | 7.0，更快 | 33.33% | 39.79% | -6.46 pp |
| 5.1–7.0 | 3.1，更慢 | 58.33% | 51.04% | +7.29 pp |

按 context 中表达的速度高低重标：

| 条件 | 成功 |
|---|---:|
| 较高速 context | 359/768 = 46.74% |
| 较低速 context | 307/768 = 39.97% |
| 差值 | +6.77 pp |

Higher-only/lower-only 为 65/13，双侧 sign `p=1.81e-9`，较高速 context
的平均最终距离低 5.66 px。

这个结果不能回写成 v1 formal success，因为查询速度组与对照历史方向
同时变化。它只产生了“当前 CEM 可能偏好高速提示”的假设。

## 8. 与 Directional v2 的关系

v1 产生的高速提示假设由预注册的 directional v2 heldout 独立检验。本协议
不重复维护 directional v2 结果，完整设计和结论见
[TwoRoom 双向速度上下文 Eval](TwoRoom_Speed_Context_Direction_Eval_Protocol.md)。

## 9. 复现入口

- 冻结配置：
  `configs/benchmark/tworoom_speed_icl_sensitive_eval_v1.yaml`
- Catalog build report：
  `artifacts/evaluation/history3/icl_sensitive_v1/catalogs/catalog_build_report.json`
- Calibration selection：
  `artifacts/evaluation/history3/icl_sensitive_v1/calibration_selection.json`
- Post-hoc diagnostics：
  `artifacts/evaluation/history3/icl_sensitive_v1/calibration_diagnostics_posthoc.json`
- 执行：
  `scripts/run_tworoom_icl_sensitive_eval.sh`
- 分析：
  `scripts/analyze_tworoom_icl_sensitive_eval.py`
