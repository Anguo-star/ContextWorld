# TwoRoom 双向速度上下文评测协议与结果

**协议版本**：v2.3
**日期**：2026-07-19  
**执行状态**：directional v2 与四模型同协议归因均已完成  
**证据级别**：预注册的 heldout 机制确认，不是最终 test

> 当前统一阶段结论和模型归因边界见
> [TwoRoom 速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)；
> 本文只保留 directional v2 的冻结协议和执行结果。

## 1. 这项实验要分清什么

上一轮 v1 每个 query 只有一档非同速历史：

- query 速度 3.1–5.0 时，另一档历史速度是更快的 7.0；
- query 速度 5.1–7.0 时，另一档历史速度是更慢的 3.1。

因此 v1 同时改变了两件事：历史与查询环境是否同速，以及历史表达的速度是更快
还是更慢。看到分数翻转后，不能判断模型偏好“同速历史”还是“更高速度历史”。

v2 在相同 query 上分别提供慢速和快速历史：

| Eval | 查询环境速度 | 同速历史 | 对照历史 |
|---|---|---|---|
| 慢速历史 Eval | 5.0 / 5.1 | 与查询环境同速 | 慢速 3.1 |
| 快速历史 Eval | 5.0 / 5.1 | 与查询环境同速 | 快速 7.0 |

5.0/5.1 位于 3.1 与 7.0 中间，对两端的速度差近似对称。这样可以在 query、
geometry 和 CEM 不变时比较慢速、同速和快速历史。

这里的速度是环境参数 `agent.speed`，表示每个原始环境步的动作—位移增益。
`frameskip/action_block` 在训练和本协议中始终固定为 5；本实验不是 frameskip
对照。旧结果字段 `correct`、`wrong_slow`、`wrong_fast` 仅为复现保留，分别解释
为同速、慢速和快速历史。

## 2. 数据和隔离

两个 catalog 都从 v1 在任何 heldout 分数出现前生成的未评分 bank 中取 geometry。
源文件 SHA-256 固定为：

```text
ab059341bb8b8c491ef9be1186de0ae02f84b1c1cb10e8448b08de2889929f20
```

冻结范围：

- same-room；
- door=49；
- 距离 72/80/88/96/104/112 px；
- 每个距离 3 个 geometry，共 18 个；
- query 速度 5.0/5.1；
- 每个 Eval 共 36 个 base queries；
- context budget 为 2 个 transitions。

两个 catalog 必须在任何模型出分前一起生成并完成 simulator replay。除对照
历史速度外，query ID、pixels、reset、goal、同速历史、context actions
和 simulator seed 必须逐项相同。

实际 build audit：

| 检查 | 结果 |
|---|---:|
| Catalog bundles | 36 + 36 |
| 每个 catalog 回放的 context rollouts | 216 |
| Simulator replay failures | 0 + 0 |
| Query payload 完全相同 | 36/36 |
| 同速历史 payload 完全相同 | 36/36 |
| 慢速与快速历史 payload 确实不同 | 36/36 |

## 3. 每个评测任务都独立使用 50×6

这项计数不能解释成“两个任务共享 300 次”。

| 项目 | 慢速历史 Eval | 快速历史 Eval |
|---|---:|---:|
| Eval seeds | 42–47 | 42–47 |
| 每条件每 seed | 50 | 50 |
| 同速历史总数 | 300 | 300 |
| 慢速/快速历史总数 | 300 | 300 |
| 每个 Eval 原始 records | 600 | 600 |

两个任务合计 1,200 records。两个 Eval 使用相同 schedule 和 CEM sub-seeds。
因为同速历史输入完全相同，两次同速历史输出也必须逐 evaluation 相同，否则
运行无效。

实际结果：

- 12/12 result files 通过；
- 每个文件恰好 50 同速历史 + 50 对照历史；
- 6/6 schedules 跨 Eval 完全一致；
- 300/300 同速历史输出跨 Eval 完全一致。

## 4. 冻结 CEM 规划配置

本协议固定以下 planner 配置：

| 参数 | 数值 |
|---|---:|
| Eval budget | 50 raw steps |
| Action block | 5 raw steps |
| Horizon | 5 action blocks |
| Receding horizon | 5 action blocks |
| CEM samples | 300 |
| CEM iterations | 30 |
| Top-k | 30 |
| Var scale | 1.0 |

所有 query 已由 direct policy 验证在 50 steps 内物理可达。

## 5. 出分前冻结的判据

### 5.1 同速历史的闭环规划收益

必须同时满足：

- 同速历史 − 慢速历史至少 5 个百分点；
- 同速历史 − 快速历史至少 5 个百分点；
- 两项 paired exact sign test 都 `p ≤ 0.05`；
- 两项都是同速-only > 对照-only。

### 5.2 快速历史的有限配置优势

必须满足：

- 快速历史 − 慢速历史至少 5 个百分点；
- paired exact sign test `p ≤ 0.05`；
- 快速-only > 慢速-only。

单个 distance、speed 或 seed 只用于解释，不替代 pooled 冻结判据。

## 6. 正式结果

### 6.1 三种上下文

| 上下文 | 成功数 | 成功率 | 平均最终距离 | 归一化进度 |
|---|---:|---:|---:|---:|
| 慢速历史 3.1 | 152/300 | 50.67% | 55.04 px | 38.68% |
| 同速历史 5.0/5.1 | 175/300 | 58.33% | 46.53 px | 49.08% |
| 快速历史 7.0 | **194/300** | **64.67%** | **43.43 px** | **52.51%** |

归一化进度为 `100 × (1 − 最终距离 / 初始距离)`。

### 6.2 配对比较

`pp` 表示百分点；“前者-only”表示同一个待测任务只在前一种上下文下成功。

| 比较 | 成功率差 | 前者-only / 后者-only | 双侧 exact sign p |
|---|---:|---:|---:|
| 同速 − 慢速 | +7.67 pp | 25 / 2 | `5.65e-6` |
| 同速 − 快速 | **-6.33 pp** | 2 / 21 | `6.60e-5` |
| 快速 − 慢速 | **+14.00 pp** | 43 / 1 | `5.12e-12` |

冻结判定：

- 同速 > 慢速：通过；
- 同速 > 快速：失败且方向相反；
- SpeedFull 速度条件化 planning ICL：**已建立**；
- 同速历史闭环最优：**未建立**；
- 快速历史在当前有限 CEM 配置下的优势：**确认**。

## 7. 分层稳健性

### 7.1 两个待测速度

| 查询环境速度 | 慢速历史 | 同速历史 | 快速历史 | 快速−慢速 |
|---:|---:|---:|---:|---:|
| 5.0 | 52.05% | 58.22% | 64.38% | +12.33 pp |
| 5.1 | 49.35% | 58.44% | 64.94% | +15.58 pp |

### 7.2 六个评测随机种子

快速−慢速依次为 `+10/+14/+20/+12/+14/+14 pp`，6/6 为正。以 seed 为单位
忽略幅度的双侧 sign test 为 `p=0.03125`。

### 7.3 距离和几何构型

6 个距离中 5 个快速−慢速为正、1 个为 0、0 个为负。18 个 geometry 中：

- 5 个三条件全部失败；
- 7 个三条件全部成功；
- 6 个对 context 敏感；
- 6 个敏感 geometry 全部快速 > 慢速，0 个反向。

所以距离不是无关变量，但单独的欧氏距离不能描述 CEM 难度。geometry 不会
混淆同一 query 内的因果比较，却会决定该 query 是否处于能测出差异的敏感区。

## 8. 结论边界

本协议结果支持：

> SpeedFull 的速度条件化规划 ICL 已建立：CEM 规划结果会随历史速度系统变化，
> 而且快速历史在当前配置下更容易成功。

本协议结果不支持：

- 同速历史在闭环成功率上稳定优于慢速和快速历史；
- 速度 support 是 SpeedFull 与单速混训之间唯一变化；
- 任意多速度训练数据或任意训练 seed 都会复现该效应；
- 上下文已经被证明改变精确 steps-to-success；
- 快速历史优势一定来自模型高估速度；
- 该优势已被分摊到 rollout、cost、top-k 或首动作；
- 更换 CEM horizon、samples 或 cost 后结论一定不变；
- same-room 结果可以直接外推到跨门任务；
- validation 结果等于最终 test。

四模型同协议归因中，原始单训、单速 5 合成单训和
原始+单速 5 混训均未通过同一个快速−慢速门，只有 SpeedFull 通过。因此，
当前证据支持：**固定训练 seed 和集成配方下，
多速度训练是稳定规划速度效应的区分因素**。这仍不是纯速度单变量消融，也
不是跨训练 seed 的普遍因果结论。完整四模型数字统一见
[速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

本协议中的 door 固定且不在必经路径上，因此不能解释配对差异。不同 geometry
对绝对难度影响很大，正式 catalog 必须按难度和结构分层匹配。

后续研究计划不在本评测协议中维护，统一见
[速度上下文学习 Benchmark 报告](../TwoRoom_Speed_Benchmark_Report.md)。

## 9. 复现入口

- 冻结配置：
  `configs/benchmark/tworoom_speed_context_direction_eval_v2.yaml`
- Catalog 构建：
  `scripts/build_tworoom_speed_context_direction_catalogs.py`
- 执行：
  `scripts/run_tworoom_speed_context_direction_eval.sh`
- 汇总：
  `scripts/analyze_tworoom_speed_context_direction_eval.py`
- Build report：
  `artifacts/evaluation/history3/icl_sensitive_v2_directional/catalogs/catalog_build_report.json`
- 正式汇总：
  `artifacts/evaluation/history3/icl_sensitive_v2_directional/formal_summary_n50x6.json`
- 四模型归因冻结配置：
  `configs/benchmark/tworoom_speed_context_model_attribution_v1.yaml`
- 四模型统一汇总：
  `artifacts/evaluation/history3/icl_sensitive_v2_directional/four_model_attribution_summary_n50x6.json`
