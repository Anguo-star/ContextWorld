# TwoRoom SpeedClean 数据与训练协议

> **文档角色**：历史数据与训练协议，不维护当前结果。当前数据比较和正式结论
> 统一见[Benchmark 主文档中的速度章节](../ContextWorld_ICL_Benchmark.md#41-速度)。

**版本**：v1.2
**日期**：2026-07-16  
**适用模型**：`H3-SpeedClean`（LeWM，history=3）

旧机器字段 `correct` 和 `wrong` 分别表示同速历史和另一档速度历史，不表示速度
正确或错误。

## 1. 目标

`TwoRoom-SpeedClean-v1` 用于检验 history-3 world model 能否从两个先验
transition 中辨识当前速度，并在未见速度上调整预测与规划。该版本保持原
TwoRoom 的视觉—动作接口，同时消除首轮 `H3-MixSpeed` 数据中的像素编码域
线索、窄起终点覆盖和高重复采样。

协议遵循“机制一致、样本隔离”：训练与 E4 使用相同的速度作用机制、图像
预处理、action block 和 history 长度，但使用互斥的速度值、reset seeds、
轨迹和 query templates。

## 2. 像素表示

合成帧以 PNG `compress_level=1` 写入 StableWM-compatible Lance。PNG 只作为
无损存储容器；StableWM 读取后得到原始 `uint8 RGB`，因此模型无法通过 JPEG
伪影区分 original 与 synthetic 数据源。

lossless smoke 已验证：

- 每个存储帧均能由固定 simulator state/action 逐字节重新编码；
- PNG 解码结果与 simulator raw render 逐像素一致；
- H5 与 Lance 经 StableWM reader 得到相同的 `pixels/action/proprio` 张量形状；
- 4 个 smoke 场景共 168 帧，pixel-byte mismatch 与 decoded-pixel mismatch 均为 0。

## 3. 速度划分

正式训练使用 32 个速度：

```text
2.6, 2.7, 2.8, 2.9, 3.0,
3.4, 3.6, 3.7, 3.8, 3.9, 4.0,
4.2, 4.4, 4.5, 4.7, 4.8,
5.2, 5.3, 5.5, 5.7,
6.0, 6.1, 6.2, 6.3,
6.6, 6.7, 6.8,
7.1, 7.2, 7.3, 7.8, 7.9
```

训练期间只用于监测的 dev 速度为：

```text
2.5, 3.2, 4.3, 4.6, 4.9, 5.4, 5.6, 5.8
```

E4 validation 速度 `3.1/3.3/3.5/4.1/5.0/5.1/5.9/7.0` 不进入
synthetic train 或 dev。原始 H5 仍包含固定速度 5.0；这一事实在结果中单独
标记，不将 5.0 解释为 synthetic held-out speed。

## 4. Paired seed crossing 与覆盖

train 使用 4 个独立 seed blocks。每个 block 含 32 个 episode seeds，并与
全部 32 个训练速度做笛卡尔交叉：

```text
4 seed blocks × 32 resets × 32 speeds = 4,096 train episodes
```

dev 使用：

```text
2 seed blocks × 16 resets × 8 speeds = 256 dev episodes
```

总计 144 个 Lance 场景、4,352 条 episodes。每个 block 内，相同 episode index
在所有速度下共享 reset seed；不同 block 和 split 的 episode seeds 互斥。

正式生成前的确定性 reset coverage 为：

| Split | Unique resets | Start bins | Goal bins | Start-goal pairs |
|---|---:|---:|---:|---:|
| train | 128 | 92 | 86 | 128 |
| dev | 32 | 29 | 32 | 32 |

网格宽度为 14 px。所有 4,352 个场景—episode reset 在最大对应速度下均满足
至少两行训练数据的安全边界；最小保证 margin 为 8.834 px。

## 5. 训练数据质量门槛

正式 `H3-SpeedClean` preflight 必须同时满足：

- pixel codec 为 lossless PNG；
- train scenarios 不少于 128，dev scenarios 不少于 16；
- frameskip=5、num_steps=4 后 synthetic raw train clips 不少于 160,000；
- 在 50/50 mixture、5 logical epochs 和 6,420 optimizer steps 下，synthetic
  平均抽样不超过 20 次/raw clip；
- catalog、manifest、StableWM commit、split 与 scenario fingerprint 完全一致；
- factor columns 不进入模型输入。

正式训练继续使用 original/speed=`50/50`、effective global batch=1,024、总
6,420 optimizer steps，以保持与 `H3-Orig` 和首轮 `H3-MixSpeed` 的计算预算
一致。训练完成后先运行 E1 与原始 ID retention；两项链路通过后执行 E4
`50×6`，并以固定 K=2 的同速−另一档历史 paired effect 作为规划 ICL 判据。

## 6. 正式数据质量结果

`TwoRoom-SpeedClean-v1` 正式生成与全量重放已完成。144 个 Lance
场景共包含 296,344 帧，其中 train 为 280,025 帧，dev 为 16,319 帧。
全量验证重放 291,992 个状态转移；pixel bytes、PNG 解码像素、状态转移、
goal 不变性与 termination flags 的 mismatch 均为 0，最大状态绝对误差为 0。

正式 history-3 训练 preflight 产生 204,211 个 synthetic raw train clips。
在 50/50 mixture 与 5 个 logical epochs 下，synthetic 的 3,287,040 次抽样
对应平均 16.096 次/raw clip，低于 20 次上限；204,211 个 clips 全部在
正式抽样映射中被覆盖。因此 codec 一致性、轨迹覆盖和复用率三项数据
条件已满足正式训练协议。

对应报告为 `artifacts/synthesis/reports/tworoom_speed_clean_v1.json` 和
`artifacts/training/reports/h3_speedclean_s3072_preflight.json`。

`H3-SpeedClean-s3072` 正式训练已在 4 张 H800 上完成：5 个 logical
epochs、每 rank 12,840 个 microbatches、共 6,420 个 optimizer/scheduler
steps。final-step 权重已通过 StableWM 原生加载的逐张量精确校验，
SHA-256 为
`2d64cf5c9beeae6578529e9fa94538dfd9e937073e5114933b7db7025ba5b35c`。
正式报告为 `artifacts/training/reports/h3_speedclean_s3072.json`。

该次已完成的正式运行启动于原生 `spt.Manager` resume 接线之前，
因此只保留完整 model-only epoch/final 权重，不伪造缺失的 optimizer
状态。后续新运行由 StableWM 原生 trainer checkpoint 支持 `train`
auto-resume 和 `resume` required 语义。

## 7. 正式评测结果

`H3-SpeedClean-s3072` 已完成全部 validation gate。E1 使用 32 个固定 query
bundles：K=2 同速历史相对无历史的 latent-MSE 改善为 0.00369，95%
scenario-bootstrap CI 为 [-0.00791, 0.01647]；另一档历史相对同速历史的损失差
为 0.01674，CI 为 [0.00625, 0.02884]。该结果表明模型输出能够区分不同速度
历史，但同速历史收益尚不稳定。

原始 TwoRoom ID retention 使用 seeds 42–47、每 seed 50 条 StableWM CEM
rollout，成功率分别为 98%/98%/100%/98%/100%/100%，合计 297/300
（99.0%）。因此后续 OOD 结果不能归因于基础 TwoRoom 规划能力丢失。

固定 K=2 的 E4 `50×6` 中，同速与另一档历史均为 73/300（24.33%）；300
个 paired queries 中 both-success=73、neither=227、同速-only=0、
另一档-only=0，双侧 paired sign test p=1.0。所有 8 个速度的 success count
也逐项相同。由此，E1 的预测分离没有转化为规划行为，数据修复正向对照未
达到 planning-level ICL 判据。

正式产物为
`artifacts/evaluation/history3/h3_speedclean_s3072/e1_speed_paired.json`、
`artifacts/evaluation/history3/h3_speedclean_s3072/id_retention_n50x6.json`
和
`artifacts/evaluation/history3/h3_speedclean_s3072/e4_speed_ctx_n50x6.json`。

## 8. 执行入口

```bash
# 仅编译并验证正式数据计划
bash scripts/run_tworoom_speedclean_data.sh compile

# lossless 小规模采集与逐帧 replay
bash scripts/run_tworoom_speedclean_data.sh smoke

# 正式采集；默认 4 分片并行，可通过 SHARDS 调整
SHARDS=4 bash scripts/run_tworoom_speedclean_data.sh generate-parallel

# 串行入口；中断后可用 resume
bash scripts/run_tworoom_speedclean_data.sh generate
bash scripts/run_tworoom_speedclean_data.sh resume

# 正式训练前的数据与采样预算检查
bash scripts/run_h3_speedclean_train.sh preflight

# 强制新训练
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedclean_train.sh fresh

# StableWM 原生 auto-resume：有 trainer checkpoint 则恢复，无则新建
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedclean_train.sh train

# 强制从 StableWM trainer checkpoint 恢复
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedclean_train.sh resume
```

所有生成数据、训练 checkpoint 和 evaluation result 默认写入
`dataset/ag_data/data/world_model/context_world/`；工程仓库只保存代码、配置、
测试和技术文档。
