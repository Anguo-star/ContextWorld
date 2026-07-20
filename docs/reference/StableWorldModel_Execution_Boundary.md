# Stable-WorldModel 执行边界与可移植入口

**版本**：v1  
**日期**：2026-07-16

## 1. 定位

ContextWorld 是 benchmark 数据、协议和启动配置层，不定义另一套 world
model 训练或评测 runtime。读者可以把 `STABLEWM_REPO` 指向自己的
Stable-WorldModel checkout；在 benchmark 锁定的 commit 上，只替换路径不应
改变模型、Trainer、checkpoint 或 planning 语义。

| 层 | 责任 |
|---|---|
| Stable-WorldModel | 模型结构、LeWM forward/loss、AMP/DDP Trainer、optimizer/scheduler state、`spt.Manager` checkpoint/resume、模型序列化与原生 planning/inference primitives |
| ContextWorld | scenario 生成、manifest/catalog、split 隔离、context/query 协议、逻辑组采样权重、评测矩阵、结果聚合与 provenance 检查 |

## 2. 训练入口

```bash
export STABLEWM_REPO=/path/to/stable-worldmodel
export STABLEWM_REF=5864b74980f6ed328fd0045e777b3865962eff43

# 只检查数据与计算预算
bash scripts/run_h3_speedclean_train.sh preflight

# 强制新运行；run 目录必须为空
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedclean_train.sh fresh

# StableWM 原生 auto-resume：有 trainer checkpoint 则恢复，无则新建
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedclean_train.sh train

# 强制恢复；缺少 trainer checkpoint 则失败
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedclean_train.sh resume
```

`train` 和 `resume` 使用 Stable-WorldModel LeWM 入口的
`get_resume_checkpoint_path` 以及 StablePretraining `Manager`。原生 checkpoint 位于：

```text
<artifact-root>/training/runs/checkpoints/<run-name>/<run-name>_weights.ckpt
```

该文件包含 model state、optimizer state、LR scheduler、global step 和 Lightning
loop state。`weights_epoch_*.pt` 只用于推理和结果固化，不得冒充断点
续训 checkpoint。

## 3. 数据适配边界

TwoRoom-SpeedClean 要求两级采样：首先在 original/speed 逻辑组间按
50/50 抽样，再在 speed 组内对 128 个 scenarios 等权。当前锁定的
Stable-WorldModel multitask loader 只支持 child-level 等权；直接传入 1 个 H5
和 128 个 Lance children 会得到约 0.8/99.2，不等价于 benchmark。

因此当前 ContextWorld 仅保留 `ContextWorldGroupedDataModule` 这一数据边界
适配；模型、forward、Trainer 状态和 resume 仍由 Stable-WorldModel/StablePretraining
执行。在 Stable-WorldModel 原生支持 hierarchical group sampling 前，不应
把直接运行 `scripts/train/lewm.py` 宣称为本 benchmark 的等价命令。

## 4. 评测入口

```bash
export STABLEWM_REPO=/path/to/stable-worldmodel
DEVICE=cuda:0 bash scripts/run_speed_stage1_eval.sh prediction
```

ContextWorld 负责将冻结的 context/query catalog 展开为 conditions 并聚合
paired metrics；模型加载、environment、CEM 与 world-model inference 从
`STABLEWM_REPO` 指向的 checkout 导入。benchmark 正式结果仍要求
`STABLEWM_REF`/runtime commit 与报告记录一致。

精确 speed-support overlap 对照提供一个完整、可改路径的参考入口：

```bash
export STABLEWM_REPO=/path/to/stable-worldmodel
export STABLEWM_REF=5864b74980f6ed328fd0045e777b3865962eff43

# 合成与全量重放
bash scripts/run_tworoom_speedseen_data.sh generate-parallel

# 数据预算检查与原生 StableWM 训练
bash scripts/run_h3_speedseen_train.sh preflight
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedseen_train.sh train

# E1、原始 TwoRoom ID 与 E4；E4 也提供四卡并行入口
bash scripts/run_h3_speedseen_eval.sh e1
bash scripts/run_h3_speedseen_eval.sh id
bash scripts/run_h3_speedseen_e4_parallel.sh
```

原始 TwoRoom ID 入口直接启动 checkout 中的
`scripts/plan/eval_wm.py`，并把 `STABLEWM_HOME` 指向原生
`checkpoints/<run>/<weights.pt> + config.json` 布局。启动器会将指定的
`STABLEWM_REPO` 前置到 `PYTHONPATH`，避免系统中另一版本的
`stable_worldmodel` 与 checkout 的 evaluator 混用；它不转换 checkpoint，
也不重新定义模型加载、CEM 或成功判定。并行 seed 分别使用 evaluation
目录下的 seed-local native cache，其中 weight/config 是指向同一冻结 checkpoint
的符号链接；这是为了隔离原生 evaluator 生成的同名 `env_*.mp4`，防止并发
seed 相互覆盖，同时避免把 evaluation 文件写回 training checkpoint 目录。

## 5. 原生 resume 验证

2026-07-16 的单卡 smoke 在 optimizer step 2 停止，然后从同一原生
checkpoint 恢复并推进至 step 4。恢复报告记录：

- `initial_global_step=2`；
- 新执行 2 个 optimizer steps；
- `global_step=4` 且 `scheduler_last_epoch=4`；
- checkpoint 包含 1 个 optimizer state 和 1 个 LR scheduler state；
- 恢复后的 model-only 权重可逐张量精确重载。

证据文件为
`artifacts/training/reports/h3_speedclean_native_resume_smoke_v2_20260716_step2.json`
和
`artifacts/training/reports/h3_speedclean_native_resume_smoke_v2_20260716_step4.json`。
