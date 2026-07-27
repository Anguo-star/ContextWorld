# TwoRoom Speed ICL Benchmark 使用指南

> **文档角色：Speed 组件复现附录。** 最终发布只维护一份
> [ContextWorld ICL Benchmark v1 统一说明](ContextWorld_ICL_Benchmark.md)；
> 本文件保留 Speed 的扩展命令与历史兼容细节。

**发布版本**：`contextworld_tworoom_speed_icl_history3_v1`

**发布范围**：Validation 训练与评测工具；正式 Test 不包含在本版本中

**当前可运行模型**：Stable-WorldModel LeWM

## 1. 这个发布包能做什么

拿到代码和数据后，使用者可以独立完成三件事：

1. 按冻结配方训练单速控制模型或多速度目标模型；
2. 用自己的 Stable-WorldModel LeWM checkpoint 运行四条速度 Eval；
3. 得到统一的单模型分数，或用三个目标种子和三个控制种子得到完整训练方法结论。

核心评测完全离线：真实未来帧在模型评分前已经生成并冻结，评分过程中不调用
TwoRoom 环境。固定候选和 CEM 是单独的规划支持层，其中 CEM 会在线运行环境。

冻结配置在：

```text
configs/benchmark/tworoom_speed_icl_release_v1.yaml
```

所有大文件使用相对于 `CONTEXTWORLD_ARTIFACT_ROOT` 的逻辑路径，代码中不依赖本机
绝对路径。

## 2. 发布内容

本仓库现在提供可执行代码、冻结清单和数据导出工具。大文件仍保存在本地 artifact
root，尚未配置公共下载地址；发布者需要先执行第 3.2 节的导出命令，再把导出目录
上传到对象存储或数据集平台。ContextWorld 仓库目前也没有声明代码和自产数据的
分发许可证，因此在许可证确定之前，这一版本是**技术上可复现的发布候选**，不是
已经可以任意再分发的数据包。上游原始 TwoRoom 数据沿用其页面标注的 MIT 许可证。

### 2.1 训练数据

| 数据 | 用途 | 当前大小 |
|---|---|---:|
| 原始 TwoRoom H5 | 原始任务数据和统一 normalizer | 解压后约 12 GB |
| 速度 5 合成数据 | 单速训练控制 | 约 6.0 GB |
| 32 个速度合成数据 | 多速度目标训练 | 约 6.1 GB |

推荐训练配方：

```text
多速度目标：50% 原始数据 + 50% 多速度合成数据
单速控制：  50% 原始数据 + 50% 速度 5 合成数据
```

两套配方共享场景请求、样本量、normalizer、网络、优化器、训练步数和 checkpoint
选择规则。完整方法提交需要三个配对训练种子：`3072/4096/5120`。

只用原始数据训练的模型是公共参考，不参与“多速度训练是否优于单速控制”的正式
归因门。它使用 9,000 个训练 episode、固定 6,420 个 optimizer step；两个混训
配方各使用 12,840 个 optimizer step，保证原始域和合成域分别获得相同的总抽样量。

原始数据来自官方
[quentinll/lewm-tworooms](https://huggingface.co/datasets/quentinll/lewm-tworooms)，
页面标记为 MIT。仓库提供的是 `tworoom.tar.zst`；解压后的 `tworoom.h5` 应满足：

```text
bytes:  12,775,849,984
sha256: 129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623
```

ContextWorld 不重复打包这份上游文件。

### 2.2 离线 Eval 数据

| 轨道 | 速度 | 每个“参考速度×历史速度”单元 |
|---|---|---:|
| 训练范围内已见 | 3.1 / 5.1 / 7.0 | 50×6=300 |
| 训练范围内未见 | 3.4 / 4.8 / 6.9 | 50×6=300 |
| 低端范围外 | 1.75 / 1.95 / 2.15 / 2.35 | 50×6=300 |
| 高端范围外 | 8.25 / 8.75 / 9.50 / 10.25 | 50×6=300 |

离线包约 462 MB，包含 4,200 个 query payload。每个 payload 保存 History-3 输入、
五个未来 action block 和五张真实未来帧。Catalog 保存逐文件 SHA256；完整审计会读取
并校验所有 4,200 个 payload。

### 2.3 规划 Eval 数据

规划层覆盖训练范围内已见和范围内未见速度，提供：

- 300 条冻结候选、10-block horizon 的动作排序；
- 50/75/100 原始步执行预算；
- 300 candidates、30 iterations、top-k 30 的冻结 CEM 配置。

每个速度、历史条件和评测种子都有 50 次执行；六个种子合计 300 次。规划分数是
支持性效用指标，不替代真实未来 latent 主指标。

## 3. 准备环境和数据

### 3.1 安装代码

```bash
git clone https://github.com/Anguo-star/ContextWorld.git
git clone https://github.com/galilai-group/stable-worldmodel ../stable-worldmodel
git -C ../stable-worldmodel checkout 5864b74980f6ed328fd0045e777b3865962eff43
pip install -e .
```

Stable-WorldModel 是上游提供的统一数据、训练和规划环境；当前发布固定到上述 commit，
避免上游接口变化改变结果。

### 3.2 设置路径

```bash
export CONTEXTWORLD_ARTIFACT_ROOT=/path/to/contextworld-speed-artifacts
export CONTEXTWORLD_TWOROOM_H5=/path/to/tworoom.h5
```

发布者生成可复制的数据树：

```bash
contextworld-speed export \
  --destination /path/to/contextworld-speed-artifacts \
  --mode copy
```

本机不复制大文件、只验证目录布局时，可使用 `--mode symlink`。导出包不包含上游
原始 H5；`release/inventory.json` 会明确记录这一点。

### 3.3 校验数据

快速检查评分器源码、文件、目录、大小和 catalog 哈希：

```bash
contextworld-speed audit --output audit.json
```

发布前完整读取原始 H5、两套合成训练数据树、4,200 个离线 Eval payload 和规划
payload：

```bash
contextworld-speed audit --full --output full-audit.json
```

只有 `passed=true` 的数据根目录才能用于正式评分。

## 4. 训练 Stable-WorldModel LeWM

先分别验证训练 mixture 能被完整构建。命令会使用冻结配方自己的 epoch 大小，不能
用多速度配方的样本预算代替原始数据参考：

```bash
contextworld-speed train-plan \
  --recipe original_reference \
  --output original-train-plan.json

contextworld-speed train-plan \
  --recipe single_speed_control \
  --output single-train-plan.json

contextworld-speed train-plan \
  --recipe multi_speed_target \
  --output multi-train-plan.json
```

如需重新训练原始数据参考模型：

```bash
TRAINING_SEED=3072 \
  bash scripts/run_h3_original_ability_train.sh origheldout train
```

训练一个多速度目标模型：

```bash
MODEL_VARIANT=multi TRAINING_SEED=3072 \
  bash scripts/run_h3_speed_isolated_train.sh train
```

训练配对的单速控制模型：

```bash
MODEL_VARIANT=single TRAINING_SEED=3072 \
  bash scripts/run_h3_speed_isolated_train.sh train
```

正式方法评测需要对 `3072/4096/5120` 分别执行两条命令。训练入口自动使用
`CONTEXTWORLD_TWOROOM_H5`，输出保存在 `CONTEXTWORLD_ARTIFACT_ROOT/training/runs`。

## 5. 评测单个模型

### 5.1 先跑 smoke

Smoke 在四条轨道的每个参考速度上各取一个 query，只验证模型接口、数据和指标链路，
不能作为正式分数。

```bash
contextworld-speed smoke \
  --checkpoint /path/to/weights_final.pt \
  --model-name my-lewm \
  --training-role multi_speed_target \
  --training-seed 3072 \
  --device cuda:0 \
  --output smoke.json
```

### 5.2 完整离线评测

不指定 `--tracks`、`--eval-seeds` 或样本上限时，默认运行完整四轨道和每格
`50×6=300`：

```bash
contextworld-speed eval \
  --checkpoint /path/to/weights_final.pt \
  --model-name my-lewm \
  --training-role multi_speed_target \
  --training-seed 3072 \
  --device cuda:0 \
  --output result-s3072.json
```

输出中的 `status=passed` 表示运行和完整性检查通过；能力是否通过应读取各轨道、各
horizon 的 `formal_within_checkpoint_pass`。单模型结果只能描述该 checkpoint，
不能单独证明提升来自多速度训练。

## 6. 汇总完整训练方法

准备三个多速度目标结果和三个同种子的单速控制结果：

```bash
contextworld-speed aggregate \
  --method-name my-speed-method \
  --target multi-s3072.json \
  --target multi-s4096.json \
  --target multi-s5120.json \
  --control single-s3072.json \
  --control single-s4096.json \
  --control single-s5120.json \
  --output method-summary.json
```

汇总器会检查：

- 六个结果是否来自同一发布版本；
- 三个训练种子是否唯一且一一配对；
- 训练种子是否严格为冻结的 `3072/4096/5120`，目标和控制角色是否放置正确；
- 每个目标模型是否在每个参考速度和六个 Eval 种子上方向一致；
- 多速度目标相对单速控制的提升是否在三个训练种子上全部为正。

只有训练范围内已见和范围内未见轨道的 h1 都通过时，完整方法输出的
`formal_claim_level` 才是 `training_attributed_speed_icl`；否则为
`speed_icl_not_demonstrated`。更长 horizon 和双向范围外能力继续逐轨道报告，不会
被一个总分掩盖。单模型输出为 `descriptive_model_score`。

## 7. 运行规划支持评测

一个固定候选单元：

```bash
contextworld-speed planning-cell \
  --mode fixed \
  --track seen_for_multi \
  --query-speed 3.1 \
  --seed 42 \
  --num-eval 50 \
  --checkpoint /path/to/weights_final.pt \
  --device cuda:0 \
  --output fixed-seen-q3.1-s42.json
```

一个 CEM 单元：

```bash
contextworld-speed planning-cell \
  --mode cem \
  --track seen_for_multi \
  --query-speed 3.1 \
  --seed 42 \
  --num-eval 50 \
  --checkpoint /path/to/weights_final.pt \
  --device cuda:0 \
  --output cem-seen-q3.1-s42.json
```

完整规划评测要遍历两条轨道、各三个速度和六个种子。每个单元的 `num-eval` 必须是
50，不能把 300 次分摊给多个速度。完成后分别汇总固定候选或 CEM 文件：

```bash
contextworld-speed aggregate-planning \
  --input result-1.json --input result-2.json \
  --output planning-summary.json
```

汇总器不会把较快历史的高成功率自动解释为预测更准。固定候选报告真实 regret；
CEM 分别报告 50/75/100 步成功率、最终距离和距离 AUC。

规划单元结果会写入 release、catalog、checkpoint、normalizer、Stable-WorldModel commit
和规划参数标记。汇总器只在两条轨道、全部速度、六个种子、每格 50 次均齐全且这些
标记一致时设置 `full_protocol=true`；少量 smoke 只能得到 `smoke_only`。

## 8. 指标怎样读

| 指标 | 含义 |
|---|---|
| 原始 latent MSE | 同一 checkpoint 内的真实未来误差，不跨模型直接比较 |
| loss 比值 | 同速历史 loss ÷ 其他历史平均 loss；越低越好 |
| query 胜率 | 同速历史优于其他历史平均值的独立 query 比例 |
| 严格胜率 | 同速历史同时优于每一种其他历史的 query 比例 |
| 固定候选 regret | 模型所选动作相对候选库真实最优动作的额外代价 |
| CEM 成功率 | 冻结规划资源和执行预算下的任务效用 |

结果不会输出从 latent 反推的像素位置或数值速度。这样的派生量会额外混入 decoder
或位置估计误差，不是本发布的速度校准主证据。

## 9. 怎样接入其他模型工程

首版命令行只正式支持 Stable-WorldModel LeWM。评测器本身依赖
`SpeedICLModelAdapter`，而不是直接依赖 LeWM 类。新 adapter 需要实现：

```python
class MyAdapter(SpeedICLModelAdapter):
    @property
    def protocol(self): ...

    @property
    def metadata(self): ...

    def encode_pixels(self, pixels, *, batch_size): ...

    def rollout_latents(
        self, input_pixels, raw_action_blocks, *, batch_size
    ): ...

    def frozen_state_hash(self): ...
```

接口输入始终是原始 `uint8 RGB` 帧和未归一化环境动作。图像预处理、动作归一化和
native latent 由 adapter 自己负责。新模型接入必须新增以下验证后，才能列为正式
支持：

1. History-3、action block 5 和五步输出形状测试；
2. 模型冻结哈希测试；
3. 完整 rollout 与截短 rollout 的共同前缀测试；
4. 至少一条完整 `50×6` 轨道与参考实现的指标一致性测试；
5. 若支持规划，再实现该工程自己的 cost/solver adapter。

跨工程时不能直接比较原始 latent MSE，因为不同 encoder 的坐标系不同。正式训练
归因必须在同一工程、同一 adapter 定义下成对比较“多速度目标”和“单速控制”；若
未来需要给不同架构做统一排名，应另加共同的冻结表征或像素级评分器，不能把当前
native latent 数值硬拼成排行榜。

## 10. 当前验证状态与边界

发布入口已经完成以下真实验证：

- 本地训练和 Eval 数据快速审计通过；
- 原始 H5 与全部 4,200 个 Eval payload 的完整 SHA256 审计通过；
- 原始、单速和多速度三套训练 mixture 均可按冻结预算构建；
- 真实多速度 LeWM checkpoint 的四轨道 smoke 通过；
- 训练已见轨道完整 `50×6` 运行通过；
- 新旧评分器的 query 胜率和严格胜率完全一致，loss 比值最大绝对差
  小于 `2.2e-5`，所有正式判定一致；
- 真实固定候选和真实 CEM 单元均运行通过。

当前版本是 Validation release。它不包含正式 Test 目标，也没有公共排行榜服务。
范围外速度数据是公开的能力边界测试；在当前参考模型上没有通过，不能据此声称
范围外速度外推。
