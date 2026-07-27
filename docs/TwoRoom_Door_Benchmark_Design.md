# TwoRoom History=3 门规则 ICL Benchmark v1

> **文档角色：Door 组件技术附录。** 最终发布的数据结构、统一命令和当前结论以
> [ContextWorld ICL Benchmark v1](ContextWorld_ICL_Benchmark.md)为准。普通使用者
> 不需要把本文件与内部协议拼接阅读。

本文件保留 Door 组件的详细任务定义、数据构造和参考结果。统一安装、数据包与发布
使用方式只在总文档中维护。

| 项目 | 当前状态 |
|---|---|
| 任务、Training、Validation、指标和门槛 | 已冻结 |
| 可直接运行的模型 | Stable-WorldModel LeWM、PLDM |
| Validation 参考结果 | 已完成 |
| 正式 Test | 继续封存 |
| 本地发布包 | 可完整审计和导出 |
| 公共数据发布 | 等待发布者填写许可证和下载地址 |

## 1. 这个 Benchmark 测什么

它只回答一个问题：

> 当前画面看不出门能否通过时，模型能否根据刚才三帧中的真实交互，预测下一次穿门
> 会成功还是会被挡住？

环境有两种隐藏规则：

| 隐藏规则 | 穿门后的真实下一帧 |
|---|---|
| 门可以通过 | 智能体穿过墙 |
| 门会阻挡 | 智能体停在墙边 |

两种规则使用完全相同的门外观、当前画面和待执行动作。模型看不到规则标签、坐标或
真实下一帧，因此不能只看当前画面猜答案。

评测分别给模型三种三帧历史：

| 历史 | 模型刚刚看见什么 | 用途 |
|---|---|---|
| 看见穿过 | 相同动作成功穿过门 | 应预测“可以通过” |
| 看见受阻 | 相同动作撞门后停下 | 应预测“会阻挡” |
| 没有试门 | 历史没有暴露规则 | 只观察模型默认倾向 |

随后三种历史都回到同一张当前画面，再执行同一个穿门动作：

```text
看见穿过 ─┐
看见受阻 ─┼─> 同一张当前画面 + 同一个动作 ─> 预测下一帧
没有试门 ─┘
```

如果模型在“看见穿过”和“看见受阻”后给出不同且正确的预测，差异只能来自历史。
这就是本任务中的 ICL：推理时利用历史，模型参数不会更新。

这不是“门位置 ICL”。门位置能从当前画面直接看见，测的是视觉几何泛化；这里隐藏
的是门的物理通行规则。

## 2. 一次评测怎样判分

每个 query 离线保存两张真实下一帧：

1. 门可以通过时的真实下一帧；
2. 门会阻挡时的真实下一帧。

模型读取一种历史，输出下一步预测 latent。评分器使用这个 checkpoint 自己的冻结
Encoder 编码两张真实下一帧，再计算两个 loss：

```text
预测 latent 与“穿过门”真实 latent 的 loss
预测 latent 与“被挡住”真实 latent 的 loss
```

loss 较小的真实下一帧就是模型选择的结果。这样不需要从 latent 反推像素位置，也不
需要在线运行环境。

Benchmark 报告两个容易解释的主分数：

| 主分数 | 怎样计算 | 说明 |
|---|---|---|
| 下一帧判断正确率 | 看见穿过后选“穿过”，看见受阻后选“受阻”的比例 | 模型是否选对结果 |
| 历史引导正确率 | 对同一个真实结果，正确历史是否比相反历史带来更低 loss | 历史是否把预测推向正确方向 |

50% 表示没有稳定区分两种规则。“没有试门”没有唯一正确答案，只用于显示默认倾向，
不参与通过判定。

原始 latent loss 不能直接跨 checkpoint 比大小，因为不同 Encoder 的 latent 尺度
可能不同。跨模型只比较上面的正确率、分组稳定性和通过种子数。

## 3. 数据怎样构造和隔离

### 3.1 Training

训练轨迹由 TwoRoom 环境真实执行动作生成，不是生成模型画出来的。每个场景成对
生成：

- 一条门可以通过的轨迹；
- 一条门会阻挡的轨迹。

一对轨迹的门位置、起点、动作和采样设置完全相同，只改变隐藏门规则。这样动作本身
不会泄漏答案。

| Training 内容 | 数量 |
|---|---:|
| 训练门位置 | 96 |
| 每种规则的训练 clip | 7,680 |
| 两种规则合计 | 15,360 |
| Loader 内部校验门位置 | 16 |
| Loader 内部校验 clip | 2,560 |

正式门规则训练使用两种规则的精确并集。只含单一规则的数据只用于检查模型是否形成
固定偏好。

所有门规则模型都从同一个原始 History=3 LeWM checkpoint 初始化，并沿用原始训练集
得到的 normalizer。门规则续训只采样上述合成数据，不混入原始 TwoRoom 样本。

“固定原始图像表示”不是冻结随机 Encoder，而是冻结该原始 LeWM checkpoint 的
`encoder` 和 `projector`，只训练后续预测部分。

### 3.2 Validation

Validation 使用训练中完全未出现的 42 个门位置：

| Validation 单元 | 数量 |
|---|---:|
| Eval seed | 6：42、43、44、45、46、47 |
| 每个 seed 的 query | 50 |
| 每个 seed 的左右方向 | 各 25 |
| 每种历史条件的 query | `50×6=300` |
| 三种历史合计模型预测 | 900 |
| 三种历史 × 两张真实下一帧的 loss | 1,800 |

每一种历史条件都有独立完整的 300 个 query，不是把多个条件平分 300。

Validation 已全部离线保存，评分时在线环境调用数为 0，也不使用 CEM。因此结果不会
混入规划时限、候选动作数量或搜索随机性的影响。

### 3.3 已完成的数据审计

发布审计确认：

- Training 和 Loader Validation 不含任何 Eval 门位置；
- 训练模板 ID、query 像素哈希与 Eval 不重合；
- 两种训练规则逐场景严格配对；
- 混合 catalog 是两个单规则 catalog 的精确并集；
- 模型输入只包含画面和动作；
- 300 个 Eval payload 的文件数、内容哈希和冻结清单一致。

## 4. 什么条件才算通过

单个 checkpoint 必须同时满足：

1. 两种规则的下一帧判断都高于 50%，不能只会固定猜一种结果；
2. 六个 Eval seed、左右两个方向的 12 个分组都朝正确方向变化；
3. 四项成对统计的 95% bootstrap 区间下界都大于 0；
4. 两张真实下一帧在该 checkpoint 的 latent 中确实可区分；
5. 不存在目标距离相同造成的虚假胜利。

单个 checkpoint 通过，只能说明这个 checkpoint 具备能力。要声明一种训练方法稳定
生效，必须使用训练种子 `3072、4096、5120`，并且三个 checkpoint 全部通过。

完整数值门槛和 bootstrap 设置以冻结发布清单为准：
`configs/benchmark/tworoom_door_icl_release_v1.yaml`。

## 5. 做了哪些对比，结果是什么

五组模型的区别如下。所有门规则续训组使用同一原始初始化、同一合成数据和同一训练
预算；只改变训练目标或是否固定图像表示。

| 模型 | 门规则训练方式 | 训练种子 |
|---|---|---:|
| 原始 History=3 LeWM | 没有使用门规则合成数据 | 1 |
| LeWM 联合训练 | 两种规则合成数据；Encoder 与预测器一起更新 | 3 |
| LeWM 固定图像表示 | 两种规则合成数据；冻结原始 Encoder/Projector | 3 |
| PLDM 联合训练 | 两种规则合成数据；使用 PLDM 训练目标并联合更新 | 3 |
| PLDM 固定图像表示 | 两种规则合成数据；使用 PLDM 目标并冻结图像表示 | 3 |

未见门位置 Validation 的最终结果：

| 模型 | 下一帧判断正确率 | 历史引导正确率 | 通过种子 |
|---|---:|---:|---:|
| 原始 History=3 LeWM | 50.00% | 51.33% | 0/1 |
| LeWM 联合训练 | 50.33% | 53.83% | 0/3 |
| LeWM 固定图像表示 | 100.00% | 100.00% | 3/3 |
| PLDM 联合训练 | 99.33% | 99.67% | 3/3 |
| PLDM 固定图像表示 | 100.00% | 100.00% | 3/3 |

从这张表可以直接得到四个阶段性结论：

1. **原始 LeWM 没有门规则 ICL。** 两个主分数接近 50%，没有按历史稳定切换。
2. **当前 LeWM 联合训练也没有学会。** 三个训练种子全部失败。
3. **合成数据本身足以支持学习。** 固定原始图像表示后的 LeWM 为 3/3 通过。
4. **门规则 ICL 不依赖冻结 Encoder 才能成立。** PLDM 联合训练在相同数据、初始化
   和预算下为 3/3 通过。

所以，当前 Benchmark 已经能够区分“没有利用历史”和“根据历史正确切换”。LeWM
联合训练失败是模型目标与优化路径的问题，不是数据或评测是否有效的问题。

这些结果只证明 History=3 的一步真实未来预测，不等同于长距离导航或 CEM 规划能力。

### 5.1 研究解释

后续表示几何、模块换件、首 batch 梯度和受控干预共同表明，LeWM 的失败不是整个
表示变成常量，而是门规则对应的条件转移方向被选择性压近：

- LeWM 联合训练后，Encoder 的整体方差和有效秩仍基本保留；
- 同一 query 下两种规则未来的相对距离缩小两个数量级；
- 收缩主要由共享在线 Encoder 的参数更新携带；
- SIGReg 的梯度方向反对收缩，其标量 loss 也会下降，但当前目标只约束无条件边缘
  分布，无法判断某个历史相关的动力学方向是否被其他视觉因素替代。

因此，当前研究把这个现象定义为**条件动力学坍缩**：边缘表示保持非退化，但
历史可推断的转移机制在联合训练中变得不可辨识。target detach 和单独的方差下限
都不能充分恢复规则切换；`std + covariance` 只在单种子 tiny pilot 中证明更强的
几何约束能够改变训练路径，尚不是正式修复或新方法。

完整的机制证据、理论边界、相关工作定位和下一阶段方法原则统一维护在
`stable-worldmodel/research/conditional_dynamics_representation/README.md`。
本 Benchmark 文档只负责冻结任务、数据、评分协议和正式参考结果。

## 6. 怎样使用

安装工程并指定本地数据根目录：

```bash
pip install -e .
export CONTEXTWORLD_ARTIFACT_ROOT=/path/to/context_world_artifacts
```

查看冻结版本并审计数据：

```bash
contextworld-door info
contextworld-door audit
contextworld-door audit --full
```

查看三个参考训练配方之一：

```bash
contextworld-door train-plan \
  --recipe pldm_joint \
  --training-seed 3072
```

评测一个 Stable-WorldModel checkpoint：

```bash
contextworld-door smoke \
  --adapter pldm \
  --checkpoint /path/to/weights.pt \
  --model-name my_model \
  --training-seed 3072 \
  --output /tmp/door_smoke.json

contextworld-door eval \
  --adapter pldm \
  --checkpoint /path/to/weights.pt \
  --model-name my_model \
  --training-recipe my_method \
  --training-seed 3072 \
  --output results/door_s3072.json
```

三个训练种子分别评测后，重新计算并汇总：

```bash
contextworld-door score \
  --method-name my_method \
  --input results/door_s3072.json \
  --input results/door_s4096.json \
  --input results/door_s5120.json \
  --output results/door_method.json
```

`eval` 会保存 1,800 条逐 query loss；`score` 从这些记录重新计算结果，不直接信任
结果文件里已有的汇总值。

### 接入其他模型

当前命令行直接支持 Stable-WorldModel LeWM 和 PLDM。其他工程只需实现统一 adapter：

1. 读取原始 `uint8 RGB` 历史画面和环境动作；
2. 输出下一步预测 latent；
3. 用同一模型的冻结 Encoder 编码真实下一帧；
4. 返回模型参数与 checkpoint 哈希，证明评分时权重没有变化。

稳定接口为：

- `contextworld.benchmarks.DoorICLEvalDataset`
- `contextworld.benchmarks.DoorICLModelAdapter`
- `contextworld.benchmarks.door_icl_score.evaluate_door_icl_model`

## 7. 当前发布边界

v1 可以用于报告：

- 模型能否根据三帧历史判断门可以通过还是会阻挡；
- 一种训练方法是否在三个训练种子上稳定获得该能力；
- 能力是否推广到训练中未出现的门位置。

v1 不能用于声称：

- 已经完成正式 Test 排行榜；
- 已经证明长距离门导航、CEM 规划或 History>3；
- `LeWM + std + covariance` 已在正式 8 卡三个种子及跨能力回归上解决联合训练失败；
- 本地导出操作自动授予公开再分发权。

代码、数据、评分器和哈希已经形成可运行的本地技术发布候选。正式公共发布还需要
发布者确定 ContextWorld 源码许可证、自产数据许可证，并填写公共下载地址。

## 8. 文件入口

普通读者只需阅读本文。其余两个入口用于运行和审计：

| 文件 | 用途 |
|---|---|
| `configs/benchmark/tworoom_door_icl_release_v1.yaml` | 冻结数据、代码、指标、结果与哈希 |
| `contextworld/benchmarks/door_icl_cli.py` | `info/audit/export/train-plan/smoke/eval/score` |

历史预注册动作和逐文件证据保留在
`docs/protocols/TwoRoom_History3_Hidden_Passage_Feasibility_v1.md`，仅供审计者使用，
不再承担当前结论说明。
