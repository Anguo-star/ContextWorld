# TwoRoom Step-1 基础数据卡

> 本文记录早期 Step-1 数据族，不是当前 Speed ICL 发布包或下一阶段门 Benchmark 的
> 正式数据清单。速度复现请以
> [Speed ICL Benchmark 使用指南](../TwoRoom_Speed_ICL_Benchmark_Release.md) 为准；
> 门部分请以 [TwoRoom 可见门位置实验报告](../TwoRoom_Door_Benchmark_Design.md)
> 为准。本文只用于理解早期门位置与组合数据的物理来源。

**日期**：2026-07-14  
**状态**：早期物理基础数据已生成并通过全量验证；不作为新门 Benchmark 的正式训练
或 Eval 数据
**环境实现**：Stable-WorldModel 5864b74980f6ed328fd0045e777b3865962eff43  
**Benchmark 规范**：[ContextWorld Benchmark 设计](../ContextWorld_Benchmark_Design.md)

旧机器字段 `correct` 和 `wrong` 仅表示同设定历史和另一设定历史，不表示因子
正确或错误。

## 1. 数据目的

该数据族为 ContextWorld 的 TwoRoom 参考实例提供受控环境分布：

1. `agent.speed`：当前像素不可直接观察、需要从 action→transition 推断的潜在动力学因素，是 ICL 正向因素；
2. `door.position`：改变单帧墙体开口、可通行位置和接触结果，是当前 query 像素可见的几何因素和 ICL 负对照；
3. `agent.speed × door.position`：每个速度值和门位置值在训练中均出现，但指定参数对只出现在 validation/test，用于训练未见组合和潜在/可见因素解耦。

物理轨迹 catalog 定义训练和测试的 scenario 分布。正式 ICL 评测还必须从这些 scenario 编译通过诊断性验证的 `ContextQueryBundle`；不能因为同一 scenario 存在多个 episode 就默认其上下文具有辨识信息。

已有原始 H5 只读保留：

~~~
/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/tworoom.h5
~~~

不重复生成原始数据，也不把原始 H5 物理复制到合成数据中。

## 2. 已验证的早期基础数据

| 数据 | Train | Validation | Test | Episodes | 实际行数 |
|---|---:|---:|---:|---:|---:|
| speed v2 | 32 场景 | 8 场景 | 24 场景 | 384 | 25,432 |
| door v1 | 16 场景 | 4 场景 | 14 场景 | 200 | 12,624 |
| speed×door v1 | 15 组合 | 5 组合 | 5 组合 | 130 | 7,540 |

数据路径：

~~~
artifacts/synthesis/data/tworoom_speed_pixel_v2
artifacts/synthesis/data/tworoom_door_pixel_v1
artifacts/synthesis/data/tworoom_speed_door_composition_v1
~~~

以上为可移植的逻辑路径，默认映射到 `/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/`；例如第一项的实际默认位置为 `.../context_world/synthesis/data/tworoom_speed_pixel_v2`。工程仓库不保存数据、checkpoint、日志或 eval 结果。其他部署可用 `CONTEXTWORLD_ARTIFACT_ROOT` 覆盖产物根目录。

其中 door v1 的训练部分只有 128 个 episode，speed×door v1 也只是早期组合网格。
它们足以验证 factor readback、渲染、碰撞和 split 逻辑，但数据规模和训练控制不足以
支持新的正式能力归因。新门实验会生成一对大规模、严格匹配的“固定门位置”和
“多门位置”v2 数据。

每个 scenario 独立为一个 Lance 表。manifest 保存原子值、seed、Stable-WM commit、输出路径和 fingerprint；catalog 分别列出 train、validation、test 与具体 regime。

## 3. Split 语义

### Speed

- train：范围 [2.5, 8.0] 内 32 个唯一值；
- validation interpolation：8 个训练未见值；
- test interpolation：16 个训练未见值；
- test extrapolation：低端 4 个、高端 4 个；
- 任意跨 split 速度差至少 0.08。

### Door

- train：49 到 169 的 16 个位置，包含原始默认值 49；
- validation interpolation：4 个训练未见位置；
- test interpolation：8 个训练未见位置；
- test extrapolation：[24, 32, 40] 和 [180, 190, 199]；
- 任意跨 split 门位置差至少 4 像素。

### Speed × Door

取速度 {3,4,5,6,7} 与门位置 {61,85,109,133,157} 的完整 5×5 网格：

- train：剩余的 15 个组合；
- validation：5 个对角组合；
- test：5 个循环错位组合；
- train/validation/test 参数对零重叠；
- validation/test 中每个单独速度值与门位置值都在 train 组合中出现；
- validation 与 test 参数对零重叠。

因此该 split 测的是组合泛化，不是未见单变量外推。

## 4. 像素和动力学准确性

所有正式 scenario 均执行逐行精确回放：

- 保存的初始 state 与 goal 重新 reset；
- 对保存 action 逐步调用原始 TwoRoom 环境；
- 每个下一状态必须逐元素完全相等；
- 每个重新渲染帧使用 Stable-WM 相同 JPEG encoder 后，字节必须完全相等；
- goal、terminated、truncated 必须一致；
- factor 列在整条轨迹中恒定且等于 manifest。

结果：

| 数据 | 回放帧 | 回放转移 | Pixel mismatch | State mismatch | 最大状态误差 |
|---|---:|---:|---:|---:|---:|
| speed v2 | 25,432 | 25,048 | 0 | 0 | 0 |
| door v1 | 12,624 | 12,424 | 0 | 0 | 0 |
| speed×door v1 | 7,540 | 7,410 | 0 | 0 | 0 |

额外 oracle：

- speed：快速度一步严格等于慢速度重复相同 action 后丢弃中间帧的结果；
- door 单帧：变化像素严格等于旧/新开口的可见对称差；
- door 边界：上/下边框最后覆写的 44 个像素被显式排除；
- door 接触：同一 state/action 下，开口对齐时穿墙，开口移走时在第 3 步夹到 x=99.5 或 x=124.5；
- 组合数据同时回读并验证 speed 与 door 两个 factor。

## 5. 报告与复现

正式报告：

~~~
artifacts/synthesis/reports/tworoom_speed_pixel_v2.json
artifacts/synthesis/reports/tworoom_door_pixel_v1.json
artifacts/synthesis/reports/tworoom_speed_door_composition_v1.json
~~~

重新生成或断点续跑：

~~~bash
python -m contextworld.synthesis.smoke \
  --config configs/synthesis/tworoom_door_pixel_v1.yaml \
  --resume --skip-loader-check

python -m contextworld.synthesis.smoke \
  --config configs/synthesis/tworoom_speed_door_composition_v1.yaml \
  --resume --skip-loader-check

python scripts/build_tworoom_step1_index.py
pytest -q
~~~

参数 --skip-loader-check 只跳过已经在 smoke 中通过的“原始 H5 + 单个合成 Lance”加载兼容性检查，不跳过任何 factor、split、oracle 或逐行精确回放。

## 6. Benchmark 使用约束

- 原始 H5 和合成 Lance 通过逻辑数据组组合；训练权重作用于 `original/speed/door/composition` 组，而不是每个 Lance scenario。
- OOD rollout 必须按 variation → state → goal 的顺序恢复 speed/door；只恢复 state/goal 的默认 evaluator 不能产生有效结果。
- 普通模型输入只包含声明的 pixels/action/proprio，不包含 factor value、scenario ID 或 privileged state。
- ICL eval 期间模型权重冻结，eval 前后 state-dict hash 必须一致。
- context 与 query 使用不重叠的 transition 或 episodes；context 不得包含 query future、label 或 goal outcome。
- 隐藏因素的不同历史条件必须共享完全相同的 query，只有历史内容可以变化。
- speed context 必须包含非零、无碰撞、未终止且能区分候选速度的转移。
- 当前 `door.position` 从 query 像素中直接可见。正式门位置泛化实验只比较“只给当前
  画面”和“同一门位置的连续 History-3”，不再把不同门位置的历史拼到相同 query
  前面。
- 物理数据 catalog、训练 mixture 和 `ContextQueryCatalog` 是不同层次的入口，不能互相替代。
