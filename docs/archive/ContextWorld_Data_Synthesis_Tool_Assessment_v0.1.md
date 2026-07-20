# ContextWorld 四任务数据合成工具评估 v0.1（已归档）

> 本文是早期工具评估快照，不代表当前 Benchmark 结论。

日期：2026-07-13

本评估对应本地 `stable-worldmodel` checkout：
`5864b74980f6ed328fd0045e777b3865962eff43`。

## 结论

不存在一个成熟工具能够直接完成四个任务的“原子参数变换、组合场景编译、
train/test 留出、可追溯采集、旧 checkpoint OOD eval”全流程。

推荐的职责划分是：

- `ContextWorld`：原子定义、组合约束、split、seed、manifest、验证和实验矩阵；
- `Stable-WorldModel`：统一环境入口、policy rollout、Lance/H5 loader、训练和评估；
- 各任务官方底层模拟器/生成器：真正修改动力学与场景参数；
- 原始 H5 保持只读，通过 loader 做逻辑混合，不制作不可追溯的大合并文件。

因此，`Stable-WorldModel` 是统一主干，但不是四任务所有参数变换的唯一工具。

## 四任务判断

| 任务 | 当前可直接使用的原子 | 成熟底层工具 | 只用 Stable-WM 是否足够 | 建议 |
| --- | --- | --- | --- | --- |
| TwoRoom | `agent.speed`、`door.position/size`、墙、颜色、初始状态等 | Stable-WM 自带 variation space 和 expert | 首轮足够 | 作为第一条端到端链路；本次 smoke 已完成速度、门位置与训练未见组合（组合留出） |
| Reacher | 当前 wrapper 已暴露 arm/finger density 和视觉变化 | dm_control + PyMJCF/MuJoCo | 只做已暴露 density 时足够；扩展 friction、damping、结构参数时不够 | 保留 Stable-WM collector，在 dm_control/MJCF 层实现原子 adapter |
| PushT | 当前 wrapper 主要暴露几何、视觉和初始状态；damping 是构造参数 | Pymunk；Diffusion Policy 官方 PushT 实现可作任务/数据语义参考 | 改视觉/几何时基本够；改质量、摩擦力时不够 | 用 Pymunk 的 body/shape/space 属性实现受约束原子，再由 Stable-WM 采集 |
| Cube | 当前 wrapper 已覆盖 cube size、初始状态、相机、灯光、颜色 | OGBench 官方环境与 `data_gen_scripts`；底层 MuJoCo | 采集格式统一足够；重做官方任务生成和广泛动力学变换时不够 | 以 OGBench generator/scripted policy 为真值，接到 Stable-WM writer |

## 工具成熟度和用途

### Stable-WorldModel

官方项目提供统一的 collect、load、train、MPC evaluate 接口，并原生支持
Lance 与 HDF5。它非常适合作为本项目的 orchestration 和数据 I/O 主干。

来源：<https://github.com/galilai-group/stable-worldmodel>

本地代码审计还确认：

- `World.collect` 能把 reset variation 写入轨迹；
- H5 与 Lance 由同一 `load_dataset` 入口读取；
- `ConcatDataset` / multitask loader 可以做逻辑混合；
- 四任务已有 collector 或 wrapper，可复用 rollout 和 expert 逻辑。

限制是 variation 的覆盖范围因任务而不同；统一 collector 并不等于已经提供了
所有物理参数的安全变换。

### OGBench（Cube）

OGBench 官方仓库包含 Cube manipulation 环境、训练/验证 dataset API、
`data_gen_scripts` 以及复现数据集的命令；官方说明 manipulation 数据生成不需要
locomotion expert 的额外训练依赖。它比自行猜测 Cube expert 或 episode 语义可靠。

来源：<https://github.com/seohongpark/ogbench>

用途：复用官方 task reset、scripted policy 和数据生成逻辑；由 ContextWorld 注入
原子组合，最终通过 Stable-WM 输出统一格式。

### dm_control / MuJoCo（Reacher 与 Cube）

dm_control 是 Reacher 的任务来源；MuJoCo 官方提供 MJCF/PyMJCF/mjSpec 的程序化
模型编辑能力。密度、质量、摩擦和结构参数应在模型定义或编译边界修改，而不是
仅在轨迹写出后改 metadata。

来源：

- <https://github.com/google-deepmind/dm_control>
- <https://mujoco.readthedocs.io/en/stable/modeling.html>
- <https://mujoco.readthedocs.io/en/stable/programming/modeledit.html>

用途：实现 `arm_density`、`finger_density`、geom friction、damping 等 adapter，
并在 reset 后回读实际模型值做验证。

### Pymunk 与 Diffusion Policy（PushT）

PushT 的物理实现基于 Pymunk。Pymunk 官方 API 原生提供 body/shape 的 mass、
density、friction、elasticity，以及 space damping；它是物理原子的正确落点，
但它本身不是 dataset split/manifest 工具。

来源：<https://www.pymunk.org/en/latest/pymunk.html>

Diffusion Policy 官方仓库提供 PushT task、dataset、runner 与原始 Zarr 数据语义，
适合用来校对任务兼容性，不建议取代 Stable-WM 成为本项目的统一编排层。

来源：<https://github.com/real-stanford/diffusion_policy>

## 实施顺序

1. TwoRoom：用 Stable-WM 原生 variation 完成第一条正式数据与 OOD eval 链路。
2. Reacher：先做当前已经原生暴露的 density 原子，验证第二种 simulator adapter。
3. PushT：实现 Pymunk friction/mass/damping adapter，并增加物理回读验证。
4. Cube：接 OGBench 官方 generator，再添加 MuJoCo 物理/视觉原子。

这能先验证实验问题，而不把第一步拖成四个环境的重构项目。

## 原子 adapter 的最低契约

每个原子必须同时实现：

1. `validate(requested_value)`：范围、类型、互斥和安全约束；
2. `apply(env, value)`：在正确的环境生命周期阶段修改参数；
3. `readback(env)`：从实际 simulator/model 回读，而不是复述配置；
4. `serialize(value)`：写入 manifest 和轨迹 metadata；
5. `compatibility(other_atoms)`：声明不可组合或需要重新编译的组合；
6. `pixel_effect`：声明属于单帧几何、单帧外观、时序动力学、接触动力学或相机投影；
7. 可执行 oracle：交齐该 `pixel_effect` 所需的参数、state、pixel 与时序证据；
8. 一个单原子测试、至少一个组合测试，以及实际轨迹的逐行 state/pixel 重放测试。

“有 API 能设置参数”只代表底层能力成熟；完成上述契约后，才算 ContextWorld 中
可复用、可做 train/test 留出的成熟合成原子。
