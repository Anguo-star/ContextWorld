# ContextWorld

ContextWorld 评测世界模型能否在**不更新权重**的情况下，从近期真实交互的图像与动作中辨认当前隐藏动力学，并预测给定动作的未来。九项任务覆盖连续响应、动作延迟、接触与附着规则、空间转移。模型只接收历史图像、已执行动作和查询动作；隐藏参数、模拟器状态与任务标签不进入模型输入。

**[技术报告](docs/ContextWorld_ICL_Benchmark.md)** 给出数据构建、评测协议和三种模型的九任务训练对照。[Public Test 总表](docs/ContextWorld_ICL_Benchmark.md#training-comparison)报告各任务的协议主指标和训练重复数；[结果附录](docs/reference/ContextWorld_Expanded_Training_Results.md)单列历史作用、条件响应、多步完整预测与原环境规划诊断。各项指标回答不同问题，不合成为一个总分。

## 任务与评分

| 能力类型 | 任务 | 环境 | 历史长度 | 模型需要判断什么 |
|---|---|---|---:|---|
| 即时连续响应 | 速度 | TwoRoom | 3 帧 | 相同动作会移动多远 |
| 即时连续响应 | 推手移动幅度 | PushT | 3 帧 | 相同动作会让推手移动较短还是较远 |
| 即时连续响应 | 机械臂质量 | Reacher | 3 帧 | 质量如何改变机械臂对力矩的响应 |
| 时间延迟动力学 | 动作延迟 | TwoRoom | 7 帧 | 动作等待多少步才生效 |
| 接触或附着条件动力学 | 接触摩擦 | PushT | 3 帧 | 接触时摩擦如何改变运动 |
| 接触或附着条件动力学 | 运动阻尼 | PushT | 3 帧 | 脱离接触后运动衰减得快还是慢 |
| 接触或附着条件动力学 | Cube 夹爪携带规则 | Cube | 3 帧 | 闭合夹爪能否携带方块 |
| 隐藏结构转移 | 门通行规则 | TwoRoom | 3 帧 | 外观相同的门能否通过 |
| 隐藏结构转移 | 传送门出口位置 | TwoRoom | 3 帧 | 进入同一入口后会从哪里离开 |

历史长度包含当前帧。参评模型输出未来 latent，评分器用该模型的冻结目标编码器编码真实未来；各模型可以使用不同的表示空间。任务主指标衡量[任务协议查询](docs/ContextWorld_ICL_Benchmark.md#task-scoring)中的未来辨别，速度、延迟和门通行规则有各自的判断方式与随机参照。多步完整误差比 E/B 是 Development 上的补充预测诊断，不是九任务 Public Test 主分，也不能直接视为物理准确率或规划成功率。

## 数据与评测

数据来自模拟器中的连续交互轨迹，每项任务改变一个隐藏动力学因素，并保存历史、查询动作和真实未来。[数据生成方法](docs/Data_Generation.md)说明配对构造与划分隔离。[HF 数据集指南](docs/HF_Dataset_Export.md)说明本地数据包格式与加载方式。

Training 用于训练；Development 用于检查接口、选择方法、训练预算与检查点；Public Test 用于方法固定后的最终离线报告。Test 采用公开离线评测，没有托管提交服务。当前[九任务 Public Test 结果](docs/ContextWorld_ICL_Benchmark.md#training-comparison)使用事前固定的检查点与评分协议，报告全部预定组合，包括低分结果。扩展动作的多步诊断只覆盖九任务主分布的 2,436 个 Development 源场景，其中速度限于未见速度插值。该诊断的结果在 Development 上报告。

训练对照使用 T0–T3：T0 仅用原环境数据，T1 从模型默认初始化直接混合原环境与任务 ICL 数据，T2 从 T0 权重继续混合训练并更新 Encoder，T3 从同一 T0 权重继续混合训练但冻结 Encoder。DINO-WM 的预训练视觉 Encoder 始终冻结，因此没有 T2。各任务分别训练检查点；总表一行不代表同一模型同时掌握九项任务。配方与初始化细节见[技术报告的训练方案](docs/ContextWorld_ICL_Benchmark.md#training-schemes)。

## 本地接入

安装与评测需要本地数据包和模型检查点；软件依赖的安装不会下载这些制品。下例是 Development 任务协议评测入口，完整的数组形状、历史对齐和外部模型实现见[Adapter 规范](docs/External_Model_Adapter_Contract.md)。

```bash
pip install -e ".[eval]"

python -m contextworld.benchmarks.external_model_cli \
  --benchmark-root /path/to/ContextWorld-bundle \
  --evaluation-split development \
  --task action_strength \
  --adapter lewm \
  --checkpoint /path/to/weights.pt \
  --model-name my-model \
  --output /path/to/development-result.json
```

参评模型实现 `contextworld.benchmarks.adapters.LatentWorldModelAdapter`。方法与检查点在 Development 上确定后，按相同任务协议报告 Test；命令参数和评测范围见[技术报告的接入说明](docs/ContextWorld_ICL_Benchmark.md#7-接入与复现)。

## 结果与发布状态

[Public Test 汇总](docs/research/data/current_reference_test_v1/overview.md)和[机器可读结果](docs/research/data/current_reference_test_v1/summary.json)已公开，可复算汇总。完整数据、模型权重和逐查询输出尚无稳定公共下载版本，因此外部读者目前无法仅凭公开制品端到端复跑全部实验。各项证据的覆盖范围与限制见[技术报告的发布状态](docs/ContextWorld_ICL_Benchmark.md#8-可复现性与结果报告)。

更多入口见[文档导航](docs/README.md)，包括[训练指南](docs/StableWM_Training.md)与[指标研究](docs/ICL_Metric_Study.md)。
