# ContextWorld

ContextWorld 用受控数据变化检验世界模型能否从近期交互中识别环境规则，并立即把
规则用于未来预测和规划。

当前第一个可运行实例是 **TwoRoom History-3 Speed ICL Benchmark**。它提供：

- 原始数据与单速、多速度合成训练配方；
- 训练已见、范围内未见、低端范围外和高端范围外速度 Eval；
- 1/2/3/5 步真实未来 latent 评分；
- 固定候选与 CEM 规划支持性评测；
- Stable-WorldModel LeWM 的训练和评测入口；
- 为其他工程模型预留的 adapter 接口。

当前版本先正式支持 Stable-WorldModel LeWM。代码和本地数据导出流程已经跑通；大
文件公共下载地址与 ContextWorld 自产内容的分发许可证仍需由发布者配置，因此请先
按使用指南准备 artifact root。

第一次使用请阅读：

1. [Speed ICL Benchmark 使用指南](docs/TwoRoom_Speed_ICL_Benchmark_Release.md)
2. [当前实验结果](docs/TwoRoom_Speed_Benchmark_Report.md)
3. [Benchmark 设计原则](docs/ContextWorld_Benchmark_Design.md)

查看冻结发布信息：

```bash
python -m contextworld.benchmarks.speed_icl_cli info
```

验证本地数据是否齐全：

```bash
python -m contextworld.benchmarks.speed_icl_cli audit
```

正式 Test 仍然封存。当前公开接口对应 Validation release，不应被描述为最终 Test
排行榜结果。
