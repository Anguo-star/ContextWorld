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

门位置研究阶段也已完成本地 Validation：固定门与多门数据、六个配对模型、真实未来
预测和规划评测均按独立 `50×6` 设计执行。多门训练在训练范围内的新门位置上出现稳定
的一步预测收益，但没有通过预注册的全部门位置判定，因此门位置 Test 继续锁定，当前
也不作为第三方发布 Benchmark。门在 query 中可见，这一结果属于视觉几何泛化，不是
门位置 ICL。

当前版本先正式支持 Stable-WorldModel LeWM。代码和本地数据导出流程已经跑通；大
文件公共下载地址与 ContextWorld 自产内容的分发许可证仍需由发布者配置，因此请先
按使用指南准备 artifact root。

第一次使用请阅读：

1. [Speed ICL Benchmark 使用指南](docs/TwoRoom_Speed_ICL_Benchmark_Release.md)
2. [Speed 当前实验结果](docs/TwoRoom_Speed_Benchmark_Report.md)
3. [可见门位置泛化结果](docs/TwoRoom_Door_Benchmark_Design.md)
4. [Benchmark 设计原则](docs/ContextWorld_Benchmark_Design.md)

查看冻结发布信息：

```bash
python -m contextworld.benchmarks.speed_icl_cli info
```

验证本地数据是否齐全：

```bash
python -m contextworld.benchmarks.speed_icl_cli audit
```

Speed 的公开接口对应 Validation release；正式 Speed Test 仍然封存。Door 当前是
Validation 阶段研究结果，预注册主门未通过，也没有解锁 Test。两者都不应被描述为
最终 Test 排行榜结果。
