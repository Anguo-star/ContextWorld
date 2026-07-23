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

门位置研究也已完成验证集实验：单门训练与多门训练共六个新模型，所有评测均按独立
`50×6` 执行。多门训练明显改善了新门位置的一步预测和动作选择，但没有满足预先规定
的全部门位置标准，因此 Test 继续锁定，当前也不作为第三方发布 Benchmark。门在当前
画面中可见，所以这项实验测的是可见几何泛化，不是门位置 ICL。

真正需要历史的门任务也已开始。当前完成的是 32 对 History-3 隐藏通行规则可行性
smoke：两种规则使用相同动作并回到相同 query，但真实未来不同。该构造依赖当前
碰撞器的恢复投影，因此下一步是端到端数据管线与模型输入 pilot，尚未进入模型训练，
也不能写成门规则 ICL 已经成立。

当前版本先正式支持 Stable-WorldModel LeWM。代码和本地数据导出流程已经跑通；大
文件公共下载地址与 ContextWorld 自产内容的分发许可证仍需由发布者配置，因此请先
按使用指南准备 artifact root。

第一次使用请阅读：

1. [Speed ICL Benchmark 使用指南](docs/TwoRoom_Speed_ICL_Benchmark_Release.md)
2. [Speed 当前实验结果](docs/TwoRoom_Speed_Benchmark_Report.md)
3. [可见门位置实验结果](docs/TwoRoom_Door_Benchmark_Design.md)
4. [隐藏通行规则可行性协议](docs/protocols/TwoRoom_History3_Hidden_Passage_Feasibility_v1.md)
5. [Benchmark 设计原则](docs/ContextWorld_Benchmark_Design.md)

查看冻结发布信息：

```bash
python -m contextworld.benchmarks.speed_icl_cli info
```

验证本地数据是否齐全：

```bash
python -m contextworld.benchmarks.speed_icl_cli audit
```

Speed 的公开接口对应验证集 release；正式 Speed Test 仍然封存。Door 当前也是验证集
阶段结果，正式通过标准未满足，因此没有解锁 Test。两者都不应被描述为最终 Test
排行榜结果。
