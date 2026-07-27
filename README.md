# ContextWorld

ContextWorld 是面向视觉世界模型的上下文规则学习基准，用于评测模型能否从短期交互
历史中识别隐藏环境规则，并在不更新参数的情况下将规则用于未来预测。

## Benchmark tasks

| 任务 | 评测能力 | v1 状态 |
|---|---|---|
| Speed | 从 History=3 识别隐藏移动速度 | Validation 已完成 |
| Door Rule | 从 History=3 判断外观相同的门能否通过 | Validation 已冻结 |

## Documentation

任务定义、数据集、指标、参考结果和运行方法统一见：

> [ContextWorld：视觉世界模型的上下文规则学习基准](docs/ContextWorld_ICL_Benchmark.md)

完整数据包包含一份 `README.md` 和一个 `benchmark/` 数据目录。Speed 与 Door 共用
相同的代码版本、数据根和审计入口。

## Quick validation

```bash
export CONTEXTWORLD_BENCHMARK=/path/to/ContextWorld-ICL-Benchmark-v1
export CONTEXTWORLD_ARTIFACT_ROOT=$CONTEXTWORLD_BENCHMARK/benchmark
export CONTEXTWORLD_TWOROOM_H5=$CONTEXTWORLD_BENCHMARK/benchmark/upstream/lewm-tworooms/tworoom.h5

contextworld-benchmark \
  --release-config $CONTEXTWORLD_BENCHMARK/benchmark/suite.yaml \
  audit --full
```

## Release scope

v1 发布 Training 与 Validation 数据，不包含隐藏 Test。当前 release candidate 已可
完整审计；正式公共分发还需补充源码与自产数据许可证及稳定下载地址。
