| 任务 | 训练方案 | 总 loss 均值变化（%） | 预测 loss 均值变化（%） | 训练 NRE ↓（5→10 轮） | Development NRE ↓（5→10 轮） |
|---|---|---:|---:|---:|---:|
| 接触摩擦 | T2 联合训练 | -9.37 | -18.05 | 1.009 → 1.011 | 0.972 → 1.004 |
| 接触摩擦 | T3 冻结 Encoder | -5.50 | -7.10 | 0.674 → 0.587 | 0.692 → 0.604 |
| 运动阻尼 | T2 联合训练 | -7.94 | -9.76 | 0.236 → 0.197 | 0.261 → 0.221 |
| 运动阻尼 | T3 冻结 Encoder | -2.95 | -0.39 | 0.940 → 0.867 | 0.959 → 0.882 |

T2：加载原环境权重并联合更新；T3：加载相同权重并冻结 Encoder。loss 为各轮已记录训练批次的均值；变化率 =（第 10 轮均值 / 第 5 轮均值 − 1）× 100%，正值表示上升。NRE 越低越好，箭头为第 5 → 10 轮端点；四行覆盖接触摩擦和运动阻尼各两种训练方案。
来源：`docs/research/data/native_training_loss_v1/summary.json`（训练损失）与`docs/research/data/native_epoch_checkpoints_v1/summary.csv`（ICL NRE）；完整曲线见 `docs/research/data/native_training_loss_v1/loss_trends.svg`，其他对照见同目录 `table.md`。
