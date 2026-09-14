# ContextWorld v3 数据发布准备（Release Readiness）

**状态：v3 基线已永久封存；本文件只跟踪把冻结数据发布为 HF 数据集的准备项。尚未上传，
公共下载地址与 revision 待确定。** 基线验收见
[最终验收（2026-09-10）](reference/Baseline_Final_Acceptance_2026-09-10.md)；正式任务定义
与参考结果见 [`ContextWorld_ICL_Benchmark.md`](ContextWorld_ICL_Benchmark.md)；打包与下载
细节见 [`HF_Dataset_Export.md`](HF_Dataset_Export.md)。

旧的 “Public v1 / Cube 外部模型试点” 发布叙事已按原字节归档于
[`archive/ContextWorld_Public_v1_Release_Readiness_Pre_V3.md`](archive/ContextWorld_Public_v1_Release_Readiness_Pre_V3.md)。
它只是历史记录，不构成当前发布门。其历史草案配置
[`contextworld_public_v1_release_readiness_draft_v1.yaml`](../configs/benchmark/contextworld_public_v1_release_readiness_draft_v1.yaml)
保持 `draft_not_release_authority` 身份与既有身份校验不变。

## 1. 基线状态：已封存，不再作为发布条件

- 冻结版本 `contextworld_joint_scratch_v1_reference_results_freeze_v3`，
  SHA-256 `01298ca407c4a72bdd9a1238879ae7109b1141a7cfdd2b25065cb0bdec87fef0`。
- 2026-09-10 验收已核对 135 个检查点×任务单元、270 份原始结果，304 项目标回归全部通过。
  该验收在记录日期完成；本文件不声称在本轮重新运行任何评测。
- 基础数据发布不需要重训、新模型或评测重跑。更高覆盖面或外部独立复现属于独立的证据
  改进，只能增强而不能阻塞已封存的不可变基线。DINO 补充证据保持既有独立身份，不在
  本次数据发布范围内重开。

## 2. 当前发布需求表

| 需求 ID | 交付物 | 验收证据 | 状态 |
|---|---|---|---|
| R1 | 冻结基线完整（freeze v3 JSON 与离线校验命令） | 2026-09-10 最终验收；freeze JSON SHA-256 与上方一致 | 已通过 |
| R2 | 仓库外的候选目录 `<CONTEXTWORLD_DATASET_ROOT>/ContextWorld-v3-hf`（合并冻结 Training/Development 与冻结 Test，原始文件字节不变） | 已构建：4,757 个数据文件、20,489,896,395 bytes；排除 614 个旧 Development 留档文件 | 已通过本地验收 |
| R3 | 离线字节校验与实际加载 | 全量 4,785 个分发文件 SHA-256 通过；9 项任务 × 3 划分读取通过，10 个 Training view 注册入口通过；9 项默认训练配方与后续 ICL 评测使用同一本地 HF 根目录 | 已通过本地验收 |
| R4 | 数据卡与元数据准确（模板 token 由实际 inventory 替换；事实性引用；不虚构 DOI 或作者） | HF YAML 元数据解析通过，inventory 已实数渲染；`viewer: false`，明确使用 native snapshot 下载 | 已通过本地验收 |
| R5 | 许可证与署名文件（`LICENSE`、`DATA_LICENSE`（CC BY 4.0）、`NOTICE`） | 三文件已复制并纳入全量哈希校验 | 已通过本地验收 |
| R6 | HF 命名空间/仓库与不可变修订 URL | **未提供**：待确定 dataset repo 并产生不可变 commit 后回填 URL/revision | **阻断（未提供）** |
| R7 | 上传后的下载冒烟测试 | `huggingface_hub.snapshot_download(repo_type='dataset', revision=<不可变 commit>, local_dir=...)` 拉取后与本仓 `--verify` 逐字节一致 | **阻断（待上传后执行）** |
| R8 | 发布记录 | 数据 repo/revision、代码 revision、manifest SHA-256 与 v3 freeze ID 的对应关系 | 待上传后生成 |

本地验收记录见 [HF 候选包验收](reference/contextworld_v3_hf_candidate_2026-09-11.json)。
该历史记录保留当时构建路径；完整数据现存放在原 `data/world_model/ContextWorld-v3-hf`
目录，仓内只保留代码、文档和小型验收记录。训练与评测均使用数据目录下的同一候选包。
迁移后的全量校验见 [2026-09-13 迁移记录](reference/contextworld_v3_hf_relocation_2026-09-13.json)。
候选 manifest SHA-256：`882bd831d5cdeb1a6f8de8fd8ac18fa4b87bac864e2fe96f5b276e090dd0299e`。
截至 2026-09-12，19 项打包测试（含迁移目录后脱离原始源目录的校验）与 5 项发布文档测试通过；本次没有训练或模型评测。加载检查发生在当前环境，
不冒充上传后的下载验证或外部独立复现。

## 3. 明确的非阻断项

- 额外任务覆盖、更多种子或外部独立复现：属于后续证据改进，不影响本表状态。
- 派生 Parquet 视图或 dataset viewer 支持：可另行独立版本化，当前原始 Lance/JSON 分发
  不依赖它。
- 旧 Cube 外部试点矩阵与 “全 Suite 跨架构覆盖” 声明：仅存在于归档文档，不是当前门。

## 4. 执行顺序

构建与逐文件校验使用 [`prepare_contextworld_hf_release.py`](../scripts/prepare_contextworld_hf_release.py)，
原生读取检查使用 [`check_contextworld_hf_loading.py`](../scripts/check_contextworld_hf_loading.py)。

1. 本地构建候选：`--development-root` + `--test-root` 先 plan，确认后 `--execute`（一次性）；
2. 离线 `--verify` 全量通过，并核对构建时已生成的数据卡、文件清单与本地加载结果；
3. 提供 HF 命名空间/仓库，上传最终评审包（本仓文档不执行上传）；
4. 上传后用不可变 revision 做下载冒烟并离线校验；
5. 保存发布记录，回填 R6/R7/R8 证据。在此之前本文件只表示准备状态，不表示已发布。
