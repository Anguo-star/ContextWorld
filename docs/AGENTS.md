# Documentation editing

These rules apply to the public ContextWorld report, guides and result appendices.

- Lead with the conclusion and scope; keep task scores, prediction diagnostics and planning measures distinct.
- Write for external readers. Preserve scientific definitions, result identities and evidence limits; do not describe unpublished internal iterations as release history.
- Treat generated BEGIN / END blocks as script-owned. Change display labels in their generator and verify that numeric results are unchanged. Never hand-edit generated values.
- Verify affected result renderers, local links and formatting after edits. Public wording must not imply that unavailable data, weights or downloads already exist.

## 文档写作与更新规范

- 对外文档只描述最终采用的任务、数据、方法和结果。未发布的内部迭代不写成版本沿革，“原协议”“补测”“初始化需求”等过程表述不进入公开阅读路径；初始化作为训练方案的一部分说明。缺项、测量范围与制品获取状态按事实简洁注明。
- 主报告说明任务、数据构建、采用的评分、训练设置、主要结果与解释边界。研究附录按研究问题组织方法验证和机制证据；不将试验流程或待办清单写入结论。
- 按研究问题、实验设置、结果和解释组织材料；不按日期或排错过程写成实验日志。先说明当前结论及适用条件，再给必要证据；明确已完成验证与尚缺证据，已有验证不重复扩写。
- 首次出现时定义术语。表格注明划分、样本数、单位、方向、分母和统计单位。
- 区分观测结果、机制解释与待验证问题；主指标、诊断指标和规划收益分开报告，不将相关指标当作独立证据或因果分解。
- 曲线定性须同时看变化幅度、多个时间点的持续性与波动范围；仅有微小端点差异时描述为数值变化或基本持平，不直接判定改善、退化或过拟合。训练批次不当作独立实验重复，四分位区间不称为置信区间。
- 规划诊断先用模拟器真实代价检查可达性和候选区分度；已知动力学控制及真实观测替换只作验证，不算模型规划成绩。
- 数值以机器可读结果为准并由脚本生成，保留复现所需的数据、检查点与评分配置标识，不混合不同数据或协议。
- 扩展实验时优先引用已有定义；长方法和复算细节放在相应指南，更新时同步校对表头、公式、正文、链接和数据范围。
- 新结果按“要验证什么、如何比较、观察到什么、能解释到哪里”组织；主报告只保留理解结论所需的表格，完整数值提供 CSV / JSON，不追加排错过程或内部执行日志。
- 区分指标的解释边界与已证实的误判；不得把单项校准失败写成整个 benchmark 无效，也不得用已有论文采用某指标替代本任务的适用性验证。
- 表头、正文、导航使用一致的指标名称和定义；任务主指标、预测诊断和决策诊断的用途必须明确。不同测量不能互相替代，不等于已证明统计独立，不用新名称重复计算证据。
- 根因研究先明确要解释的表现和可区分的机制；评分验证达到该对照所需范围后即可推进，不把统一物理重建或通用综合分设为无关前置任务。
- 机制分析分别写清观测、已排除解释和待验证原因；“可区分”不等于“表示充分”。首步、末步必须注明物理时间端点，模型与训练方案的范围须与表格一致。
- 综合结论按证据组织，直接链接结果与反例。局部对照不推广为九任务的共同根因；尚未区分的机制列为开放问题，不把实验待办或寻求通用修复写成公开报告的主线。

- 新增评分或数据分布须标明验证阶段与覆盖范围。根因对照固定场景、动作、目标表示和误差分母，并说明一次干预同时改变了什么。
- 同一指标全篇使用统一名称；百分数、误差比及其换算分明确区分。每张结果表就近说明数据划分和预测时域，目录不混入执行日志或维护规则。
