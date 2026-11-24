# Edit-Grounded Adversarial Consensus 设计

## 目标

在同一批 30 张测试图片上仅运行优化后的 `Adversarial Consensus`，抛弃
`Specialist Cascade` 和 `Evidence Seeking Agent`。优化重点不是让 bbox 尽可能小，
而是让每个 bbox 成为可直接交给图片编辑模型局部重绘的独立编辑区域。

使用部署方提供的 OpenAI-compatible 模型服务，具体地址通过运行参数传入。

模型名称保持：

`Qwen3.6-27B-agen-0722-agentv4`

## 编辑区域原则

- 每个 finding 必须对应一个可明确描述的编辑目标，例如人物的一只手、车辆前轮、
  海报标题区域或发生材质断裂的局部表面。
- bbox 应覆盖完整问题实体、缺陷像素以及局部重绘需要的少量上下文，允许适度冗余，
  不追求紧贴缺陷边缘。
- bbox 应排除与修复无关的其他实体和大面积场景；不能为了方便定位而直接框整图。
- 不设置 bbox 最大面积、最大宽度或最大高度。主体即使占据画面大部分，只要它是独立
  编辑实体且区域可用于重绘，就允许使用大框。
- 程序仅识别贴近四条图像边界的近整图框。此类候选不能直接进入最终结果，必须由
  Skeptic 给出符合编辑原则的修正区域，否则只能进入不确定结果。
- 多个空间上分离的问题必须拆成多个 finding，不得使用一个大框把它们合并。

## 方案

采用“实体锚定的 Proposer + 编辑区域复核的 Skeptic + 受约束 Judge”。

不采用仅修改提示词的方案，因为它无法稳定阻止整图框；不引入额外检测或分割模型，
因为这会改变本轮所比较的 critic 模型能力，并增加新的模型依赖。

## 管线

### Proposer

保留三个角色：

- anatomy / structure / topology；
- relations / physics；
- rendering / text。

每条候选除原有字段外，还必须给出：

- `edit_target`：中文描述的独立编辑目标；
- `edit_region_rationale`：为何该区域适合局部重绘，以及包含了哪些必要上下文；
- `bbox_norm`：归一化编辑区域。

提示词明确要求：先确定独立编辑目标，再给出 bbox；不允许用整图框代替定位；空间分离
的问题必须拆开。

### Skeptic

Skeptic 对每条候选同时检查：

1. 视觉质量问题是否有直接证据；
2. bbox 是否是适合重绘的独立编辑区域；
3. bbox 是否覆盖完整问题实体和必要上下文；
4. bbox 是否混入过多无关实体或背景。

返回字段增加：

- `grounding_status`：`valid`、`needs_revision` 或 `invalid`；
- `revised_bbox_norm`：需要修正时返回新的编辑区域；
- `grounding_rationale`：中文说明区域是否适合重绘。

修正不是单纯“收紧”：可以缩小无关区域，也可以为完整实体、连接处或纹理过渡增加少量
上下文。修正后的区域若仍是近整图框，则不能标记为有效。

### Judge

Judge 仍只能选择已有 candidate ID，不得创建新问题。只有视觉问题得到支持、且
`grounding_status=valid` 的候选才能进入 accepted findings。视觉问题成立但编辑区域
仍无法可靠确定的候选只能进入 uncertain findings。

最终结果保留：

- artifact taxonomy；
- 中文 observation、abnormality、counterargument 和 rationale；
- `edit_target`、`edit_region_rationale`、grounding 复核信息；
- 最终用于重绘的 bbox。

## 中文与命名

- 管线名 `Adversarial Consensus` 和技术 stage ID 保留英文。
- 模型所有自然语言输出和页面中的过程解释使用简体中文。
- JSON 字段名、枚举值、candidate ID 和 bbox 数值保持英文或 schema 原值。
- 英文原始请求与响应仅在 HTML 折叠区保留，便于审计。

## 运行与输出

- 复用现有匿名输入发现与目录标签隔离逻辑，人工目录标签绝不发送给模型。
- 先运行 1 张图片，检查 structured output、编辑区域和中文轨迹。
- 验证通过后，断点续跑同一批全部 30 张图片。
- 新结果写入独立目录，不覆盖现有三管线结果。
- `results.jsonl`、`traces.jsonl`、`errors.jsonl` 和 `run_config.json` 均保留。
- 运行期间持续输出已完成数量、失败数量、耗时和预计剩余时间。

## HTML 可视化

- 页面仅展示 `Adversarial Consensus`，使用绿色作为管线颜色。
- 每张图片只显示一套原图和该管线最终 bbox，bbox 由前端绘制。
- finding 同时显示问题分类、中文描述、编辑目标、编辑区域理由、grounding 复核结论、
  最终 bbox 和置信度。
- 默认显示中文中间轨迹，英文原始输入输出折叠保留。
- 保留人工参考分类筛选、最终结论筛选、问题分类筛选和失败结果查看。
- 新报告验证后，将 11223 服务切换到新输出目录；旧报告和旧结果不删除。

## 错误与恢复

- Proposer 返回缺少编辑目标、无效 bbox 或近整图 bbox 时，候选必须进入 grounding
  修正流程，不能直接 accepted。
- Skeptic 未能返回合法修正区域时，该候选降为 uncertain，不静默伪造或裁剪 bbox。
- 单张图片失败写入 `errors.jsonl`，其他图片继续运行；重跑只补失败或未完成单元。
- API 重试、result-tool 优先级、敏感路径与目录标签隔离沿用现有实现。

## 验证标准

- 自动化测试证明仅运行 `Adversarial Consensus`。
- 自动化测试证明近整图框不能直接进入 accepted findings。
- 自动化测试证明大面积但语义独立的实体区域不会仅因面积大而被拒绝。
- 自动化测试证明 Skeptic 可以把候选 bbox 改成适合重绘、带少量上下文的区域。
- 自动化测试证明模型自然语言字段被强制为简体中文。
- 1 张试跑结果人工可检查后，才继续 30 张完整运行。
- 完整运行结束后，结果数量、失败数量、bbox 合法性、中文覆盖和 11223 页面响应均通过验证。
