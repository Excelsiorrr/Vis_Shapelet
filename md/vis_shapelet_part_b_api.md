# Shapelet Part B API 契约（v1）

- 关联 PRD: `vis_shapelet_spec.md`（2.2 Part B）
- 主 API 文档: `vis_shapelet_api.md`
- 适用范围: Part B（Shapelet Library Panel）
- 范围边界: 本文件仅维护 Part B 专属接口；跨 Part 通用接口暂不纳入，统一留在主文档维护。

## 1. API 契约（v1）

## 1.1 通用约定
- Base URL: `/api/v1`
- 数据格式: `application/json`
- 错误返回:
  - `{"code":"ERR_xxx","message":"...","trace_id":"..."}`
- 版本控制:
  - 响应包含 `spec_version`
- Part B 约束:
  - 不提供 `/overview` 或等价全量打包接口
  - `omega` 调整只触发解释后处理统计重算，不触发训练，不修改模型参数
  - `seg_threshold` 调整只触发 candidate segment preview 重算，不触发训练，不修改模型参数
  - 默认 `scope=test`，并在响应中回显 `scope/omega`
  - Part B 采用双阈值结构:
    - `omega`: 样本级触发统计阈值
    - `seg_threshold`: 候选段生成阈值
  - 历史固定阈值常量（`0.5/0.4`）不再直接暴露为业务口径，对外统一收敛为 `seg_threshold`
  - `omega_default` 单一真源为解释配置（`players.omega`，可按数据集覆盖），不得复用 `margin_threshold`
  - `Part B -> Part C` 联动契约已冻结:
    - 必带字段: `dataset`, `sample_id`, `shapelet_id`, `scope`, `omega`, `source_panel='part_b'`
    - 可选字段: `trigger_score`, `rank`, `rank_metric`
    - 禁止仅凭 `shapelet_id` 跳转 Part C，必须携带具体 `sample_id`
    - `sample_id` 的唯一正式来源是 `top-hits` 接口；`sample_ids_preview` 仅用于预览

## 1.2 核心对象 Schema

```json
{
  "ApiWarning": {
    "code": "string",
    "message": "string"
  },
  "ShapeletLibraryMetaResponse": {
    "spec_version": "string",
    "dataset": "string",
    "scope_default": "test|train|all",
    "omega_default": "float",
    "trigger_rule": "trigger_{n,p}(omega)=1{max_t I_{n,p,t}>=omega}",
    "histogram_default": {
      "mode": "per_shapelet|global",
      "bins": "int",
      "density": "bool"
    },
    "warnings": "ApiWarning[]"
  },
  "ShapeletGalleryItem": {
    "shapelet_id": "string",
    "shapelet_len": "int",
    "ckpt_id": "string",
    "prototype": "float[L][D]",
    "sample_ids_preview": "string[]"
  },
  "ShapeletGalleryListResponse": {
    "spec_version": "string",
    "dataset": "string",
    "total": "int",
    "offset": "int",
    "limit": "int",
    "items": "ShapeletGalleryItem[]",
    "warnings": "ApiWarning[]"
  },
  "ShapeletDetailResponse": {
    "spec_version": "string",
    "dataset": "string",
    "shapelet": "ShapeletGalleryItem",
    "warnings": "ApiWarning[]"
  },
  "ShapeletStatsSummaryResponse": {
    "spec_version": "string",
    "dataset": "string",
    "shapelet_id": "string",
    "scope": "test|train|all",
    "omega": "float",
    "global_trigger_rate": "float",
    "class_trigger_rate": {
      "class_id": "float"
    },
    "class_coverage": {
      "class_id": "float"
    },
    "lift": {
      "class_id": "float|null"
    },
    "support": {
      "triggered_samples": "int",
      "total_samples": "int",
      "min_support": "int",
      "alpha": "float"
    },
    "warnings": "ApiWarning[]"
  },
  "ShapeletHistogramResponse": {
    "spec_version": "string",
    "dataset": "string",
    "scope": "test|train|all",
    "hist_mode": "per_shapelet|global",
    "shapelet_id": "string|null",
    "bins": "int",
    "density": "bool",
    "range": "float[2]",
    "counts": "float[bins]",
    "bin_edges": "float[bins+1]",
    "warnings": "ApiWarning[]"
  },
  "ShapeletSegmentPreviewResponse": {
    "spec_version": "string",
    "dataset": "string",
    "shapelet_id": "string",
    "scope": "test|train|all",
    "seg_threshold": "float",
    "signal_type": "string",
    "time_axis": "int[]",
    "curve": "float[]",
    "segments": [
      {
        "start": "int",
        "end": "int",
        "length": "int",
        "peak_value": "float"
      }
    ],
    "segment_count": "int",
    "covered_ratio": "float",
    "longest_segment": "int",
    "warnings": "ApiWarning[]"
  },
  "ShapeletEvidenceMatchItem": {
    "sample_id": "string",
    "rank": "int",
    "label": "int|null",
    "pred_class": "int|null",
    "margin": "float|null",
    "peak_t": "int",
    "peak_activation": "float",
    "t_start": "int",
    "t_end": "int",
    "raw_window": "float[L][D]",
    "activation_window": "float[L]"
  },
  "ShapeletEvidenceTopMatchesResponse": {
    "spec_version": "string",
    "dataset": "string",
    "shapelet_id": "string",
    "scope": "test|train|all",
    "shapelet_length": "int",
    "rank_metric": "peak_activation",
    "limit": "int",
    "items": "ShapeletEvidenceMatchItem[]",
    "warnings": "ApiWarning[]"
  },
  "ShapeletClassStatsItem": {
    "class_id": "int",
    "prior": "float",
    "trigger_rate": "float",
    "coverage": "float",
    "lift": "float|null"
  },
  "ShapeletClassStatsResponse": {
    "spec_version": "string",
    "dataset": "string",
    "shapelet_id": "string",
    "scope": "test|train|all",
    "omega": "float",
    "items": "ShapeletClassStatsItem[]",
    "warnings": "ApiWarning[]"
  },
  "TopHitSampleItem": {
    "sample_id": "string",
    "trigger_score": "float",
    "rank": "int",
    "label": "int|null",
    "pred_class": "int|null",
    "margin": "float|null"
  },
  "ShapeletTopHitsResponse": {
    "spec_version": "string",
    "dataset": "string",
    "shapelet_id": "string",
    "scope": "test|train|all",
    "omega": "float",
    "total": "int",
    "offset": "int",
    "limit": "int",
    "rank_metric": "max_i|trigger_score",
    "items": "TopHitSampleItem[]",
    "warnings": "ApiWarning[]"
  },
  "PartBToCLink": {
    "dataset": "string",
    "sample_id": "string",
    "shapelet_id": "string",
    "shapelet_len": "int|null",
    "scope": "test|train|all",
    "omega": "float",
    "source_panel": "part_b",
    "trigger_score": "float|null",
    "rank": "int|null",
    "rank_metric": "max_i|trigger_score|null"
  }
}
```

## 1.2.1 Part B Schema 前端理解

### ApiWarning
- 这是什么:
  - 后端主动返回的告警信息。
- 一般出现在哪:
  - Part B 各响应的 `warnings` 字段。
- 前端通常怎么用:
  - 渲染提示条或可展开告警列表。
  - 也可根据 `code` 做条件提示（例如低支持度、scope 提示）。
- 字段说明:
  - `code`: 稳定的告警代码，适合前端做分支逻辑或埋点统计。
  - `message`: 面向用户的解释文本，可直接展示。

### ShapeletLibraryMetaResponse
- 这是什么:
  - Part B 首屏基础配置响应。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/meta` 的响应体。
- 前端通常怎么用:
  - 初始化 `scope/omega` 默认值和直方图默认参数。
  - 展示 trigger 规则说明，减少口径误解。
  - 保存 `dataset` 和 `spec_version` 作为页面状态校验信息。
- 字段说明:
  - `spec_version`: 接口契约版本号，前端可用于兼容性检查。
  - `dataset`: 当前数据集名称，用于页面标题、路由状态或请求参数回显。
  - `scope_default`: 默认统计范围，当前后端默认返回 `test`。
  - `omega_default`: 默认触发阈值，来自解释配置 `players.omega`，用于阈值滑条初始值。
  - `trigger_rule`: 触发规则公式字符串，用于帮助文案或 tooltip。
  - `histogram_default`: 直方图默认配置对象。
  - `warnings`: 与当前数据集相关的告警列表。

### histogram_default（ShapeletLibraryMetaResponse 内部对象）
- 这是什么:
  - Part B 直方图组件的默认配置。
- 一般出现在哪:
  - `ShapeletLibraryMetaResponse.histogram_default`
- 前端通常怎么用:
  - 初始化直方图组件的模式、分箱数量和计数口径（密度/频数）。
- 字段说明:
  - `mode`: 默认直方图模式，`per_shapelet` 或 `global`。
  - `bins`: 默认分箱数量。
  - `density`: 是否按密度归一化。

### ShapeletGalleryItem
- 这是什么:
  - 单个 shapelet 的静态信息。
- 一般出现在哪:
  - `ShapeletGalleryListResponse.items`
  - `ShapeletDetailResponse.shapelet`
- 前端通常怎么用:
  - 列表卡片展示 `shapelet_id`、长度、prototype 预览。
  - 点击后作为 stats 请求的主键。
  - 可缓存该对象，避免在阈值变化时重复请求静态信息。
- 字段说明:
  - `shapelet_id`: shapelet 标识符（例如 `s0003`），用于后续详情与统计请求。
  - `shapelet_len`: shapelet 时间长度，可用于 UI 尺寸提示或窗口默认值提示。
  - `ckpt_id`: 模型 checkpoint 标识，用于展示模型来源或调试信息。
  - `prototype`: shapelet 原型序列，形状 `L x D`，可用于绘制波形预览。
  - `sample_ids_preview`: 样本 id 预览列表，当前后端可能返回空数组，前端应容错处理。
  - 说明:
    - `sample_ids_preview` 仅用于轻量预览，不作为 B->C 跳转的正式数据源。
    - 正式跳转必须使用 `top-hits` 接口返回的 `sample_id + trigger_score + rank`。

### ShapeletGalleryListResponse
- 这是什么:
  - shapelet 静态列表分页响应。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets?offset=...&limit=...` 的响应体。
- 前端通常怎么用:
  - 渲染主列表与分页控件。
  - 注意该接口与 `omega` 无关，不应在阈值滑动时重复请求。
  - 可按 `offset/limit` 驱动分页器状态。
- 字段说明:
  - `spec_version`: 接口契约版本号。
  - `dataset`: 当前数据集名称。
  - `total`: shapelet 总数，用于计算页数。
  - `offset`: 当前页起始偏移。
  - `limit`: 当前页大小。
  - `items`: 当前页 shapelet 列表。
  - `warnings`: 告警信息列表。

### ShapeletDetailResponse
- 这是什么:
  - 单个 shapelet 的静态详情。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}` 的响应体。
- 前端通常怎么用:
  - 在右侧详情面板展示 prototype 大图和补充信息。
  - 列表场景可不请求该接口，按需加载即可。
- 字段说明:
  - `spec_version`: 接口契约版本号。
  - `dataset`: 当前数据集名称。
  - `shapelet`: shapelet 详情对象，结构与 `ShapeletGalleryItem` 相同。
  - `warnings`: 告警信息列表。

### ShapeletStatsSummaryResponse
- 这是什么:
  - 单个 shapelet 的动态统计摘要（阈值相关）。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary?...` 的响应体。
- 前端通常怎么用:
  - 展示 `global_trigger_rate`、`class_trigger_rate`、`lift` 的摘要卡片。
  - 用 `support.min_support` 和 `lift=null` 判定是否显示“不稳定”提示。
  - `scope` 或 `omega` 变化后应重新请求该接口。
- 字段说明:
  - `spec_version`: 接口契约版本号。
  - `dataset`: 当前数据集名称。
  - `shapelet_id`: 当前统计目标 shapelet id。
  - `scope`: 当前统计范围（`test|train|all`）。
  - `omega`: 当前触发阈值回显。
  - `global_trigger_rate`: 全样本触发比例（`triggered_samples / total_samples`）。
  - `class_trigger_rate`: 按类别触发率映射，key 为类别 id 字符串，value 为触发率。
  - `class_coverage`: 按类别覆盖率映射，当前后端实现与 `class_trigger_rate` 口径一致。
  - `lift`: 按类别提升度映射；当支持度不足时 value 可能为 `null`。
  - `support`: 支持度摘要对象。
  - `warnings`: 告警信息列表。

### support（ShapeletStatsSummaryResponse 内部对象）
- 这是什么:
  - 用于说明 lift 统计稳定性的支持度信息。
- 一般出现在哪:
  - `ShapeletStatsSummaryResponse.support`
- 前端通常怎么用:
  - 判断是否展示“统计不稳定”提示。
  - 在 tooltip 中解释 lift 为 `null` 的原因。
- 字段说明:
  - `triggered_samples`: 触发样本数。
  - `total_samples`: 当前 `scope` 下总样本数。
  - `min_support`: 最小支持度阈值，低于该值时 lift 不稳定。
  - `alpha`: 平滑参数（拉普拉斯平滑）回显。

### ShapeletHistogramResponse
- 这是什么:
  - `I` 分布直方图数据。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/stats/histogram?...` 的响应体。
- 前端通常怎么用:
  - 默认 `hist_mode=per_shapelet` 展示单个 shapelet 分布。
  - 切换 `hist_mode=global` 做全局概览。
  - 直接使用 `counts` 和 `bin_edges` 绘图，无需前端二次分箱。
- 字段说明:
  - `spec_version`: 接口契约版本号。
  - `dataset`: 当前数据集名称。
  - `scope`: 当前统计范围。
  - `hist_mode`: 直方图模式（`per_shapelet|global`）。
  - `shapelet_id`: 当 `hist_mode=per_shapelet` 时为目标 id；`global` 时为 `null`。
  - `bins`: 分箱数量。
  - `density`: 是否密度归一化。
  - `range`: 本次直方图使用的值域 `[min, max]`。
  - `counts`: 每个 bin 的计数或密度值，长度为 `bins`。
  - `bin_edges`: bin 边界数组，长度为 `bins + 1`。
  - `warnings`: 告警信息列表。

### ShapeletSegmentPreviewResponse
- 这是什么:
  - 单个 shapelet 的候选段预览响应（`seg_threshold` 相关）。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/segments/preview?...` 的响应体。
- 前端通常怎么用:
  - 作为 Part B / Part E 的候选段辅助数据源
  - 可驱动后续证据面板中的“候选段来源说明”或 Part E 入口前的阈值解释
  - 当前不再要求单独实现一个低信息增益的 `BHistogramDynamicPanel`
- 字段说明:
  - `seg_threshold`: 当前切段阈值回显。
  - `signal_type`: 当前切段信号类型说明，例如 `actions_sum`、`prototype0_smoothed`。
  - `time_axis`: 横轴时间索引。
  - `curve`: 用于切段的时间曲线。
  - `segments`: 当前阈值下切出的候选段列表。
  - `segment_count`: 候选段数量。
  - `covered_ratio`: 被候选段覆盖的时间点占比。
  - `longest_segment`: 最长候选段长度。
  - `warnings`: 告警信息列表。

### ShapeletEvidenceTopMatchesResponse
- 这是什么:
  - 单个 shapelet 的真实命中证据响应。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/evidence/top-matches?...` 的响应体。
- 前端通常怎么用:
  - 作为 `BShapeletEvidencePanel` 的主数据源。
  - 展示当前 shapelet 在真实样本里命中最强的若干局部片段。
  - 选中某个条目后，可进一步生成 Part B -> Part C 的精确定位上下文。
- 字段说明:
  - `shapelet_length`: 当前 shapelet 长度，用于解释命中窗口如何裁切。
  - `rank_metric`: 第一版固定为 `peak_activation`。
  - `limit`: 本次返回条目数上限回显。
  - `items`: 命中证据条目数组。
  - `warnings`: 告警信息列表。

### ShapeletEvidenceMatchItem
- 这是什么:
  - 单个命中证据条目。
- 一般出现在哪:
  - `ShapeletEvidenceTopMatchesResponse.items[]`
- 前端通常怎么用:
  - 渲染 top matched subsequences 列表。
  - 点击后展示局部 raw / activation 对照。
- 字段说明:
  - `sample_id`: 命中证据所属样本。
  - `rank`: 当前排序下名次（1-based）。
  - `label / pred_class / margin`: 当前样本的真实标签、预测类别、分类 margin。
  - `peak_t`: 当前 shapelet 在该样本上的最强命中时间位置。
  - `peak_activation`: 当前 shapelet 在该样本上的最强命中强度。
  - `t_start / t_end`: 以 `peak_t` 为中心裁出的局部窗口边界。
  - `raw_window`: 原始序列局部窗口。
  - `activation_window`: 当前 shapelet 在同一窗口上的 activation 局部曲线。

### ShapeletSegmentPreviewResponse 的语义冻结
- `curve` 必须是后端真正用于切段的那条曲线，不允许为了好看单独换成另一条可视化曲线。
- `segments` 必须与后端切段结果一一对应，不能在前端二次重算。
- `peak_value` 定义为该 segment 在 `curve[start:end]` 内的最大值。
- `covered_ratio` 定义为被全部 segments 覆盖的时间点数 / `curve.length`。

### `signal_type` 当前推荐枚举
- `actions_sum_filled`
  - 对应 `shapeX.py` 的 ECG 分支
  - 语义：`sum(actions, dim=-1)` 后经过 `fill_short_negative_sequences`
- `prototype0_smoothed`
  - 对应 `shapeX.py` 的 SNC 分支
  - 语义：`actions[:,:,0]` 后经过 `moving_average_centered(..., 100)`
- v1 第一版不开放前端切换 `signal_type`，只在响应中回显

### ShapeletClassStatsItem
- 这是什么:
  - 单个类别下的 shapelet 统计明细行。
- 一般出现在哪:
  - `ShapeletClassStatsResponse.items[]`
- 前端通常怎么用:
  - 渲染类别表格的一行。
  - 支持按 `lift` 或 `trigger_rate` 排序。
- 字段说明:
  - `class_id`: 类别 id。
  - `prior`: 类别先验占比（该类样本数 / 总样本数）。
  - `trigger_rate`: 该类别内触发率。
  - `coverage`: 该类别覆盖率，当前后端实现与 `trigger_rate` 口径一致。
  - `lift`: 该类别 lift，低支持度时可能为 `null`。

### ShapeletClassStatsResponse
- 这是什么:
  - 单个 shapelet 的按类明细统计响应。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes?...` 的响应体。
- 前端通常怎么用:
  - 渲染类别统计表格（`prior/trigger_rate/coverage/lift`）。
  - 与 summary 保持同一 `scope/omega`，作为明细视图。
- 字段说明:
  - `spec_version`: 接口契约版本号。
  - `dataset`: 当前数据集名称。
  - `shapelet_id`: 当前统计目标 shapelet id。
  - `scope`: 当前统计范围。
  - `omega`: 当前触发阈值回显。
  - `items`: 按类别统计明细数组，元素类型为 `ShapeletClassStatsItem`。
  - `warnings`: 告警信息列表。

### TopHitSampleItem
- 这是什么:
  - 单个 shapelet 在给定 `scope + omega` 下的高触发样本条目。
- 一般出现在哪:
  - `ShapeletTopHitsResponse.items[]`
- 前端通常怎么用:
  - 作为 Part B -> Part C 跳转前的样本选择列表。
  - 选中条目后按冻结契约组装跳转参数。
- 字段说明:
  - `sample_id`: 目标样本 id（B->C 必带）。
  - `trigger_score`: 当前 shapelet 在该样本上的触发分数。
  - `rank`: 当前排序下名次（1-based）。
  - `label/pred_class/margin`: 可选辅助展示字段。

### ShapeletTopHitsResponse
- 这是什么:
  - 单个 shapelet 的高触发样本分页响应（联动专用）。
- 一般出现在哪:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/samples/top-hits?...` 的响应体。
- 前端通常怎么用:
  - 在 Part B 中先选样本，再触发跳转到 Part C。
  - 与 `summary/classes` 使用同一 `scope + omega`，保证口径一致。

### PartBToCLink
- 这是什么:
  - 从 Part B 跳转 Part C 的冻结参数对象。
- 前端通常怎么用:
  - 作为路由 query 或导航 state 统一结构。
- 字段说明:
  - `dataset/sample_id/shapelet_id/scope/omega/source_panel` 为必填核心字段。
  - `trigger_score/rank/rank_metric` 为可选追溯字段。
  - `shapelet_len` 为可选加速字段；Part C 计算高亮窗口时长度来源优先级为:
    - 优先 `match` 响应中的 `shapelet_lens`
    - 次选 `PartBToCLink.shapelet_len`
    - 兜底读取 `shapelet detail/meta`

## 1.3 端点列表

### 0) 获取 Part B 首屏基础信息
- `GET /api/v1/part-b/datasets/{dataset_name}/meta`
- 返回:
  - `scope_default`（默认 `test`）
  - `omega_default`
  - `trigger_rule`
  - 直方图默认参数（`mode/bins/density`）
  - warnings
- 前端理解:
  - 这是 Part B 页面初始化接口，建议作为首个请求。
  - 返回的默认值应写入页面状态（如 `scope`、`omega`、直方图配置）。
  - `trigger_rule` 适合放在帮助说明或 tooltip，避免统计口径误读。
- 参数说明:
  - `dataset_name`: 数据集名称，需在后端支持列表内。

### 1) 获取 shapelet gallery 分页列表（静态）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets?offset=0&limit=100`
- 返回:
  - `ShapeletGalleryListResponse`
- 说明:
  - 必须分页
  - 与 `omega` 无关
- 前端理解:
  - 这是 shapelet 列表主数据源，用于渲染左侧/中间 gallery。
  - 页面切换分页时请求；滑动 `omega` 时不应重复请求该接口。
  - `items[].shapelet_id` 通常作为后续统计接口主键。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `offset`: 分页偏移，`>= 0`。
  - `limit`: 分页大小，后端约束 `1 ~ 500`。

### 2) 获取单个 shapelet 静态详情（可选）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}`
- 返回:
  - `ShapeletDetailResponse`
- 前端理解:
  - 用于按需加载右侧详情面板，不是首屏必请求接口。
  - 若列表接口已携带足够字段，可先不请求该接口。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `shapelet_id`: shapelet 标识符（例如 `s0003`）。

### 3) 获取单个 shapelet 的动态统计摘要
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary?scope=test&omega=0.1`
- 返回:
  - `global_trigger_rate`
  - `class_trigger_rate`
  - `class_coverage`
  - `lift`（小支持度时可为 `null`）
  - `support`（含 `min_support` 与 `alpha`）
- 前端理解:
  - 这是摘要卡片核心数据源，通常与直方图、按类统计一起刷新。
  - 用户切换 `shapelet`、`scope`、`omega` 时都应重新请求。
  - 当 `lift` 为 `null` 或 `warnings` 含低支持度告警时，前端应展示“不稳定”提示。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `shapelet_id`: 目标 shapelet 标识符。
  - `scope`: 统计范围，`test|train|all`。
  - `omega`: 触发阈值。

### 4) 获取 `I` 分布直方图（per_shapelet/global）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/stats/histogram?scope=test&hist_mode=per_shapelet&shapelet_id=s_001&bins=50&density=true`
- 返回:
  - `ShapeletHistogramResponse`
- 说明:
  - 默认 `hist_mode=per_shapelet`
  - `hist_mode=global` 时 `shapelet_id` 可为空
- 前端理解:
  - 这是直方图组件数据源，`counts + bin_edges` 可直接绘图。
  - 选择单个 shapelet 时用 `per_shapelet`，做总览时切到 `global`。
  - 与 summary 类似，`scope` 变化后需要同步刷新。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `scope`: 统计范围，`test|train|all`。
  - `hist_mode`: 直方图模式，`per_shapelet|global`。
  - `shapelet_id`: `per_shapelet` 模式必填；`global` 可不传。
  - `bins`: 分箱数量，后端约束 `5 ~ 500`。
  - `density`: 是否返回密度值（`true`）或频数（`false`）。
  - `range_min/range_max`: 可选自定义值域；仅当 `range_min < range_max` 时生效。

### 4.1) 获取单个 shapelet 的候选段预览（R3 规划契约）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/segments/preview?scope=test&seg_threshold=0.4`
- 返回:
  - `ShapeletSegmentPreviewResponse`
- 说明:
  - 这是 Part B / Part E 共用的候选段预览契约，当前用于冻结字段语义
  - 该接口保留，但不再要求用它单独构建一个 Part B 主组件
- 前端理解:
  - 这块主要服务于 Part E 的候选段来源解释，而不是 B->C 的样本选择
  - 也可作为 Part B 证据面板中的辅助信息，而不是主视图
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `shapelet_id`: 目标 shapelet 标识符。
  - `scope`: 统计范围，`test|train|all`。
  - `seg_threshold`: 候选段阈值。

### 4.2) 获取单个 shapelet 的真实命中证据（R3 主接口建议）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/evidence/top-matches?scope=test&limit=12`
- 返回:
  - `ShapeletEvidenceTopMatchesResponse`
- 说明:
  - 这是 `BShapeletEvidencePanel` 的推荐主接口。
  - 它服务“这个 shapelet 在真实样本里最强命中了什么局部片段”。
  - 它不等价于 `shapeX.py` 的候选段生成接口。
- 前端理解:
  - 这是 Part B 高价值解释层的主数据源。
  - 返回的是按 `peak_activation` 排序的 top matched subsequences。
  - 选中条目后可作为 Part C 精确定位和 Part E 候选 span 的上游上下文。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `shapelet_id`: 目标 shapelet 标识符。
  - `scope`: 统计范围，`test|train|all`。
  - `limit`: 返回条目数，第一版建议 `6 ~ 20`。

### 4.2.1 `evidence/top-matches` 的后端实现口径（冻结建议）

- 目标:
  - 直接贴近模型真实命中机制，而不是复用候选段 proposal 口径。

- 第一版命中片段选择规则:
  1. 固定一个 shapelet `p`
  2. 对每个样本 `n` 计算：
     - `peak_t = argmax_t I[n,t,p]`
     - `peak_activation = max_t I[n,t,p]`
  3. 以 `peak_t` 为中心，按 `shapelet_length = L` 裁切局部窗口：
     - `t_start = peak_t - floor(L/2)`
     - `t_end = t_start + L - 1`
     - 越界时做 clamp
  4. 返回：
     - `raw_window = x[n, t_start:t_end]`
     - `activation_window = I[n, t_start:t_end, p]`
  5. 按 `peak_activation` 降序排序，取 Top-K

- 第一版明确不做:
  - 不直接沿用 `shapeX.py` 的 segment proposal 结果
  - 不让前端本地根据热图或聚合曲线自己重算命中片段

### 4.1.1 `segments/preview` 的后端实现口径（冻结建议）

- 目标：
  - 尽量与 `shapeX.py` 当前切段逻辑保持一致，避免 Part B 候选段预览和 Part E / saliency 链路漂移。

- 推荐数据集分支：
  - `mitecg` 或名称中包含 `ecg`
    - 对齐 [shapeX.py:131](d:/shapelet/ShapeX-new/shapeX.py:131) 到 [shapeX.py:186](d:/shapelet/ShapeX-new/shapeX.py:186)
    - `curve = sum(actions, dim=-1)`
    - 再执行 `fill_short_negative_sequences`
    - 默认 `seg_threshold = 0.5`
    - `signal_type = actions_sum_filled`
  - `mcce / mcch / mtce / mtch`
    - 对齐 [shapeX.py:190](d:/shapelet/ShapeX-new/shapeX.py:190) 到 [shapeX.py:231](d:/shapelet/ShapeX-new/shapeX.py:231)
    - `curve = actions[:,:,0]`
    - 再执行 `moving_average_centered(..., 100)`
    - 默认 `seg_threshold = 0.4`
    - `signal_type = prototype0_smoothed`

- 推荐响应补充说明：
  - 若当前数据集命中了哪条分支，应通过 `signal_type` 明确回显
  - 若后端对默认 `seg_threshold` 做了数据集级覆盖，应通过响应原样回显
  - 若切段结果为空，不应报错；应返回：
    - `segments = []`
    - `segment_count = 0`
    - `covered_ratio = 0`
    - `longest_segment = 0`

- 第一版明确不做：
  - 不让前端自己复刻 `fill_short_negative_sequences`
  - 不让前端自己复刻 `moving_average_centered`
  - 不让前端在本地重新根据 `curve` 计算 segments

### 4.1.2 `segments/preview` 推荐错误与 warning

- 错误：
  - `ERR_SHAPELET_NOT_FOUND`
  - `ERR_SCOPE_INVALID`
  - `ERR_SEG_THRESHOLD_INVALID`
- warning：
  - `WARN_EMPTY_SEGMENTS`
    - 含义：当前 `seg_threshold` 下没有任何候选段
  - `WARN_SIGNAL_FALLBACK`
    - 含义：当前数据集未命中特化分支，后端退回通用切段信号
  - `WARN_SEG_THRESHOLD_CLIPPED`
    - 含义：请求阈值超出后端允许范围，已裁剪

### 5) 获取单个 shapelet 的按类统计明细
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes?scope=test&omega=0.1`
- 返回:
  - `ShapeletClassStatsResponse`
- 前端理解:
  - 这是类别表格数据源，和 summary 同口径但粒度更细。
  - 与 summary 一样，`shapelet/scope/omega` 变化时应同步刷新。
  - 适合支持按 `lift`、`trigger_rate` 排序或筛选。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `shapelet_id`: 目标 shapelet 标识符。
  - `scope`: 统计范围，`test|train|all`。
  - `omega`: 触发阈值。

### 6) 获取单个 shapelet 的高触发样本列表（用于 B->C 联动）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/samples/top-hits?scope=test&omega=0.1&offset=0&limit=20&rank_metric=max_i`
- 返回:
  - `ShapeletTopHitsResponse`
- 说明:
  - 该接口是 `Part B -> Part C` 的正式样本来源接口。
  - 当前若后端尚未实现，可按本契约先 mock，字段语义不再变更（联动契约冻结）。
- 前端理解:
  - 先请求该接口拿到候选样本，用户选中后再按 `PartBToCLink` 跳转 Part C。
  - `scope + omega` 必须与当前 B 页面状态一致，避免口径漂移。
- 参数说明:
  - `dataset_name`: 数据集名称。
  - `shapelet_id`: 目标 shapelet 标识符。
  - `scope`: 统计范围，`test|train|all`。
  - `omega`: 触发阈值。
  - `offset/limit`: 分页参数。
  - `rank_metric`: 排序指标，默认 `max_i`。

## 2. 推荐请求流
- `meta -> gallery list -> (shapelet detail + histogram panel + segment preview + stats summary + class stats + top-hits)`
- `omega` 或 `scope` 变化时，仅刷新 `stats` 相关接口（summary / histogram / class stats）与 `top-hits`
- `seg_threshold` 变化时，仅刷新 `segment preview`
- `offset/limit` 变化仅刷新 gallery list

## 3. 当前现状与下一轮规划的边界

- 当前已实现 / 已冻结契约:
  - `meta`
  - `gallery list`
  - `shapelet detail`
  - `stats/summary`
  - `stats/histogram`
  - `evidence/top-matches`
  - `stats/classes`
  - `samples/top-hits`

- 当前已进入接口层、作为辅助链路保留:
  - `segments/preview`

- 解释:
  - Part B 现在已经明确拆成两条线：
    1. `omega` 线：统计解释
    2. `seg_threshold` 线：候选段生成
  - 第 2 条线保留为后端候选段口径与 Part E 上下文，不再强制落成独立的 Part B 主组件
