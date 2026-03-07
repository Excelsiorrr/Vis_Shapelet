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
  - 默认 `scope=test`，并在响应中回显 `scope/omega`

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
  - `omega_default`: 默认触发阈值，通常作为阈值滑条初始值。
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

## 2. 推荐请求流
- `meta -> gallery list -> (shapelet detail + stats summary + histogram + class stats)`
- `omega` 变化仅刷新 `stats` 相关接口（summary / histogram / class stats）
- `offset/limit` 变化仅刷新 gallery list
