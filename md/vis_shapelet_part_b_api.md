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

### ShapeletLibraryMetaResponse
- 这是什么:
  - Part B 首屏基础配置响应。
- 一般出现在哪:
  - `GET /part-b/.../meta`。
- 前端通常怎么用:
  - 初始化 `scope/omega` 默认值和直方图默认参数。
  - 展示 trigger 规则说明，减少口径误解。

### ShapeletGalleryItem
- 这是什么:
  - 单个 shapelet 的静态信息。
- 一般出现在哪:
  - `ShapeletGalleryListResponse.items`
  - `ShapeletDetailResponse.shapelet`
- 前端通常怎么用:
  - 列表卡片展示 `shapelet_id`、长度、prototype 预览。
  - 点击后作为 stats 请求的主键。

### ShapeletGalleryListResponse
- 这是什么:
  - shapelet 静态列表分页响应。
- 前端通常怎么用:
  - 渲染主列表与分页控件。
  - 注意该接口与 `omega` 无关，不应在阈值滑动时重复请求。

### ShapeletDetailResponse
- 这是什么:
  - 单个 shapelet 的静态详情。
- 前端通常怎么用:
  - 在右侧详情面板展示 prototype 大图和补充信息。

### ShapeletStatsSummaryResponse
- 这是什么:
  - 单个 shapelet 的动态统计摘要（阈值相关）。
- 前端通常怎么用:
  - 展示 `global_trigger_rate`、`class_trigger_rate`、`lift` 的摘要卡片。
  - 用 `support.min_support` 和 `lift=null` 判定是否显示“不稳定”提示。
- 字段说明:
  - `scope/omega`: 当前统计口径回显，前端可用于状态一致性校验。
  - `class_trigger_rate`: 按类别触发率。
  - `class_coverage`: 按类别覆盖率。
  - `lift`: 不平衡修正后指标，低支持度可为空。

### ShapeletHistogramResponse
- 这是什么:
  - `I` 分布直方图数据。
- 前端通常怎么用:
  - 默认 `hist_mode=per_shapelet` 展示单个 shapelet 分布。
  - 切换 `hist_mode=global` 做全局概览。
- 字段说明:
  - `counts + bin_edges`: 直接用于绘图。
  - `range/bins/density`: 图表配置与口径回显。

### ShapeletClassStatsItem / ShapeletClassStatsResponse
- 这是什么:
  - 单个 shapelet 的按类明细统计。
- 前端通常怎么用:
  - 渲染类别统计表格（`prior/trigger_rate/coverage/lift`）。
  - 与 summary 口径保持一致，仅展示粒度更细。

## 1.3 端点列表

### 0) 获取 Part B 首屏基础信息
- `GET /api/v1/part-b/datasets/{dataset_name}/meta`
- 返回:
  - `scope_default`（默认 `test`）
  - `omega_default`
  - `trigger_rule`
  - 直方图默认参数（`mode/bins/density`）
  - warnings

### 1) 获取 shapelet gallery 分页列表（静态）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets?offset=0&limit=100`
- 返回:
  - `ShapeletGalleryListResponse`
- 说明:
  - 必须分页
  - 与 `omega` 无关

### 2) 获取单个 shapelet 静态详情（可选）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}`
- 返回:
  - `ShapeletDetailResponse`

### 3) 获取单个 shapelet 的动态统计摘要
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary?scope=test&omega=0.1`
- 返回:
  - `global_trigger_rate`
  - `class_trigger_rate`
  - `class_coverage`
  - `lift`（小支持度时可为 `null`）
  - `support`（含 `min_support` 与 `alpha`）

### 4) 获取 `I` 分布直方图（per_shapelet/global）
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/stats/histogram?scope=test&hist_mode=per_shapelet&shapelet_id=s_001&bins=50&density=true`
- 返回:
  - `ShapeletHistogramResponse`
- 说明:
  - 默认 `hist_mode=per_shapelet`
  - `hist_mode=global` 时 `shapelet_id` 可为空

### 5) 获取单个 shapelet 的按类统计明细
- `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes?scope=test&omega=0.1`
- 返回:
  - `ShapeletClassStatsResponse`

## 2. 推荐请求流
- `meta -> gallery list -> (shapelet detail + stats summary + histogram + class stats)`
- `omega` 变化仅刷新 `stats` 相关接口
