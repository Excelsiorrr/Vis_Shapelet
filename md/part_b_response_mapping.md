# Part B 接口与响应类型对照表

- 代码来源: [backend/api/part_b.py](/d:/shapelet/ShapeX-new/backend/api/part_b.py)
- 目的:
  - 把 Part B 当前实际接口与响应模型的一一对应关系单独列清楚
  - 方便前端、后端、文档对照，不必在 `vis_shapelet_part_b_api.md` 中来回跳转

## 1. 总表

| 接口 | 顶层响应类型 | 列表/嵌套类型 | 说明 |
| --- | --- | --- | --- |
| `GET /api/v1/part-b/datasets/{dataset_name}/meta` | `ShapeletLibraryMetaResponse` | `HistogramDefault` `ApiWarning[]` | Part B 首屏基础信息 |
| `GET /api/v1/part-b/datasets/{dataset_name}/shapelets?offset=...&limit=...` | `ShapeletGalleryListResponse` | `ShapeletGalleryItem[]` `ApiWarning[]` | shapelet gallery 分页列表（静态） |
| `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}` | `ShapeletDetailResponse` | `ShapeletGalleryItem` `ApiWarning[]` | 单个 shapelet 静态详情 |
| `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary?scope=...&omega=...` | `ShapeletStatsSummaryResponse` | `SupportSummary` `dict[str,float]` `dict[str,float|null]` `ApiWarning[]` | 单个 shapelet 动态统计摘要 |
| `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/stats/histogram?scope=...&hist_mode=...&shapelet_id=...&bins=...&density=...` | `ShapeletHistogramResponse` | `counts: float[]` `bin_edges: float[]` `ApiWarning[]` | `I` 分布直方图（per_shapelet/global） |
| `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes?scope=...&omega=...` | `ShapeletClassStatsResponse` | `ShapeletClassStatsItem[]` `ApiWarning[]` | 单个 shapelet 按类统计明细 |
| `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/samples/top-hits?scope=...&omega=...&offset=...&limit=...&rank_metric=...` | `ShapeletTopHitsResponse` | `TopHitSampleItem[]` `ApiWarning[]` | B->C 联动样本来源（契约冻结，待实现） |

## 2. 类型说明

### 2.1 首屏基础信息

#### `ShapeletLibraryMetaResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/meta`
- 结构:
  - `spec_version`
  - `dataset`
  - `scope_default`
  - `omega_default`
  - `trigger_rule`
  - `histogram_default: HistogramDefault`
  - `warnings: ApiWarning[]`

#### `HistogramDefault`
- 典型字段:
  - `mode`
  - `bins`
  - `density`

#### `ApiWarning`
- 典型字段:
  - `code`
  - `message`

### 2.2 shapelet 静态列表与详情

#### `ShapeletGalleryListResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets?offset=...&limit=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `total`
  - `offset`
  - `limit`
  - `items: ShapeletGalleryItem[]`
  - `warnings: ApiWarning[]`

#### `ShapeletDetailResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}`
- 结构:
  - `spec_version`
  - `dataset`
  - `shapelet: ShapeletGalleryItem`
  - `warnings: ApiWarning[]`

#### `ShapeletGalleryItem`
- 典型字段:
  - `shapelet_id`
  - `shapelet_len`
  - `ckpt_id`
  - `prototype`
  - `sample_ids_preview`

- 说明:
  - 这是静态对象，不受 `omega` 变化影响
  - `sample_ids_preview` 当前后端通常返回空数组，前端需容错
  - `sample_ids_preview` 仅用于预览，不作为 B->C 正式跳转数据源

### 2.3 动态统计摘要

#### `ShapeletStatsSummaryResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary?scope=...&omega=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `shapelet_id`
  - `scope`
  - `omega`
  - `global_trigger_rate`
  - `class_trigger_rate: dict[str, float]`
  - `class_coverage: dict[str, float]`
  - `lift: dict[str, float | null]`
  - `support: SupportSummary`
  - `warnings: ApiWarning[]`

#### `SupportSummary`
- 典型字段:
  - `triggered_samples`
  - `total_samples`
  - `min_support`
  - `alpha`

- 说明:
  - 当 `triggered_samples < min_support` 时，`lift` 可能返回 `null`

### 2.4 直方图

#### `ShapeletHistogramResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/stats/histogram?scope=...&hist_mode=...&shapelet_id=...&bins=...&density=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `scope`
  - `hist_mode`
  - `shapelet_id`
  - `bins`
  - `density`
  - `range: float[2]`
  - `counts: float[bins]`
  - `bin_edges: float[bins+1]`
  - `warnings: ApiWarning[]`

- 说明:
  - `hist_mode=per_shapelet` 时必须传 `shapelet_id`
  - `hist_mode=global` 时 `shapelet_id` 为 `null`

### 2.5 按类统计明细

#### `ShapeletClassStatsResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes?scope=...&omega=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `shapelet_id`
  - `scope`
  - `omega`
  - `items: ShapeletClassStatsItem[]`
  - `warnings: ApiWarning[]`

#### `ShapeletClassStatsItem`
- 典型字段:
  - `class_id`
  - `prior`
  - `trigger_rate`
  - `coverage`
  - `lift`

### 2.6 B->C 联动样本（契约冻结）

#### `ShapeletTopHitsResponse`
- 对应接口:
  - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/samples/top-hits?scope=...&omega=...&offset=...&limit=...&rank_metric=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `shapelet_id`
  - `scope`
  - `omega`
  - `total`
  - `offset`
  - `limit`
  - `rank_metric`
  - `items: TopHitSampleItem[]`
  - `warnings: ApiWarning[]`

#### `TopHitSampleItem`
- 典型字段:
  - `sample_id`
  - `trigger_score`
  - `rank`
  - `label`
  - `pred_class`
  - `margin`

#### `PartBToCLink`（前端导航对象）
- 必带字段:
  - `dataset`
  - `sample_id`
  - `shapelet_id`
  - `scope`
  - `omega`
  - `source_panel='part_b'`
- 可选字段:
  - `trigger_score`
  - `rank`
  - `rank_metric`
- 说明:
  - 禁止仅凭 `shapelet_id` 跳转 Part C，必须携带 `sample_id`

## 3. 两个容易混淆的点

### 3.1 `ShapeletStatsSummaryResponse` 和 `ShapeletClassStatsResponse` 不是一回事
- `ShapeletStatsSummaryResponse`
  - 用于摘要卡片
  - 返回全局触发率 + 各类映射字典 + `support`
  - 更适合顶部 KPI 和简要解读

- `ShapeletClassStatsResponse`
  - 用于明细表格
  - 返回 `items[]`（每类一行）
  - 更适合排序、筛选、导出

### 3.2 `ShapeletGalleryItem` 和 `ShapeletDetailResponse` 不是一回事
- `ShapeletGalleryItem`
  - 是一个“条目类型”
  - 出现在列表的 `items[]` 里，也作为详情中的 `shapelet` 嵌套对象

- `ShapeletDetailResponse`
  - 是“接口顶层响应类型”
  - 除了 `shapelet` 外，还包含 `spec_version`、`dataset`、`warnings`

## 4. 当前 Part B 的推荐前端请求顺序

1. `GET /api/v1/part-b/datasets/{dataset_name}/meta`
2. `GET /api/v1/part-b/datasets/{dataset_name}/shapelets?offset=0&limit=100`
3. 用户选中某个 `shapelet_id` 后并行请求:
   - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}`
   - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/summary?scope=test&omega=0.1`
   - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/stats/histogram?scope=test&hist_mode=per_shapelet&shapelet_id={shapelet_id}&bins=50&density=true`
   - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/stats/classes?scope=test&omega=0.1`
   - `GET /api/v1/part-b/datasets/{dataset_name}/shapelets/{shapelet_id}/samples/top-hits?scope=test&omega=0.1&offset=0&limit=20&rank_metric=max_i`（契约冻结，待实现）
4. 用户调整 `omega` 时，仅刷新:
   - `stats/summary`
   - `stats/histogram`
   - `stats/classes`
   - `samples/top-hits`
5. 用户翻页列表时，仅刷新:
   - `shapelets?offset=...&limit=...`
6. 用户执行 B->C 跳转时，使用 `PartBToCLink` 组装路由参数:
   - 必带: `dataset + sample_id + shapelet_id + scope + omega + source_panel`
   - 可选: `trigger_score + rank + rank_metric`
