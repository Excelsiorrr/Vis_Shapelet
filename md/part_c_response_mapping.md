# Part C 接口与响应类型对照表

- 契约来源:
  - [md/vis_shapelet_part_c_api.md](/d:/shapelet/ShapeX-new/md/vis_shapelet_part_c_api.md)
  - [md/vis_shapelet_api.md](/d:/shapelet/ShapeX-new/md/vis_shapelet_api.md)
- 目的:
  - 把 Part C 当前接口契约与响应模型的一一对应关系单独列清楚
  - 方便前端、后端、文档对照，不必在大文档中来回跳转
- 说明:
  - 当前 Part C 以后端契约为准（v1 文档冻结口径），代码实现可按本表落地

## 1. 总表

| 接口 | 顶层响应类型 | 列表/嵌套类型 | 说明 |
| --- | --- | --- | --- |
| `GET /api/v1/part-c/datasets/{dataset_name}/meta` | `PartCMetaResponse` | `ApiWarning[]` | Part C 首屏基础信息 |
| `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches` | `MatchTensorResponse` | `HighlightWindow[]|null` `PinnedShapeletStatus` `PredictionSummary|null` `ApiWarning[]` | 单样本匹配张量与定位结果（Part C 核心） |
| `POST /api/v1/part-c/navigation/from-part-b` | `PartCFromPartBResponse` | `PartBToCLink` `PartCMatchRequest` `MatchTensorResponse` `ApiWarning[]` | B->C 联动聚合加载（v1 必做） |

## 2. 类型说明

### 2.1 首屏基础信息

#### `PartCMetaResponse`
- 对应接口:
  - `GET /api/v1/part-c/datasets/{dataset_name}/meta`
- 结构:
  - `spec_version`
  - `dataset`
  - `scope_default`（`test|train`）
  - `omega_default`
  - `time_index_base`
  - `peak_alignment`
  - `match_score_semantics`
  - `shapelet_len_source_priority`
  - `warnings: ApiWarning[]`

#### `ApiWarning`
- 典型字段:
  - `code`
  - `message`

### 2.2 核心匹配接口

#### `PartCMatchRequest`（请求体）
- 对应接口:
  - `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches`
- 结构:
  - `scope`（`test|train`，不支持 `all`）
  - `omega`
  - `shapelet_ids: string[]|null`
  - `topk_shapelets: int|null`
  - `pinned_shapelet_id: string|null`
  - `include_sequence: bool`
  - `include_prediction: bool`
  - `include_windows: bool`
- 规则（冻结）:
  - `shapelet_ids` 非空时优先级最高，忽略 `topk_shapelets`
  - `shapelet_ids` 为空且 `topk_shapelets` 有值时，按 `peak_score` 降序截取 Top-K；并列按 `shapelet_id` 升序
  - 两者都为空时，返回全部 shapelet
  - `include_*` 默认值均为 `true`

#### `MatchTensorResponse`
- 对应接口:
  - `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches`
- 结构:
  - `spec_version`
  - `dataset`
  - `sample_id`
  - `split`（`test|train`）
  - `scope`（`test|train`）
  - `omega`
  - `shapelet_ids: string[P]`
  - `shapelet_lens: int[P]`
  - `I: float[P][T]`
  - `peak_t: int[P]`
  - `windows: HighlightWindow[]|null`
  - `pinned_shapelet: PinnedShapeletStatus`
  - `sequence: float[T][D]|null`
  - `prediction: PredictionSummary|null`
  - `params`
  - `warnings: ApiWarning[]`

#### `HighlightWindow`
- 典型字段:
  - `shapelet_id`
  - `shapelet_len`
  - `peak_t`
  - `start`
  - `end`
  - `peak_score`
  - `triggered`
- 说明:
  - `include_windows=true` 且正常计算时返回数组；无命中返回 `[]`
  - `include_windows=false` 或后端降级未计算时可返回 `null`

#### `PinnedShapeletStatus`
- 典型字段:
  - `shapelet_id`
  - `is_present_in_tensor`
  - `peak_t`
  - `triggered`

#### `PredictionSummary`
- 典型字段:
  - `pred_class`
  - `probs`
  - `margin`

### 2.3 B->C 联动聚合

#### `PartCFromPartBRequest`（请求体）
- 对应接口:
  - `POST /api/v1/part-c/navigation/from-part-b`
- 结构:
  - `link: PartBToCLink`
  - `include_sequence`
  - `include_prediction`
  - `include_windows`

#### `PartBToCLink`
- 必带字段:
  - `dataset`
  - `sample_id`
  - `shapelet_id`
  - `scope`
  - `omega`
  - `source_panel='part_b'`
- 可选字段:
  - `shapelet_len`
  - `trigger_score`
  - `rank`
  - `rank_metric`
- 说明:
  - `link.scope` 若为 `all`，应返回 `400 ERR_INVALID_SCOPE`

#### `PartCFromPartBResponse`
- 对应接口:
  - `POST /api/v1/part-c/navigation/from-part-b`
- 结构:
  - `spec_version`
  - `link: PartBToCLink`
  - `resolved_match_request: PartCMatchRequest`
  - `match: MatchTensorResponse`
  - `warnings: ApiWarning[]`

## 3. 两个容易混淆的点

### 3.1 `PartCMatchRequest` 和 `MatchTensorResponse` 不是一回事
- `PartCMatchRequest`
  - 请求体类型
  - 表达“这次要按什么口径拿数据”
  - 例如 `scope/omega/shapelet_ids/topk/include_*`

- `MatchTensorResponse`
  - 响应体类型
  - 返回“实际匹配与定位结果”
  - 例如 `I/peak_t/windows/sequence/prediction`

### 3.2 `windows = null` 和 `windows = []` 不是一回事
- `windows = null`
  - 表示本次未返回窗口计算结果
  - 常见于 `include_windows=false` 或后端降级

- `windows = []`
  - 表示已执行窗口计算，但当前没有可返回的命中窗口
  - 语义是“算过了，结果为空”

## 4. 当前 Part C 的推荐前端请求顺序

1. `GET /api/v1/part-c/datasets/{dataset_name}/meta`
2. 从 Part A 进入时:
   - `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches`
3. 从 Part B 进入时（v1 推荐）:
   - `POST /api/v1/part-c/navigation/from-part-b`
4. 用户调整 `omega` 或切换样本时:
   - 仅刷新 `matches` 或 `navigation/from-part-b`（按当前入口路径）
5. 用户仅切换是否展示序列/预测/窗口时:
   - 优先通过 `include_sequence/include_prediction/include_windows` 控制回包裁剪
