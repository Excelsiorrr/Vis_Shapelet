# Shapelet Part C API 契约（v1 草案）

- 关联 PRD: `vis_shapelet_spec.md`（2.3 Part C）
- 主 API 文档: `vis_shapelet_api.md`
- 适用范围: Part C（Match & Locate Panel）
- 范围边界:
  - 本文件仅维护 Part C 专属接口与跨 Part 联动载荷
  - Part A 的样本详情（原始序列/预测）仍以 `part-a` 接口为主数据源
  - Part B 的 shapelet 库静态信息仍以 `part-b` 接口为主数据源

## 1. API 契约（v1 草案）

## 1.1 通用约定
- Base URL: `/api/v1`
- 数据格式: `application/json`
- 错误返回:
  - `{"code":"ERR_xxx","message":"...","trace_id":"..."}`
- 版本控制:
  - 响应包含 `spec_version`
- Part C 冻结约束:
  - v1 后端仅返回并展示 `I`（模型原生匹配分数），不返回 `A`
  - `peak_t` 语义固定为 center-aligned 索引，不是窗口起点
  - 高亮窗口计算固定口径:
    - `start = peak_t - floor(L/2)`
    - `end = start + L - 1`
    - 最终裁剪到 `[0, T-1]`
  - 证据高亮阈值统一使用全局 `omega`（与 Part B/Part E 一致）
  - `Part B -> Part C` 联动契约冻结:
    - 必带字段: `dataset`, `sample_id`, `shapelet_id`, `scope`, `omega`, `source_panel='part_b'`
    - 可选字段: `trigger_score`, `rank`, `rank_metric`
    - 禁止仅凭 `shapelet_id` 跳转 Part C，必须携带具体 `sample_id`
  - `Part E -> Part C` 回跳契约冻结:
    - 必带字段: `dataset`, `sample_id`, `shapelet_id`, `t_start`, `t_end`, `scope`, `omega`, `source_panel='part_e'`
    - 可选字段: `baseline`, `value_type`, `target_class`
    - `span=[t_start,t_end]` 为闭区间（0-based，含两端）
    - 禁止仅凭 `shapelet_id` 回跳 Part C，必须携带具体 `sample_id + span`
  - Part C 支持的 `scope`（冻结）:
    - 仅支持 `test|train`，不开放 `all`
    - 当接收到 `scope=all` 时返回 `400 ERR_INVALID_SCOPE`
  - shapelet 返回优先级（冻结）:
    - 当 `shapelet_ids` 非空时，按 `shapelet_ids` 返回，并忽略 `topk_shapelets`
    - 当 `shapelet_ids` 为空且 `topk_shapelets` 有值时，按 `peak_score` 降序返回 Top-K
    - 若 `peak_score` 相同，按 `shapelet_id` 升序稳定排序
    - 当两者都为空时，返回该样本下全部 shapelet（Part C 默认行为）
  - `navigation/from-part-b` 接口为 v1 必做接口（冻结）
  - `include_*` 默认值（冻结）:
    - `include_sequence=true`
    - `include_prediction=true`
    - `include_windows=true`
  - `windows` 返回语义（冻结）:
    - `include_windows=true` 且正常计算: 返回数组；无命中时返回 `[]`
    - `include_windows=false` 或后端降级未计算: 返回 `null`

## 1.2 核心对象 Schema

```json
{
  "ApiWarning": {
    "code": "string",
    "message": "string"
  },
  "PredictionSummary": {
    "pred_class": "int",
    "probs": "float[C]",
    "margin": "float"
  },
  "PartCMetaResponse": {
    "spec_version": "string",
    "dataset": "string",
    "scope_default": "test|train",
    "omega_default": "float",
    "time_index_base": "0",
    "peak_alignment": "center",
    "match_score_semantics": "model_native_match_score",
    "shapelet_len_source_priority": "match_response > part_b_link > shapelet_detail",
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
  },
  "PartCMatchRequest": {
    "scope": "test|train",
    "omega": "float",
    "shapelet_ids": "string[]|null",
    "topk_shapelets": "int|null",
    "pinned_shapelet_id": "string|null",
    "include_sequence": "bool",
    "include_prediction": "bool",
    "include_windows": "bool"
  },
  "HighlightWindow": {
    "shapelet_id": "string",
    "shapelet_len": "int",
    "peak_t": "int",
    "start": "int",
    "end": "int",
    "peak_score": "float",
    "triggered": "bool"
  },
  "PinnedShapeletStatus": {
    "shapelet_id": "string|null",
    "is_present_in_tensor": "bool",
    "peak_t": "int|null",
    "triggered": "bool|null"
  },
  "MatchTensorResponse": {
    "spec_version": "string",
    "dataset": "string",
    "sample_id": "string",
    "split": "test|train",
    "scope": "test|train",
    "omega": "float",
    "shapelet_ids": "string[P]",
    "shapelet_lens": "int[P]",
    "I": "float[P][T]",
    "peak_t": "int[P]",
    "windows": "HighlightWindow[]|null",
    "pinned_shapelet": "PinnedShapeletStatus",
    "sequence": "float[T][D]|null",
    "prediction": "PredictionSummary|null",
    "params": {
      "similarity_type": "string",
      "shapelet_temperature": "float",
      "normalize": "string",
      "score_semantics": "model_native_match_score"
    },
    "warnings": "ApiWarning[]"
  },
  "PartCFromPartBRequest": {
    "link": "PartBToCLink",
    "include_sequence": "bool",
    "include_prediction": "bool",
    "include_windows": "bool"
  },
  "PartCFromPartBResponse": {
    "spec_version": "string",
    "link": "PartBToCLink",
    "resolved_match_request": "PartCMatchRequest",
    "match": "MatchTensorResponse",
    "warnings": "ApiWarning[]"
  },
  "PartEToCLink": {
    "dataset": "string",
    "sample_id": "string",
    "shapelet_id": "string",
    "t_start": "int",
    "t_end": "int",
    "scope": "test|train",
    "omega": "float",
    "source_panel": "part_e",
    "baseline": "linear_interp|zero|dataset_mean|null",
    "value_type": "prob|logit|null",
    "target_class": "int|null"
  },
  "PartCFromPartERequest": {
    "link": "PartEToCLink",
    "include_sequence": "bool",
    "include_prediction": "bool",
    "include_windows": "bool"
  },
  "PartCFromPartEResponse": {
    "spec_version": "string",
    "link": "PartEToCLink",
    "resolved_match_request": "PartCMatchRequest",
    "match": "MatchTensorResponse",
    "warnings": "ApiWarning[]"
  }
}
```

## 1.2.1 Part C Schema 前端理解

### ApiWarning
- 这是什么:
  - 后端主动返回的告警信息。
- 一般出现在哪:
  - Part C 各响应的 `warnings` 字段。
- 前端通常怎么用:
  - 渲染页面顶部提示、数据降级提示、联动参数修正提示。
- 字段说明:
  - `code`: 稳定告警代码，适合做埋点和条件分支。
  - `message`: 面向用户的解释文本，可直接展示。

### PredictionSummary
- 这是什么:
  - 单样本预测摘要对象，与 Part A 语义一致。
- 一般出现在哪:
  - `MatchTensorResponse.prediction`
- 前端通常怎么用:
  - 在 Part C 顶部信息栏显示预测类别和 margin。
  - 在高亮解释时给出“当前样本不确定性”上下文。
- 字段说明:
  - `pred_class`: 预测类别 id。
  - `probs`: 各类别概率数组。
  - `margin`: Top1 与 Top2 概率差值。

### PartCMetaResponse
- 这是什么:
  - Part C 页面的初始化配置响应。
- 一般出现在哪:
  - `GET /api/v1/part-c/datasets/{dataset_name}/meta` 的响应体。
- 前端通常怎么用:
  - 初始化 `scope/omega` 默认值。
  - 展示 `peak_t` 对齐与窗口计算规则说明。
  - 记录 score 语义，避免把 `I` 误认为概率值。
- 字段说明:
  - `scope_default`: 默认统计范围，推荐 `test`。
  - `omega_default`: 默认证据阈值，与 Part B/Part E 统一。
  - `time_index_base`: 时间索引基准，固定 `0`。
  - `peak_alignment`: 固定 `center`，指明 `peak_t` 口径。
  - `match_score_semantics`: `I` 的语义说明（模型原生匹配分数）。
  - `shapelet_len_source_priority`: 高亮窗口长度来源优先级说明。

### PartBToCLink
- 这是什么:
  - 从 Part B 进入 Part C 的冻结导航对象。
- 一般出现在哪:
  - 前端路由参数、导航 state、`PartCFromPartBRequest.link`。
- 前端通常怎么用:
  - 统一承接 B 页选中的样本和 shapelet 上下文，避免跨页口径漂移。
- 字段说明:
  - `dataset/sample_id/shapelet_id/scope/omega/source_panel` 为必填字段。
  - `trigger_score/rank/rank_metric` 为可选追溯字段。
  - `shapelet_len` 为可选加速字段，不能替代 `match` 响应中的正式长度。

### PartCMatchRequest
- 这是什么:
  - Part C 拉取匹配张量的请求体。
- 一般出现在哪:
  - `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches` 请求体。
- 前端通常怎么用:
  - 首次进入可请求全量或 `topk` 子集；交互时按需增量刷新。
  - `pinned_shapelet_id` 用于首屏自动聚焦或从 B->C 带入焦点。
- 字段说明:
  - `scope`: 当前样本来源范围，`test|train`（不支持 `all`）。
  - `omega`: 证据阈值。
  - `shapelet_ids`: 指定 shapelet 子集；非空时优先级最高。
  - `topk_shapelets`: 限制返回 shapelet 数量，用于性能控制；仅在 `shapelet_ids` 为空时生效。
  - `pinned_shapelet_id`: UI 当前 pin 的 shapelet。
  - `include_sequence`: 是否携带原始序列。
  - `include_prediction`: 是否携带预测摘要。
  - `include_windows`: 是否返回已计算好的高亮窗口。

### HighlightWindow
- 这是什么:
  - 单个 shapelet 在当前样本上的可视化高亮窗口。
- 一般出现在哪:
  - `MatchTensorResponse.windows[]`
- 前端通常怎么用:
  - 在主图叠加证据区间（`start~end`）。
  - hover 某行热图时快速定位到原序列窗口。
- 字段说明:
  - `shapelet_id`: 对应的 shapelet。
  - `shapelet_len`: 当前 shapelet 长度 `L`。
  - `peak_t`: 峰值中心位置。
  - `start/end`: 按中心对齐推导并裁剪后的显示区间。
  - `peak_score`: `I[p, peak_t]`。
  - `triggered`: 是否满足 `max_t I[p,t] >= omega`。

### PinnedShapeletStatus
- 这是什么:
  - 当前 pin 的 shapelet 在本次响应中的解析状态。
- 一般出现在哪:
  - `MatchTensorResponse.pinned_shapelet`
- 前端通常怎么用:
  - 判断 pin 是否有效。
  - 若 `is_present_in_tensor=false`，可提示用户切换 shapelet 或放宽筛选。
- 字段说明:
  - `shapelet_id`: 当前 pin 目标。
  - `is_present_in_tensor`: 是否出现在本次返回集合中。
  - `peak_t`: pin shapelet 的峰值位置（若存在）。
  - `triggered`: pin shapelet 在当前 `omega` 下是否触发。

### MatchTensorResponse
- 这是什么:
  - Part C 核心响应，承载单样本的 `shapelet x time` 匹配张量与定位结果。
- 一般出现在哪:
  - `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches` 响应体。
- 前端通常怎么用:
  - 直接绘制 heatmap（`I`）。
  - 用 `peak_t + shapelet_lens` 或 `windows` 同步高亮主图。
  - 用 `pinned_shapelet` 完成跨页自动定位。
- 字段说明:
  - `split`: 当前样本实际来源（`train|test`），用于和 `scope` 做一致性提示。
  - `scope/omega`: 本次统计口径回显（`scope` 仅 `test|train`）。
  - `shapelet_ids/shapelet_lens`: 行索引与长度数组，按同一顺序对齐。
  - `I`: 匹配分数矩阵，形状 `P x T`。
  - `peak_t`: 每个 shapelet 的峰值中心索引，长度 `P`。
  - `windows`: 可选预计算高亮窗口列表；当 `include_windows=false` 时可返回 `null`。
  - `sequence`: 可选样本原始序列，形状 `T x D`。
  - `prediction`: 可选预测摘要。
  - `params`: 本次匹配配置回显。

### PartCFromPartBRequest
- 这是什么:
  - Part B -> Part C 聚合加载请求体。
- 一般出现在哪:
  - `POST /api/v1/part-c/navigation/from-part-b` 请求体。
- 前端通常怎么用:
  - 把 B 页面已有导航参数一次性交给后端，避免前端二次拼接错误。
  - 可通过 `include_*` 控制首屏负载大小。

### PartCFromPartBResponse
- 这是什么:
  - B->C 聚合加载响应（包含解析后的匹配请求和 match 结果）。
- 一般出现在哪:
  - `POST /api/v1/part-c/navigation/from-part-b` 响应体。
- 前端通常怎么用:
  - 作为 C 页首屏一次请求结果，减少串行等待。
  - 用 `resolved_match_request` 对照当前页面状态，保证口径一致。

### PartEToCLink
- 这是什么:
  - 从 Part E 回跳 Part C 的冻结导航对象。
- 一般出现在哪:
  - `PartCFromPartERequest.link`。
- 前端通常怎么用:
  - 把 what-if 页当前上下文（样本、shapelet、span、口径）完整带回 C 复核。
- 字段说明:
  - `dataset/sample_id/shapelet_id/t_start/t_end/scope/omega/source_panel` 为必填。
  - `baseline/value_type/target_class` 为可选透传字段（用于 UI 回显，不影响 Match 主计算）。

### PartCFromPartERequest / PartCFromPartEResponse
- 这是什么:
  - Part E -> Part C 的聚合加载请求/响应。
- 前端通常怎么用:
  - 回跳时优先用聚合接口，避免前端重复拼接 `matches` 请求。

## 1.3 端点列表

### 0) 获取 Part C 首屏基础信息
- `GET /api/v1/part-c/datasets/{dataset_name}/meta`
- 返回:
  - `scope_default`（`test` 或 `train`，推荐 `test`）
  - `omega_default`
  - `peak_t` 对齐语义与时间索引约定
  - 高亮窗口长度来源优先级说明
  - warnings
- 前端理解:
  - 建议作为 Part C 首个请求，用于初始化页面状态和帮助文案。
  - 与 Part B 联动进入时，若 query 已携带 `scope/omega`，则以联动参数优先。
- 参数说明:
  - `dataset_name`: 数据集名称，需在后端支持列表内。

### 1) 获取单样本匹配张量（Part C 核心接口）
- `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches`
- 请求体:
  - `PartCMatchRequest`
- 返回:
  - `MatchTensorResponse`
- 前端理解:
  - 这是 Part C 主数据源，负责热图、定位和证据高亮。
  - 若从 Part A 进入，通常先请求该接口并设置 `pinned_shapelet_id=null`。
  - 若从 Part B 进入，通常携带 `pinned_shapelet_id=shapelet_id`。
- 参数说明:
  - Path:
    - `dataset_name`: 数据集名称。
    - `sample_id`: 样本 id。
  - Body:
    - `scope`: `test|train`。
    - `omega`: 全局阈值。
    - `shapelet_ids`: 可选 shapelet 子集。
    - `topk_shapelets`: 可选返回数量上限，建议用于性能保护；按 `peak_score` 降序截取，分数并列时按 `shapelet_id` 升序。
    - `pinned_shapelet_id`: 可选 pin 目标。
    - `include_sequence/include_prediction/include_windows`: 可选回包裁剪开关。
  - 参数边界与默认值（冻结）:
    - `omega`: 必须为有限浮点数（非 `NaN/Inf`）
    - `shapelet_ids`: 可选；若提供需去重，最大长度 `512`
    - `topk_shapelets`: 可选；取值范围 `1 ~ 512`
    - `include_sequence/include_prediction/include_windows`: 默认均为 `true`
  - 错误码（冻结）:
    - `ERR_INVALID_SCOPE`: `scope` 非 `test|train`（含 `all`）
    - `ERR_INVALID_OMEGA`: `omega` 非法（`NaN/Inf`）
    - `ERR_INVALID_TOPK`: `topk_shapelets` 越界
    - `ERR_TOO_MANY_SHAPELETS`: `shapelet_ids` 超过上限
    - `ERR_SAMPLE_NOT_FOUND`: `sample_id` 无效
    - `ERR_SHAPELET_NOT_FOUND`: `shapelet_ids` 或 `pinned_shapelet_id` 中存在无效 id

### 2) Part B -> Part C 联动聚合加载（v1 必做）
- `POST /api/v1/part-c/navigation/from-part-b`
- 请求体:
  - `PartCFromPartBRequest`
- 返回:
  - `PartCFromPartBResponse`
- 说明:
  - 用于减少“解析 link + 拼 match 请求 + 拉取 match”的前端重复逻辑。
  - v1 要求后端实现该接口。
  - 当 `link.scope=all` 时应返回 `400 ERR_INVALID_SCOPE`，由前端回退到 `test` 或 `train` 后重试。
- 前端理解:
  - 已有 B->C 冻结字段时，优先使用该聚合接口可降低联动错误率。
  - 响应中的 `link` 与 `resolved_match_request` 可作为调试和埋点依据。

### 3) Part E -> Part C 回跳聚合加载（v1 建议）
- `POST /api/v1/part-c/navigation/from-part-e`
- 请求体:
  - `PartCFromPartERequest`
- 返回:
  - `PartCFromPartEResponse`
- 说明:
  - 用于承接 Part E 的 what-if 上下文并回到 C 做匹配复核。
  - `t_start/t_end` 仅作为回跳上下文；`match` 计算仍以 `dataset/sample_id/scope/omega` 与 pin 逻辑为准。
  - `link.scope=all` 时应返回 `400 ERR_INVALID_SCOPE`。

## 2. 推荐请求流
- Part A -> Part C:
  - `meta -> matches`
- Part B -> Part C（使用聚合接口）:
  - `meta -> navigation/from-part-b`
- Part E -> Part C（回跳复核）:
  - `meta -> navigation/from-part-e`
- 交互刷新建议:
  - `omega` 变化:
    - 仅刷新 `matches`（不重拉 `meta`）
  - `sample_id` 变化:
    - 重新请求 `matches`
  - 仅切换是否显示原始序列/预测:
    - 优先使用 `include_sequence/include_prediction` 控制回包，避免重复请求无关字段

## 3. 一致性与校验规则（实现建议）
- B/C 一致性:
  - 同一 `dataset + sample_id + shapelet_id + scope + omega` 下：
    - Part B 的触发结论应与 Part C `windows[].triggered` 一致
- 长度一致性:
  - `len(shapelet_ids) == len(shapelet_lens) == len(peak_t) == I.shape[0]`
- 时间维一致性:
  - `I.shape[1] == sequence.shape[0]`（当 `sequence != null`）
- pin 一致性:
  - 若 `pinned_shapelet_id != null` 且不在返回 `shapelet_ids` 中，必须回填:
    - `pinned_shapelet.is_present_in_tensor = false`
    - `warnings` 增加可解释提示
