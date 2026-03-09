# Shapelet 可视分析系统 API 契约（v1）

- 关联 PRD: `vis_shapelet_spec.md`
- 适用范围: Part A - Part E
- 说明:
  - 本文件承载接口契约、Schema、请求响应示例
  - PRD 与算法口径保留在 `vis_shapelet_spec.md`
  - Part E 专属细化文档: `vis_shapelet_part_e_api.md`

## 4. API 契约（v1）

## 4.1 通用约定
- Base URL: `/api/v1`
- 数据格式: `application/json`
- 时间索引: 0-based，区间采用闭区间 `[start, end]`
- 错误返回:
  - `{"code":"ERR_xxx","message":"...","trace_id":"..."}`
- 版本控制:
  - 响应包含 `spec_version`

## 4.2 核心对象 Schema

### Part A 新增 Schema

前端阅读说明:
- 本节 JSON 仍然是正式契约，字段名和类型以 JSON 为准。
- 本节后面的“前端理解”仅解释字段语义、典型展示位置和交互用途，不改变接口含义。
- 若字段语义与页面理解发生冲突，以本文件契约和后端返回值为准。

```json
{
  "DatasetMeta": {
    "dataset": "string",
    "sampling_rate": "hour|min|day|second|unknown",
    "seq_len": "int"
  },
  "TrainingMeta": {
    "shapelet_num": "int",
    "shapelet_len": "int",
    "classifier_num_classes": "int"
  },
  "MetaResponse": {
    "spec_version": "string",
    "dataset_meta": "DatasetMeta",
    "training_meta": "TrainingMeta",
    "train_split_mapping": "SplitMappingNote",
    "test_split_mapping": "SplitMappingNote",
    "warnings": "ApiWarning[]"
  },
  "ClusterProfile": {
    "cluster_id": "int",
    "size": "int",
    "sample_ids_preview": "string[]",
    "centroid_sequence": "float[T][D]",
    "median_sequence": "float[T][D]",
    "q25_sequence": "float[T][D]",
    "q75_sequence": "float[T][D]"
  },
  "PredictionSummary": {
    "pred_class": "int",
    "probs": "float[C]",
    "margin": "float"
  },
  "TestSampleSummary": {
    "sample_id": "string",
    "label": "int|null",
    "prediction": "PredictionSummary"
  },
  "LowMarginSample": {
    "sample_id": "string",
    "label": "int|null",
    "pred_class": "int",
    "probs": "float[C]",
    "margin": "float",
    "sequence": "float[T][D]"
  },
  "SplitMappingNote": {
    "requested_role": "train|test",
    "actual_source": "string"
  },
  "ApiWarning": {
    "code": "string",
    "message": "string"
  },
  "MetricsResponse": {
    "spec_version": "string",
    "test_metrics": {
      "acc": "float",
      "f1": "float",
      "auc": "float|null"
    },
    "sample_count": "int",
    "class_distribution": {
      "class_id": "int"
    },
    "low_margin_count": "int"
  },
  "ClustersResponse": {
    "spec_version": "string",
    "dataset": "string",
    "cluster_k": "int",
    "train_cluster_profiles": "ClusterProfile[]",
    "warnings": "ApiWarning[]"
  },
  "LowMarginSampleSummary": {
    "sample_id": "string",
    "label": "int|null",
    "pred_class": "int",
    "probs": "float[C]",
    "margin": "float",
    "sequence": "float[T][D]"
  },
  "LowMarginSamplesResponse": {
    "spec_version": "string",
    "dataset": "string",
    "margin_threshold": "float",
    "total": "int",
    "offset": "int",
    "limit": "int",
    "items": "LowMarginSampleSummary[]",
    "warnings": "ApiWarning[]"
  },
  "ClassSamplesResponse": {
    "spec_version": "string",
    "dataset": "string",
    "label": "int",
    "total": "int",
    "offset": "int",
    "limit": "int",
    "items": "TestSampleSummary[]",
    "warnings": "ApiWarning[]"
  },
  "SampleDetailResponse": {
    "spec_version": "string",
    "dataset": "string",
    "split": "train|test",
    "sample_id": "string",
    "label": "int|null",
    "prediction": "PredictionSummary",
    "sequence": "float[T][D]",
    "suggested_window_len": "int",
    "warnings": "ApiWarning[]"
  }
}
```

#### Part A Schema 前端理解

##### DatasetMeta
- 这是什么:
  - 数据集本身的基础信息，不是模型输出，也不是单样本数据。
- 一般出现在哪:
  - `MetaResponse.dataset_meta`
- 前端通常怎么用:
  - 展示在总览页顶部信息卡片。
  - 给图表或详情面板显示全局上下文。
- 字段说明:
  - `dataset`: 数据集名称，可直接显示，也可作为当前页面路由状态。
  - `sampling_rate`: 时间粒度，不是数值采样频率。当前约定返回 `hour|min|day|second|unknown` 之一，表示 `1 step` 对应的时间单位。
  - `seq_len`: 单条序列长度。前端可用来设置横轴范围、滚动窗口上限。

##### TrainingMeta
- 这是什么:
  - 模型训练阶段得到的元信息，描述当前模型规模和关键配置。
- 一般出现在哪:
  - `MetaResponse.training_meta`
- 前端通常怎么用:
  - 展示在“模型信息”卡片中。
  - 为后续交互提供默认值，例如默认窗口长度。
- 字段说明:
  - `shapelet_num`: 当前模型中 shapelet 的数量。常用于展示模型规模，不对应单个样本。
  - `shapelet_len`: 每个 shapelet 的时间长度。前端可把它作为默认窗口长度或默认 brush 长度。
  - `classifier_num_classes`: 分类头实际输出的类别数。前端可用它校验 `probs` 数组长度、构建类别图例。

##### ClusterProfile
- 这是什么:
  - 一个训练簇的统计摘要，不是单个样本点。
  - 聚类时先对每条原始序列做 z-normalized，再把归一化后的序列展平后送入 KMeans。
  - 可视化时不画 PCA 散点图，而是回到原始序列空间，画每簇“中心线 + 分位带”。
- 一般出现在哪:
  - `ClustersResponse.train_cluster_profiles`
- 前端通常怎么用:
  - 每个簇画一张折线图或一个小面板。
  - 可同时画两条代表线:
    - `centroid_sequence`: KMeans 簇中心线
    - `median_sequence`: 更抗异常值的典型形态线
  - 用 `q25_sequence` 到 `q75_sequence` 之间的区域作为分位带。
  - `sample_ids_preview` 可用于展示少量成员样本 id 预览。
- 字段说明:
  - `cluster_id`: 聚类后的簇编号。
  - `size`: 当前簇包含的训练样本数。
  - `sample_ids_preview`: 当前簇内成员样本 id 预览列表，默认仅返回少量样本用于前端展示。
  - `centroid_sequence`: 当前簇成员在原始序列空间的 centroid 线，可理解为逐时间点均值得到的簇中心线。
  - `median_sequence`: 在原始序列空间中，对簇内所有样本逐时间点取中位数得到的中心线。
  - `q25_sequence`: 在原始序列空间中，对簇内所有样本逐时间点取 25% 分位数。
  - `q75_sequence`: 在原始序列空间中，对簇内所有样本逐时间点取 75% 分位数。

##### PredictionSummary
- 这是什么:
  - 单个样本的预测摘要，是前端最常展示的预测结果块。
- 一般出现在哪:
  - `TestSampleSummary.prediction`
  - `SampleDetailResponse.prediction`
- 前端通常怎么用:
  - 展示预测类别、概率条形图、置信度提示。
  - 用 `margin` 标记“不确定样本”。
- 字段说明:
  - `pred_class`: 预测类别 id。
  - `probs`: 各类别概率数组，长度通常等于 `classifier_num_classes`。
  - `margin`: 最大概率与第二大概率之差。越小表示模型越犹豫。

##### TestSampleSummary
- 这是什么:
  - 测试样本在列表视图中的简化摘要。
- 一般出现在哪:
  - `ClassSamplesResponse.items`
- 前端通常怎么用:
  - 渲染分类后的样本列表。
  - 点击某个样本后再请求详情接口。
- 字段说明:
  - `sample_id`: 当前样本 id，可直接拼到样本详情接口路径中。
  - `label`: 真实标签。
  - `prediction`: 当前样本的预测摘要。

##### LowMarginSample
- 这是什么:
  - 预测不够稳定的测试样本详情。
- 一般出现在哪:
  - `SampleDetailResponse.sequence` 的上游选样结果，或后端内部计算过程。
- 前端通常怎么用:
  - 在“重点关注样本”区域展示。
  - 直接画出其原始序列，无需再次请求详情接口。
- 字段说明:
  - `sample_id`, `label`, `pred_class`, `probs`, `margin`: 含义与前文一致。
  - `sequence`: 原始时序，形状为 `T x D`。单变量时也统一写成二维数组。

##### SplitMappingNote
- 这是什么:
  - 一个解释性对象，说明“接口名义上的 split”和“底层实际使用的数据源”是否一致。
- 一般出现在哪:
  - `MetaResponse.train_split_mapping`
  - `MetaResponse.test_split_mapping`
- 前端通常怎么用:
  - 在页面上提示用户存在已知映射问题。
  - 不建议拿它做主业务逻辑判断，更适合作为说明信息展示。
- 字段说明:
  - `requested_role`: 接口层想表达的角色，例如 `train` 或 `test`。
  - `actual_source`: 底层真实取到的数据源名称。

##### ApiWarning
- 这是什么:
  - 后端主动返回的告警信息。
- 一般出现在哪:
  - `MetaResponse.warnings`
  - `ClustersResponse.warnings`
  - `LowMarginSamplesResponse.warnings`
  - `ClassSamplesResponse.warnings`
  - `SampleDetailResponse.warnings`
- 前端通常怎么用:
  - 渲染成页面顶部提示条、弹窗说明或可展开告警列表。
- 字段说明:
  - `code`: 稳定的告警代码，适合前端做条件判断。
  - `message`: 面向人的解释文本，适合直接展示。

##### MetaResponse
- 这是什么:
  - Part A 首屏基础信息响应。
- 前端通常怎么用:
  - 页面初始化时优先请求它。
  - 请求完成后即可填充顶部信息卡片、全局 warning 和 split 映射说明。

##### MetricsResponse
- 这是什么:
  - 测试集评估摘要响应。
- 前端通常怎么用:
  - 作为异步卡片数据源，展示 `acc/f1/auc` 与样本数量摘要。

##### ClustersResponse
- 这是什么:
  - 训练集聚类统计响应。
- 前端通常怎么用:
  - 在总览页主体区域异步加载训练簇图，不阻塞首屏。

##### LowMarginSamplesResponse
- 这是什么:
  - 低 margin 样本分页响应。
- 前端通常怎么用:
  - 填充“重点关注样本”列表。
  - 可直接用返回的 `sequence` 画小图，必要时再请求样本详情补更多字段。

##### ClassSamplesResponse
- 这是什么:
  - 按类别分页返回的测试样本摘要列表。
- 前端通常怎么用:
  - 渲染按类查看的样本列表或分页表格。

##### SampleDetailResponse
- 这是什么:
  - 单个样本详情页或详情侧栏所需数据。
- 前端通常怎么用:
  - 点击样本后按需请求。
  - 用于绘制大图、显示预测信息、初始化时间窗口长度。
- 字段说明:
  - `dataset`: 所属数据集。
  - `split`: 当前样本来自 `train` 还是 `test`。
  - `sample_id`: 当前样本 id。
  - `label`: 真实标签。
  - `prediction`: 预测摘要。
  - `sequence`: 原始序列，形状为 `T x D`。
  - `suggested_window_len`: 推荐的初始窗口长度，当前后端约定等于 `shapelet_len`。
  - `warnings`: 与当前数据集相关的告警信息。

```json
{
  "PredictionSummary": {
    "pred_class": "int",
    "probs": "float[C]",
    "margin": "float"
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
  "PartEWhatIfRequest": {
    "shapelet_id": "string",
    "t_start": "int",
    "t_end": "int",
    "scope": "test|train",
    "omega": "float",
    "baseline": "linear_interp|zero|dataset_mean",
    "value_type": "prob|logit",
    "target_class": "int|null",
    "seed": "int|null",
    "include_perturbed_sequence": "bool"
  },
  "PartEWhatIfResponse": {
    "spec_version": "string",
    "dataset": "string",
    "sample_id": "string",
    "shapelet_id": "string",
    "t_start": "int",
    "t_end": "int",
    "scope": "test|train",
    "omega": "float",
    "baseline": "linear_interp|zero|dataset_mean",
    "value_type": "prob|logit",
    "target_class": "int|null",
    "seed": "int",
    "p_original": "float",
    "p_whatif": "float",
    "delta": "float",
    "delta_target": "float|null",
    "pred_class_original": "int",
    "pred_class_whatif": "int",
    "y_true": "int|null",
    "perturbed_sequence": "float[T][D]|null",
    "warnings": "ApiWarning[]"
  },
  "ExplainResult": {
    "status": "deprecated_for_v1",
    "available_in": "v1.1+"
  }
}
```

## 4.3 端点列表

### Part A 新增端点

#### 0) 获取内置数据集列表
- `GET /api/v1/part-a/datasets`
- 返回:
  - 可选内置数据集列表
  - `dataset_file` 当前未启用的说明
  - 若为 `mcce | mcch | mtce | mtch`，额外返回 split 映射和 `num_classes` 说明
- 前端理解:
  - 这是数据集选择器的初始化接口。
  - 页面首次加载时可先请求它，决定下拉框、卡片列表或默认数据集。
  - 若只做 Part A 页面，这通常是第一个请求。

#### 1) 获取 Part A 首屏基础信息
- `GET /api/v1/part-a/datasets/{dataset_name}/meta`
- 返回:
  - 数据集元信息: `sampling_rate`, `seq_len`
  - 训练元信息: `shapelet_num`, `shapelet_len`, `classifier_num_classes`
  - split 映射说明
  - warnings
- 前端理解:
  - 这是 Part A 首屏同步接口，应尽量轻量。
  - 页面应优先用它渲染基础信息，再异步加载评估、聚类和样本列表。

#### 2) 获取测试集评估摘要
- `GET /api/v1/part-a/datasets/{dataset_name}/metrics?margin_threshold=0.1`
- 返回:
  - `acc/f1/auc`
  - 测试集样本总数 `sample_count`
  - 类别分布 `class_distribution`
  - 低 margin 样本数量 `low_margin_count`
- 前端理解:
  - 这是评估摘要接口，适合作为异步指标卡片数据源。
  - `margin_threshold` 会影响 `low_margin_count`。

#### 3) 获取训练集聚类统计
- `GET /api/v1/part-a/datasets/{dataset_name}/clusters?cluster_k=4`
- 返回:
  - `cluster_k`
  - 训练集聚类统计曲线: `train_cluster_profiles`
  - warnings
- 前端理解:
  - 这是训练簇视图的数据源。
  - 聚类输入使用 z-normalized 后的训练序列。
  - 可视化展示使用原始训练序列的统计曲线，不使用 PCA 散点图。

#### 4) 获取低 margin 样本摘要分页列表
- `GET /api/v1/part-a/datasets/{dataset_name}/samples/low-margin?threshold=0.1&offset=0&limit=50`
- 返回:
  - 低 margin 样本列表 `items`
  - `total / offset / limit`
- 前端理解:
  - 这是“重点关注样本”区的数据源。
  - 列表返回摘要和原始序列，可直接用于小图或快速预览。

#### 5) 获取按类别分页的测试样本列表
- `GET /api/v1/part-a/datasets/{dataset_name}/samples?label=1&offset=0&limit=50`
- 返回:
  - 指定类别下的测试样本摘要列表 `items`
  - `total / offset / limit`
- 前端理解:
  - 这是“按类分页”展示的数据源。
  - 若前端需要展示所有类别，可先请求 `metrics` 获取类别分布，再按类分页请求本接口。

#### 6) 获取单样本详情
- `GET /api/v1/part-a/datasets/{dataset_name}/samples/{sample_id}?split=test`
- 返回:
  - 样本原始序列
  - `pred_class`, `probs`, `margin`
  - 切时间窗建议 `suggested_window_len = shapelet_len`
  - warnings（同 meta 接口）
- 前端理解:
  - 这是按需详情接口，适合点击样本后再请求。
  - `sample_id` 通常来自低 margin 列表或按类分页列表。
  - `sequence` 可直接用来画折线图或热图。
  - `suggested_window_len` 可作为前端初始时间窗或刷选长度默认值。

### Part C 端点（草案，遵循 `/api/v1/part-*` 体系）

#### 1) 获取单样本匹配张量（I）
- `POST /api/v1/part-c/datasets/{dataset_name}/samples/{sample_id}/matches`
- 请求体: `PartCMatchRequest`
- 返回: `MatchTensorResponse`
- 说明:
  - `scope` 仅支持 `test|train`，不开放 `all`
  - `shapelet_ids` 与 `topk_shapelets` 同时传入时，以 `shapelet_ids` 为准
  - 当仅提供 `topk_shapelets` 时，按 `peak_score` 降序返回，分数并列按 `shapelet_id` 升序

#### 2) Part B -> Part C 联动加载（v1 必做）
- `POST /api/v1/part-c/navigation/from-part-b`
- 说明:
  - 请求体可使用 `PartCFromPartBRequest`（内部 `link` 使用 `PartBToCLink` 冻结字段: `dataset/sample_id/shapelet_id/scope/omega/source_panel`）。
  - v1 要求实现该聚合接口。
  - 当 `link.scope=all` 时，返回 `400 ERR_INVALID_SCOPE`。

### Part D 端点（草案，遵循 `/api/v1/part-*` 体系）

#### 3) 由阈值生成 players
- `POST /api/v1/part-d/datasets/{dataset_name}/samples/{sample_id}/players:derive`
- 返回: `players + union_spans + stats`

### Part E 端点（草案，遵循 `/api/v1/part-*` 体系）

#### 4) what-if 评估
- `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate`
- 请求体: `PartEWhatIfRequest`
- 返回: `PartEWhatIfResponse`
- v1 冻结约束:
  - 路径必填: `dataset_name/sample_id`
  - 请求体必填: `shapelet_id/t_start/t_end/scope/omega`
  - `span=[t_start,t_end]` 采用闭区间（0-based，含两端）
  - `delta = p_whatif - p_original`
  - 若请求携带 `target_class`，响应额外返回 `delta_target`
- 错误码:
  - `ERR_INVALID_SCOPE`
  - `ERR_INVALID_OMEGA`
  - `ERR_INVALID_SPAN`
  - `ERR_SHAPELET_NOT_FOUND`
  - `ERR_SAMPLE_NOT_FOUND`
  - `SHAPELET_SPAN_MISMATCH`（warning，不中断）

#### 5) Shapley 估计（v1.1+）
- `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/shapley:estimate`
- 返回: `ExplainResult`
- 说明:
  - 不属于 v1 必做范围；待 Part D players 稳定后恢复。

### 会话端点（草案，遵循 `/api/v1/part-*` 体系）

#### 6) 会话保存与回放
- `POST /api/v1/part-e/sessions`
- `GET /api/v1/part-e/sessions/{session_id}`

### 废弃说明
- 旧草案路由 `/samples/*`、`/shapelets/*`、`/sessions/*` 不再作为 v1 契约的一部分。
- v1 统一保留 `/api/v1/part-a/*`、`/api/v1/part-b/*`，以及后续补齐的 `/api/v1/part-c/*`、`/api/v1/part-d/*`、`/api/v1/part-e/*`。
