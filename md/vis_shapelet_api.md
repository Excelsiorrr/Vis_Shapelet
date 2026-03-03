# Shapelet 可视分析系统 API 契约（v1）

- 关联 PRD: `vis_shapelet_spec.md`
- 适用范围: Part A - Part E
- 说明:
  - 本文件承载接口契约、Schema、请求响应示例
  - PRD 与算法口径保留在 `vis_shapelet_spec.md`

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
  "ClusterProfile": {
    "cluster_id": "int",
    "size": "int",
    "sample_ids": "string[]",
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
  "DatasetOverviewResponse": {
    "spec_version": "string",
    "dataset_meta": "DatasetMeta",
    "training_meta": "TrainingMeta",
    "cluster_k": "int",
    "train_split_mapping": "SplitMappingNote",
    "test_split_mapping": "SplitMappingNote",
    "train_cluster_profiles": "ClusterProfile[]",
    "test_metrics": {
      "acc": "float",
      "f1": "float",
      "auc": "float|null"
    },
    "test_by_class": {
      "class_id": "TestSampleSummary[]"
    },
    "low_margin_samples": "LowMarginSample[]",
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
  - `DatasetOverviewResponse.dataset_meta`
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
  - `DatasetOverviewResponse.training_meta`
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
  - `DatasetOverviewResponse.train_cluster_profiles`
- 前端通常怎么用:
  - 每个簇画一张折线图或一个小面板。
  - 可同时画两条代表线:
    - `centroid_sequence`: KMeans 簇中心线
    - `median_sequence`: 更抗异常值的典型形态线
  - 用 `q25_sequence` 到 `q75_sequence` 之间的区域作为分位带。
  - `sample_ids` 可用于联动样本列表或后续高亮。
- 字段说明:
  - `cluster_id`: 聚类后的簇编号。
  - `size`: 当前簇包含的训练样本数。
  - `sample_ids`: 当前簇内成员样本 id 列表。
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
  - `DatasetOverviewResponse.test_by_class`
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
  - `DatasetOverviewResponse.low_margin_samples`
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
  - `DatasetOverviewResponse.train_split_mapping`
  - `DatasetOverviewResponse.test_split_mapping`
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
  - `DatasetOverviewResponse.warnings`
  - `SampleDetailResponse.warnings`
- 前端通常怎么用:
  - 渲染成页面顶部提示条、弹窗说明或可展开告警列表。
- 字段说明:
  - `code`: 稳定的告警代码，适合前端做条件判断。
  - `message`: 面向人的解释文本，适合直接展示。

##### DatasetOverviewResponse
- 这是什么:
  - Part A 总览页的聚合响应。一个接口返回总览页大部分所需数据。
- 前端通常怎么用:
  - 页面初始化时优先请求它。
  - 请求完成后即可同时填充总览卡片、训练簇统计图、分类列表、低置信样本列表。
- 字段说明:
  - `spec_version`: 响应契约版本。
  - `dataset_meta`: 数据集基础信息。
  - `training_meta`: 模型训练元信息。
  - `cluster_k`: 本次请求实际使用的聚类数，通常与查询参数一致。
  - `train_split_mapping`, `test_split_mapping`: split 映射说明。
  - `train_cluster_profiles`: 训练集各簇的统计曲线数据。
  - `test_metrics`: 测试集整体指标。
  - `test_by_class`: 以类别 id 为 key 的测试样本分组。
  - `low_margin_samples`: 低置信样本明细列表。
  - `warnings`: 全局告警。

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
  "Sample": {
    "id": "string",
    "x": "float[T][D]",
    "label": "int|null",
    "meta": {
      "sampling_rate_hz": "float",
      "seq_len": "int",
      "dataset": "string"
    }
  },
  "Prediction": {
    "pred_class": "int",
    "logits": "float[C]",
    "probs": "float[C]",
    "margin": "float"
  },
  "MatchTensor": {
    "sample_id": "string",
    "shapelet_ids": "string[P]",
    "I": "float[P][T]",
    "peak_t": "int[P]",
    "params": {
      "similarity_type": "string",
      "shapelet_temperature": "float",
      "normalize": "string",
      "score_semantics": "model_native_match_score"
    }
  },
  "Player": {
    "player_id": "string",
    "shapelet_id": "string",
    "start": "int",
    "end": "int",
    "peak_score": "float",
    "mean_score": "float"
  },
  "ExplainResult": {
    "sample_id": "string",
    "coalition": "string[player_id]",
    "p_original": "float",
    "p_whatif": "float",
    "delta": "float",
    "phi": "float[n_players]",
    "stderr": "float[n_players]",
    "saliency": "float[T]",
    "config": {
      "target_class": "int",
      "value_type": "logit|prob",
      "baseline": "linear_interp|zero|dataset_mean",
      "seed": "int",
      "perm_count": "int"
    }
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

#### 1) 获取 Part A 数据集总览
- `GET /api/v1/part-a/datasets/{dataset_name}/overview?cluster_k=4&margin_threshold=0.1`
- 返回:
  - 数据集元信息: `sampling_rate`, `seq_len`
  - 训练元信息: `shapelet_num`, `shapelet_len`, `classifier_num_classes`
  - 训练集聚类统计曲线: `train_cluster_profiles`
  - 测试集指标: `acc/f1/auc`
  - 测试集按类别样本列表: `test_by_class`
  - 低 `margin` 样本明细: `low_margin_samples`
  - warnings:
    - `dataset_file` MVP 未启用
    - `mcce | mcch | mtce | mtch` 的 split 映射问题
    - `mcce | mcch | mtce | mtch` 的 `4 类 checkpoint` vs `yaml 中 2 类配置`
- 前端理解:
  - 这是总览页主接口，建议把它当成“页面主数据源”。
  - `cluster_k` 会影响训练簇的数量。
  - `margin_threshold` 会影响低置信样本列表长度。
  - 聚类前后处理口径:
    - 聚类输入使用 z-normalized 后的训练序列。
    - 可视化展示使用原始训练序列的统计曲线，不使用 PCA 散点图。
  - 典型页面映射:
    - `dataset_meta` + `training_meta`: 顶部信息卡片
    - `train_cluster_profiles`: 每簇 centroid + 中位数线 + IQR 分位带图
    - `test_metrics`: 指标卡片
    - `test_by_class`: 分类样本列表
    - `low_margin_samples`: 重点关注样本区
    - `warnings`: 页面提示条

#### 2) 获取单样本详情
- `GET /api/v1/part-a/datasets/{dataset_name}/samples/{sample_id}?split=test`
- 返回:
  - 样本原始序列
  - `pred_class`, `probs`, `margin`
  - 切时间窗建议 `suggested_window_len = shapelet_len`
  - warnings（同总览接口）
- 前端理解:
  - 这是按需详情接口，适合点击样本后再请求。
  - `sample_id` 通常来自 `test_by_class[*].sample_id` 或 `low_margin_samples[*].sample_id`。
  - `sequence` 可直接用来画折线图或热图。
  - `suggested_window_len` 可作为前端初始时间窗或刷选长度默认值。

### 1) 获取样本列表
- `GET /samples?split=test&offset=0&limit=50`
- 返回: `sample_id`, `label`, `meta`, 可选缩略统计

### 2) 获取样本详情与预测
- `GET /samples/{sample_id}/prediction?target_class=1`
- 返回: `Sample + Prediction`

### 3) 获取匹配张量（I）
- `POST /samples/{sample_id}/matches`
- 请求:
```json
{
  "shapelet_ids": ["s1", "s2"],
  "similarity_type": "cosine",
  "shapelet_temperature": 1.0,
  "normalize": "znorm"
}
```
- 返回: `MatchTensor`

### 4) 由阈值生成 players
- `POST /samples/{sample_id}/players:derive`
- 请求:
```json
{
  "I": "optional server-side ref",
  "omega": 0.02,
  "min_len": 5,
  "fill_gap_len": 2,
  "merge_iou": 0.6
}
```
- 返回:
```json
{
  "players": [],
  "union_spans": [[10, 28], [72, 90]],
  "stats": {
    "n_players": 6
  }
}
```

### 5) what-if 评估
- `POST /samples/{sample_id}/whatif:evaluate`
- 请求:
```json
{
  "coalition": ["p1", "p3"],
  "target_class": 1,
  "value_type": "prob",
  "baseline": "linear_interp"
}
```
- 返回:
```json
{
  "p_original": 0.73,
  "p_whatif": 0.81,
  "delta": 0.08
}
```

### 6) Shapley 估计
- `POST /samples/{sample_id}/shapley:estimate`
- 请求:
```json
{
  "players": ["p1", "p2", "p3"],
  "target_class": 1,
  "value_type": "prob",
  "baseline": "linear_interp",
  "min_perm": 64,
  "max_perm": 512,
  "stderr_tol": 0.01,
  "seed": 2026
}
```
- 返回: `ExplainResult`

### 7) shapelet 库查询（Part B）
- `GET /shapelets?offset=0&limit=100`
- `GET /shapelets/{shapelet_id}/stats?split=test`

### 8) 会话保存与回放
- `POST /sessions`
- `GET /sessions/{session_id}`
