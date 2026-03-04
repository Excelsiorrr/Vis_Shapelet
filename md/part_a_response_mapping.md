# Part A 接口与响应类型对照表

- 代码来源: [backend/part_a.py](/d:/shapelet/ShapeX-new/backend/part_a.py)
- 目的:
  - 把 Part A 当前实际接口与响应模型的一一对应关系单独列清楚
  - 方便前端、后端、文档对照，不必在 `vis_shapelet_api.md` 中来回跳转

## 1. 总表

| 接口 | 顶层响应类型 | 列表/嵌套类型 | 说明 |
| --- | --- | --- | --- |
| `GET /api/v1/part-a/datasets` | `DatasetListResponse` | `DatasetListItem[]` | 内置数据集列表 |
| `GET /api/v1/part-a/datasets/{dataset_name}/meta` | `MetaResponse` | `DatasetMeta` `TrainingMeta` `SplitMappingNote` `ApiWarning[]` | 首屏基础信息 |
| `GET /api/v1/part-a/datasets/{dataset_name}/metrics?margin_threshold=...` | `MetricsResponse` | `MetricSummary` | 测试集评估摘要 |
| `GET /api/v1/part-a/datasets/{dataset_name}/clusters?cluster_k=...` | `ClustersResponse` | `ClusterProfile[]` `ApiWarning[]` | 训练集聚类统计 |
| `GET /api/v1/part-a/datasets/{dataset_name}/samples/low-margin?threshold=...&offset=...&limit=...` | `LowMarginSamplesResponse` | `LowMarginSampleSummary[]` `ApiWarning[]` | 低 margin 样本分页 |
| `GET /api/v1/part-a/datasets/{dataset_name}/samples?label=...&offset=...&limit=...` | `ClassSamplesResponse` | `TestSampleSummary[]` `PredictionSummary` `ApiWarning[]` | 按类别分页的测试样本列表 |
| `GET /api/v1/part-a/datasets/{dataset_name}/samples/{sample_id}?split=test` | `SampleDetailResponse` | `PredictionSummary` `ApiWarning[]` | 单样本完整详情 |

## 2. 类型说明

### 2.1 列表入口

#### `DatasetListResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets`
- 结构:
  - `spec_version`
  - `datasets: DatasetListItem[]`

#### `DatasetListItem`
- 典型字段:
  - `dataset`
  - `display_name`
  - `dataset_file_supported`
  - `notes`

### 2.2 首屏基础信息

#### `MetaResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets/{dataset_name}/meta`
- 结构:
  - `spec_version`
  - `dataset_meta: DatasetMeta`
  - `training_meta: TrainingMeta`
  - `train_split_mapping: SplitMappingNote`
  - `test_split_mapping: SplitMappingNote`
  - `warnings: ApiWarning[]`

#### `DatasetMeta`
- 典型字段:
  - `dataset`
  - `sampling_rate`
  - `seq_len`

#### `TrainingMeta`
- 典型字段:
  - `shapelet_num`
  - `shapelet_len`
  - `classifier_num_classes`

#### `SplitMappingNote`
- 典型字段:
  - `requested_role`
  - `actual_source`

#### `ApiWarning`
- 典型字段:
  - `code`
  - `message`

### 2.3 评估摘要

#### `MetricsResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets/{dataset_name}/metrics?margin_threshold=...`
- 结构:
  - `spec_version`
  - `test_metrics: MetricSummary`
  - `sample_count`
  - `class_distribution: dict[str, int]`
  - `low_margin_count`

#### `MetricSummary`
- 典型字段:
  - `acc`
  - `f1`
  - `auc`

### 2.4 训练集聚类

#### `ClustersResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets/{dataset_name}/clusters?cluster_k=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `cluster_k`
  - `train_cluster_profiles: ClusterProfile[]`
  - `warnings: ApiWarning[]`

#### `ClusterProfile`
- 典型字段:
  - `cluster_id`
  - `size`
  - `sample_ids_preview`
  - `centroid_sequence`
  - `median_sequence`
  - `q25_sequence`
  - `q75_sequence`

- 说明:
  - 现在不再返回完整 `sample_ids`
  - 仅返回 `sample_ids_preview` 用于前端展示少量成员预览

### 2.5 低 margin 列表

#### `LowMarginSamplesResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets/{dataset_name}/samples/low-margin?threshold=...&offset=...&limit=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `margin_threshold`
  - `total`
  - `offset`
  - `limit`
  - `items: LowMarginSampleSummary[]`
  - `warnings: ApiWarning[]`

#### `LowMarginSampleSummary`
- 典型字段:
  - `sample_id`
  - `label`
  - `pred_class`
  - `probs`
  - `margin`
  - `sequence`

- 说明:
  - 当前会返回原始 `sequence`
  - 适合低 margin 样本列表直接画小图或做快速预览
  - 如果还需要 `suggested_window_len` 等字段，继续调用样本详情接口

### 2.6 按类别分页样本列表

#### `ClassSamplesResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets/{dataset_name}/samples?label=...&offset=...&limit=...`
- 结构:
  - `spec_version`
  - `dataset`
  - `label`
  - `total`
  - `offset`
  - `limit`
  - `items: TestSampleSummary[]`
  - `warnings: ApiWarning[]`

#### `TestSampleSummary`
- 典型字段:
  - `sample_id`
  - `label`
  - `prediction: PredictionSummary`

#### `PredictionSummary`
- 典型字段:
  - `pred_class`
  - `probs`
  - `margin`

### 2.7 单样本详情

#### `SampleDetailResponse`
- 对应接口:
  - `GET /api/v1/part-a/datasets/{dataset_name}/samples/{sample_id}?split=test`
- 结构:
  - `spec_version`
  - `dataset`
  - `split`
  - `sample_id`
  - `label`
  - `prediction: PredictionSummary`
  - `sequence`
  - `suggested_window_len`
  - `warnings: ApiWarning[]`

## 3. 两个容易混淆的点

### 3.1 `LowMarginSampleSummary` 和 `SampleDetailResponse` 不是一回事
- `LowMarginSampleSummary`
  - 出现在低 margin 列表接口
  - 带 `sequence`
  - 用于重点关注样本列表和快速预览

- `SampleDetailResponse`
  - 出现在单样本详情接口
  - 带完整 `sequence`
  - 还带 `split`、`suggested_window_len`、更完整的详情语义
  - 用于点击后的详情展示

### 3.2 `TestSampleSummary` 和 `LowMarginSampleSummary` 不是一回事
- `TestSampleSummary`
  - 用于按类别分页样本列表
  - 结构里嵌套 `prediction: PredictionSummary`

- `LowMarginSampleSummary`
  - 用于低 margin 分页列表
  - 把 `pred_class/probs/margin` 直接平铺到当前对象

## 4. 当前 Part A 的推荐前端请求顺序

1. `GET /api/v1/part-a/datasets`
2. `GET /api/v1/part-a/datasets/{dataset_name}/meta`
3. 并行请求:
   - `GET /api/v1/part-a/datasets/{dataset_name}/metrics`
   - `GET /api/v1/part-a/datasets/{dataset_name}/clusters`
   - `GET /api/v1/part-a/datasets/{dataset_name}/samples/low-margin`
4. 用户切换到按类浏览时:
   - `GET /api/v1/part-a/datasets/{dataset_name}/samples?label=...&offset=...&limit=...`
5. 用户点击具体样本时:
   - `GET /api/v1/part-a/datasets/{dataset_name}/samples/{sample_id}?split=test`
