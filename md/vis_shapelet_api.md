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

```json
{
  "DatasetMeta": {
    "dataset": "string",
    "sampling_rate": "float",
    "seq_len": "int"
  },
  "TrainingMeta": {
    "shapelet_num": "int",
    "shapelet_len": "int",
    "classifier_num_classes": "int"
  },
  "ClusterPoint": {
    "sample_id": "string",
    "label": "int|null",
    "cluster_id": "int",
    "projection": "float[2]"
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
    "train_clusters": "ClusterPoint[]",
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

#### 1) 获取 Part A 数据集总览
- `GET /api/v1/part-a/datasets/{dataset_name}/overview?cluster_k=4&margin_threshold=0.1`
- 返回:
  - 数据集元信息: `sampling_rate`, `seq_len`
  - 训练元信息: `shapelet_num`, `shapelet_len`, `classifier_num_classes`
  - 训练集聚类点集: `train_clusters`
  - 测试集指标: `acc/f1/auc`
  - 测试集按类别样本列表: `test_by_class`
  - 低 `margin` 样本明细: `low_margin_samples`
  - warnings:
    - `dataset_file` MVP 未启用
    - `mcce | mcch | mtce | mtch` 的 split 映射问题
    - `mcce | mcch | mtce | mtch` 的 `4 类 checkpoint` vs `yaml 中 2 类配置`

#### 2) 获取单样本详情
- `GET /api/v1/part-a/datasets/{dataset_name}/samples/{sample_id}?split=test`
- 返回:
  - 样本原始序列
  - `pred_class`, `probs`, `margin`
  - 切时间窗建议 `suggested_window_len = shapelet_len`
  - warnings（同总览接口）

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
