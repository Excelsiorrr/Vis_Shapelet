# Part E 接口与响应类型对照表

- 契约来源:
  - [md/vis_shapelet_part_e_api.md](/d:/shapelet/ShapeX-new/md/vis_shapelet_part_e_api.md)
  - [md/vis_shapelet_api.md](/d:/shapelet/ShapeX-new/md/vis_shapelet_api.md)
- 目的:
  - 把 Part E 当前接口契约与响应模型的一一对应关系单独列清楚
  - 方便前端、后端、文档对照，不必在大文档中来回跳转
- 说明:
  - 当前 Part E 以后端契约文档为准（v1 冻结口径）
  - v1 仅交付单一 `shapelet_id + span` 的 what-if，Shapley 为 v1.1+

## 1. 总表

| 接口 | 顶层响应类型 | 列表/嵌套类型 | 说明 |
| --- | --- | --- | --- |
| `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate` | `PartEWhatIfResponse` | `perturbed_sequence: float[T][D]\|null` `ApiWarning[]` | 单样本单 segment what-if（Part E 核心） |
| `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/shapley:estimate`（v1.1+） | `ExplainResult`（v1.1+） | `phi/stderr/saliency`（v1.1+） | Shapley 估计，不属于 v1 必做 |

## 2. 类型说明

### 2.1 核心 what-if 接口

#### `PartEWhatIfRequest`（请求体）
- 对应接口:
  - `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate`
- 结构:
  - `shapelet_id`
  - `t_start`
  - `t_end`
  - `scope`（`test|train`）
  - `omega`
  - `baseline`
  - `value_type`
  - `target_class`
  - `seed`
  - `include_perturbed_sequence`
- 规则（冻结）:
  - `span` 在 API 中拆分为 `t_start/t_end` 两字段
  - `span=[t_start,t_end]` 为闭区间（含两端）
  - `0 <= t_start <= t_end <= T-1`
  - 越界先裁剪再执行，并在响应中回显裁剪后的区间

#### `PartEWhatIfResponse`
- 对应接口:
  - `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate`
- 结构:
  - `spec_version`
  - `dataset`
  - `sample_id`
  - `shapelet_id`
  - `t_start`
  - `t_end`
  - `scope`
  - `omega`
  - `baseline`
  - `value_type`
  - `target_class`
  - `seed`
  - `p_original`
  - `p_whatif`
  - `delta`
  - `delta_target`
  - `pred_class_original`
  - `pred_class_whatif`
  - `y_true`
  - `perturbed_sequence: float[T][D]|null`
  - `warnings: ApiWarning[]`

#### `ApiWarning`
- 典型字段:
  - `code`
  - `message`

### 2.2 C->E 联动对象（导航载荷）

#### `PartCToELink`（前端导航对象）
- 典型字段:
  - `dataset`
  - `sample_id`
  - `shapelet_id`
  - `t_start`
  - `t_end`
  - `scope`
  - `omega`
  - `source_panel='part_c'`
- 说明:
  - `t_start/t_end` 可为空；为空时 E 侧可按 C 的 `peak_t + L_p` 推导默认 span
  - 推荐 `Part A/B -> Part C -> Part E` 链路中使用该对象，保证跨页口径一致

### 2.3 v1.1+ 预留类型

#### `ExplainResult`（v1.1+）
- 对应接口:
  - `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/shapley:estimate`
- 说明:
  - 当前不属于 v1 必做；用于 v1.1+ Shapley 输出

## 3. 两个容易混淆的点

### 3.1 `target_class` 和 `y_true` 不是一回事
- `target_class`
  - 请求侧可选观察目标类别
  - 影响 `delta_target` 的计算与返回

- `y_true`
  - 响应侧真实标签（若存在）
  - 仅用于展示，不作为 `target_class` 默认值

### 3.2 `perturbed_sequence = null` 和 `perturbed_sequence != null` 不是一回事
- `perturbed_sequence = null`
  - 常见于 `include_perturbed_sequence=false`
  - 表示本次不返回扰动后序列明细，只返回概率变化结果

- `perturbed_sequence != null`
  - 常见于 `include_perturbed_sequence=true`
  - 可用于前端做原始序列 vs 扰动序列对照展示

## 4. 当前 Part E 的推荐前端请求顺序

1. 在 Part C 完成样本与证据定位（拿到 `sample_id + shapelet_id + span + scope + omega`）
2. 调用:
   - `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate`
3. 用户切换 `baseline/value_type/target_class/span/omega` 时:
   - 仅重新请求 `whatif:evaluate`
4. 用户需要序列对照视图时:
   - 将 `include_perturbed_sequence` 设为 `true` 重新请求
5. 用户执行 E->C 复核时:
   - 携带 `sample_id + shapelet_id + span + scope + omega` 回跳 Part C

## 5. v1 错误码与 warning 对照

- 错误码:
  - `ERR_INVALID_SCOPE`
  - `ERR_INVALID_OMEGA`
  - `ERR_INVALID_SPAN`
  - `ERR_SAMPLE_NOT_FOUND`
  - `ERR_SHAPELET_NOT_FOUND`
- warning 码:
  - `SHAPELET_SPAN_MISMATCH`（不阻断执行）
