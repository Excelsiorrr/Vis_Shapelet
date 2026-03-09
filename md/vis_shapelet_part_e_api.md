# Shapelet Part E API 契约（v1 草案）

- 关联 PRD: `vis_shapelet_spec.md`（2.5 Part E, 3.3, 3.5）
- 主 API 文档: `vis_shapelet_api.md`
- 适用范围: Part E（Perturbation Panel）
- 范围边界:
  - 本文件仅维护 Part E 专属接口与跨 Part 联动载荷
  - v1 仅覆盖单一 `shapelet_id + span` 的 what-if
  - v1 不依赖 Part D players/coalition
  - Shapley 估计下沉到 v1.1+

## 1. API 契约（v1 草案）

## 1.1 通用约定
- Base URL: `/api/v1`
- 数据格式: `application/json`
- 时间索引: 0-based，区间采用闭区间 `[start, end]`
- 错误返回:
  - `{"code":"ERR_xxx","message":"...","trace_id":"..."}`
- 版本控制:
  - 响应包含 `spec_version`
- Part E 冻结约束:
  - 路径必填: `dataset_name`, `sample_id`
  - 请求体必填: `shapelet_id`, `t_start`, `t_end`, `scope`, `omega`
  - 可选字段: `baseline`, `value_type`, `target_class`, `seed`, `include_perturbed_sequence`
  - `delta = p_whatif - p_original`
  - `y_true` 仅展示，不作为 `target_class` 默认值
  - 当 `shapelet_id` 与 `span` 存在语义不一致时，默认返回 warning 并继续执行

## 1.2 核心对象 Schema

```json
{
  "ApiWarning": {
    "code": "string",
    "message": "string"
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
  "PartCToELink": {
    "dataset": "string",
    "sample_id": "string",
    "shapelet_id": "string",
    "t_start": "int|null",
    "t_end": "int|null",
    "scope": "test|train",
    "omega": "float",
    "source_panel": "part_c"
  }
}
```

## 1.2.1 Part E Schema 前端理解

### ApiWarning
- 这是什么:
  - 后端主动返回的告警信息。
- 一般出现在哪:
  - `PartEWhatIfResponse.warnings`。
- 前端通常怎么用:
  - 渲染提示条，标注“结果可用但存在口径提醒”。
- 字段说明:
  - `code`: 稳定告警代码，适合做分支处理与埋点。
  - `message`: 面向用户的解释文本，可直接展示。

### PartEWhatIfRequest
- 这是什么:
  - Part E 执行单次 what-if 的请求体。
- 一般出现在哪:
  - `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate` 请求体。
- 前端通常怎么用:
  - 从 Part C 已选中的 `shapelet + span` 直接构造请求。
  - `span` 在 API 中拆成 `t_start/t_end` 两个字段传递。
- 字段说明:
  - `shapelet_id`: 本次干预目标 shapelet。
  - `t_start/t_end`: 干预时间区间，闭区间，含两端。
  - `scope`: 当前样本来源范围，`test|train`（不支持 `all`）。
  - `omega`: 解释阈值回显字段，与 Part B/Part C 口径一致。
  - `baseline`: 基线类型，默认 `linear_interp`。
  - `value_type`: 输出口径，默认 `prob`，可选 `logit`。
  - `target_class`: 可选目标类别；用于额外返回 `delta_target`。
  - `seed`: 随机种子，不传则后端使用默认值并回显。
  - `include_perturbed_sequence`: 是否返回扰动后序列，默认 `false`。

### PartEWhatIfResponse
- 这是什么:
  - 单次 what-if 的标准结果对象。
- 一般出现在哪:
  - `whatif:evaluate` 响应体。
- 前端通常怎么用:
  - 展示 `p_original/p_whatif/delta`、类别变化、可选扰动序列对比。
- 字段说明:
  - `dataset/sample_id/shapelet_id/t_start/t_end/scope/omega`: 本次实验上下文回显。
  - `p_original/p_whatif/delta`: 核心结果，`delta = p_whatif - p_original`。
  - `delta_target`: 仅当请求传入 `target_class` 时返回，否则为 `null`。
  - `pred_class_original/pred_class_whatif`: 预测类别变化。
  - `y_true`: 样本真实标签（若有）。
  - `perturbed_sequence`: 当 `include_perturbed_sequence=true` 时返回。

### PartCToELink
- 这是什么:
  - 从 Part C 进入 Part E 的冻结导航对象。
- 一般出现在哪:
  - 前端路由参数、导航 state、会话记录。
- 前端通常怎么用:
  - 复用 C 页上下文，避免 E 页自行猜测样本或 shapelet。
- 字段说明:
  - `dataset/sample_id/shapelet_id/scope/omega/source_panel` 为必填。
  - `t_start/t_end` 可为空；为空时允许 Part E 按 C 的 `peak_t + L_p` 规则推导默认 span。

## 1.3 端点列表

### 1) what-if 评估（v1 必做）
- `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/whatif:evaluate`
- 请求体:
  - `PartEWhatIfRequest`
- 返回:
  - `PartEWhatIfResponse`
- 前端理解:
  - 这是 Part E 主接口。
  - 建议在 C 页确定 `shapelet + span` 后调用，避免 E 页二次推断。
- 参数说明:
  - Path:
    - `dataset_name`: 数据集名称。
    - `sample_id`: 样本 id。
  - Body:
    - `shapelet_id`: 干预目标 shapelet。
    - `t_start/t_end`: 干预区间（闭区间，含两端）。
    - `scope`: `test|train`。
    - `omega`: 全局阈值口径回显。
    - `baseline/value_type/target_class/seed/include_perturbed_sequence`: 可选。
- 参数边界与默认值（冻结）:
  - `t_start/t_end`: 合法范围 `0 <= t_start <= t_end <= T-1`；越界时裁剪并回显裁剪后值。
  - `scope`: 仅支持 `test|train`。
  - `omega`: 必须为有限浮点数（非 `NaN/Inf`）。
  - `baseline`: 默认 `linear_interp`。
  - `value_type`: 默认 `prob`。
  - `include_perturbed_sequence`: 默认 `false`。
- 执行规则（冻结）:
  - 仅允许单一 `shapelet_id + span` 干预。
  - `linear_interp` 按通道独立插值。
  - 端点使用 `x[t_start-1]` 与 `x[t_end+1]`；边界缺失按单端延拓。
  - 单点段（`t_start == t_end`）按邻点均值或单端值退化处理。
- 错误码（冻结）:
  - `ERR_INVALID_SCOPE`: `scope` 非 `test|train`（含 `all`）
  - `ERR_INVALID_OMEGA`: `omega` 非法（`NaN/Inf`）
  - `ERR_INVALID_SPAN`: 区间非法（空/反向/不可解析）
  - `ERR_SAMPLE_NOT_FOUND`: `sample_id` 无效
  - `ERR_SHAPELET_NOT_FOUND`: `shapelet_id` 无效
  - `SHAPELET_SPAN_MISMATCH`（warning，不中断）

### 2) Shapley 估计（v1.1+）
- `POST /api/v1/part-e/datasets/{dataset_name}/samples/{sample_id}/shapley:estimate`
- 说明:
  - 不属于 v1 必做范围。

## 2. 推荐请求流
- Part A/B -> Part C -> Part E:
  - 在 C 页确定 `sample_id + shapelet_id + span + scope + omega` 后调用 what-if。
- Part C -> Part E:
  - 优先传入 `PartCToELink`。
  - 若 link 未提供 span，E 侧可按 `peak_t + L_p` 推导默认 span。
- Part E -> Part C:
  - 回跳应携带 `sample_id + shapelet_id + span + scope + omega`，用于复核匹配一致性。

## 3. 一致性与性能验收
- 一致性:
  - 同一 `dataset + sample_id + shapelet_id + span + scope + omega + baseline + seed` 下，结果必须一致。
  - `delta` 计算口径固定，不允许前后端各自定义。
- 性能:
  - `P95 <= 1.0s`（`T<=2000, P<=64, D<=16`, 单样本 what-if, 命中缓存）
  - `P99 <= 1.5s`（同上条件）
