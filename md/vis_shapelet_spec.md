# Shapelet 可视分析系统开发规格（PRD + API + 算法口径）

## 0. 文档信息
- 版本: `v0.1-template`
- 状态: `Draft -> Ready for Dev`
- 项目路径: `d:/shapelet/ShapeX-new`
- 关联文档: `vis_shapelet.md`
- 适用数据集: `mitecg | mcce | mcch | mtce | mtch`（可扩展）
- 读者: 算法、后端、前端、测试

## 1. 目标与范围（PRD）

### 1.1 产品目标
- 目标1: 基于 ShapeX/ProtoPTST 输出，提供样本级可解释分析闭环。
- 目标2: 支持从“模型预测”到“证据定位”到“干预验证（what-if）”到“Shapley 归因”的完整链路。
- 目标3: 保证解释结果可复现（固定随机种子、固定口径、可回放会话）。

### 1.2 MVP 范围（首版）
- 包含: `Part A + Part C + Part E` 的可用闭环（v1 暂不依赖 Part D）。
- 包含: `Part B` 的只读能力（shapelet 浏览、分布统计、阈值调节）。
- 不含: 在线训练和在线编辑 shapelet 原型（只加载 checkpoint）。
- 不含: 多用户权限体系（先单用户/本地）。

### 1.3 非目标（当前版本不做）
- 不做模型结构搜索与自动调参。
- 不做跨项目统一解释平台。
- 不做大规模分布式计算（先单机 GPU/CPU）。

### 1.4 关键用户问题
- Q1: 模型在该样本上“看到了哪里”？
- Q2: 哪些 shapelet 对某类别有区分力？
- Q3: 某段证据被保留/屏蔽后，预测如何变化？
- Q4: 贡献分解是否稳定（Shapley stderr 是否可接受）？

### 1.5 验收标准（MVP）
- A. 样本加载后，2 秒内返回基础预测与 margin。
- B. 指定样本可展示 `I[p,t]` heatmap 与对应证据高亮。
- C. 调整阈值 `Omega` 后，1 秒内完成单一 shapelet/segment 的 what-if 可视验证准备（`T <= 2000, P <= 64`）。
- D. what-if 单次评估 1 秒内返回 `P(original), P(what-if), delta`。
- E. v1 不强制交付 Shapley；若启用实验开关，需返回采样配置与随机种子。

## 2. 页面需求拆解（PRD）

## 2.1 Part A: Data Panel
- 输入:
  - 用户传入的数据文件 `dataset_file`
  - 初版支持从内置数据集选择: `mitecg | mcce | mcch | mtce | mtch`
  - 聚类参数: `cluster_k`
  - 输出:
  - 数据集元信息（只读）: `sampling_rate`, `seq_len`
    - 其中 `sampling_rate` 表示时间粒度字符串，当前约定为 `hour|min|day|second|unknown`，含义是 `1 step = ?`
  - 训练元信息（只读）: `shapelet_num`, `shapelet_len`
  - 训练集可视化:
    - 聚类前先对每条训练序列做 z-normalized
    - 将归一化后的序列展平后做 `kmeans`
    - 可视化时不画 PCA 散点图，而是对每个簇回到原始序列空间画“中心线 + 分位带”
    - 每簇同时提供 KMeans 的簇中心线（centroid）和中位数曲线，并用 25%~75% 分位带（IQR）表示簇内波动
  - 测试集分类结果指标展示: `acc/f1/auc`（数据集级）
  - 测试集可视化: 按类别展示测试集样本分布/样本列表，并对低 `margin` 样本提供特殊可视化
  - 测试集深度可视化（新增）:
    - 以模型预测类别 `pred_class` 为分组维度
    - 每组输出 PID 深度分数 `depth`、最大深度代表曲线、50% 中心区域
    - 深度算法在后端统一计算；前端仅消费接口结果并绘图
  - 接口拆分原则:
    - 首屏同步只加载基础信息与 warning
    - 测试集指标、训练集聚类、低 margin 列表、按类样本列表均采用异步加载
    - 深度可视化通过独立异步接口按 `pred_class` 按需加载
    - 列表类接口必须支持分页
- 交互:
  - 选择内置数据集或上传数据文件
  - 调整聚类类别数 `cluster_k`
  - 选择样本
  - 在训练集聚类视图与测试集类别视图中联动高亮样本
  - 切时间窗建议 `shapelet_len`（MVP 只记录，不触发在线训练）
  - 保存分析会话
- 验收:
  - 同一 `dataset_file/internal_dataset + model_ckpt + cluster_k` 下，训练集聚类结果一致
  - 同一 `sample_id + model_ckpt` 的预测结果一致
- MVP 备注:
  - `dataset_file` 在 PRD 中保留，但首版后端暂不落地；首版仅支持内置数据集选择
  - `Part A` 不再提供单一重型 `/overview` 接口；首版前端应按 `meta -> metrics / clusters / lists -> sample detail` 的分层请求流实现
- 已知问题:
  - 对 `mcce | mcch | mtce | mtch`，Part A 首版后端严格复用当前 `get_saliency_data(return_dict=True)` 的返回口径，因此 `TRAIN/TEST` 与底层原始 split 的语义存在映射反转问题；接口响应中必须显式返回该提示
  - 对 `mcce | mcch | mtce | mtch`，分类 checkpoint 的实际输出类别数为 `4`，而当前 yaml 仍保留 `num_classes: 2` 这一另一种选择；Part A 首版以后端实际推理所依赖的 checkpoint 输出为准

## 2.2 Part B: Shapelet Library Panel（MVP 只读）
- 输入:
  - checkpoint 中 shapelet/prototype
  - 匹配分数 `I[p,t]`（模型原生输出）
  - 触发定义（冻结）:
    - `trigger_{n,p}(Omega) = 1{max_t I_{n,p,t} >= Omega}`
    - 含义: 对样本 `n` 的 shapelet `p`，只要存在任一时间点达到阈值即记为触发
  - 统计范围 `scope in {test, train, all}`（默认 `test`）
    - `train`: 用于观察模型学习到的模式，允许偏乐观
    - `test`: 用于默认展示与对外解释，代表泛化表现
    - `all`: 用于探索分析，不作为默认评估口径
  - 统计粒度 `granularity in {sample, time}`（概念口径，v1 不作为 API 请求参数）
    - `sample`（样本级）: 回答“有多少样本触发该 shapelet / 与类别关系如何”
    - `time`（时间点级）: 回答“shapelet 在时间轴的活跃区间分布”
- 输出:
  - shapelet gallery（静态，阈值无关）:
    - `shapelet_id`, `prototype`, `shapelet_len`, `ckpt_id`（可附 `sample_ids_preview`）
  - shapelet stats（动态，阈值相关）:
    - 全局触发率 `global_trigger_rate`
    - 按类触发率 `class_trigger_rate`
    - 触发次数统计、类别覆盖统计（含类别不平衡修正）
    - `I` 分布直方图（默认单个 shapelet 视图）
    - 统计口径回显: `scope`, `omega`（`granularity` 在 v1 由接口类型隐式表达，不单独回显）
  - 接口拆分原则（禁止重型总接口）:
    - 首屏仅加载 `meta`（轻量、阈值无关）
    - `gallery list` 与 `gallery detail` 分开；列表必须分页
    - `stats summary`、`histogram`、`class stats` 分开按需异步请求
    - `top-hits samples` 作为 B->C 联动专用接口，分页按需请求（唯一正式样本来源）
    - 阈值 `omega` 或 `scope` 变化时，仅刷新 `stats` 相关接口与 `top-hits samples`，不重拉 `gallery/meta`
    - 不提供单一 `/overview` 或等价“全量打包”接口
  - 建议请求流:
    - `meta -> gallery list -> (shapelet detail + stats summary + histogram + class stats + top-hits samples)`
- 交互:
  - 选择 shapelet 后，先拉取该 shapelet 的 `top-hits samples`，再选择样本进入 Part C
  - 阈值预览（边界冻结）:
    - 不触发训练流程
    - 不修改模型参数与 checkpoint
    - 仅影响 Part B / Part C / Part E 的统计与可视化结果
  - 支持切换 `scope`（默认 `test`）
  - `I` 直方图支持“单个 shapelet / 全局汇总”视图切换（默认单个 shapelet）
  - B->C 联动契约（冻结）:
    - 必带字段: `dataset`, `sample_id`, `shapelet_id`, `scope`, `omega`, `source_panel='part_b'`
    - 可选字段: `trigger_score`, `rank`, `rank_metric`
    - 禁止仅凭 `shapelet_id` 跳转 Part C，必须携带具体 `sample_id`
    - `sample_id` 必须来自 `top-hits samples` 接口；`sample_ids_preview` 仅用于预览展示
- 验收:
  - 可追溯性: 任一 `shapelet_id` 的统计结果可回溯到样本 ID 列表与计数过程
  - 一致性: 同一 `dataset + ckpt + scope + omega + seed` 下 `global_trigger_rate / class_trigger_rate / lift` 结果一致（误差 0）
  - 首屏时延: `meta` 与 `gallery list` 在 1.0 秒内返回（P95，本地单机基线）
  - 阈值交互时延: 仅重算 `stats` 时 500ms 内返回（P95，`T<=2000, P<=64`）
  - 分页正确性: `offset/limit` 下无重复、无漏项，排序稳定
  - 稳健性: 当 `N_{p,trig} < min_support` 时 `lift=null` 且返回 warning
  - 联动正确性: `top-hits samples` 可稳定返回可跳转样本；B->C 跳转参数满足冻结契约
- MVP 实现口径:
  - 采用“预计算 + 实时聚合”混合策略:
    - 预计算（或缓存）模型前向得到的 `activations(I)`（按 `scope` 分开）
    - 可额外缓存 `max_t I_{n,p,t}` 以加速触发率统计
    - `trigger` 不预存固定结果；在请求时按当前 `omega` 计算
    - 滑动 `omega` 时仅做阈值化与聚合重算，不重复模型前向
  - 目标: 保持交互响应速度，同时保证统计口径一致
- 已知问题:
  - 不同数据集的 `I` 数值范围可能差异较大；跨数据集直接比较同一 `omega` 不具可比性
  - `scope=train` 统计可能偏乐观，仅用于诊断，不作为默认对外结论
  - 小样本类别下 `class_trigger_rate/lift` 方差较大；当前通过 `alpha=1.0` 与 `min_support=20` 缓解，但仍需 warning 提示
  - `global` 直方图会掩盖单个 shapelet 差异，默认应使用 `per_shapelet` 视图
  - `sample_ids_preview` 仅为轻量预览字段，不作为 B->C 正式跳转数据源
  - 历史固定阈值常量路径（`shapeX.py` 中按数据集分支使用 `0.5/0.4`）已废弃；v1 统一由全局 `omega` 驱动 B/C/E

- 默认值来源（冻结）:
  - `omega_default` 的单一真源为解释配置（`explain.omega`，可按数据集覆盖）
  - `omega_default` 不得复用 `margin_threshold` 或其他分类指标阈值

## 2.3 Part C: Match & Locate Panel
- 输入: 单样本 `x[t,d]`, `I[p,t]`, `peak_t[p]`
- 输出:
  - 主图: 原始序列 + 证据高亮
  - 副图: `shapelet x time` heatmap
- 页面入口与联动优先级（冻结）:
  - 主入口: `Part A -> Part C`（先选样本再看匹配定位）
  - 次入口: `Part B -> Part C`（先选 shapelet，再带样本进入定位）
  - 复核入口: `Part E -> Part C`（从高贡献 segment 回看匹配合理性）
- MVP 口径（冻结）:
  - v1 不支持后端返回 `A`（softmax activation）；Part C 仅使用并展示 `I`
  - 若前端需要“相对强度”视图，可在前端基于 `I` 做派生归一化显示，但该结果不作为模型原生输出口径
  - 证据高亮阈值使用全局 `omega`（与 Part B/Part E 统一），不提供 Part C 局部覆盖阈值
  - 时间定位按底层实现采用“中心对齐”:
    - `peak_t` 是 center-aligned 索引，不是窗口起点
    - 对应 shapelet 长度 `L_p` 的高亮窗口可按 `start = peak_t - floor(L_p/2)`, `end = start + L_p - 1` 计算，并在 `[0, T-1]` 内裁剪
    - `L_p` 来源优先级（冻结）:
      - 优先使用 `match` 响应携带的 `shapelet_len/shapelet_lens`（推荐）
      - 若 `match` 未携带，则回退到 `shapelet detail/meta` 读取
- Part B -> Part C 联动契约（冻结）:
  - 必带字段: `dataset`, `sample_id`, `shapelet_id`, `scope`, `omega`, `source_panel='part_b'`
  - 可选字段: `trigger_score`, `rank`, `rank_metric`
  - 约束: Part B 不能仅凭 `shapelet_id` 直接跳转；必须携带具体 `sample_id`（例如 top-hit sample）
- 跳转后的默认行为:
  - 自动加载 `sample_id` 对应样本并 pin `shapelet_id`
  - C 页沿用 B 传入的 `scope` 与 `omega`
  - 首次进入 C 时自动定位到该 `shapelet_id` 的 `peak_t`
- 交互:
  - hover 热力图单元定位原始曲线
  - 多选 shapelet 同屏展示
- 验收:
  - 交互定位误差不超过 1 个时间步
  - 同一 `dataset + sample_id + shapelet_id + scope + omega` 下，Part B 触发结论与 Part C 高亮结果一致

<!--
## 2.4 Part D: Players Builder Panel
- 输入: `I[p,t]` 和阈值 `Omega`
- 输出:
  - 每个 shapelet 的 segments（players）
  - 多 shapelet 并集 `T(G')`
  - segment 列表（区间、长度、强度摘要）
- 交互:
  - `Omega` 滑条实时重算
  - 合并/拆分建议（基于重叠率与最短长度）
- 验收:
  - 同一输入和阈值下 players 完全可复现
-->

## 2.5 Part E: Perturbation Panel（v1: 无 Part D 依赖）
- 输入:
  - 必填请求字段: `dataset`, `sample_id`, `shapelet_id`, `span=[t_start,t_end]`, `scope`, `omega`
  - 可选请求字段: `baseline`（默认 `linear_interp`）, `value_type`（默认 `prob`）, `target_class`（默认不传）
- `span` 定义（冻结）:
  - `span=[t_start,t_end]` 是时间索引闭区间（含两端），单位为当前序列采样点 index
  - 合法约束: `0 <= t_start <= t_end <= T-1`
  - 越界处理: 服务端先裁剪到 `[0,T-1]` 再执行，并在响应回显裁剪后的区间
- 输出:
  - perturbed 序列（或可选返回其摘要）
  - `P(original), P(what-if), delta`
  - what-if 上下文回显（`dataset`, `sample_id`, `shapelet_id`, `t_start`, `t_end`, `scope`, `omega`, `baseline`, `value_type`, `seed`）
  - 类别信息:
    - 返回 `pred_class_original/pred_class_whatif`
    - 若样本有标注则返回 `y_true`（仅展示，不作为 `target_class` 默认值）
    - 若请求携带 `target_class`，额外返回 `delta_target`
- segment 来源规则（冻结）:
  - 优先使用前端明确传入的 `segment/span`
  - 若未传入，则使用 Part C 当前 pin 的 `shapelet_id + peak_t + L_p` 推导窗口
  - v1 不依赖 Part D 的 players/coalition 生成
  - 兼容关系: 若底层或离线流程已得到多段 `players={g_1,...,g_n}`，v1 可只选其中一段 `g_i` 执行 what-if（等价于 `G'={g_i}`）
- 冲突处理（冻结）:
  - 当 `shapelet_id` 与 `span` 不一致（例如不覆盖该 shapelet 的 `peak_t`）时，默认继续执行并返回 warning `ERR_SHAPELET_SPAN_MISMATCH`
  - 当 `shapelet_id` 非法或 `span` 非法（空区间/反向）时，返回错误并拒绝执行
- 扰动算子（v1）:
  - `linear_interp`: 按通道独立插值
  - 端点使用 `x[t_start-1]` 与 `x[t_end+1]`；边界缺失时按单端延拓
  - 当 `t_start == t_end`（单点）时，按邻点均值或单端值退化处理
- 交互:
  - 单一 segment 开关 what-if（on/off）
  - baseline 切换（v1 至少支持 `linear_interp`）
  - 输出口径切换（`prob` 默认，可选 `logit`）
- 错误码（冻结）:
  - `ERR_INVALID_SPAN`
  - `ERR_SHAPELET_NOT_FOUND`
  - `ERR_SAMPLE_NOT_FOUND`
  - `ERR_SCOPE_MISMATCH`
  - `ERR_SHAPELET_SPAN_MISMATCH`（warning，不中断）
- 验收:
  - 时延指标（本地单机基线）:
    - `P95 <= 1.0s`（`T<=2000, P<=64, D<=16`, 单样本 what-if, 命中缓存）
    - `P99 <= 1.5s`（同上条件）
  - 同一 `dataset + sample_id + shapelet_id + span + baseline + seed` 结果可复现

## 3. 算法口径（统一定义）

## 3.1 符号与张量形状
- 输入序列: `x in R^{T x D}`
- shapelet 字典: `S in R^{P x L x D}`
- 匹配分数: `I in R^{P x T}`（超界时间步按 padding 或忽略）
- 峰值位置: `peak_t[p] = argmax_t I[p,t]`，且 `peak_t` 定义为 center-aligned 时间索引（不是窗口起点）
- `L_p`: 第 `p` 个 shapelet 的长度，用于把 `peak_t[p]` 映射为可视化高亮窗口
- segment（v1）: `segment = (shapelet_id, t_start, t_end)`

## 3.2 匹配分数 I（建议口径）
- 当前代码口径:
  - `I` 直接取底层 shapelet matcher 的输出，即模型前向传播返回的 `activations`
  - 在 `use_shapelet_layer=True` 时，`I` 来自 `LearningShapeletsSeg`
  - 在 `use_shapelet_layer=False` 时，`I` 来自 `conv1d + LayerNorm + LeakyReLU`
- 统一约定:
  - `I` 是“模型原生匹配分数”，不是额外定义的纯原始距离
  - API 只暴露 `I`，不再单独暴露 `A`
  - Part C 的热力图与证据高亮以 `I` 为唯一后端数据源
- 默认配置:
  - `dist_measure='cosine'`
  - `shapelet_znorm=True`
  - `shapelet_temperature=1.0`
- 兼容说明:
  - 若后续更换底层匹配实现，只要求 `I[p,t]` 满足“值越大表示匹配越强”

<!--
## 3.3 从 I 到 players（Part D）
- 二值化:
  - `B[p,t] = 1{I[p,t] >= Omega}`
- 连通段提取:
  - 对每个 `p`，从 `B[p,:]` 提取连续 1 区间
- 过滤:
  - 最短长度 `min_len`
  - 可选填洞 `fill_gap_len`
- 重叠处理:
  - `IoU >= merge_iou` 可合并
- 输出:
  - `players` 和并集时间集合 `T(G')`
-->

## 3.3 v1 单 segment 选择口径（Part E）
- 目标:
  - 在不依赖 Part D 的前提下，为 what-if 提供单一可复现的干预区间
- 选择对象:
  - 单个 `shapelet_id`
  - 单个时间区间 `segment = [t_start, t_end]`
- 默认推导（当未显式给出 span）:
  - 由 Part C 的 `peak_t` 与 `L_p` 推导:
  - `t_start = peak_t - floor(L_p/2)`
  - `t_end = t_start + L_p - 1`
  - 边界裁剪到 `[0, T-1]`
- 约束:
  - v1 一次 what-if 仅允许一个 segment
  - 多 segment/multi-shapelet coalition 保留到后续版本
  - 与 players 口径兼容: 单 segment 模式是多 player 集合干预的单元素特例，不改变底层 `I` 与预测模型定义

## 3.4 Part B 统计输入口径（冻结）
- 输出分层约束:
  - `gallery` 仅包含阈值无关字段
  - `global_trigger_rate / class_trigger_rate` 仅在 `stats` 中返回，且必须依赖并回传 `omega`
- 统计范围:
  - 默认 `scope = test`
  - `scope = train` 仅用于训练行为诊断
  - `scope = all` 仅用于探索，不作为默认对比或验收口径
- 统计粒度（概念口径，v1 不作为 API 参数）:
  - 样本级统计（默认主指标，对应 `summary/classes` 接口）:
    - 触发定义: `trigger_{n,p} = 1{max_t I_{n,p,t} >= Omega}`
    - 用于计算触发率、类别覆盖、lift 等
  - 时间点级统计（默认细节视图，对应 `histogram` 接口）:
    - 使用 `I[p,t]` 在时间轴上的分布，用于直方图/热区展示
- 计算策略:
  - 底层前向阶段只负责产出 `activations(I)`，不固定产出某个 `omega` 下的 trigger 统计
  - 不在 `omega` 交互时重复模型前向
  - 通过预计算或缓存的 `I`（可含 `max_t I`）在服务端做阈值化和聚合
  - `trigger` 由服务端按请求参数实时计算: `trigger_{n,p}(Omega) = 1{max_t I_{n,p,t} >= Omega}`
  - `omega` 调整属于解释后处理: 不触发训练、不更新模型权重，仅重算 B/C/E 的统计与可视化派生结果
  - 响应中必须回传 `scope`、`omega` 以保证可追溯；`granularity` 在 v1 由接口类型隐式确定
- 类别覆盖与不平衡修正（冻结）:
  - 记号:
    - `N`: 当前 `scope` 的样本数
    - `C`: 类别数
    - `N_c = Σ_n 1(y_n = c)`
    - `N_{p,trig} = Σ_n 1(trigger_{n,p}=1)`
    - `N_{p,c} = Σ_n 1(trigger_{n,p}=1 and y_n=c)`
  - 统计量:
    - 全局触发率: `global_trigger_rate(p) = N_{p,trig} / N`
    - 按类触发率: `class_trigger_rate(p,c) = N_{p,c} / N_c`
    - 类别覆盖率: `class_coverage(p,c) = N_{p,c} / N_c`
  - 不平衡修正（lift）:
    - 先验: `P(y=c) = (N_c + alpha) / (N + C*alpha)`
    - 条件概率: `P(y=c | trigger_p=1) = (N_{p,c} + alpha) / (N_{p,trig} + C*alpha)`
    - `lift(p,c) = P(y=c | trigger_p=1) / P(y=c)`
  - 默认稳健参数:
    - `alpha = 1.0`（Laplace 平滑）
    - `min_support = 20`；当 `N_{p,trig} < min_support` 时 `lift` 标记为不稳定（返回 `null` 并附 warning）
- `I` 直方图口径（冻结）:
  - 视图模式:
    - 默认 `mode = per_shapelet`（单个 shapelet 直方图）
    - 可切换 `mode = global`（全局汇总，所有 shapelet 合并）
  - 统计对象:
    - `per_shapelet`: 使用当前 `shapelet_id` 的 `I_{n,p,t}`
    - `global`: 使用当前 `scope` 下所有 `I_{n,p,t}`
  - 默认参数:
    - `bins = 50`
    - 范围默认按当前统计对象的最小值/最大值确定（可通过参数覆写）
    - 归一化默认开启（density）
  - 响应回显:
    - 必须回传 `hist_mode`, `bins`, `range`, `density`, `scope`

## 3.5 扰动函数（SDSL what-if）
- 价值函数:
  - `v(g) = f(x_g)`，其中 `g` 为单一 segment 干预配置
- 序列构造:
  - 对 `t in [t_start, t_end]` 按规则执行干预（默认替换为 baseline）
  - 其他位置保留原值
- baseline 类型:
  - `linear_interp`（默认）
  - `zero`
  - `dataset_mean`
- 输出:
  - `P(original)`, `P(what-if)`, `delta = P(what-if)-P(original)`

## 3.6 Shapley（降级为 v1.1+）
- v1 不作为必做项:
  - 首版 Part E 仅覆盖单一 shapelet/segment what-if
  - 不要求在 v1 输出 `phi/stderr`
- v1.1+ 恢复条件:
  - 当 Part D players 稳定后，再恢复多 player 的 Shapley 估计

## 3.7 segment 归因回填到时间点（v1.1+）
- 依赖 Shapley 输出 `phi_i`，故不纳入 v1 范围

## 3.8 分类与 margin 口径
- `pred_class = argmax_c probs[c]`
- `margin = probs[top1] - probs[top2]`
- 低 margin 样本定义:
  - `margin <= margin_threshold`（默认 `0.1`）

## 3.9 Part A 训练集聚类展示口径
- 聚类输入:
  - 使用训练集原始序列 `x in R^{T x D}`
  - 对每条样本分别做 z-normalized，再作为聚类输入
  - 推荐口径:
    - `x_norm[t,d] = (x[t,d] - mean_d(x[:,d])) / std_d(x[:,d])`
    - 若某维 `std_d = 0`，该维按 `1` 处理，避免除零
  - 将 `x_norm` 展平成一维向量后输入 `kmeans`
- 聚类输出:
  - 输出簇编号 `cluster_id`
  - `cluster_k` 仅控制簇数，不改变后续预测逻辑
- 可视化口径:
  - 不使用 PCA/UMAP 等二维投影散点图作为 Part A 首版训练集主视图
  - 对每个簇，回到原始序列空间统计簇内样本
  - 每个簇返回一条 `centroid` 线，表示该簇成员在原始序列空间中的均值中心线
  - 每个时间点逐维统计:
    - 中位数 `median`
    - 25% 分位数 `q25`
    - 75% 分位数 `q75`
  - 前端展示:
    - 一条线画 `centroid`
    - 一条粗线画 `median`
    - 阴影带画 `[q25, q75]`
  - 该视图的目标是展示“这一簇典型形态 + 簇内波动”，而不是样本点在二维平面中的相对位置

## 3.10 Part A 测试集 PID 深度口径（新增）
- 目标:
  - 为测试集“按预测类别”的可视化提供统一深度排序与中心区域表达，替代前端本地计算。
- 分组:
  - 使用 `pred_class = argmax(probs)` 对测试样本分组。
- 输入:
  - 某一 `pred_class` 下的序列集合 `X in R^{N x T}`（v1 口径取单通道 `D=1`，多通道时使用第 1 通道）。
- 计算:
  - `mu_t = mean_i X_{i,t}`
  - `sigma_t = std_i X_{i,t} + 1e-6`
  - `p_{i,t} = NormalPDF(X_{i,t}; mu_t, sigma_t)`
  - `s1_i = mean_t p_{i,t}`
  - `s2_i = s1_i / mean_t NormalPDF(mu_t; mu_t, sigma_t)`
  - `depth_i = min(s1_i, s2_i)`
- 派生输出:
  - 代表曲线: `argmax_i depth_i` 对应样本序列
  - 50% 中心区域阈值: `median(depth)`
  - 中心区域边界: 取 `depth_i >= median(depth)` 的样本集合，逐时间点给出 `lower=min`、`upper=max`
- 工程约束:
  - 该算法由后端接口统一计算并返回；
  - 前端禁止自行复刻 PID 口径作为业务结果，避免口径漂移。
  - 绘图序列统一使用按样本时间轴 z-normalized 后的数据（与训练聚类口径一致）。
  - 细线背景不绘制全量样本，默认仅绘制均匀采样 10% 子集；全量样本仅用于列表与交互详情。

## 4. API 契约（v1）
- API 契约已拆分到独立文档: `vis_shapelet_api.md`
- 建议阅读顺序:
  - 先读本文件中的 PRD 与算法口径
  - 再读 `vis_shapelet_api.md` 中的 Schema 与 endpoints

## 5. 后端模块划分（建议）
- `loader`: 数据与 checkpoint 读取
- `matcher`: 计算 `I/peak_t`
- `segment_selector`: 维护单一 `shapelet_id + span`（可由 Part C 推导）
- `perturb`: 构造 `x_g` 与 `v(g)`（单 segment what-if）
- `shapley`: v1.1+（暂缓）
- `service`: API 编排与缓存
- `session_store`: 分析会话存储

## 6. 前端状态契约（建议）
- 全局状态:
  - `currentSample`
  - `currentShapelets`
  - `currentMatchScore = I`
  - `omega`
  - `currentSegment`
  - `whatifResult`
- 联动规则:
  - Part A 选样本 -> 刷新 C/E
  - Part B 进入 C 必须携带 `dataset + sample_id + shapelet_id + scope + omega`，C 自动 pin 对应 shapelet 并定位到 `peak_t`
  - Part E 回跳 C 时必须携带 `sample_id` 与目标 `segment/span`（可附 `shapelet_id`），用于匹配复核
  - Part E 切换 `omega` 或 `span` -> 重新执行单 segment what-if

## 7. 性能与工程约束
- 最大输入建议: `T<=4000`, `P<=128`, `D<=16`
- 缓存键: `sample_id + ckpt_id + similarity_type + shapelet_temperature + normalize`
- 并发策略:
  - what-if 与 shapley 支持任务队列
  - 提供任务状态查询接口（可在 v1.1 增加）

## 8. 测试计划（必须）
- 单元测试:
  - `I` 计算正确性
  - baseline 构造正确性
  - 单 segment what-if 可复现性（seed 固定）
- 集成测试:
  - A->C->E 全链路
  - session 保存后可回放一致
- 回归测试:
  - 不同数据集配置切换
  - 长序列与高 shapelet 数压力测试

## 9. 关键决策记录（v1）
- R1: `I` 在 v1 中定义为底层 shapelet matcher 直接输出的模型原生匹配分数；默认实现采用 `use_shapelet_layer=True` 和 `dist_measure='cosine'`；统一语义为“值越大表示匹配越强”
  - 后续改进空间: 可增加 `-L2`、cross-correlation 等替代口径，并提供跨口径对照实验，评估其对热力图、segment 选择与归因结果稳定性的影响
- R2: `shapelet_temperature (tau)` 在 v1 中全数据集统一为固定值；`Omega` 按数据集配置，初始值依据该数据集的 `I` 分布直方图确定
- R3: baseline 在 v1 中统一为 `linear_interp`
  - 后续改进空间: 可增加 `zero`、`dataset_mean`、局部样本库插值、类条件 baseline，并比较不同 baseline 对 what-if / Shapley 稳定性的影响
- R4: v1 的 Part E 仅交付单一 shapelet/segment what-if；Shapley 估计下沉到 v1.1+
  - 后续改进空间: 恢复 `prob/logit` 双口径 Shapley，对比不同口径下归因排序一致性
- R5: Part B 的类别不平衡修正在 v1 中采用 `lift`（`alpha=1.0` 的 Laplace 平滑，`min_support=20`）
  - 后续改进空间: 可补充 `PMI`、加权 lift、置信区间或显著性检验，用于减少小样本类别的统计波动
- R6: 在线训练完全移出首版；v1 只支持加载已有 checkpoint 做推理、匹配与单 segment what-if 解释计算
  - 后续改进空间: 可在后续版本增加异步训练任务、训练进度查询、训练结果版本管理与训练后自动刷新可视分析结果
- R7: 阈值统一策略
  - v1: 统一全局 `omega`，用于 Part B 统计、Part C 证据高亮、Part E what-if
  - v1: 历史固定阈值常量路径（`0.5/0.4`）不再作为业务口径，仅保留迁移期兼容说明
- R8: `granularity` 参数化策略
  - v1: `granularity` 仅作为概念口径，不作为 Part B API 的请求参数或回显字段
  - v1.1 目标: 评估是否引入显式 `granularity` 参数（或聚合接口）以统一样本级/时间点级统计入口

## 10. 默认配置（可直接落地）

```yaml
spec_version: v1
dataset: mitecg
matching:
  similarity_type: cosine
  normalize: znorm
  shapelet_temperature: 1.0
explain:
  omega: 0.02
whatif:
  baseline: linear_interp
  value_type: prob
  seed: 2026
margin_threshold: 0.1
```

## 11. 里程碑（建议）
- M1（1 周）: 离线计算链路跑通（`I -> single-segment what-if`）
- M2（1 周）: API v1 落地与缓存
- M3（1-2 周）: 前端联动与会话回放
- M4（1 周）: 测试补齐与性能调优

## 12. 开发前 Checklist
- [ ] 所有算法口径在本文件“第3章”确认并冻结
- [ ] API 字段命名冻结并生成接口 mock
- [ ] 至少 1 个数据集完成端到端 dry run
- [ ] `seed`、`config`、`ckpt_id` 写入 explain 结果
- [ ] 验收指标（响应时间、稳定性）可自动测试
