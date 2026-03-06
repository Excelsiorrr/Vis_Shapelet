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
- 包含: `Part A + Part C + Part D + Part E` 的可用闭环。
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
- C. 调整阈值 `Omega` 后，1 秒内重算 players（`T <= 2000, P <= 64`）。
- D. what-if 单次评估 1 秒内返回 `P(original), P(what-if), delta`。
- E. Shapley 估计返回 `phi + stderr`，并记录采样配置与随机种子。

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
  - 接口拆分原则:
    - 首屏同步只加载基础信息与 warning
    - 测试集指标、训练集聚类、低 margin 列表、按类样本列表均采用异步加载
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
  - 统计粒度 `granularity in {sample, time}`（默认主视图 `sample`）
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
    - 统计口径回显: `scope`, `granularity`, `omega`
  - 接口拆分原则（禁止重型总接口）:
    - 首屏仅加载 `meta`（轻量、阈值无关）
    - `gallery list` 与 `gallery detail` 分开；列表必须分页
    - `stats summary`、`histogram`、`class stats` 分开按需异步请求
    - 阈值 `omega` 变化时仅刷新 `stats` 相关接口，不重拉 `gallery/meta`
    - 不提供单一 `/overview` 或等价“全量打包”接口
  - 建议请求流:
    - `meta -> gallery list -> (shapelet detail + stats summary + histogram + class stats)`
- 交互:
  - 选择 shapelet（与 Part C 的联动契约暂列为留存项，待 Part C 实现时冻结）
  - 阈值预览（边界冻结）:
    - 不触发训练流程
    - 不修改模型参数与 checkpoint
    - 仅影响 Part B / Part C / Part D 的统计与可视化结果
  - 支持切换 `scope`（默认 `test`）
  - `I` 直方图支持“单个 shapelet / 全局汇总”视图切换（默认单个 shapelet）
- 验收:
  - 可追溯性: 任一 `shapelet_id` 的统计结果可回溯到样本 ID 列表与计数过程
  - 一致性: 同一 `dataset + ckpt + scope + omega + seed` 下 `global_trigger_rate / class_trigger_rate / lift` 结果一致（误差 0）
  - 首屏时延: `meta` 与 `gallery list` 在 1.0 秒内返回（P95，本地单机基线）
  - 阈值交互时延: 仅重算 `stats` 时 500ms 内返回（P95，`T<=2000, P<=64`）
  - 分页正确性: `offset/limit` 下无重复、无漏项，排序稳定
  - 稳健性: 当 `N_{p,trig} < min_support` 时 `lift=null` 且返回 warning
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
  - Part B -> Part C 联动字段仍为留存项，待 Part C 实现时冻结

## 2.3 Part C: Match & Locate Panel
- 输入: 单样本 `x[t,d]`, `I[p,t]`, `peak_t[p]`
- 输出:
  - 主图: 原始序列 + 证据高亮
  - 副图: `shapelet x time` heatmap
- 交互:
  - hover 热力图单元定位原始曲线
  - 多选 shapelet 同屏展示
- 验收:
  - 交互定位误差不超过 1 个时间步

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

## 2.5 Part E: Perturbation & Shapley Panel
- 输入: 原序列 + players + coalition `G'`
- 输出:
  - perturbed 序列
  - `P(original), P(what-if), delta`
  - `phi_i`, `stderr_i`, 回填 saliency
- 交互:
  - segment 开关 what-if
  - 选择归因口径（logit/prob）
- 验收:
  - 返回结果包含随机种子、采样次数、baseline 类型

## 3. 算法口径（统一定义）

## 3.1 符号与张量形状
- 输入序列: `x in R^{T x D}`
- shapelet 字典: `S in R^{P x L x D}`
- 匹配分数: `I in R^{P x T}`（超界时间步按 padding 或忽略）
- 峰值位置: `peak_t[p] = argmax_t I[p,t]`
- players: `player_i = (shapelet_id, t_start, t_end, score_summary)`

## 3.2 匹配分数 I（建议口径）
- 当前代码口径:
  - `I` 直接取底层 shapelet matcher 的输出，即模型前向传播返回的 `activations`
  - 在 `use_shapelet_layer=True` 时，`I` 来自 `LearningShapeletsSeg`
  - 在 `use_shapelet_layer=False` 时，`I` 来自 `conv1d + LayerNorm + LeakyReLU`
- 统一约定:
  - `I` 是“模型原生匹配分数”，不是额外定义的纯原始距离
  - API 只暴露 `I`，不再单独暴露 `A`
- 默认配置:
  - `dist_measure='cosine'`
  - `shapelet_znorm=True`
  - `shapelet_temperature=1.0`
- 兼容说明:
  - 若后续更换底层匹配实现，只要求 `I[p,t]` 满足“值越大表示匹配越强”

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

## 3.4 Part B 统计输入口径（冻结）
- 输出分层约束:
  - `gallery` 仅包含阈值无关字段
  - `global_trigger_rate / class_trigger_rate` 仅在 `stats` 中返回，且必须依赖并回传 `omega`
- 统计范围:
  - 默认 `scope = test`
  - `scope = train` 仅用于训练行为诊断
  - `scope = all` 仅用于探索，不作为默认对比或验收口径
- 统计粒度:
  - 样本级统计（默认主指标）:
    - 触发定义: `trigger_{n,p} = 1{max_t I_{n,p,t} >= Omega}`
    - 用于计算触发率、类别覆盖、lift 等
  - 时间点级统计（默认细节视图）:
    - 使用 `I[p,t]` 在时间轴上的分布，用于直方图/热区展示
- 计算策略:
  - 底层前向阶段只负责产出 `activations(I)`，不固定产出某个 `omega` 下的 trigger 统计
  - 不在 `omega` 交互时重复模型前向
  - 通过预计算或缓存的 `I`（可含 `max_t I`）在服务端做阈值化和聚合
  - `trigger` 由服务端按请求参数实时计算: `trigger_{n,p}(Omega) = 1{max_t I_{n,p,t} >= Omega}`
  - `omega` 调整属于解释后处理: 不触发训练、不更新模型权重，仅重算 B/C/D 的统计与可视化派生结果
  - 响应中必须回传 `scope`、`granularity`、`omega` 以保证可追溯
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
  - `v(S) = f(x_S)`，其中 `S` 为保留的 player 集合
- 序列构造:
  - `t in T(S)` 位置保留原值
  - 其他位置替换为 baseline
- baseline 类型:
  - `linear_interp`（默认）
  - `zero`
  - `dataset_mean`
- 输出:
  - `P(original)`, `P(what-if)`, `delta = P(what-if)-P(original)`

## 3.6 Shapley 估计
- player 集合: `N = {1..n}`
- 定义:
  - `phi_i = E_pi[ v(Pre_pi(i) U {i}) - v(Pre_pi(i)) ]`
- 估计:
  - 置换采样 `M` 次
  - 记录每次边际增量，输出均值与标准误 `stderr`
- 停止条件（建议）:
  - `M >= min_perm` 且 `max(stderr_i) <= stderr_tol`
  - 或达到 `max_perm`
- 可复现:
  - 固定 `seed`
  - 返回 `perm_count_actual`

## 3.7 segment 归因回填到时间点
- `saliency[t]` 由覆盖 `t` 的 `phi_i` 聚合
- 聚合方式:
  - `sum_abs`（默认）
  - `max_abs`
- 输出可视化:
  - 时间点热度 + segment 边界

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

## 4. API 契约（v1）
- API 契约已拆分到独立文档: `vis_shapelet_api.md`
- 建议阅读顺序:
  - 先读本文件中的 PRD 与算法口径
  - 再读 `vis_shapelet_api.md` 中的 Schema 与 endpoints

## 5. 后端模块划分（建议）
- `loader`: 数据与 checkpoint 读取
- `matcher`: 计算 `I/peak_t`
- `player_builder`: 阈值化与 segments 管理
- `perturb`: 构造 `x_S` 与 `v(S)`
- `shapley`: 置换采样与统计
- `service`: API 编排与缓存
- `session_store`: 分析会话存储

## 6. 前端状态契约（建议）
- 全局状态:
  - `currentSample`
  - `currentShapelets`
  - `currentMatchScore = I`
  - `omega`
  - `players`
  - `coalition`
  - `explainResult`
- 联动规则:
  - Part A 选样本 -> 刷新 C/D/E
  - Part B 点 shapelet -> C 高亮并 pin（留存项，待 Part C 实现时冻结字段与交互细节）
  - Part D 改 `omega` -> players 重算并清空旧 shapley

## 7. 性能与工程约束
- 最大输入建议: `T<=4000`, `P<=128`, `D<=16`
- 缓存键: `sample_id + ckpt_id + similarity_type + shapelet_temperature + normalize`
- 并发策略:
  - what-if 与 shapley 支持任务队列
  - 提供任务状态查询接口（可在 v1.1 增加）

## 8. 测试计划（必须）
- 单元测试:
  - `I` 计算正确性
  - segments 提取边界
  - baseline 构造正确性
  - shapley 可复现性（seed 固定）
- 集成测试:
  - A->C->D->E 全链路
  - session 保存后可回放一致
- 回归测试:
  - 不同数据集配置切换
  - 长序列与高 shapelet 数压力测试

## 9. 关键决策记录（v1）
- R1: `I` 在 v1 中定义为底层 shapelet matcher 直接输出的模型原生匹配分数；默认实现采用 `use_shapelet_layer=True` 和 `dist_measure='cosine'`；统一语义为“值越大表示匹配越强”
  - 后续改进空间: 可增加 `-L2`、cross-correlation 等替代口径，并提供跨口径对照实验，评估其对热力图、players 和归因结果稳定性的影响
- R2: `shapelet_temperature (tau)` 在 v1 中全数据集统一为固定值；`Omega` 按数据集配置，初始值依据该数据集的 `I` 分布直方图确定
- R3: baseline 在 v1 中统一为 `linear_interp`
  - 后续改进空间: 可增加 `zero`、`dataset_mean`、局部样本库插值、类条件 baseline，并比较不同 baseline 对 what-if / Shapley 稳定性的影响
- R4: Shapley 输出口径在 v1 中以前端展示友好的 `prob` 为主，后端保留 `logit` 作为可选参数
  - 后续改进空间: 可增加 `prob` 与 `logit` 双输出对照视图，并评估不同口径下归因排序的一致性
- R5: Part B 的类别不平衡修正在 v1 中采用 `lift`（`alpha=1.0` 的 Laplace 平滑，`min_support=20`）
  - 后续改进空间: 可补充 `PMI`、加权 lift、置信区间或显著性检验，用于减少小样本类别的统计波动
- R6: 在线训练完全移出首版；v1 只支持加载已有 checkpoint 做推理、匹配、players 构建与解释计算
  - 后续改进空间: 可在后续版本增加异步训练任务、训练进度查询、训练结果版本管理与训练后自动刷新可视分析结果

## 10. 默认配置（可直接落地）

```yaml
spec_version: v1
dataset: mitecg
matching:
  similarity_type: cosine
  normalize: znorm
  shapelet_temperature: 1.0
players:
  omega: 0.02
  min_len: 5
  fill_gap_len: 2
  merge_iou: 0.6
whatif:
  baseline: linear_interp
  value_type: prob
shapley:
  min_perm: 64
  max_perm: 512
  stderr_tol: 0.01
  seed: 2026
margin_threshold: 0.1
```

## 11. 里程碑（建议）
- M1（1 周）: 离线计算链路跑通（`I -> players -> what-if -> shapley`）
- M2（1 周）: API v1 落地与缓存
- M3（1-2 周）: 前端联动与会话回放
- M4（1 周）: 测试补齐与性能调优

## 12. 开发前 Checklist
- [ ] 所有算法口径在本文件“第3章”确认并冻结
- [ ] API 字段命名冻结并生成接口 mock
- [ ] 至少 1 个数据集完成端到端 dry run
- [ ] `seed`、`config`、`ckpt_id` 写入 explain 结果
- [ ] 验收指标（响应时间、稳定性）可自动测试
