# 系统信息架构

## PartA:Data Panel ：数据与预测（Input → Black-box Output）

### 展现

#### 原始时序展示

##### 按特征做聚类，用户设定聚类数量展示

#### 显示时序元信息：采样率：1 step = h、min、长度：T_total 训练参数元信息： shaplet_num\shaplet_len，设置之后点击训练，后端模型运行，给出新的运行结果

##### 模型训练运行时展示Lcls（分类损失） / Lmatch（匹配/原型贴合损失） / Ldiv（多样性损失）来调节模型参数 理想状态：Lcls 稳定下降，Lmatch 同步下降，Ldiv 下降后趋稳。

#### 黑盒预测类别(按照训练参数元信息)

##### 显示运行分类的评价指标

##### 按分类结果类别可视化（原始序列）

##### 显示置信度和margin（margin 越小表示越“纠结/接近决策边界”），用其他可视化设计呈现出margin小的样本形态

###### 接口

###### 是否可以取这里的参数矩阵可视化

### 交互

#### 在具体样本上切时间窗来更改shaplet_len

#### 保存当前样本为“分析会话”（后续页面联动）

### 对应 ShapeX模型 输入 x[t]（原始序列/预处理后的序列） 分类器输出：logits / softmax 概率 

## Part B:Shapelet Library Panel

### 展现

#### Shapelet 列表（gallery）

##### 读取checkpoint文件里保存的shaplet，波形展示

##### 每个样本得到对应每个shaplet的一维的激活度曲线，累积得到activation map矩阵，展示对每个shapelet激活值的分布直方图，并让用户根据这个分布来自行设置阈值

###### 测试集设置的阈值决定了哪些被划分为重要的点或片段

###### shapex的实现只保留了一个shaplet的segment；具体来说合成数据集是只保留一个，mitecg是异常类进行sum处理

##### 平均触发次数

######  触发点变成连续区间段（segments），展示某shaplet在某序列上的平均触发次数，区分偶尔触发一次还是经常反复出现的周期性模式

##### 覆盖的类别分布：判断某个shaplet是否具有类相关性

###### 避免数据本身类别不平衡带来的偏差 

###### 设定阈值之后，某个shaplet不一定出现在每个序列里，进行统计

#### 交互调节shaplet

##### 根据交互调节后的形状计算新的数据点，作为新shaplet,使用训练集中所有相似片段作为集合，提供先验参考

### 交互

#### 点选 shapelet：联动到 Page C 高亮其在样本上的激活位置与证据片段

#### 相似度阈值：筛掉冗余 shapelets / 合并候选（工程上用于库维护）

### 补充

#### 出现频率

##### 全局出现频率：回答“普遍性/常见性”

###### 

##### 分类别出现频率：回答“类相关性/判别性”

###### 

##### 

### 问题，这个阈值和模型shapex有什么关系，shapex是通过activation  map进行分类的，从这个过程里是否可以实现可视化

### 对应 ShapeX模型哪部分 shapelet 原型参数（dictionary / prototypes） shapelet-to-class 的全局统计（不是模型层，而是基于模型输出/匹配结果做的解释统计） 

## Part C:Match & Locate Panel：匹配与定位（Activation Map + Detected Evidence）

### 展现

#### 主视图：原始时序 + 叠加高亮shaplet（候选证据区）

#### 副视图 1：Shapelet × 时间 的 activation heatmap

##### 可选shaplet数量

### 交互

#### hover heatmap 单元：主折线图可以同步定位局部窗口

#### shapelet 多选：同时显示多个 shapelets 的证据片段

#### 可选对比：softmax activation A vs conv/raw similarity I

### 目的

#### 模型“到底在看哪里”？

##### 通过 heatmap + hover 联动到原始曲线，看到： 1.哪些时间位置出现了强匹配（亮带） 2.这些亮带对应原始曲线上的什么形态（峰/谷/阶跃/趋势） 

#### shapelet 匹配是稳定还是“到处都像”

##### softmax vs raw 切换帮助判断： 亮点是“真实高相似度”还是“softmax 强行挑出来的相对高点” 这能帮助你发现： 某些 shapelet 其实不具备判别力（raw 没明显峰，但 softmax 仍会制造对比） 某些 shapelet 过于噪声敏感（raw 很乱） 

#### 这里画的是单个样本的shaplet,可以看多个shaplet

##### 入口

###### 从 PartA 进入 Part C 你在 Part A 选定一个 sample（比如 Sample #42），点击“解释/查看匹配” → Part C 载入这条样本的原始序列。 这条样本就是分类输入，所以左侧显示它。

###### 从 Part B 进入 Part C 你在 Part B 点某个 shapelet 的“Top hits examples” → 选择一个触发最强的样本 → 跳 Part C 并自动 pin 那个 shapelet。 这时左侧依然是该样本的原始序列，但焦点已经锁定到某个 shapelet。

###### 从 Part E 回跳（复核） 用户在 Part E 发现某个贡献大的 segment，对应某个 shapelet/时间段 → 回到 Part C 去看“匹配定位是否真的合理”。

### 对应 ShapeX模型哪部分 测试集shapelet matching 相似度：I[p][t] activation 归一化：softmax over time 得到 A[p][t] detector 输出：peak_t（以及由 L 得到的窗口） 

### 做what-if的并非shaplet,而是shaplet相似度计算出的激活片段

## Part D: Players Builder Panel：Shapelet-driven 分段（Segments as Explanation Units）

### 展现

#### 由阈值 Ω 把每个 shapelet 的激活曲线切出 run：A[p][t] ≥ Ω → intervals

#### 多 shapelets 的 union：T(G′)（用于 coalition/扰动的集合定义）

#### segment 列表：长度、覆盖区间、对应 shapelet、激活强度摘要

### 交互

#### Ω 阈值滑条：实时重算 segments（解释粒度控制）

#### segment 合并/拆分建议（当不同 shapelets 高度重叠或碎片化时提示）

### 目的：把 Page C 的“匹配热力图”转换成后续因果/归因所需的“可控单位”：players（segments）

### 对应 ShapeX/SHAPEX 模型哪部分 从 A[p][t] 到 segments 的后处理（阈值化 + 连通段提取） players = shapelet evidence units（SHAPEX 的核心建模选择之一：解释单元来自 shapelet 触发） 

## Part E:Perturbation & Shapley Panel：干预与归因（SDSL Perturbation → Shapley → Saliency/CATE）

### 展现

#### 原始序列 ，以及 Part D 得到的一组 players（shapelet-aligned segments）

##### UI 做的事：勾选一组 segments（一个 coalition G′）

##### 在被勾选的 segments 内：保留原始 x[t] 在未勾选的部分：用 线性插值 填充

##### 通过勾选不同segments,获得信息：在各种可能的上下文 S（别的 segments 已经保留的集合）下，再加入 segment i 会让预测平均改变多少？

##### 展示 P(original)：模型在原始输入序列 xxx 上，对目标类别（或正类）的预测概率 P(what-if kept set)：勾选的一组 segments（coalition G′G'G′）“被保留”，其他时间点都被替换成 baseline 后，模型的预测概率。 Δ=P(G′​)−P(x)。Δ > 0：保留的这组 segments 让模型更倾向目标类                         Δ < 0：你保留的 segments 反而削弱目标类

#### 原始曲线 vs perturbed 曲线两条叠在一起，同时显示预测概率变化 Δ。

#### Shapley（φ + stderr） permutations 采样：随机顺序加入 player，记录边际增量平均得 φ stderr：边际增量样本的标准误（估计稳定性） 

#### Saliency 回填 segment-level：φ_i timestep-level：把 |φ_i| 分配到 span（sum 或 max 聚合，作为可选项） 

### 交互（核心）

#### segment 开关（what-if）：用户手动选择保留/屏蔽某些 segments，实时查看预测概率变化

#### CATE 口径生成：“在上下文 G′ 下，保留 segment n 平均使 f(X) 变化 Δ（概率/logit）”

### 目的

#### 先做“集合级证据测试”：只保留这些证据，模型还坚持这个分类吗？（what-if, Δ）

#### 再做“归因分摊”：到底哪个 segment 真正在推动预测？（Shapley φ）

#### 最后把 φ 回填到时间轴形成 saliency，用户能把解释再映射回原始现象。

### 对应 ShapeX/SHAPEX 模型哪部分 SDSL perturbation / baseline：定义 v(S)=f(x_S) 的关键（解释的“可验证性”来源） Shapley value estimation：segment 贡献（φ） saliency mapping：segment → time 的可视化回填（解释输出形式） 

## 页面间的数据接口

### Sample：id, x[t], meta Shapelet dict：shapeletId → prototype, length L Match tensor：I[p][t], A[p][t], peak_t[p] Players：playerId, shapeletId, span, peakScore Explain result：coalition G′, x_G′, P(original), P(what-if), Δ, φ, stderr, Rt 

# Workflow

## Data Panel 选样本、看原始时序、看元数据

### 模型看的是什么数据？这批数据长什么样？要选哪个样本继续分析？

## Shapelet Library Panel shapelet 列表、激活分布直方图、阈值 Ω、类分布/lift 

### 模型学到了哪些 shapelet？哪些 shapelet 有区分力/偏向哪个类？

## Match & Locate Panel 原始序列 + heatmap(A/I切换) + detected segments + brush→top hits

### 在单个样本里：哪些 shapelet 在哪里匹配？匹配分数可信么？

## Players Builder Panel 从 A[p][t]≥Ω 得到每个 shapelet 的 evidence segment（players），并管理/筛选

### 把“匹配结果”变成可解释单元（players = shapelet-aligned segments）

## Perturbation & Shapley Panel what-if（keep set）、Δ、Shapley φ/stderr、saliency 回填

### 通过可控干预（what-if）+ Shapley 归因，得到可验证 saliency ﻿

