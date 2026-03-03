### 第一阶段：初始化与调度 (`runer.py`)

这是程序的入口。

1. **读取总任务单**：
程序首先加载 `configs/run_configs.yaml`。它会查看 `datasets` 列表（例如 `mcce, mcch`）以及开关 `train` (是否训练) 和 `test` (是否测试)。
2. **任务循环**：
程序开始遍历数据集列表。对于每一个数据集（比如 `mcce`）：
    - 调用 `build_config` 构建基础配置。
    - **实例化流水线**：初始化 `ShapeXPipline(config)`。
        - **关键动作**：此时会调用 `exp_saliency_general.py` 中的 `get_args`。它会尝试加载具体的数据集配置文件（如 `configs/mcce.yaml`），从而确定模型的超参数（如 `num_prototypes=2`, `prototype_len=100`）。

### 第二阶段：训练解释器 (`train_shapex`)

*(如果 `run_configs.yaml` 中 `train: True`，则执行此阶段)*

这一步的目标是训练 `ProtoPTST` 模型，让它学会“什么样的时间序列片段是重要的”。

1. **准备环境**：
`ShapeXPipline` 调用 `train_shapex()`。
    - 加载训练数据 (`get_saliency_data`)。
    - 初始化实验引擎 `Exp_Classification`。
2. **构建模型**：
初始化 `ProtoPTST` 模型。此时原型向量 (`prototype_vectors`) 被随机初始化。
3. **训练循环 (Epoch Loop)**：
    - **前向传播**：数据输入 `ProtoPTST`，计算出 `activations`（数据与原型的相似度）和 `outputs`（分类结果）。
    - **计算损失**：
        - **分类 Loss**：保证模型能分对类。
        - **原型 Loss (`seg_prototype_loss`)**：强迫原型向量长得像真实的训练数据片段。
    - **反向传播**：更新模型参数。
    - **早停 (EarlyStopping)**：如果验证集效果不再提升，提前结束训练并保存模型到 `checkpoints/`。

### 第三阶段：生成解释与评估 (`eval_shapex`)

*(如果 `run_configs.yaml` 中 `test: True`，则执行此阶段)*

这一步是 ShapeX 的核心：利用训练好的 `ProtoPTST` 和预训练的黑盒分类器来生成显著性图（Saliency Map）。

1. **加载双模型**：
    - **加载解释器 (`seg_model`)**：从硬盘加载刚才训练好的 `ProtoPTST` 模型。
    - **加载黑盒模型 (`class_model`)**：从 `checkpoints/classification_models/` 加载预训练好的 Transformer 或 CNN。这个模型是**冻结的**，只负责做裁判。
2. **计算显著性 (`ScoreSubsequences`)**：
程序跳转到 `shapeX.py` 进行核心计算。
    - **扫描 (`get_seg_unified`)**：`seg_model` 扫描整条时间序列，根据原型匹配程度，找出“嫌疑片段”。
    - **扰动验证 (`get_score_vector`)**：
        - 依次把这些“嫌疑片段”抹掉（Mask/Perturbation）。
        - 把残缺的数据喂给 `class_model`。
        - 如果 `class_model` 的预测发生了剧烈变化，说明这个片段非常重要。
    - **生成分数**：结合扰动结果和原型的注意力权重，生成最终的 Saliency Score。
3. **对比真值 (`ground_truth_xai_eval`)**：
    - 将生成的 Saliency Score 与数据集中自带的 `gt_exps` (Ground Truth) 进行对比。
    - 计算 **AUPRC**, **AUROC** 等指标。
4. **输出结果**：
    - 在控制台打印指标。
    - 将结果追加写入 `results.txt` 文件。