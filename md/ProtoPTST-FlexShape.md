# ProtoPTST-ls 使用指南

## 目录
- [1. 模型对比：ProtoPTST-ls vs ProtoPTST](#1-模型对比protoptst-ls-vs-protoptst)
- [2. 已完成的代码修改清单](#2-已完成的代码修改清单)
- [3. 使用 ProtoPTST-ls 替换 ProtoPTST 的步骤](#3-使用-protoptst-ls-替换-protoptst-的步骤)
- [4. 配置文件示例](#4-配置文件示例)
- [5. 实验对比建议](#5-实验对比建议)

---

## 1. 模型对比：ProtoPTST-ls vs ProtoPTST

### 1.1 核心架构差异

#### **ProtoPTST (原始版本)**

**特点：**
- 使用 `nn.Parameter` 直接学习原型向量
- 通过一维卷积 (`F.conv1d`) 计算输入序列与原型的匹配度
- 简单的 LayerNorm + LeakyReLU 激活函数
- 初始化方式：Kaiming 或 Xavier

**代码实现：**
```python
# 使用可训练的 prototype_vectors 参数
self.prototype_vectors = nn.Parameter(
    torch.randn(self.prototype_shape), 
    requires_grad=True
)

# 通过 conv1d 计算距离
def convolution_distance_x_as_input(self, x, prototype_vectors):
    conv_out = F.conv1d(x, prototype_vectors, padding=..., stride=1)
    # LayerNorm + LeakyReLU 激活
    return conv_out
```

---

#### **ProtoPTST-ls (Learning Shapelets 版本)**

**核心创新：双模式设计**
- 通过 `use_shapelet_layer` 开关在两种方式间切换
- 模式 A：使用专门的 `LearningShapeletsSeg` 模块
- 模式 B：降级为原始 ProtoPTST 方式（完全向后兼容）

**代码实现：**
```python
# 控制开关
self.use_shapelet_layer = getattr(configs, 'use_shapelet_layer', False)

if self.use_shapelet_layer:
    # 模式 A：使用 LearningShapeletsSeg
    self.shapelet_layer = LearningShapeletsSeg(
        configs=configs,
        num_shapelets=configs.num_prototypes,
        shapelet_len=configs.prototype_len,
        in_channels=configs.enc_in,
        dist_measure='euclidean',  # 可选：euclidean, cosine, correlation
        znorm=True,                 # Z-normalization
        temperature=1.0             # Softmax 温度参数
    )
else:
    # 模式 B：原始方式
    self.prototype_vectors_param = nn.Parameter(...)
```

**新增特性：**
| 特性 | ProtoPTST | ProtoPTST-ls |
|------|-----------|--------------|
| **距离度量** | 仅 conv1d | 欧氏距离、余弦相似度、互相关 |
| **归一化** | 手动 LayerNorm | 自动 Z-normalization |
| **温度参数** | 无 | Softmax 温度控制 |
| **代码复用** | 重复逻辑多 | 模块化设计 |
| **消融实验** | 需改多处代码 | 仅需修改配置文件 |
| **向后兼容** | N/A | 完全兼容（默认模式） |

---

### 1.2 前向传播差异

#### **ProtoPTST**
```python
def forward(self, x_enc, ...):
    # 1. 处理原型
    prototype = self.prototype_layer(self.prototype_vectors)  
    
    # 2. 计算激活
    activations = self.convolution_distance_x_as_input(x, prototype)
    
    # 3. 分类
    output = self.projection(...)
    return output, activations, prototype
```

#### **ProtoPTST-ls**
```python
def forward(self, x_enc, ...):
    if self.use_shapelet_layer:
        # 模式 A：LearningShapeletsSeg
        activations = self.shapelet_layer(x)
        prototype = self.shapelet_layer.get_shapelets()
    else:
        # 模式 B：原始方式
        prototype = self.prototype_layer(self.prototype_vectors_param)
        activations = self.convolution_distance_x_as_input(x, prototype)
    
    # 统一的后续处理
    output = self.projection(...)
    return output, activations, prototype
```

**优势：**
1. ✅ **向后兼容**：`use_shapelet_layer=False` 时与 ProtoPTST 完全一致
2. ✅ **灵活切换**：通过配置文件轻松切换实现
3. ✅ **统一接口**：返回格式 `(output, activations, prototype)` 保持一致

---

### 1.3 属性兼容性处理

```python
@property
def prototype_vectors(self):
    """向后兼容：根据模式返回对应的 prototype vectors"""
    if self.use_shapelet_layer:
        return self.shapelet_layer.get_shapelets()  # [P, L, C]
    else:
        return self.prototype_vectors_param  # [P, L, C]
```

**作用：**
- 保证外部代码（如 `seg_prototype_loss`、可视化脚本）可以统一访问 `model.prototype_vectors`
- 避免修改依赖此属性的下游代码

---

## 2. 已完成的代码修改清单

### 2.1 模型文件修改 ✅

**文件：** `shapelet_encoder/models/ProtoPTST-ls.py`

**主要修改：**
1. 导入 `LearningShapeletsSeg` 模块
2. 添加 `use_shapelet_layer` 控制参数
3. 条件初始化两种模式
4. 修改 `forward` 方法支持双模式
5. 添加 `@property prototype_vectors` 兼容访问器

**关键代码：**
```python
from .models.LearningShapeletsSeg import LearningShapeletsSeg

class Model(PatchTST.Model):
    def __init__(self, configs):
        # ...
        self.use_shapelet_layer = getattr(configs, 'use_shapelet_layer', False)
        
        if self.use_shapelet_layer:
            self.shapelet_layer = LearningShapeletsSeg(...)
        else:
            self.prototype_vectors_param = nn.Parameter(...)
```

---

### 2.2 需要修改的配置文件

**文件列表：**
- `configs/mcce.yaml`
- `configs/mcch.yaml`
- `configs/mitecg.yaml`
- `configs/mtce.yaml`
- `configs/mtch.yaml`

**需要添加的参数：**
```yaml
# ProtoPTST-ls 相关配置
use_shapelet_layer: false          # 默认 false 使用原方式
dist_measure: 'euclidean'          # 距离度量：euclidean, cosine, correlation
shapelet_znorm: true               # 是否 Z-normalization
shapelet_temperature: 1.0          # Softmax 温度参数
```

---

### 2.3 需要修改的训练文件

**文件：** `exp_saliency_general.py`

#### 修改 0：KMeans 初始化（新增）

当 `use_shapelet_layer=true` 时，训练开始前会基于训练数据做一次 KMeans 初始化（参考 `Learning-Shapelets/demo/demo.ipynb` 的思路）。

**行为：**
- 数据来源：`get_saliency_data()` 返回的 `train_data.X`（形状 `[N, T, C]`）
- 片段抽样：从所有滑动窗口中随机采样，默认最多 `10000` 段
- 聚类：`tslearn.TimeSeriesKMeans`，`metric="euclidean"`，`max_iter=50`
- 结果写回：`model.shapelet_layer.set_shapelets(centers)`（形状 `[P, L, C]`）

**依赖：**
- 需要安装 `tslearn`（当前 `requirements.txt` 已包含 `tslearn==0.6.3`）
- 若环境无法导入 `tslearn`，会打印提示并保持随机初始化

#### 修改 1：添加参数解析

在 `get_args()` 函数中添加：

```python
def get_args(config):
    # ...existing code...
    
    # ProtoPTST args
    parser.add_argument("--num_prototypes", "-n_pro", type=int, default=N_PROTOS)
    parser.add_argument("--prototype_len", "-pro_len", type=int, default=PROTO_LEN)
    parser.add_argument("--prototype_init", default="kaiming", type=str)
    parser.add_argument("--prototype_activation", default="linear", type=str)
    parser.add_argument("--ablation", default="none", type=str)
    
    # 新增 LearningShapeletsSeg 相关参数
    parser.add_argument("--use_shapelet_layer", type=bool, default=False,
                       help="Use LearningShapeletsSeg instead of conv1d")
    parser.add_argument("--dist_measure", type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'correlation'],
                       help="Distance measure for shapelet matching")
    parser.add_argument("--shapelet_znorm", type=bool, default=True,
                       help="Apply z-normalization to shapelets")
    parser.add_argument("--shapelet_temperature", type=float, default=1.0,
                       help="Temperature for softmax in shapelet layer")
    
    # ...existing code...
```

#### 修改 2：更新模型导入（可选）

如果你重命名了文件，需要修改导入路径：

```python
# 方案 1：将 ProtoPTST-ls.py 重命名为 ProtoPTST.py 覆盖原文件
# 无需修改导入

# 方案 2：保持两个文件独立
from shapelet_encoder.models.ProtoPTST_ls import Model as ProtoPTST
```

**训练循环无需修改**（接口保持一致）：
```python
for i, (batch_x, label, padding_mask) in enumerate(train_loader):
    outputs, place_holder_1, prototype = self.model(batch_x, padding_mask, None, None)
    
    # loss 计算保持不变
    prototype_loss = self.model.seg_prototype_loss(
        place_holder_1, batch_x, prototype, outputs
    )
    
    loss = criterion(outputs, label) + 10 * prototype_loss
```

---

### 2.4 可视化文件修改（建议）

**文件：** `vis_ecg.py`

**修改函数：** `plot_proto_vs_matches()`

```python
def plot_proto_vs_matches(seg_model, X_cpu, proto_idx=0, K=None, mode="z", seed=0):
    # 计算 activations
    acts = compute_activations(seg_model, X_cpu)
    t_star = torch.argmax(acts[:, :, proto_idx], dim=1)

    # 获取 prototypes - 兼容两种模式
    with torch.no_grad():
        if hasattr(seg_model, 'use_shapelet_layer') and seg_model.use_shapelet_layer:
            # 使用 LearningShapeletsSeg 模式
            protos_used = seg_model.shapelet_layer.get_shapelets()  # [P, L, C]
        else:
            # 使用原始模式
            protos_used = seg_model.prototype_layer(seg_model.prototype_vectors)
        
        protos_used = protos_used.detach().cpu()
        pv = protos_used[:, :, 0] if protos_used.dim() == 3 else protos_used.squeeze(-1)[:, :, 0]
    
    # ...existing code...
```

---

### 2.5 数据提取脚本修改（建议）

**文件：** `extract_for_viz.py`

```python
def extract_data():
    # ...existing code...
    
    # 兼容两种模式获取 prototypes
    if hasattr(seg_model, 'use_shapelet_layer') and seg_model.use_shapelet_layer:
        prototypes = seg_model.shapelet_layer.get_shapelets().detach().cpu().numpy()
    else:
        prototypes = seg_model.prototype_vectors.detach().cpu().numpy()
    
    print(f"【模块1】已提取 Shapelet 字典，形状: {prototypes.shape}")
    
    # ...existing code...
```

---

### 2.6 ShapeX 推理脚本修改（可选）

**文件：** `shapeX.py`

**说明：** 此文件的 `get_seg_ProtopTST()` 和 `get_seg_ProtopTST_SNC()` 函数**无需修改**，因为：
1. `forward` 方法的接口保持一致
2. 返回的 `(out, actions, prototype)` 格式不变
3. 外部代码只关心输出，不关心内部实现

**现有代码已兼容：**
```python
def get_seg_ProtopTST(signal, seg_model):
    device = ProtopTST.device
    x_in = signal.reshape(1, -1, 1).to(device)
    
    # forward 接口保持一致，无论使用哪种模式
    out, actions, prototype = ProtopTST(x_in, x_in, x_in, x_in)
    
    # 后续处理不变
    actions_sum = torch.sum(actions, dim=-1).reshape(-1)
    # ...
```

---

## 3. 使用 ProtoPTST-ls 替换 ProtoPTST 的步骤

### 方案 A：直接替换（推荐）

#### 步骤 1：备份原文件
```powershell
# 备份原始 ProtoPTST.py
Copy-Item "shapelet_encoder\models\ProtoPTST.py" "shapelet_encoder\models\ProtoPTST_original_backup.py"
```

#### 步骤 2：重命名新文件
```powershell
# 将 ProtoPTST-ls.py 重命名为 ProtoPTST.py
Move-Item "shapelet_encoder\models\ProtoPTST-ls.py" "shapelet_encoder\models\ProtoPTST.py" -Force
```

#### 步骤 3：修改配置文件

**使用原始方式（默认，兼容旧代码）：**
```yaml
# configs/mitecg.yaml
use_shapelet_layer: false  # 或不设置此参数
```

**使用新的 LearningShapeletsSeg：**
```yaml
# configs/mitecg.yaml
use_shapelet_layer: true
dist_measure: 'euclidean'
shapelet_znorm: true
shapelet_temperature: 1.0
```

#### 步骤 4：运行训练
```powershell
# 使用原始方式
python exp_saliency_general.py --data mitecg

# 使用 LearningShapeletsSeg 方式（需先修改配置文件）
python exp_saliency_general.py --data mitecg --use_shapelet_layer true --dist_measure euclidean
```

---

### 方案 B：保持两个文件独立

如果你想保留原始 ProtoPTST.py 用于对比实验：

#### 步骤 1：修改导入路径

在 `exp_saliency_general.py` 中：
```python
# 原来
from shapelet_encoder.models import ProtoPTST

# 改为
from shapelet_encoder.models.ProtoPTST_ls import Model as ProtoPTST
```

#### 步骤 2：修改其他导入位置

搜索所有导入 ProtoPTST 的文件并更新：
```powershell
# 搜索所有导入
Select-String -Path *.py -Pattern "from.*ProtoPTST"
```

可能需要修改的文件：
- `exp_saliency_general.py`
- `vis_ecg.py`
- `extract_for_viz.py`
- `shapeX.py`（如果有显式导入）

---

## 4. 配置文件示例

### 4.1 MIT-BIH ECG 数据集配置

**文件：** `configs/mitecg.yaml`

```yaml
base:
  root_dir: D:/shapelet/ShapeX

dataset:
  name: mitecg
  meta_dataset: default

mitecg:
  num_classes: 2
  seq_len: 360
  proto_len: 30
  num_prototypes: 4
  
  # ProtoPTST-ls 配置
  use_shapelet_layer: false         # 使用原方式
  dist_measure: 'euclidean'
  shapelet_znorm: true
  shapelet_temperature: 1.0
  
  # 原有配置保持不变
  prototype_init: 'kaiming'
  ablation: 'none'
```

### 4.2 启用 LearningShapeletsSeg 的配置

```yaml
mitecg:
  # ...其他配置...
  
  # 启用 LearningShapeletsSeg
  use_shapelet_layer: true
  dist_measure: 'euclidean'         # 可选：euclidean, cosine, correlation
  shapelet_znorm: true              # 启用 Z-normalization
  shapelet_temperature: 1.0         # 温度参数
```

### 4.3 不同距离度量的消融实验配置

#### 欧氏距离（推荐）
```yaml
use_shapelet_layer: true
dist_measure: 'euclidean'
shapelet_znorm: true
shapelet_temperature: 1.0
```

#### 余弦相似度
```yaml
use_shapelet_layer: true
dist_measure: 'cosine'
shapelet_znorm: false              # 余弦相似度自带归一化
shapelet_temperature: 1.0
```

#### 互相关（最快）
```yaml
use_shapelet_layer: true
dist_measure: 'correlation'        # 使用 conv1d 实现
shapelet_znorm: true
shapelet_temperature: 1.0
```

---

## 5. 实验对比建议

### 5.1 基准实验

**目标：** 验证 ProtoPTST-ls 在原始模式下与 ProtoPTST 结果一致

```yaml
# 配置
use_shapelet_layer: false
```

**运行：**
```powershell
python exp_saliency_general.py --data mitecg
```

**预期结果：** 准确率、损失值应与原始 ProtoPTST 完全一致

---

### 5.2 距离度量消融实验

**实验组：**
1. **原始 conv1d** (`use_shapelet_layer: false`)
2. **欧氏距离** (`dist_measure: 'euclidean'`)
3. **余弦相似度** (`dist_measure: 'cosine'`)
4. **互相关** (`dist_measure: 'correlation'`)

**运行脚本：**
```powershell
# 实验 1：原始方式
python exp_saliency_general.py --data mitecg --use_shapelet_layer false

# 实验 2：欧氏距离
python exp_saliency_general.py --data mitecg --use_shapelet_layer true --dist_measure euclidean

# 实验 3：余弦相似度
python exp_saliency_general.py --data mitecg --use_shapelet_layer true --dist_measure cosine

# 实验 4：互相关
python exp_saliency_general.py --data mitecg --use_shapelet_layer true --dist_measure correlation
```

**对比指标：**
- 分类准确率
- Prototype loss
- 训练时间
- 解释质量（AOPC, Sufficiency, Comprehensiveness）

---

