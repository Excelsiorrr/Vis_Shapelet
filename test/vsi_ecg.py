import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data_folder = r"d:\shapelet\ShapeX\datasets\mitecg"
X = torch.load(os.path.join(data_folder, "X.pt"))
y = torch.load(os.path.join(data_folder, "y.pt"))
saliency = torch.load(os.path.join(data_folder, "saliency.pt"))

# 调整数据维度并转换类型
X = X.permute(1, 0, 2)  # [360, 100012, 1] -> [100012, 360, 1]
y = y.squeeze().long()  # [100012, 1] -> [100012]
saliency = saliency  # [100012, 360] 保持不变

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"saliency shape: {saliency.shape}")
print(f"类别分布: {torch.bincount(y)}")

# 找出类别0和类别1的索引
idx_class_0 = torch.where(y == 0)[0]
idx_class_1 = torch.where(y == 1)[0]

print(f"\n类别0样本数: {len(idx_class_0)}")
print(f"类别1样本数: {len(idx_class_1)}")

# 选择每个类别的前10个样本
num_samples = 10
samples_class_0 = idx_class_0[:num_samples]
samples_class_1 = idx_class_1[:num_samples]

# 创建可视化
fig, axes = plt.subplots(num_samples, 2, figsize=(16, 20))
fig.suptitle('MIT-ECG 数据可视化 (左: 类别0, 右: 类别1)', fontsize=16, y=0.995)

for i in range(num_samples):
    # 类别0
    idx_0 = samples_class_0[i]
    ax_0 = axes[i, 0]
    
    # 绘制时间序列
    if len(X.shape) == 3:  # (N, T, C) 格式
        time_series_0 = X[idx_0, :, 0].numpy()
    else:  # (N, T) 格式
        time_series_0 = X[idx_0, :].numpy()
    
    sal_0 = saliency[idx_0, :].numpy()
    time_points = np.arange(len(time_series_0))
    
    # 绘制ECG信号
    ax_0.plot(time_points, time_series_0, 'b-', linewidth=1.5, label='ECG信号')
    
    # 标记显著性区域（使用颜色强度表示）
    # 归一化显著性值
    sal_norm_0 = (sal_0 - sal_0.min()) / (sal_0.max() - sal_0.min() + 1e-8)
    
    # 用红色阴影标记高显著性区域
    threshold = 0.5  # 显著性阈值
    high_sal_mask = sal_norm_0 > threshold
    ax_0.fill_between(time_points, time_series_0.min(), time_series_0.max(), 
                       where=high_sal_mask, alpha=0.3, color='red', label='高显著性区域')
    
    # 在顶部绘制显著性曲线
    ax_twin_0 = ax_0.twinx()
    ax_twin_0.plot(time_points, sal_norm_0, 'r--', alpha=0.6, linewidth=1, label='显著性')
    ax_twin_0.set_ylabel('显著性', color='r', fontsize=8)
    ax_twin_0.tick_params(axis='y', labelcolor='r', labelsize=7)
    ax_twin_0.set_ylim([0, 1])
    
    ax_0.set_ylabel('振幅', fontsize=8)
    ax_0.set_title(f'类别0 - 样本{i+1} (索引: {idx_0.item()})', fontsize=10)
    ax_0.grid(True, alpha=0.3)
    ax_0.tick_params(labelsize=7)
    if i == 0:
        ax_0.legend(loc='upper left', fontsize=7)
        ax_twin_0.legend(loc='upper right', fontsize=7)
    
    # 类别1
    idx_1 = samples_class_1[i]
    ax_1 = axes[i, 1]
    
    if len(X.shape) == 3:
        time_series_1 = X[idx_1, :, 0].numpy()
    else:
        time_series_1 = X[idx_1, :].numpy()
    
    sal_1 = saliency[idx_1, :].numpy()
    
    # 绘制ECG信号
    ax_1.plot(time_points, time_series_1, 'b-', linewidth=1.5, label='ECG信号')
    
    # 标记显著性区域
    sal_norm_1 = (sal_1 - sal_1.min()) / (sal_1.max() - sal_1.min() + 1e-8)
    high_sal_mask_1 = sal_norm_1 > threshold
    ax_1.fill_between(time_points, time_series_1.min(), time_series_1.max(), 
                       where=high_sal_mask_1, alpha=0.3, color='red', label='高显著性区域')
    
    # 显著性曲线
    ax_twin_1 = ax_1.twinx()
    ax_twin_1.plot(time_points, sal_norm_1, 'r--', alpha=0.6, linewidth=1, label='显著性')
    ax_twin_1.set_ylabel('显著性', color='r', fontsize=8)
    ax_twin_1.tick_params(axis='y', labelcolor='r', labelsize=7)
    ax_twin_1.set_ylim([0, 1])
    
    ax_1.set_ylabel('振幅', fontsize=8)
    ax_1.set_title(f'类别1 - 样本{i+1} (索引: {idx_1.item()})', fontsize=10)
    ax_1.grid(True, alpha=0.3)
    ax_1.tick_params(labelsize=7)
    if i == 0:
        ax_1.legend(loc='upper left', fontsize=7)
        ax_twin_1.legend(loc='upper right', fontsize=7)
    
    # 只在最后一行显示x轴标签
    if i == num_samples - 1:
        ax_0.set_xlabel('时间点', fontsize=8)
        ax_1.set_xlabel('时间点', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(data_folder, 'mitecg_visualization.png'), dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存到: {os.path.join(data_folder, 'mitecg_visualization.png')}")
plt.show()