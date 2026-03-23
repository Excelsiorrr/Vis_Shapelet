import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_ts_pid_depth(X):
    """
    计算 1D 时序数据的概率包含深度 (PID)
    X 形状: (n_samples, n_time_steps)
    """
    n_samples, n_steps = X.shape
    
    # 1. 计算每个时间点的群体统计量 (构建概率中心 C)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-6
    
    # 2. 计算每个样本在每个时间点的“包含概率” (Gaussian PDF)
    probs = norm.pdf(X, loc=mu, scale=sigma)
    
    # 3. 计算 s1: 样本自身的平均包含密度
    s1 = np.mean(probs, axis=1)
    
    # 4. 计算 s2: 归一化得分 (相对于该位置的最大可能密度)
    max_probs = norm.pdf(mu, loc=mu, scale=sigma)
    s2 = np.mean(probs, axis=1) / np.mean(max_probs)
    
    # 5. 深度 = min(s1, s2)
    depths = np.minimum(s1, s2)
    return depths, mu, sigma

# --- 1. 合成数据 ---
np.random.seed(42)
t = np.linspace(0, 10, 200)
n_normal = 80

# 正常数据: 略有噪声的 Sine 波
normal_data = np.array([np.sin(t) + np.random.normal(0, 0.1, len(t)) for _ in range(n_normal)])

# 异常数据 1: 整体偏移 (Offset) - 5条
outlier_shift = np.array([np.sin(t) + 1.5 for _ in range(5)])

# 异常数据 2: 局部尖峰 (Spike) - 5条
spike = np.zeros_like(t)
spike[100] = 5
outlier_spike = np.array([np.sin(t) + spike for _ in range(5)])

# 异常数据 3: 频率变快 (Frequency) - 5条
outlier_freq = np.array([np.sin(2 * t) for _ in range(5)])

# 合并所有数据
X = np.vstack([normal_data, outlier_shift, outlier_spike, outlier_freq])

# --- 2. 计算深度 ---
depths, mu, sigma = compute_ts_pid_depth(X)

# --- 3. 对比可视化 ---
# 创建 1行2列的子图，共享Y轴
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# =========================================================
# 图 1 (左): 标准可视化 (按已知分组上色)
# 用于展示数据的实际组成，方便对比
# =========================================================
ax0 = axes[0]
ax0.set_title("1. Standard Visualization (Colored by True Group)", fontsize=14)

# 绘制正常数据 (浅蓝色，低透明度)
for i in range(n_normal):
    ax0.plot(t, X[i], color='C0', alpha=0.2, linewidth=1)
# 添加一个用于图例的空线条
ax0.plot([], [], color='C0', alpha=0.5, label='Normal Data')

# 绘制异常值 (使用鲜艳的颜色突出显示)
# 偏移异常 (橙色)
for i in range(n_normal, n_normal + 5):
    ax0.plot(t, X[i], color='C1', linewidth=2)
ax0.plot([], [], color='C1', linewidth=2, label='Outlier: Shift')

# 尖峰异常 (绿色)
for i in range(n_normal + 5, n_normal + 10):
    ax0.plot(t, X[i], color='C2', linewidth=2)
ax0.plot([], [], color='C2', linewidth=2, label='Outlier: Spike')

# 频率异常 (红色)
for i in range(n_normal + 10, n_normal + 15):
    ax0.plot(t, X[i], color='C3', linewidth=2)
ax0.plot([], [], color='C3', linewidth=2, label='Outlier: Freq')

ax0.set_xlabel("Time")
ax0.set_ylabel("Value")
ax0.legend(loc='upper left')
ax0.grid(True, alpha=0.3)


# =========================================================
# 图 2 (右): PID 统计深度可视化 (完全根据算法结果上色)
# 用于展示算法如何在无监督状态下识别离群点
# =========================================================
ax1 = axes[1]
ax1.set_title("2. PID Statistical Depth Visualization (Colored by Algorithm)", fontsize=14)

# 颜色映射逻辑
norm_depths = (depths - depths.min()) / (depths.max() - depths.min())
colors = plt.cm.viridis(norm_depths)

# 背景：画出所有序列
for i in range(len(X)):
    # 差异化透明度：低深度（离群点）更显眼
    alpha = 0.3 if norm_depths[i] < 0.5 else 0.1
    ax1.plot(t, X[i], color=colors[i], alpha=alpha, linewidth=1)

# 突出显示中值序列 (Median Curve)
median_idx = np.argmax(depths)
ax1.plot(t, X[median_idx], color='red', linewidth=3, label='Median (Max Depth)')

# 画出 50% 中心区域 (相当于 Boxplot 的 Box)
threshold = np.median(depths)
central_samples = X[depths >= threshold]
upper_bound = np.max(central_samples, axis=0)
lower_bound = np.min(central_samples, axis=0)
ax1.fill_between(t, lower_bound, upper_bound, color='gray', alpha=0.3, label='50% Central Region')

ax1.set_xlabel("Time")
# ax1.set_ylabel("Value") # 共享Y轴，不需要重复标签
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 添加 Colorbar
sm = plt.cm.ScalarMappable(norm=plt.Normalize(depths.min(), depths.max()), cmap='viridis')
fig.colorbar(sm, ax=ax1, label='PID Depth Score', pad=0.02)

plt.tight_layout()
plt.show()