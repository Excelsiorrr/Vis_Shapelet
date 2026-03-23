import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def compute_soft_depth_against_mean(X, eps=1e-8):
    """
    基于“与均值曲线重叠”的软深度（单向，不做双向）。

    Parameters
    ----------
    X : np.ndarray, shape (N, T)
        N条时序，每条长度为T
    eps : float
        防止某些时间点标准差为0时除零

    Returns
    -------
    depth : np.ndarray, shape (N,)
        每条时序的软深度，越大越接近均值曲线
    mean_curve : np.ndarray, shape (T,)
        均值曲线
    sigma_t : np.ndarray, shape (T,)
        每个时间点上的标准差
    W : np.ndarray, shape (N, T)
        每条曲线在每个时间点对均值曲线的贴近程度
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, T), but got {X.shape}")

    # 1) 均值曲线 c(t)
    mean_curve = np.mean(X, axis=0)

    # 2) 每个时间点的波动尺度 sigma_t
    sigma_t = np.std(X, axis=0)
    sigma_t = np.maximum(sigma_t, eps)

    # 3) 局部贴近程度 W[i, t]
    W = np.exp(-((X - mean_curve) ** 2) / (2 * sigma_t ** 2))

    # 4) 整条曲线的软深度
    depth = np.mean(W, axis=1)

    return depth, mean_curve, sigma_t, W


def get_central_band_by_depth(X, depth, central_ratio=0.5, band_mode="quantile"):
    """
    取深度最高的一部分曲线，构造中心带。

    Parameters
    ----------
    X : np.ndarray, shape (N, T)
    depth : np.ndarray, shape (N,)
    central_ratio : float
        取前多少比例的高深度曲线，例如 0.5 表示前50%
    band_mode : str
        "minmax" : 中心带上下界 = 高深度曲线的逐时刻最小/最大值
        "quantile" : 中心带上下界 = 高深度曲线的逐时刻 25%/75% 分位数

    Returns
    -------
    central_idx : np.ndarray
        高深度曲线对应的原始索引
    central_curves : np.ndarray
        高深度曲线本身
    lower_band : np.ndarray, shape (T,)
    upper_band : np.ndarray, shape (T,)
    """
    X = np.asarray(X, dtype=float)
    depth = np.asarray(depth, dtype=float)

    N = X.shape[0]
    k = max(1, int(np.ceil(N * central_ratio)))

    order_high_to_low = np.argsort(-depth)
    central_idx = order_high_to_low[:k]
    central_curves = X[central_idx]

    if band_mode == "minmax":
        lower_band = np.min(central_curves, axis=0)
        upper_band = np.max(central_curves, axis=0)
    elif band_mode == "quantile":
        lower_band = np.quantile(central_curves, 0.25, axis=0)
        upper_band = np.quantile(central_curves, 0.75, axis=0)
    else:
        raise ValueError("band_mode must be 'minmax' or 'quantile'")

    return central_idx, central_curves, lower_band, upper_band


def plot_soft_depth_3panels(
    X,
    time=None,
    central_ratio=0.5,
    band_mode="quantile",
    cmap_name="viridis",
    line_alpha=0.75,
    line_lw=1.4,
    mean_lw=3.2,
    figsize=(18, 15),
    panel1_title="(1) All Time Series + Mean Curve + Colored by Soft Depth",
    panel2_title="(2) Local Similarity Heatmap W (rows sorted by depth)",
    panel3_title="(3) Mean Curve + Central Band from Top-Depth Curves",
    show=False,
    save_path=None,
    dpi=220,
):
    """
    画三张推荐图：
    1) 所有时序 + 均值曲线 + 按深度着色
    2) 按深度排序的局部贴近热图 W
    3) 均值曲线 + 高深度前50%曲线形成的中心带

    Returns
    -------
    result : dict
        包含 depth, mean_curve, sigma_t, W, 排序结果, 中心带结果
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, T), but got {X.shape}")

    N, T = X.shape
    if time is None:
        time = np.arange(T)
    else:
        time = np.asarray(time)
        if len(time) != T:
            raise ValueError("time length must match X.shape[1]")

    # 计算软深度
    depth, mean_curve, sigma_t, W = compute_soft_depth_against_mean(X)

    # 排序
    order_low_to_high = np.argsort(depth)
    order_high_to_low = np.argsort(-depth)

    # 中心带
    central_idx, central_curves, lower_band, upper_band = get_central_band_by_depth(
        X, depth, central_ratio=central_ratio, band_mode=band_mode
    )

    # 颜色映射
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=depth.min(), vmax=depth.max())

    # 画图
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # -------------------------
    # Panel 1
    # -------------------------
    ax1 = axes[0]
    for i in order_low_to_high:
        color = cmap(norm(depth[i]))
        ax1.plot(time, X[i], color=color, alpha=line_alpha, linewidth=line_lw)

    ax1.plot(time, mean_curve, color="black", linewidth=mean_lw, label="Mean curve")
    ax1.set_title(panel1_title, fontsize=14)
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("Value", fontsize=11)
    ax1.legend()

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=ax1, pad=0.01)
    cbar1.set_label("Soft depth", fontsize=11)

    # -------------------------
    # Panel 2
    # -------------------------
    ax2 = axes[1]
    W_sorted = W[order_high_to_low]  # 深度高的放上面
    im = ax2.imshow(
        W_sorted,
        aspect="auto",
        interpolation="nearest",
        origin="upper"
    )
    ax2.set_title(panel2_title, fontsize=14)
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylabel("Curves sorted by depth (high → low)", fontsize=11)

    # 自定义 x ticks 到真实 time
    xtick_positions = np.linspace(0, T - 1, num=min(8, T), dtype=int)
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels([str(time[idx]) for idx in xtick_positions])

    cbar2 = fig.colorbar(im, ax=ax2, pad=0.01)
    cbar2.set_label("Local similarity W[i, t]", fontsize=11)

    # -------------------------
    # Panel 3
    # -------------------------
    ax3 = axes[2]

    # 所有曲线先淡灰画出来，帮助看整体背景
    for i in range(N):
        ax3.plot(time, X[i], color="lightgray", alpha=1.0, linewidth=0.8, zorder=1)

    # 中心带
    ax3.fill_between(
        time,
        lower_band,
        upper_band,
        color="tab:orange",
        alpha=0.55,
        zorder=3,
        label=f"Central band (top {int(np.ceil(central_ratio*100))}% by depth)",
    )
    # Draw band boundaries so the central band is clearer.
    ax3.plot(time, lower_band, color="tab:orange", linewidth=2.0, alpha=0.9, zorder=4)
    ax3.plot(time, upper_band, color="tab:orange", linewidth=2.0, alpha=0.9, zorder=4)

    # 高深度曲线也可以轻微画一下
    for idx in central_idx:
        ax3.plot(time, X[idx], alpha=0.7, linewidth=1.2, zorder=2)

    # 均值曲线
    ax3.plot(time, mean_curve, color="black", linewidth=3.5, zorder=5, label="Mean curve")

    ax3.set_title(panel3_title, fontsize=14)
    ax3.set_xlabel("Time", fontsize=11)
    ax3.set_ylabel("Value", fontsize=11)
    ax3.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "depth": depth,
        "mean_curve": mean_curve,
        "sigma_t": sigma_t,
        "W": W,
        "order_low_to_high": order_low_to_high,
        "order_high_to_low": order_high_to_low,
        "central_idx": central_idx,
        "central_curves": central_curves,
        "lower_band": lower_band,
        "upper_band": upper_band,
    }


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    # 构造示例数据
    N = 60
    T = 120
    time = np.arange(T)

    base = (
        0.8 * np.sin(2 * np.pi * time / 40)
        + 0.4 * np.sin(2 * np.pi * time / 18)
        + 0.01 * time
    )

    X = []
    for _ in range(N):
        noise = np.random.normal(scale=0.22, size=T)
        shift = np.random.normal(scale=0.15)
        scale = np.random.normal(loc=1.0, scale=0.08)
        curve = scale * base + shift + noise
        X.append(curve)

    X = np.array(X)

    # 加几个明显偏离均值的例子
    X[0] += 1.8
    X[1] -= 1.5
    X[2, 50:80] += 2.0
    X[3] *= 1.8

    result = plot_soft_depth_3panels(
        X,
        time=time,
        central_ratio=0.5,
        band_mode="quantile",   # 可改成 "quantile"
        cmap_name="viridis",
        line_alpha=0.8,
        line_lw=1.4,
        mean_lw=3.2,
        figsize=(18, 15),
        show=False,
        save_path="toy/soft_depth_3panels.png",
    )

    depth = result["depth"]
    top_idx = np.argsort(-depth)[:10]
    print("Top 10 deepest curves:")
    for idx in top_idx:
        print(f"curve {idx:2d} | depth = {depth[idx]:.4f}")
