"""
实验主程序
完整实验流水线：加载数据 -> 预算分配 -> DP发布 -> 后处理 -> 评估 -> 可视化
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
INNER_FONT_SIZE = 16
OUTER_FONT_SIZE = 19
SMALL_LEGEND_FONT_SIZE = 10
matplotlib.rcParams.update({
    "font.size": INNER_FONT_SIZE,
    "axes.titlesize": OUTER_FONT_SIZE,
    "axes.labelsize": OUTER_FONT_SIZE,
    "xtick.labelsize": INNER_FONT_SIZE,
    "ytick.labelsize": INNER_FONT_SIZE,
    "legend.fontsize": INNER_FONT_SIZE,
    "figure.titlesize": OUTER_FONT_SIZE,
})

from src.data_loader import DynamicGraphLoader, compute_graph_stats
from src.privacy_budget import PrivacyBudgetManager
from src.dp_mechanism import (
    dp_publish_dynamic_graph,
    dp_publish_statistics_series,
    dp_graph_statistics,
)
from src.post_processing import smooth_statistics_series, KalmanFilter1D
from src.evaluation import (
    evaluate_dynamic_publication,
    evaluate_statistics_series,
    rmse,
    trend_correlation,
)

# ============================================================
# 实验配置
# ============================================================

DATASETS = ["college", "bitcoin"]  # facebook 数据量大，先用小数据集
NUM_SNAPSHOTS = 10
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
STRATEGIES = ["uniform", "exponential", "adaptive"]
SMOOTHING_METHODS = ["none", "sliding", "exponential", "kalman"]
OUTPUT_DIR = "results"
SEED = 42


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# ============================================================
# 实验1: 不同隐私预算下的统计量发布误差
# ============================================================

def experiment_epsilon_vs_error(dataset_name: str):
    """不同 ε 值下统计量发布的 RMSE"""
    print(f"\n{'='*60}")
    print(f"实验1: ε vs 误差 - 数据集: {dataset_name}")
    print(f"{'='*60}")

    loader = DynamicGraphLoader()
    snapshots, labels = loader.load_dataset(dataset_name, NUM_SNAPSHOTS)

    # 计算真实统计量
    true_stats = []
    for G in snapshots:
        true_stats.append({
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "clustering_coeff": __import__("networkx").average_clustering(G) if G.number_of_nodes() > 0 else 0,
        })

    results = {eps: {} for eps in EPSILON_VALUES}

    for eps in EPSILON_VALUES:
        print(f"\n  ε = {eps}")
        for strategy in STRATEGIES:
            np.random.seed(SEED)
            mgr = PrivacyBudgetManager(eps, NUM_SNAPSHOTS)
            budgets = mgr.get_allocation(strategy, snapshots=snapshots)

            # DP 统计量发布
            noisy_stats = dp_publish_statistics_series(snapshots, budgets)

            # 后处理（卡尔曼滤波）
            # dp_graph_statistics 将预算4等分，实际每个统计量的 ε = budget_t / 4
            avg_budget = eps / NUM_SNAPSHOTS
            actual_eps_per_stat = avg_budget / 4
            R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps_per_stat)
            # Q 基于真实数据变化幅度估计
            edge_counts = np.array([G.number_of_edges() for G in snapshots])
            Q = max(np.var(np.diff(edge_counts)), 1.0)
            smoothed_stats = smooth_statistics_series(
                noisy_stats, method="kalman", process_noise=Q, measurement_noise=R
            )

            # 评估
            eval_raw = evaluate_statistics_series(true_stats, noisy_stats)
            eval_smooth = evaluate_statistics_series(true_stats, smoothed_stats)

            results[eps][strategy] = {
                "raw": eval_raw,
                "smoothed": eval_smooth,
            }
            print(f"    {strategy}: 边数RMSE(raw)={eval_raw['edge_count_rmse']:.2f}, "
                  f"边数RMSE(smooth)={eval_smooth['edge_count_rmse']:.2f}")

    # 可视化
    _plot_epsilon_vs_rmse(results, dataset_name, "edge_count")
    _plot_epsilon_vs_rmse(results, dataset_name, "node_count")
    return results


def _plot_epsilon_vs_rmse(results, dataset_name, metric_key):
    """绘制 ε-RMSE 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, label_suffix, data_key in zip(axes, ["原始", "平滑后"], ["raw", "smoothed"]):
        for strategy in STRATEGIES:
            rmses = [results[eps][strategy][data_key][f"{metric_key}_rmse"]
                     for eps in EPSILON_VALUES]
            ax.plot(EPSILON_VALUES, rmses, marker="o", label=strategy)

        ax.set_xlabel("隐私预算 ε")
        ax.set_ylabel(f"{metric_key} RMSE")
        ax.set_title(f"{dataset_name} - {metric_key} RMSE ({label_suffix})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks(EPSILON_VALUES)
        ax.set_xticklabels(["0.1", "0.5", "1", "2", "5"])
        ax.minorticks_off()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{dataset_name}_eps_vs_{metric_key}_rmse.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {dataset_name}_eps_vs_{metric_key}_rmse.png")


# ============================================================
# 实验2: 不同预算分配策略对比
# ============================================================

def experiment_strategy_comparison(dataset_name: str, epsilon: float = 1.0):
    """固定 ε，对比三种预算分配策略"""
    print(f"\n{'='*60}")
    print(f"实验2: 策略对比 - 数据集: {dataset_name}, ε={epsilon}")
    print(f"{'='*60}")

    loader = DynamicGraphLoader()
    snapshots, labels = loader.load_dataset(dataset_name, NUM_SNAPSHOTS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for strategy in STRATEGIES:
        np.random.seed(SEED)
        mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
        budgets = mgr.get_allocation(strategy, snapshots=snapshots)

        # 预算分配可视化
        axes[0, 0].bar(
            np.arange(NUM_SNAPSHOTS) + STRATEGIES.index(strategy) * 0.25,
            budgets, width=0.25, label=strategy
        )

        # 累积消耗
        cumulative = mgr.get_cumulative_consumption(budgets)
        axes[0, 1].plot(range(NUM_SNAPSHOTS), cumulative, marker="o", label=strategy)

        # DP 统计量发布
        noisy_stats = dp_publish_statistics_series(snapshots, budgets)
        true_edge_counts = [G.number_of_edges() for G in snapshots]
        noisy_edge_counts = [s["edge_count"] for s in noisy_stats]

        axes[1, 0].plot(range(NUM_SNAPSHOTS), noisy_edge_counts, marker="s",
                        label=f"{strategy} (DP)", alpha=0.7)

        # 后处理
        avg_budget = epsilon / NUM_SNAPSHOTS
        actual_eps_per_stat = avg_budget / 4
        R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps_per_stat)
        edge_counts = np.array([G.number_of_edges() for G in snapshots])
        Q = max(np.var(np.diff(edge_counts)), 1.0)
        smoothed_stats = smooth_statistics_series(
            noisy_stats, method="kalman", process_noise=Q, measurement_noise=R
        )
        smoothed_edge_counts = [s["edge_count"] for s in smoothed_stats]
        axes[1, 1].plot(range(NUM_SNAPSHOTS), smoothed_edge_counts, marker="s",
                        label=f"{strategy} (平滑后)", alpha=0.7)

    # 真实值
    true_edge_counts = [G.number_of_edges() for G in snapshots]
    axes[1, 0].plot(range(NUM_SNAPSHOTS), true_edge_counts, "k--", linewidth=2, label="真实值")
    axes[1, 1].plot(range(NUM_SNAPSHOTS), true_edge_counts, "k--", linewidth=2, label="真实值")

    axes[0, 0].set_title("预算分配")
    axes[0, 0].set_xlabel("快照编号")
    axes[0, 0].set_ylabel("ε_t")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("累积隐私消耗")
    axes[0, 1].set_xlabel("快照编号")
    axes[0, 1].set_ylabel("累积 ε")
    axes[0, 1].axhline(y=epsilon, color="r", linestyle="--", label=f"总预算 ε={epsilon}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("边数发布（原始）")
    axes[1, 0].set_xlabel("快照编号")
    axes[1, 0].set_ylabel("边数")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("边数发布（卡尔曼滤波后）")
    axes[1, 1].set_xlabel("快照编号")
    axes[1, 1].set_ylabel("边数")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{dataset_name} - 预算分配策略对比 (ε={epsilon})", fontsize=OUTER_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{dataset_name}_strategy_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {dataset_name}_strategy_comparison.png")


# ============================================================
# 实验3: 后处理平滑方法对比
# ============================================================

def experiment_smoothing_comparison(dataset_name: str, epsilon: float = 1.0):
    """对比不同后处理平滑方法的效果"""
    print(f"\n{'='*60}")
    print(f"实验3: 平滑方法对比 - 数据集: {dataset_name}, ε={epsilon}")
    print(f"{'='*60}")

    loader = DynamicGraphLoader()
    snapshots, labels = loader.load_dataset(dataset_name, NUM_SNAPSHOTS)

    np.random.seed(SEED)
    mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
    budgets = mgr.get_allocation("adaptive", snapshots=snapshots)
    noisy_stats = dp_publish_statistics_series(snapshots, budgets)

    true_edge_counts = np.array([G.number_of_edges() for G in snapshots])
    noisy_edge_counts = np.array([s["edge_count"] for s in noisy_stats])

    avg_budget = epsilon / NUM_SNAPSHOTS
    actual_eps_per_stat = avg_budget / 4
    R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps_per_stat)
    edge_counts_arr = np.array([G.number_of_edges() for G in snapshots])
    Q = max(np.var(np.diff(edge_counts_arr)), 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(NUM_SNAPSHOTS), true_edge_counts, "k-", linewidth=2, label="真实值", marker="o")
    ax.plot(range(NUM_SNAPSHOTS), noisy_edge_counts, "r--", alpha=0.6, label="DP加噪", marker="x")

    colors = {"sliding": "blue", "exponential": "green", "kalman": "purple"}
    for method in ["sliding", "exponential", "kalman"]:
        if method == "kalman":
            smoothed = smooth_statistics_series(
                noisy_stats, method=method, process_noise=Q, measurement_noise=R
            )
        else:
            smoothed = smooth_statistics_series(noisy_stats, method=method)
        smoothed_edge = np.array([s["edge_count"] for s in smoothed])
        rmse_val = np.sqrt(np.mean((true_edge_counts - smoothed_edge) ** 2))
        ax.plot(range(NUM_SNAPSHOTS), smoothed_edge, color=colors[method],
                marker="s", alpha=0.8, label=f"{method} (RMSE={rmse_val:.1f})")

    ax.set_xlabel("快照编号")
    ax.set_ylabel("边数")
    ax.set_title(f"{dataset_name} - 后处理平滑方法对比 (ε={epsilon})")
    ax.legend(loc="lower right", fontsize=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{dataset_name}_smoothing_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {dataset_name}_smoothing_comparison.png")


# ============================================================
# 实验4: 综合雷达图
# ============================================================

def experiment_radar_chart(dataset_name: str, epsilon: float = 1.0):
    """多维度雷达图对比（基于统计量发布，避免大图边扰动的性能问题）"""
    print(f"\n{'='*60}")
    print(f"实验4: 雷达图 - 数据集: {dataset_name}, ε={epsilon}")
    print(f"{'='*60}")

    loader = DynamicGraphLoader()
    snapshots, labels = loader.load_dataset(dataset_name, NUM_SNAPSHOTS)

    import networkx as nx_
    # 真实统计量序列
    true_stats = []
    for G in snapshots:
        true_stats.append({
            "node_count": float(G.number_of_nodes()),
            "edge_count": float(G.number_of_edges()),
            "clustering_coeff": float(nx_.average_clustering(G)) if G.number_of_nodes() > 0 else 0.0,
            "avg_degree": 2.0 * G.number_of_edges() / max(G.number_of_nodes(), 1),
        })

    metrics_all = {}
    for strategy in STRATEGIES:
        np.random.seed(SEED)
        mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
        budgets = mgr.get_allocation(strategy, snapshots=snapshots)

        # DP 统计量发布
        noisy_stats = dp_publish_statistics_series(snapshots, budgets)

        # 卡尔曼后处理
        avg_budget = epsilon / NUM_SNAPSHOTS
        actual_eps_per_stat = avg_budget / 4
        R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps_per_stat)
        edge_counts = np.array([G.number_of_edges() for G in snapshots])
        Q = max(np.var(np.diff(edge_counts)), 1.0)
        smoothed_stats = smooth_statistics_series(
            noisy_stats, method="kalman", process_noise=Q, measurement_noise=R
        )

        # 计算各维度指标
        true_ec = np.array([s["edge_count"] for s in true_stats])
        true_nc = np.array([s["node_count"] for s in true_stats])
        true_cc = np.array([s["clustering_coeff"] for s in true_stats])

        smooth_ec = np.array([s["edge_count"] for s in smoothed_stats])
        smooth_nc = np.array([s["node_count"] for s in smoothed_stats])
        smooth_cc = np.array([s["clustering_coeff"] for s in smoothed_stats])

        metrics_all[strategy] = {
            "edge_rmse": rmse(true_ec, smooth_ec),
            "node_rmse": rmse(true_nc, smooth_nc),
            "cc_rmse": rmse(true_cc, smooth_cc),
            "edge_trend_corr": trend_correlation(true_ec, smooth_ec),
            "node_trend_corr": trend_correlation(true_nc, smooth_nc),
        }
        print(f"  {strategy}: edge_rmse={metrics_all[strategy]['edge_rmse']:.2f}, "
              f"edge_trend_corr={metrics_all[strategy]['edge_trend_corr']:.3f}")

    # 雷达图
    categories = ["边数RMSE", "节点数RMSE", "聚类系数RMSE", "边数趋势相关", "节点趋势相关"]
    keys = ["edge_rmse", "node_rmse", "cc_rmse", "edge_trend_corr", "node_trend_corr"]
    invert = [False, False, False, True, True]  # 相关性越高越好，需要反转

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # 收集所有策略的原始值用于归一化
    all_values = {k: [] for k in keys}
    for strategy in STRATEGIES:
        for key in keys:
            all_values[key].append(metrics_all[strategy][key])

    for strategy in STRATEGIES:
        values_norm = []
        for key, inv in zip(keys, invert):
            v = metrics_all[strategy][key]
            vmin = min(all_values[key])
            vmax = max(all_values[key])
            if vmax - vmin > 1e-10:
                normed = (v - vmin) / (vmax - vmin)
            else:
                normed = 0.5
            if inv:
                normed = 1 - normed  # 反转：相关性越高 -> 归一化误差越小
            values_norm.append(normed)
        values_norm += values_norm[:1]

        ax.plot(angles, values_norm, "o-", linewidth=2, label=strategy)
        ax.fill(angles, values_norm, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"{dataset_name} - 多维度对比 (ε={epsilon})\n(值越小越好)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{dataset_name}_radar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {dataset_name}_radar.png")


# ============================================================
# 主入口
# ============================================================

def run_all_experiments():
    """运行全部实验"""
    ensure_output_dir()

    for dataset in DATASETS:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {dataset}")
        print(f"{'#'*60}")

        experiment_epsilon_vs_error(dataset)
        experiment_strategy_comparison(dataset, epsilon=1.0)
        experiment_smoothing_comparison(dataset, epsilon=1.0)
        experiment_radar_chart(dataset, epsilon=1.0)

    print(f"\n{'='*60}")
    print("全部实验完成！结果保存在 results/figures/ 目录下")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_experiments()
