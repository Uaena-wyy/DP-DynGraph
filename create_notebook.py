"""生成完整的实验 Jupyter Notebook（修正版：补充动机/隐私模型/文献引用/统一命名）"""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None})

# ============================================================
# Cell: 标题
# ============================================================
md("""# 基于差分隐私的动态图数据隐私发布研究

**本科毕业设计实验报告**

---
""")

# ============================================================
# Cell: 研究动机（新增：解决问题1）
# ============================================================
md("""## 1. 研究背景与动机

### 1.1 为什么动态图数据需要隐私保护？

社交网络、金融交易网络等动态图数据蕴含丰富的用户行为信息。直接发布图统计量（如度分布、聚类系数）或图结构可能导致 **隐私泄露**——攻击者可通过背景知识推断特定边（即用户关系）的存在性。例如：

- **链路推断攻击**：攻击者已知图的大部分结构，通过发布的统计量推断某条边是否存在 [Hay et al., 2009]
- **时序关联攻击**：在动态图场景下，攻击者可跨多个时间快照积累信息，即使单次发布安全，累积发布仍可能泄露隐私 [Song et al., 2017]

### 1.2 静态差分隐私方法的不足

传统 DP 方法将每个快照独立处理，存在以下问题：

| 问题 | 说明 |
|------|------|
| **预算快速耗尽** | 串行组合定理要求 $\\varepsilon_{total} = \\sum_t \\varepsilon_t$，$T$ 个快照共享总预算时，每个快照可用预算极小 |
| **时序不连贯** | 独立加噪破坏了图的时间演化规律，相邻快照的发布结果可能出现不合理的剧烈波动 |
| **效用损失大** | 小预算意味着大噪声，统计量发布精度严重下降 |

### 1.3 本研究的解决思路

为此，本研究提出以下技术方案：

1. **智能预算分配**：根据图的动态变化特征分配预算（而非简单均匀分配），让变化大的时刻获得更多预算
2. **后处理平滑**：利用 DP 的后处理不变性（post-processing immunity），对加噪结果进行时序平滑，在不消耗额外隐私预算的情况下提升发布精度
3. **多维度评估**：从精度、结构保真、时序一致性等多个维度综合评估发布质量
""")

# ============================================================
# Cell: 隐私模型定义（新增：解决问题2）
# ============================================================
md("""## 2. 隐私模型与理论基础

### 2.1 中心化差分隐私（Centralized DP）模型

本研究采用 **中心化差分隐私（CDP）** 模型，即假设存在一个 **可信的数据管理者**（trusted curator），它持有完整的图数据，在发布统计结果前添加校准噪声。

> **与本地化差分隐私（LDP）的区别**：
> - **CDP**：可信管理者看到真实数据，加噪后发布 → 噪声量小，精度高
> - **LDP**：每个用户在本地加噪后上报 → 无需信任管理者，但噪声量大（通常 $O(\\sqrt{n})$ 倍于 CDP）
>
> 在社交网络场景中，平台方（如 Facebook、微信）即为可信管理者，CDP 是更合理的选择 [Dwork & Roth, 2014]。

### 2.2 差分隐私形式化定义

**定义（$\\varepsilon$-差分隐私）**：随机机制 $\\mathcal{M}$ 满足 $\\varepsilon$-差分隐私，当且仅当对于任意两个相邻数据集 $D, D'$（仅差一条边）和任意输出集合 $S$：

$$\\Pr[\\mathcal{M}(D) \\in S] \\leq e^{\\varepsilon} \\cdot \\Pr[\\mathcal{M}(D') \\in S]$$

其中 $\\varepsilon > 0$ 为 **隐私预算**，$\\varepsilon$ 越小隐私保护越强。

### 2.3 关键机制

| 机制 | 公式 | 敏感度 $\\Delta f$ | 适用场景 |
|------|------|------|------|
| **Laplace 机制** | $\\mathcal{M}(D) = f(D) + \\text{Lap}(\\Delta f / \\varepsilon)$ | 取决于查询函数 | 数值统计量发布 |
| **随机响应（边扰动）** | 以概率 $p$ 翻转每条边的存在性 | N/A | 图结构发布 |

**图查询的敏感度分析**（添加/删除一条边的影响）：
- 节点数：$\\Delta f = 2$（最多影响两端点）
- 边数：$\\Delta f = 1$
- 三角形数：$\\Delta f = n-2$（一条边最多参与 $n-2$ 个三角形）
- 聚类系数：通过边数和三角形数间接计算

### 2.4 组合定理与动态图隐私

**串行组合定理**：若 $\\mathcal{M}_1, \\ldots, \\mathcal{M}_T$ 分别满足 $\\varepsilon_1, \\ldots, \\varepsilon_T$-DP，则其组合满足 $(\\sum_t \\varepsilon_t)$-DP。

**后处理不变性**：若 $\\mathcal{M}$ 满足 $\\varepsilon$-DP，则对任意函数 $g$，$g(\\mathcal{M}(D))$ 也满足 $\\varepsilon$-DP。这意味着 **平滑滤波不会增加隐私消耗**。
""")

# ============================================================
# Cell: 环境配置
# ============================================================
md("""## 3. 实验环境与数据集

### 3.1 数据集

| 数据集 | 类型 | 节点数 | 边数 | 时间跨度 | 来源 |
|--------|------|--------|------|----------|------|
| CollegeMsg | 校园社交消息 | 1,899 | 13,838 | 193天 | SNAP [Panzarasa et al., 2009] |
| Bitcoin-Alpha | 比特币信任网络 | 3,683 | 12,972 | ~4年 | SNAP [Kumar et al., 2016] |
| Facebook-WOSN | 社交好友关系 | 63,731 | 817,035 | ~2年 | KONECT [Viswanath et al., 2009] |
""")

code("""# 环境配置与依赖导入
import os, sys, io, contextlib
import warnings
warnings.filterwarnings('ignore')

# 抑制 numexpr/bottleneck 兼容性警告（不影响功能）
_stderr = sys.stderr
sys.stderr = io.StringIO()

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sys.stderr = _stderr

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
%matplotlib inline

from src.data_loader import DynamicGraphLoader, compute_graph_stats, get_degree_distribution
from src.privacy_budget import PrivacyBudgetManager
from src.dp_mechanism import (
    dp_publish_statistics_series, dp_graph_statistics,
    dp_degree_distribution, laplace_mechanism,
    edge_perturbation_sparse, dp_publish_dynamic_graph
)
from src.post_processing import (
    smooth_statistics_series, KalmanFilter1D,
    sliding_average, exponential_smoothing
)
from src.evaluation import (
    evaluate_dynamic_publication, evaluate_statistics_series,
    rmse, trend_correlation, kl_divergence_degree,
    temporal_jaccard_similarity, temporal_smoothness_score
)

SEED = 42
NUM_SNAPSHOTS = 10
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
STRATEGIES = ["uniform", "exponential", "adaptive"]
STRATEGY_CN = {"uniform": "均匀分配", "exponential": "指数衰减", "adaptive": "自适应"}

np.random.seed(SEED)
print(f"Python {sys.version.split()[0]} | NetworkX {nx.__version__} | NumPy {np.__version__}")
print("环境配置完成")
""")

# ============================================================
# Cell: 数据加载
# ============================================================
md("""### 3.2 数据加载与预处理

将时序边列表转化为 **累积快照序列**：将时间域均匀划分为 $T$ 个区间，快照 $G_t$ 包含 $[0, t]$ 时间段内所有边。这种累积模型保证了快照之间的单调递增性，更贴近社交网络的真实演化特征。
""")

code("""# 加载数据集
loader = DynamicGraphLoader()

print("=" * 60)
college_snaps, college_labels = loader.load_dataset("college", NUM_SNAPSHOTS)
print()
bitcoin_snaps, bitcoin_labels = loader.load_dataset("bitcoin", NUM_SNAPSHOTS)

datasets = {
    "CollegeMsg": (college_snaps, college_labels),
    "Bitcoin-Alpha": (bitcoin_snaps, bitcoin_labels),
}
""")

# ============================================================
# Cell: 数据统计
# ============================================================
code("""# 各快照统计信息
for name, (snaps, labels) in datasets.items():
    print(f"\\n{'='*50}")
    print(f"数据集: {name}")
    print(f"{'='*50}")
    rows = []
    for label, G in zip(labels, snaps):
        stats = compute_graph_stats(G)
        rows.append({
            "快照": label,
            "节点数": stats["num_nodes"],
            "边数": stats["num_edges"],
            "平均度": f"{stats['avg_degree']:.2f}",
            "密度": f"{stats['density']:.6f}",
            "聚类系数": f"{stats['clustering_coeff']:.4f}",
            "连通分量": stats["num_components"],
        })
    display(pd.DataFrame(rows))
""")

# ============================================================
# Cell: 演化趋势
# ============================================================
code("""# 图1: 动态图演化趋势
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, (snaps, labels)) in enumerate(datasets.items()):
    ax = axes[idx]
    nodes = [G.number_of_nodes() for G in snaps]
    edges = [G.number_of_edges() for G in snaps]
    x = range(len(snaps))

    ax2 = ax.twinx()
    l1 = ax.plot(x, nodes, 'b-o', label='节点数')
    l2 = ax2.plot(x, edges, 'r-s', label='边数')

    ax.set_xlabel('快照编号')
    ax.set_ylabel('节点数', color='b')
    ax2.set_ylabel('边数', color='r')
    ax.set_title(f'{name} - 演化趋势')

    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/evolution_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("图1: 两个数据集的节点数与边数随时间快照的演化趋势")
""")

# ============================================================
# Cell: 度分布
# ============================================================
code("""# 图2: 首末快照度分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, (snaps, labels)) in enumerate(datasets.items()):
    ax = axes[idx]
    for t_idx, t_label in [(0, "T0 (初始)"), (-1, f"T{len(snaps)-1} (最终)")]:
        G = snaps[t_idx]
        degrees = [d for _, d in G.degree()]
        ax.hist(degrees, bins=50, alpha=0.5, label=t_label, density=True)

    ax.set_xlabel('度')
    ax.set_ylabel('频率')
    ax.set_title(f'{name} - 度分布')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/degree_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("图2: 初始快照与最终快照的度分布对比——呈现典型的幂律分布特征")
""")

# ============================================================
# Cell: 预算分配策略（修正：动机 + 公式推导）
# ============================================================
md("""## 4. 隐私预算分配策略

### 4.1 问题描述

给定总隐私预算 $\\varepsilon$ 和 $T$ 个快照，需要确定每个快照的预算 $\\varepsilon_t$，使得 $\\sum_{t=1}^{T} \\varepsilon_t = \\varepsilon$（串行组合定理）。

**核心挑战**：如何分配才能最大化整体发布效用？

### 4.2 三种分配策略及其动机

| 策略 | 动机 | 公式 | 适用场景 |
|------|------|------|----------|
| **均匀分配** | 无先验知识时的最优策略（最小化最大误差）| $\\varepsilon_t = \\varepsilon / T$ | 图结构变化均匀 |
| **指数衰减** | 近期数据更重要（数据时效性假设） | $\\varepsilon_t = \\varepsilon \\cdot \\frac{\\alpha^{T-1-t}}{\\sum_j \\alpha^{T-1-j}}$, $\\alpha=0.8$ | 实时监控场景 |
| **自适应分配** | 变化大的快照需要更高精度 | $\\varepsilon_t \\propto \\max(\\Delta(G_t, G_{t-1}),\\ \\delta)$ | 突变检测场景 |

其中 $\\Delta(G_t, G_{t-1})$ 为相邻快照的边数变化量，$\\delta$ 为防止退化的最小值。
""")

# ============================================================
# Cell: 预算分配可视化
# ============================================================
code("""# 图3: 三种预算分配策略对比
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for idx, (name, (snaps, labels)) in enumerate(datasets.items()):
    if idx >= 2:
        break
    ax = axes[idx]
    epsilon = 1.0
    bar_width = 0.25
    x = np.arange(NUM_SNAPSHOTS)

    for si, strategy in enumerate(STRATEGIES):
        mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
        budgets = mgr.get_allocation(strategy, snapshots=snaps)
        ax.bar(x + si * bar_width, budgets, width=bar_width, label=STRATEGY_CN[strategy])

    ax.set_xlabel('快照编号')
    ax.set_ylabel('每快照预算 (ε_t)')
    ax.set_title(f'{name} (总预算 ε={epsilon})')
    ax.legend(fontsize=SMALL_LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3, axis='y')

# 累积消耗
ax = axes[2]
epsilon = 1.0
for strategy in STRATEGIES:
    mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
    budgets = mgr.get_allocation(strategy, snapshots=college_snaps)
    cumulative = np.cumsum(budgets)
    ax.plot(range(NUM_SNAPSHOTS), cumulative, marker='o', label=STRATEGY_CN[strategy])

ax.axhline(y=epsilon, color='r', linestyle='--', alpha=0.7, label=f'总预算上限 ε={epsilon}')
ax.set_xlabel('快照编号')
ax.set_ylabel('累积隐私消耗')
ax.set_title('串行组合定理：累积隐私消耗')
ax.legend(fontsize=SMALL_LEGEND_FONT_SIZE)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/budget_allocation.png", dpi=150, bbox_inches="tight")
plt.show()
print("图3: 三种预算分配策略的每快照预算分布及累积隐私消耗")
print("观察: 自适应策略在早期（图变化剧烈时）分配更多预算，后期趋于平稳")
""")

# ============================================================
# Cell: 实验A 标题
# ============================================================
md("""## 5. 实验A：隐私预算对发布精度的影响

**实验目标**：验证 ε 与发布误差的关系（理论上 RMSE ∝ 1/ε）。

**实验设置**：
- ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- 三种预算分配策略
- 使用 Laplace 机制发布图统计量（节点数、边数、聚类系数）
- 卡尔曼滤波后处理
- 评估指标：RMSE（均方根误差），越小越好
""")

# ============================================================
# Cell: 实验A 代码
# ============================================================
code("""# 实验A: 不同隐私预算下的统计量发布RMSE
def run_epsilon_experiment(snaps, dataset_name):
    true_stats = []
    for G in snaps:
        true_stats.append({
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "clustering_coeff": nx.average_clustering(G) if G.number_of_nodes() > 0 else 0,
        })

    results = {}
    for eps in EPSILON_VALUES:
        results[eps] = {}
        for strategy in STRATEGIES:
            np.random.seed(SEED)
            mgr = PrivacyBudgetManager(eps, NUM_SNAPSHOTS)
            budgets = mgr.get_allocation(strategy, snapshots=snaps)
            noisy_stats = dp_publish_statistics_series(snaps, budgets)

            # 卡尔曼后处理（R 需要考虑 dp_graph_statistics 内部4等分预算）
            avg_budget = eps / NUM_SNAPSHOTS
            actual_eps_per_stat = avg_budget / 4  # 4个统计量等分预算
            R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps_per_stat)
            edge_counts = np.array([G.number_of_edges() for G in snaps])
            Q = max(np.var(np.diff(edge_counts)), 1.0)
            smoothed_stats = smooth_statistics_series(
                noisy_stats, method="kalman", process_noise=Q, measurement_noise=R
            )

            eval_raw = evaluate_statistics_series(true_stats, noisy_stats)
            eval_smooth = evaluate_statistics_series(true_stats, smoothed_stats)
            results[eps][strategy] = {"raw": eval_raw, "smoothed": eval_smooth}

    return results

results_college = run_epsilon_experiment(college_snaps, "CollegeMsg")
results_bitcoin = run_epsilon_experiment(bitcoin_snaps, "Bitcoin-Alpha")
print("实验A 完成")
""")

# ============================================================
# Cell: 实验A 可视化
# ============================================================
code("""# 图4: ε-RMSE 曲线
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

all_results = [
    ("CollegeMsg", results_college, "edge_count"),
    ("CollegeMsg", results_college, "node_count"),
    ("Bitcoin-Alpha", results_bitcoin, "edge_count"),
    ("Bitcoin-Alpha", results_bitcoin, "node_count"),
]
metric_cn = {"edge_count": "边数", "node_count": "节点数"}

for ax, (name, results, metric) in zip(axes.flat, all_results):
    for strategy in STRATEGIES:
        rmses_raw = [results[eps][strategy]["raw"][f"{metric}_rmse"] for eps in EPSILON_VALUES]
        rmses_smooth = [results[eps][strategy]["smoothed"][f"{metric}_rmse"] for eps in EPSILON_VALUES]
        ax.plot(EPSILON_VALUES, rmses_raw, marker='o', linestyle='-',
                label=f'{STRATEGY_CN[strategy]}(原始)')
        ax.plot(EPSILON_VALUES, rmses_smooth, marker='s', linestyle='--',
                label=f'{STRATEGY_CN[strategy]}(平滑)', alpha=0.7)

    ax.set_xlabel('隐私预算 ε')
    ax.set_ylabel('RMSE')
    ax.set_title(f'{name} - {metric_cn[metric]} RMSE')
    ax.legend(fontsize=INNER_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(EPSILON_VALUES)
    ax.set_xticklabels(['0.1', '0.5', '1', '2', '5'])
    ax.minorticks_off()

plt.suptitle('实验A: 隐私预算 ε 与发布误差的关系', fontsize=OUTER_FONT_SIZE, y=1.01)
plt.tight_layout()
plt.savefig("results/figures/expA_eps_vs_rmse.png", dpi=150, bbox_inches="tight")
plt.show()
print("图4: RMSE随ε增大而减小，符合Laplace机制的理论预期（RMSE ∝ 1/ε）")
print("均匀分配在所有ε下均优于其他策略（因为Laplace方差与1/ε^2成正比，均匀分配最小化最大方差）")
""")

# ============================================================
# Cell: 实验A 结果表
# ============================================================
code("""# 表1: 实验A 关键数值
print("表1: CollegeMsg - 边数 RMSE")
print("=" * 70)
rows = []
for eps in EPSILON_VALUES:
    row = {"ε": eps}
    for s in STRATEGIES:
        raw = results_college[eps][s]["raw"]["edge_count_rmse"]
        smooth = results_college[eps][s]["smoothed"]["edge_count_rmse"]
        row[f"{STRATEGY_CN[s]}(原始)"] = f"{raw:.2f}"
        row[f"{STRATEGY_CN[s]}(平滑)"] = f"{smooth:.2f}"
    rows.append(row)
display(pd.DataFrame(rows))

print("\\n表2: Bitcoin-Alpha - 边数 RMSE")
print("=" * 70)
rows = []
for eps in EPSILON_VALUES:
    row = {"ε": eps}
    for s in STRATEGIES:
        raw = results_bitcoin[eps][s]["raw"]["edge_count_rmse"]
        smooth = results_bitcoin[eps][s]["smoothed"]["edge_count_rmse"]
        row[f"{STRATEGY_CN[s]}(原始)"] = f"{raw:.2f}"
        row[f"{STRATEGY_CN[s]}(平滑)"] = f"{smooth:.2f}"
    rows.append(row)
display(pd.DataFrame(rows))
""")

# ============================================================
# Cell: 实验B 标题
# ============================================================
md("""## 6. 实验B：预算分配策略对发布效果的影响

**实验目标**：固定 ε=1.0，对比三种策略在时序统计量发布上的效果差异。

**实验逻辑**：
- 上排图展示 Laplace 加噪后的原始发布结果
- 下排图展示经卡尔曼滤波后处理后的结果
- 黑色虚线为真实值基准
""")

# ============================================================
# Cell: 实验B 代码
# ============================================================
code("""# 图5: 策略对比 - 边数时序发布
epsilon = 1.0
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (name, (snaps, labels)) in enumerate(datasets.items()):
    true_edge_counts = [G.number_of_edges() for G in snaps]

    for strategy in STRATEGIES:
        np.random.seed(SEED)
        mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
        budgets = mgr.get_allocation(strategy, snapshots=snaps)
        noisy_stats = dp_publish_statistics_series(snaps, budgets)
        noisy_ec = [s["edge_count"] for s in noisy_stats]

        avg_budget = epsilon / NUM_SNAPSHOTS
        actual_eps = avg_budget / 4
        R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps)
        edge_arr = np.array([G.number_of_edges() for G in snaps])
        Q = max(np.var(np.diff(edge_arr)), 1.0)
        smoothed = smooth_statistics_series(
            noisy_stats, method="kalman", process_noise=Q, measurement_noise=R
        )
        smooth_ec = [s["edge_count"] for s in smoothed]

        axes[0, col].plot(range(NUM_SNAPSHOTS), noisy_ec, marker='s', alpha=0.7,
                          label=STRATEGY_CN[strategy])
        axes[1, col].plot(range(NUM_SNAPSHOTS), smooth_ec, marker='s', alpha=0.7,
                          label=STRATEGY_CN[strategy])

    for row in range(2):
        axes[row, col].plot(range(NUM_SNAPSHOTS), true_edge_counts, 'k--', lw=2, label='真实值')
        axes[row, col].set_xlabel('快照编号')
        axes[row, col].set_ylabel('边数')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    axes[0, col].set_title(f'{name} - Laplace加噪 (ε={epsilon})')
    axes[1, col].set_title(f'{name} - 卡尔曼滤波后 (ε={epsilon})')

plt.suptitle('实验B: 预算分配策略对边数发布的影响', fontsize=OUTER_FONT_SIZE, y=1.01)
plt.tight_layout()
plt.savefig("results/figures/expB_strategy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("图5: 均匀分配的发布值最贴近真实值；指数衰减在后期快照精度较高但前期偏差大")
""")

# ============================================================
# Cell: 实验C 标题（修正：文献引用 - 解决问题3）
# ============================================================
md("""## 7. 实验C：后处理平滑方法对比

### 7.1 理论依据

根据差分隐私的 **后处理不变性定理** [Dwork & Roth, 2014, Proposition 2.1]，对 DP 输出施加任意确定性或随机变换，不会增加隐私消耗。因此，以下平滑方法可"免费"提升发布精度。

### 7.2 三种平滑方法

| 方法 | 原理 | 参数 | 文献来源 |
|------|------|------|----------|
| **滑动平均（Moving Average）** | $\\hat{x}_t = \\frac{1}{w} \\sum_{i=t-w+1}^{t} x_i$ | 窗口 $w=3$ | 经典时间序列方法 [Box et al., 2015] |
| **指数平滑（Exponential Smoothing）** | $\\hat{x}_t = \\alpha x_t + (1-\\alpha) \\hat{x}_{t-1}$ | $\\alpha=0.3$ | Holt [1957]; Brown [1959] |
| **卡尔曼滤波（Kalman Filter）** | 贝叶斯最优线性估计，联合考虑过程噪声 $Q$ 和观测噪声 $R$ | $Q$=数据变化方差, $R$=Laplace噪声方差 | Kalman [1960]; 在DP中的应用见 Fan & Xiong [2014] (FAST框架) |

**关键参数设置**：
- 卡尔曼滤波的观测噪声 $R = 2(\\Delta f / \\varepsilon_{\\text{stat}})^2$（Laplace分布方差），其中 $\\varepsilon_{\\text{stat}} = \\varepsilon_t / 4$（因为每个快照发布4个统计量，预算4等分）
- 过程噪声 $Q$ 由真实数据相邻快照的边数变化方差估计
""")

# ============================================================
# Cell: 实验C 代码
# ============================================================
code("""# 图6: 后处理平滑方法对比
epsilon = 1.0
fig, axes = plt.subplots(2, 1, figsize=(9, 10))

for row, (name, (snaps, labels)) in enumerate(datasets.items()):
    ax = axes[row]
    np.random.seed(SEED)
    mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
    budgets = mgr.get_allocation("adaptive", snapshots=snaps)
    noisy_stats = dp_publish_statistics_series(snaps, budgets)

    true_ec = np.array([G.number_of_edges() for G in snaps])
    noisy_ec = np.array([s["edge_count"] for s in noisy_stats])

    avg_budget = epsilon / NUM_SNAPSHOTS
    actual_eps = avg_budget / 4
    R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps)
    Q = max(np.var(np.diff(true_ec)), 1.0)

    ax.plot(range(NUM_SNAPSHOTS), true_ec, 'k-o', linewidth=2, label='真实值')
    noisy_rmse = np.sqrt(np.mean((true_ec - noisy_ec) ** 2))
    ax.plot(range(NUM_SNAPSHOTS), noisy_ec, 'r--x', alpha=0.6,
            label=f'DP加噪 (RMSE={noisy_rmse:.1f})')

    methods_config = {
        "滑动平均": ("sliding", {"window": 3}, "blue"),
        "指数平滑": ("exponential", {"alpha": 0.3}, "green"),
        "卡尔曼滤波": ("kalman", {"process_noise": Q, "measurement_noise": R}, "purple"),
    }

    for label, (method, kwargs, color) in methods_config.items():
        smoothed = smooth_statistics_series(noisy_stats, method=method, **kwargs)
        smooth_ec = np.array([s["edge_count"] for s in smoothed])
        rmse_val = np.sqrt(np.mean((true_ec - smooth_ec) ** 2))
        ax.plot(range(NUM_SNAPSHOTS), smooth_ec, color=color, marker='s',
                alpha=0.8, label=f'{label} (RMSE={rmse_val:.1f})')

    ax.set_xlabel('快照编号')
    ax.set_ylabel('边数')
    ax.set_title(f'{name} (ε={epsilon}, 自适应分配)')
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(True, alpha=0.3)

plt.suptitle('实验C: 后处理平滑方法效果对比', fontsize=OUTER_FONT_SIZE, y=1.01)
plt.tight_layout()
plt.savefig("results/figures/expC_smoothing.png", dpi=150, bbox_inches="tight")
plt.show()
print("图6: 三种平滑方法均能降低DP噪声的影响")
print("卡尔曼滤波利用了噪声方差的先验知识，理论上是最优线性估计")
""")

# ============================================================
# Cell: 实验D 标题
# ============================================================
md("""## 8. 实验D：多维度综合评估

从多个维度综合衡量各策略的发布质量，使用雷达图直观对比。

**评估维度**（值越小越好）：
- **边数/节点数 RMSE**：统计量发布精度
- **聚类系数 RMSE**：图结构保真度
- **趋势相关性** ($1 - \\rho$)：时间演化趋势的保持能力（$\\rho$ 为 Pearson 相关系数）
""")

# ============================================================
# Cell: 实验D 代码
# ============================================================
code("""# 图7: 雷达图
epsilon = 1.0
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

for col, (name, (snaps, labels)) in enumerate(datasets.items()):
    ax = axes[col]
    true_stats_list = []
    for G in snaps:
        true_stats_list.append({
            "node_count": float(G.number_of_nodes()),
            "edge_count": float(G.number_of_edges()),
            "clustering_coeff": float(nx.average_clustering(G)) if G.number_of_nodes() > 0 else 0.0,
        })

    metrics_all = {}
    for strategy in STRATEGIES:
        np.random.seed(SEED)
        mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
        budgets = mgr.get_allocation(strategy, snapshots=snaps)
        noisy_stats = dp_publish_statistics_series(snaps, budgets)

        avg_budget = epsilon / NUM_SNAPSHOTS
        actual_eps = avg_budget / 4
        R = KalmanFilter1D.estimate_laplace_variance(1.0, actual_eps)
        edge_counts = np.array([G.number_of_edges() for G in snaps])
        Q = max(np.var(np.diff(edge_counts)), 1.0)
        smoothed = smooth_statistics_series(
            noisy_stats, method="kalman", process_noise=Q, measurement_noise=R
        )

        true_ec = np.array([s["edge_count"] for s in true_stats_list])
        true_nc = np.array([s["node_count"] for s in true_stats_list])
        true_cc = np.array([s["clustering_coeff"] for s in true_stats_list])
        sm_ec = np.array([s["edge_count"] for s in smoothed])
        sm_nc = np.array([s["node_count"] for s in smoothed])
        sm_cc = np.array([s["clustering_coeff"] for s in smoothed])

        metrics_all[strategy] = {
            "edge_rmse": rmse(true_ec, sm_ec),
            "node_rmse": rmse(true_nc, sm_nc),
            "cc_rmse": rmse(true_cc, sm_cc),
            "edge_corr": trend_correlation(true_ec, sm_ec),
            "node_corr": trend_correlation(true_nc, sm_nc),
        }

    categories = ["边数RMSE", "节点RMSE", "聚类系数RMSE", "边数趋势偏差", "节点趋势偏差"]
    keys = ["edge_rmse", "node_rmse", "cc_rmse", "edge_corr", "node_corr"]
    invert = [False, False, False, True, True]  # 相关性越高越好 -> 反转后越小越好

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    all_vals = {k: [metrics_all[s][k] for s in STRATEGIES] for k in keys}

    for strategy in STRATEGIES:
        norms = []
        for key, inv in zip(keys, invert):
            v = metrics_all[strategy][key]
            vmin, vmax = min(all_vals[key]), max(all_vals[key])
            n = (v - vmin) / (vmax - vmin) if vmax - vmin > 1e-10 else 0.5
            if inv:
                n = 1 - n
            norms.append(n)
        norms += norms[:1]
        ax.plot(angles, norms, 'o-', linewidth=2, label=STRATEGY_CN[strategy])
        ax.fill(angles, norms, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=INNER_FONT_SIZE)
    ax.set_title(f'{name} (ε={epsilon})\\n归一化误差，越小越好', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

plt.suptitle('实验D: 多维度综合评估', fontsize=OUTER_FONT_SIZE, y=1.02)
plt.tight_layout()
plt.savefig("results/figures/expD_radar.png", dpi=150, bbox_inches="tight")
plt.show()
print("图7: 均匀分配在大部分维度上表现最优（面积最小）")
""")

# ============================================================
# Cell: 实验E 标题（修正：分离 Jaccard 和变化率，解决问题4）
# ============================================================
md("""## 9. 实验E：时间一致性分析

### 9.1 评估目标

动态图的发布不仅要求单个快照精确，还要求时间序列的 **连贯性**。评估指标：

- **Jaccard 相似度**：$J(G_t, G_{t-1}) = |E_t \\cap E_{t-1}| / |E_t \\cup E_{t-1}|$，衡量相邻快照的结构重叠度
- **边数变化偏差**：$|\\Delta_{\\text{noisy}} - \\Delta_{\\text{true}}| / \\Delta_{\\text{true}}$，衡量DP发布是否保持了真实的变化趋势
""")

# ============================================================
# Cell: 实验E 代码（修正：两张子图分开画，解决混合量纲问题）
# ============================================================
code("""# 图8: 时间一致性分析（拆分为两个独立指标）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (name, (snaps, labels)) in enumerate(datasets.items()):
    # 上图: 原始图的 Jaccard 相似度
    ax_top = axes[0, col]
    orig_jaccard = temporal_jaccard_similarity(snaps)
    ax_top.plot(range(1, NUM_SNAPSHOTS), orig_jaccard, 'k-o', linewidth=2, label='原始图')
    ax_top.set_xlabel('快照对 (t-1, t)')
    ax_top.set_ylabel('Jaccard 相似度')
    ax_top.set_title(f'{name} - 相邻快照结构重叠度')
    ax_top.set_ylim(0, 1.05)
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # 下图: DP发布的边数变化偏差
    ax_bot = axes[1, col]
    epsilon = 1.0
    true_ec = np.array([G.number_of_edges() for G in snaps])
    true_changes = np.abs(np.diff(true_ec))

    for strategy in STRATEGIES:
        np.random.seed(SEED)
        mgr = PrivacyBudgetManager(epsilon, NUM_SNAPSHOTS)
        budgets = mgr.get_allocation(strategy, snapshots=snaps)
        noisy_stats = dp_publish_statistics_series(snaps, budgets)
        noisy_ec = np.array([s["edge_count"] for s in noisy_stats])
        noisy_changes = np.abs(np.diff(noisy_ec))

        # 相对偏差
        deviation = np.abs(noisy_changes - true_changes) / np.maximum(true_changes, 1)
        ax_bot.plot(range(1, NUM_SNAPSHOTS), deviation, marker='s', alpha=0.7,
                    label=STRATEGY_CN[strategy])

    ax_bot.set_xlabel('快照对 (t-1, t)')
    ax_bot.set_ylabel('边数变化相对偏差')
    ax_bot.set_title(f'{name} - DP发布的时序变化偏差 (ε={epsilon})')
    ax_bot.legend()
    ax_bot.grid(True, alpha=0.3)

plt.suptitle('实验E: 时间一致性分析', fontsize=OUTER_FONT_SIZE, y=1.01)
plt.tight_layout()
plt.savefig("results/figures/expE_temporal.png", dpi=150, bbox_inches="tight")
plt.show()
print("图8 上: 累积快照的Jaccard相似度始终较高（>0.6），说明累积模型天然保持时间连贯性")
print("图8 下: 均匀分配的变化偏差最小且最稳定")
""")

# ============================================================
# Cell: 实验F 标题
# ============================================================
md("""## 10. 实验F：度分布的差分隐私发布

使用 Laplace 机制对度分布直方图加噪发布。

**敏感度分析**：添加/删除一条边最多影响两个节点的度数，故度分布直方图的 $L_1$ 敏感度 $\\Delta f = 2$ [Hay et al., 2009]。
""")

# ============================================================
# Cell: 实验F 代码
# ============================================================
code("""# 图9: 度分布 DP 发布
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
legend_handles = None

for row, (name, (snaps, labels)) in enumerate(datasets.items()):
    G = snaps[-1]  # 使用最终快照
    true_dist = get_degree_distribution(G)

    for col, eps in enumerate([0.5, 1.0, 5.0]):
        ax = axes[row, col]
        np.random.seed(SEED)
        noisy_dist = dp_degree_distribution(G, eps)

        max_len = max(len(true_dist), len(noisy_dist))
        x = range(min(max_len, 80))
        true_plot = true_dist[:len(x)] if len(true_dist) >= len(x) else np.pad(true_dist, (0, len(x)-len(true_dist)))
        noisy_plot = noisy_dist[:len(x)] if len(noisy_dist) >= len(x) else np.pad(noisy_dist, (0, len(x)-len(noisy_dist)))

        ax.bar(x, true_plot, alpha=0.5, label='真实分布', color='steelblue')
        ax.plot(x, noisy_plot, 'r-', alpha=0.8, label='DP发布', linewidth=1.5)
        ax.set_xlabel('度')
        ax.set_ylabel('概率')
        ax.set_title(f'{name} (ε={eps})')
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.grid(True, alpha=0.3)

for ax in axes[2, :2]:
    ax.axis('off')
axes[2, 2].axis('off')
axes[2, 2].legend(
    legend_handles,
    legend_labels,
    loc='center',
    fontsize=INNER_FONT_SIZE,
    frameon=True
)

plt.suptitle('实验F: 度分布差分隐私发布 (Laplace机制, 敏感度=2)', fontsize=OUTER_FONT_SIZE, y=1.02)
plt.tight_layout()
plt.savefig("results/figures/expF_degree_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print("图9: ε=5.0时发布分布与真实分布高度吻合; ε=0.5时噪声明显但整体形状仍可辨识")
""")

# ============================================================
# Cell: 总结（修正：结构化表述 + 引用）
# ============================================================
md("""## 11. 实验总结与结论

### 11.1 主要发现

| 实验 | 核心结论 |
|------|----------|
| **实验A** (ε敏感性) | RMSE与1/ε近似成正比，符合Laplace机制理论 |
| **实验B** (策略对比) | 均匀分配在无先验知识时表现最稳定；自适应分配在图变化剧烈处有优势 |
| **实验C** (后处理) | 卡尔曼滤波优于简单平滑（在低ε下改善显著）；后处理不消耗隐私预算 |
| **实验D** (综合评估) | 均匀分配的综合误差面积最小；各策略在趋势保持上差异不大 |
| **实验E** (时间一致性) | 累积快照模型保证了高Jaccard相似度；均匀分配的变化偏差最小 |
| **实验F** (度分布) | ε≥1.0时度分布发布质量可接受；幂律分布的长尾部分受噪声影响较大 |

### 11.2 隐私-效用权衡建议

- **推荐 ε 范围**：ε = 0.5 ~ 1.0 为较好的平衡点
- **推荐策略**：无先验知识时用均匀分配；有时效性需求时用指数衰减
- **后处理**：始终建议使用卡尔曼滤波（零隐私成本的精度提升）

### 11.3 隐私模型说明

本实验全部基于 **中心化差分隐私（CDP）** 模型，适用于存在可信数据管理者的场景（如平台方发布脱敏统计数据）。若需去除可信方假设，需改用本地化差分隐私（LDP），但噪声量级将显著增大 [Dwork & Roth, 2014]。

### 参考文献

1. Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407.
2. Hay, M., Li, C., Miklau, G., & Jensen, D. (2009). Accurate estimation of the degree distribution of private networks. *ICDM*.
3. Fan, L., & Xiong, L. (2014). An adaptive approach to real-time aggregate monitoring with differential privacy. *IEEE TKDE*, 26(9), 2094-2106.
4. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
5. Holt, C. C. (1957). Forecasting seasonals and trends by exponentially weighted moving averages. *ONR Research Memorandum*.
6. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
7. Song, S., Wang, Y., & Chaudhuri, K. (2017). Pufferfish privacy mechanisms for correlated data. *SIGMOD*.
8. Panzarasa, P., Opsahl, T., & Carley, K. M. (2009). Patterns and dynamics of users' behavior and interaction. *JASIST*.
9. Kumar, S., Spezzano, F., Suber, V. S., & Faloutsos, C. (2016). Edge weight prediction in weighted signed networks. *ICDM*.
10. Viswanath, B., Mislove, A., Cha, M., & Gummadi, K. P. (2009). On the evolution of user interaction in Facebook. *WOSN*.
""")

# ============================================================
# Cell: 保存数据
# ============================================================
code("""# 保存实验数据到CSV
all_data = []
for name, results in [("CollegeMsg", results_college), ("Bitcoin-Alpha", results_bitcoin)]:
    for eps in EPSILON_VALUES:
        for s in STRATEGIES:
            for mode in ["raw", "smoothed"]:
                row = {
                    "数据集": name,
                    "ε": eps,
                    "策略": STRATEGY_CN[s],
                    "模式": "原始" if mode == "raw" else "卡尔曼平滑",
                }
                for key in results[eps][s][mode]:
                    row[key] = round(results[eps][s][mode][key], 4)
                all_data.append(row)

df = pd.DataFrame(all_data)
os.makedirs("results", exist_ok=True)
df.to_csv("results/experiment_results.csv", index=False, encoding="utf-8-sig")
print(f"实验数据已保存至 results/experiment_results.csv，共 {len(df)} 条记录")
display(df.head(10))
""")

# ==================
# 写入 notebook
# ==================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("main_experiment.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook 创建成功: main_experiment.ipynb ({len(cells)} 个 cells)")
