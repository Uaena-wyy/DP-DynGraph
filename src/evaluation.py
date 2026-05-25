"""
评估模块
实现多维度评估指标：隐私保护能力、结构保持性、动态查询误差、时间一致性
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
from scipy.special import rel_entr


# ============================================================
# 结构保持性评估
# ============================================================

def kl_divergence_degree(G_orig: nx.Graph, G_pub: nx.Graph) -> float:
    """
    计算度分布的 KL 散度
    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    Args:
        G_orig: 原始图
        G_pub: 发布图
    Returns:
        KL 散度值（越小越好）
    """
    dist_orig = _get_degree_dist(G_orig)
    dist_pub = _get_degree_dist(G_pub)

    # 对齐长度
    max_len = max(len(dist_orig), len(dist_pub))
    p = np.zeros(max_len)
    q = np.zeros(max_len)
    p[:len(dist_orig)] = dist_orig
    q[:len(dist_pub)] = dist_pub

    # 避免零值导致 log 问题，添加小量平滑
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(rel_entr(p, q)))


def clustering_coefficient_error(G_orig: nx.Graph, G_pub: nx.Graph) -> float:
    """
    聚类系数相对误差
    |CC_orig - CC_pub| / max(CC_orig, 1e-10)
    """
    cc_orig = nx.average_clustering(G_orig) if G_orig.number_of_nodes() > 0 else 0
    cc_pub = nx.average_clustering(G_pub) if G_pub.number_of_nodes() > 0 else 0
    return abs(cc_orig - cc_pub) / max(cc_orig, 1e-10)


def component_count_error(G_orig: nx.Graph, G_pub: nx.Graph) -> float:
    """连通分量数量的绝对误差"""
    c_orig = nx.number_connected_components(G_orig) if G_orig.number_of_nodes() > 0 else 0
    c_pub = nx.number_connected_components(G_pub) if G_pub.number_of_nodes() > 0 else 0
    return abs(c_orig - c_pub)


def edge_count_relative_error(G_orig: nx.Graph, G_pub: nx.Graph) -> float:
    """边数相对误差"""
    e_orig = G_orig.number_of_edges()
    e_pub = G_pub.number_of_edges()
    return abs(e_orig - e_pub) / max(e_orig, 1)


# ============================================================
# 动态查询误差评估
# ============================================================

def rmse(true_series: np.ndarray, noisy_series: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(np.mean((true_series - noisy_series) ** 2)))


def mean_absolute_error(true_series: np.ndarray, noisy_series: np.ndarray) -> float:
    """平均绝对误差"""
    return float(np.mean(np.abs(true_series - noisy_series)))


def trend_correlation(true_series: np.ndarray, noisy_series: np.ndarray) -> float:
    """
    趋势相关性（Pearson 相关系数）
    衡量发布序列是否保持了真实序列的趋势方向
    """
    if len(true_series) < 2:
        return 1.0
    if np.std(true_series) == 0 or np.std(noisy_series) == 0:
        return 0.0
    corr = np.corrcoef(true_series, noisy_series)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def degree_series_rmse(
    orig_snapshots: List[nx.Graph], pub_snapshots: List[nx.Graph]
) -> float:
    """
    度序列的时序 RMSE
    对每个快照计算平均度，然后计算整个序列的 RMSE
    """
    avg_deg_orig = np.array([_avg_degree(G) for G in orig_snapshots])
    avg_deg_pub = np.array([_avg_degree(G) for G in pub_snapshots])
    return rmse(avg_deg_orig, avg_deg_pub)


def edge_change_rate_error(
    orig_snapshots: List[nx.Graph], pub_snapshots: List[nx.Graph]
) -> float:
    """
    边变化率估计误差
    变化率 = |E_t - E_{t-1}| / max(|E_{t-1}|, 1)
    """
    orig_rates = _compute_edge_change_rates(orig_snapshots)
    pub_rates = _compute_edge_change_rates(pub_snapshots)
    if len(orig_rates) == 0:
        return 0.0
    return rmse(orig_rates, pub_rates)


# ============================================================
# 时间一致性评估
# ============================================================

def temporal_jaccard_similarity(snapshots: List[nx.Graph]) -> np.ndarray:
    """
    计算相邻快照间的 Jaccard 相似度序列
    J(t, t-1) = |E_t ∩ E_{t-1}| / |E_t ∪ E_{t-1}|

    Returns:
        长度为 T-1 的相似度数组
    """
    T = len(snapshots)
    similarities = np.zeros(T - 1)

    for t in range(1, T):
        edges_prev = set(snapshots[t - 1].edges())
        edges_curr = set(snapshots[t].edges())
        union = edges_prev | edges_curr
        if len(union) == 0:
            similarities[t - 1] = 1.0
        else:
            intersection = edges_prev & edges_curr
            similarities[t - 1] = len(intersection) / len(union)

    return similarities


def temporal_smoothness_score(snapshots: List[nx.Graph]) -> float:
    """时间平滑性得分（Jaccard 相似度的均值）"""
    sims = temporal_jaccard_similarity(snapshots)
    return float(np.mean(sims)) if len(sims) > 0 else 1.0


def count_abrupt_changes(
    snapshots: List[nx.Graph], threshold: float = 0.3
) -> int:
    """
    统计突变次数
    当相邻快照 Jaccard 相似度低于 threshold 时视为突变
    """
    sims = temporal_jaccard_similarity(snapshots)
    return int(np.sum(sims < threshold))


# ============================================================
# 综合评估
# ============================================================

def evaluate_snapshot_pair(G_orig: nx.Graph, G_pub: nx.Graph) -> dict:
    """评估单个快照的发布质量"""
    return {
        "kl_divergence": kl_divergence_degree(G_orig, G_pub),
        "clustering_error": clustering_coefficient_error(G_orig, G_pub),
        "component_error": component_count_error(G_orig, G_pub),
        "edge_error": edge_count_relative_error(G_orig, G_pub),
    }


def evaluate_dynamic_publication(
    orig_snapshots: List[nx.Graph],
    pub_snapshots: List[nx.Graph]
) -> dict:
    """
    对整个动态图发布进行综合评估

    Returns:
        包含多维度指标的字典
    """
    T = len(orig_snapshots)

    # 逐快照结构评估
    kl_scores = []
    cc_errors = []
    edge_errors = []

    for t in range(T):
        kl_scores.append(kl_divergence_degree(orig_snapshots[t], pub_snapshots[t]))
        cc_errors.append(clustering_coefficient_error(orig_snapshots[t], pub_snapshots[t]))
        edge_errors.append(edge_count_relative_error(orig_snapshots[t], pub_snapshots[t]))

    # 动态查询误差
    deg_rmse = degree_series_rmse(orig_snapshots, pub_snapshots)
    edge_change_err = edge_change_rate_error(orig_snapshots, pub_snapshots)

    # 时间一致性
    orig_smoothness = temporal_smoothness_score(orig_snapshots)
    pub_smoothness = temporal_smoothness_score(pub_snapshots)
    pub_abrupt = count_abrupt_changes(pub_snapshots)

    # 度序列趋势相关性
    avg_deg_orig = np.array([_avg_degree(G) for G in orig_snapshots])
    avg_deg_pub = np.array([_avg_degree(G) for G in pub_snapshots])
    deg_trend_corr = trend_correlation(avg_deg_orig, avg_deg_pub)

    return {
        "avg_kl_divergence": float(np.mean(kl_scores)),
        "avg_clustering_error": float(np.mean(cc_errors)),
        "avg_edge_relative_error": float(np.mean(edge_errors)),
        "degree_series_rmse": deg_rmse,
        "edge_change_rate_error": edge_change_err,
        "degree_trend_correlation": deg_trend_corr,
        "orig_temporal_smoothness": orig_smoothness,
        "pub_temporal_smoothness": pub_smoothness,
        "pub_abrupt_changes": pub_abrupt,
        "kl_per_snapshot": kl_scores,
        "cc_error_per_snapshot": cc_errors,
        "edge_error_per_snapshot": edge_errors,
    }


def evaluate_statistics_series(
    true_stats_series: List[dict],
    noisy_stats_series: List[dict]
) -> dict:
    """
    评估统计量序列发布的精度

    Returns:
        每个统计量的 RMSE 和趋势相关性
    """
    keys = true_stats_series[0].keys()
    result = {}

    for key in keys:
        true_vals = np.array([s[key] for s in true_stats_series])
        noisy_vals = np.array([s[key] for s in noisy_stats_series])
        result[f"{key}_rmse"] = rmse(true_vals, noisy_vals)
        result[f"{key}_trend_corr"] = trend_correlation(true_vals, noisy_vals)
        result[f"{key}_mae"] = mean_absolute_error(true_vals, noisy_vals)

    return result


# ============================================================
# 辅助函数
# ============================================================

def _get_degree_dist(G: nx.Graph) -> np.ndarray:
    """获取归一化度分布"""
    if G.number_of_nodes() == 0:
        return np.array([1.0])
    degrees = [d for _, d in G.degree()]
    max_deg = max(degrees)
    dist = np.zeros(max_deg + 1)
    for d in degrees:
        dist[d] += 1
    total = dist.sum()
    if total > 0:
        dist = dist / total
    return dist


def _avg_degree(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    return 2 * G.number_of_edges() / G.number_of_nodes()


def _compute_edge_change_rates(snapshots: List[nx.Graph]) -> np.ndarray:
    T = len(snapshots)
    if T < 2:
        return np.array([])
    rates = np.zeros(T - 1)
    for t in range(1, T):
        e_prev = snapshots[t - 1].number_of_edges()
        e_curr = snapshots[t].number_of_edges()
        rates[t - 1] = abs(e_curr - e_prev) / max(e_prev, 1)
    return rates
