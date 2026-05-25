"""
差分隐私机制模块
实现 Laplace 机制（统计量发布）和 随机边扰动（图结构发布）
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
from copy import deepcopy


# ============================================================
# Laplace 机制：用于图统计量的 DP 发布
# ============================================================

def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """
    Laplace 机制
    发布值 = 真实值 + Lap(sensitivity / epsilon)

    Args:
        true_value: 真实统计量
        sensitivity: 全局灵敏度 Δf
        epsilon: 隐私预算
    Returns:
        加噪后的值
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise


def dp_degree_distribution(G: nx.Graph, epsilon: float) -> np.ndarray:
    """
    差分隐私度分布发布
    灵敏度分析：添加/删除一条边最多影响2个节点的度，度分布直方图最多变化2
    全局灵敏度 Δf = 2 (L1范数)

    Args:
        G: 原始图
        epsilon: 隐私预算
    Returns:
        加噪后的度分布（频次向量）
    """
    if G.number_of_nodes() == 0:
        return np.array([0.0])

    degrees = [d for _, d in G.degree()]
    max_deg = max(degrees) if degrees else 0
    hist = np.zeros(max_deg + 1)
    for d in degrees:
        hist[d] += 1

    # 灵敏度 = 2（一条边变动影响两个节点的度）
    sensitivity = 2.0
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=len(hist))
    noisy_hist = hist + noise

    # 后处理：非负约束 + 归一化
    noisy_hist = np.maximum(noisy_hist, 0)
    total = noisy_hist.sum()
    if total > 0:
        noisy_hist = noisy_hist / total
    return noisy_hist


def dp_node_count(G: nx.Graph, epsilon: float) -> float:
    """DP 节点数发布，灵敏度=1"""
    return laplace_mechanism(G.number_of_nodes(), sensitivity=1.0, epsilon=epsilon)


def dp_edge_count(G: nx.Graph, epsilon: float) -> float:
    """DP 边数发布，灵敏度=1（添加/删除一条边）"""
    return laplace_mechanism(G.number_of_edges(), sensitivity=1.0, epsilon=epsilon)


def dp_triangle_count(G: nx.Graph, epsilon: float) -> float:
    """
    DP 三角形计数发布
    灵敏度 = max_degree（删除一条边最多影响 min(d_u, d_v) 个三角形）
    为安全起见使用上界 max_degree
    """
    triangles = sum(nx.triangles(G).values()) // 3
    max_deg = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
    sensitivity = max(max_deg, 1)
    return max(0, laplace_mechanism(triangles, sensitivity, epsilon))


def dp_clustering_coefficient(G: nx.Graph, epsilon: float) -> float:
    """
    DP 聚类系数发布
    聚类系数范围 [0, 1]，灵敏度上界 = 1/n（单节点对全局平均的最大影响）
    """
    if G.number_of_nodes() == 0:
        return 0.0
    true_cc = nx.average_clustering(G)
    sensitivity = 1.0 / G.number_of_nodes()
    result = laplace_mechanism(true_cc, sensitivity, epsilon)
    return np.clip(result, 0, 1)


def dp_graph_statistics(G: nx.Graph, epsilon: float) -> dict:
    """
    发布一组图统计量，通过并行组合分配预算
    将 epsilon 均分给各统计量
    """
    num_queries = 4
    eps_each = epsilon / num_queries
    return {
        "node_count": dp_node_count(G, eps_each),
        "edge_count": dp_edge_count(G, eps_each),
        "triangle_count": dp_triangle_count(G, eps_each),
        "clustering_coeff": dp_clustering_coefficient(G, eps_each),
    }


# ============================================================
# 随机边扰动：用于图结构的 DP 发布
# ============================================================

def edge_perturbation(G: nx.Graph, epsilon: float) -> nx.Graph:
    """
    基于随机响应的边扰动算法（满足 ε-边差分隐私）

    对邻接矩阵的每个上三角元素独立翻转：
    - 边存在时以概率 p 保留
    - 边不存在时以概率 q 添加
    其中 p = e^ε / (1 + e^ε)，q = 1 / (1 + e^ε)

    Args:
        G: 原始图
        epsilon: 隐私预算
    Returns:
        扰动后的图
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    p_keep = np.exp(epsilon) / (1 + np.exp(epsilon))  # 边存在时保留概率
    p_add = 1.0 / (1 + np.exp(epsilon))               # 边不存在时添加概率

    G_perturbed = nx.Graph()
    G_perturbed.add_nodes_from(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            u, v = nodes[i], nodes[j]
            if G.has_edge(u, v):
                # 边存在：以 p_keep 概率保留
                if np.random.random() < p_keep:
                    G_perturbed.add_edge(u, v)
            else:
                # 边不存在：以 p_add 概率添加
                if np.random.random() < p_add:
                    G_perturbed.add_edge(u, v)

    return G_perturbed


def edge_perturbation_sparse(G: nx.Graph, epsilon: float) -> nx.Graph:
    """
    稀疏图优化版边扰动算法
    对于大规模稀疏图，不遍历所有 O(n^2) 节点对，
    而是只处理已有边（翻转删除）+ 采样添加新边

    Args:
        G: 原始图
        epsilon: 隐私预算
    Returns:
        扰动后的图
    """
    nodes = list(G.nodes())
    n = len(nodes)
    edges = set(G.edges())

    p_keep = np.exp(epsilon) / (1 + np.exp(epsilon))
    p_add = 1.0 / (1 + np.exp(epsilon))

    G_perturbed = nx.Graph()
    G_perturbed.add_nodes_from(nodes)

    # 处理已有边：以 p_keep 概率保留
    for u, v in edges:
        if np.random.random() < p_keep:
            G_perturbed.add_edge(u, v)

    # 非边采样添加：期望添加 p_add * (n*(n-1)/2 - |E|) 条边
    total_possible = n * (n - 1) // 2
    num_non_edges = total_possible - len(edges)
    expected_add = int(p_add * num_non_edges)

    # 随机采样节点对作为候选新边
    added = 0
    max_attempts = expected_add * 3
    attempts = 0
    while added < expected_add and attempts < max_attempts:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            u, v = nodes[i], nodes[j]
            if not G.has_edge(u, v) and not G_perturbed.has_edge(u, v):
                G_perturbed.add_edge(u, v)
                added += 1
        attempts += 1

    return G_perturbed


def dp_publish_dynamic_graph(
    snapshots: List[nx.Graph],
    budgets: np.ndarray,
    method: str = "sparse"
) -> List[nx.Graph]:
    """
    动态图的差分隐私结构发布

    Args:
        snapshots: 原始图快照序列
        budgets: 每个快照的隐私预算
        method: "full" 使用完整边扰动, "sparse" 使用稀疏优化版
    Returns:
        扰动后的图快照序列
    """
    perturb_func = edge_perturbation if method == "full" else edge_perturbation_sparse
    published = []
    for t, (G, eps) in enumerate(zip(snapshots, budgets)):
        G_pub = perturb_func(G, eps)
        published.append(G_pub)
        print(f"  快照 T{t}: ε={eps:.4f}, 原始边={G.number_of_edges()}, 发布边={G_pub.number_of_edges()}")
    return published


def dp_publish_statistics_series(
    snapshots: List[nx.Graph],
    budgets: np.ndarray
) -> List[dict]:
    """
    动态图的差分隐私统计量序列发布

    Args:
        snapshots: 原始图快照序列
        budgets: 每个快照的隐私预算
    Returns:
        每个快照的 DP 统计量字典列表
    """
    stats_series = []
    for t, (G, eps) in enumerate(zip(snapshots, budgets)):
        stats = dp_graph_statistics(G, eps)
        stats_series.append(stats)
    return stats_series


if __name__ == "__main__":
    # 简单测试
    G = nx.erdos_renyi_graph(50, 0.1, seed=42)
    print(f"原始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # Laplace 统计量发布
    stats = dp_graph_statistics(G, epsilon=1.0)
    print(f"DP统计量 (ε=1.0): {stats}")

    # 边扰动
    G_pub = edge_perturbation_sparse(G, epsilon=1.0)
    print(f"扰动图: {G_pub.number_of_nodes()} 节点, {G_pub.number_of_edges()} 边")
