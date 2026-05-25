"""
动态图数据加载模块
将不同格式的时序图数据统一转换为图快照序列 (List[nx.Graph])
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Optional


class DynamicGraphLoader:
    """动态图数据加载器，支持多种数据集格式"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_college_msg(self, num_snapshots: int = 10) -> Tuple[List[nx.Graph], List[str]]:
        """
        加载 CollegeMsg 数据集
        格式: SRC TGT UNIX_TIMESTAMP (空格分隔)
        返回: (图快照列表, 时间标签列表)
        """
        filepath = os.path.join(self.data_dir, "CollegeMsg.txt")
        df = pd.read_csv(filepath, sep=" ", header=None, names=["src", "tgt", "timestamp"])
        return self._build_snapshots(df, num_snapshots, dataset_name="CollegeMsg")

    def load_bitcoin_alpha(self, num_snapshots: int = 10) -> Tuple[List[nx.Graph], List[str]]:
        """
        加载 Bitcoin-Alpha 数据集
        格式: SOURCE,TARGET,RATING,TIMESTAMP (逗号分隔)
        返回: (图快照列表, 时间标签列表)
        """
        filepath = os.path.join(self.data_dir, "soc-sign-bitcoinalpha.csv")
        df = pd.read_csv(filepath, header=None, names=["src", "tgt", "rating", "timestamp"])
        # 只保留正向信任关系（rating > 0）构建无向图
        df = df[df["rating"] > 0][["src", "tgt", "timestamp"]]
        return self._build_snapshots(df, num_snapshots, dataset_name="Bitcoin-Alpha")

    def load_facebook_wosn(self, num_snapshots: int = 10) -> Tuple[List[nx.Graph], List[str]]:
        """
        加载 Facebook WOSN 数据集
        格式: SRC TGT WEIGHT TIMESTAMP (空格分隔，前2行注释以%开头)
        返回: (图快照列表, 时间标签列表)
        """
        filepath = os.path.join(self.data_dir, "facebook-wosn-links", "out.facebook-wosn-links")
        df = pd.read_csv(
            filepath, sep=r"\s+", header=None, comment="%",
            names=["src", "tgt", "weight", "timestamp"]
        )
        # 过滤掉时间戳为0的边（初始边无时间信息）
        df_with_ts = df[df["timestamp"] > 0][["src", "tgt", "timestamp"]]
        if len(df_with_ts) < 100:
            # 如果有效时序边太少，使用全部边并人工分配时间
            df["timestamp"] = range(len(df))
            df_with_ts = df[["src", "tgt", "timestamp"]]
        return self._build_snapshots(df_with_ts, num_snapshots, dataset_name="Facebook-WOSN")

    def _build_snapshots(
        self, df: pd.DataFrame, num_snapshots: int, dataset_name: str = ""
    ) -> Tuple[List[nx.Graph], List[str]]:
        """
        将边列表按时间均匀划分为 num_snapshots 个快照（累积模式）
        每个快照包含该时间点及之前的所有边
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        t_min, t_max = df["timestamp"].min(), df["timestamp"].max()

        # 时间区间划分
        boundaries = np.linspace(t_min, t_max, num_snapshots + 1)
        snapshots = []
        time_labels = []

        for i in range(num_snapshots):
            # 累积快照：包含到当前时间窗口为止的所有边
            mask = df["timestamp"] <= boundaries[i + 1]
            edges_df = df[mask]

            G = nx.Graph()
            for _, row in edges_df.iterrows():
                G.add_edge(int(row["src"]), int(row["tgt"]))

            snapshots.append(G)
            time_labels.append(f"T{i}")

        print(f"[{dataset_name}] 加载完成: {num_snapshots} 个快照")
        for i, G in enumerate(snapshots):
            print(f"  {time_labels[i]}: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

        return snapshots, time_labels

    def load_dataset(
        self, name: str, num_snapshots: int = 10
    ) -> Tuple[List[nx.Graph], List[str]]:
        """统一加载接口"""
        loaders = {
            "college": self.load_college_msg,
            "bitcoin": self.load_bitcoin_alpha,
            "facebook": self.load_facebook_wosn,
        }
        if name not in loaders:
            raise ValueError(f"未知数据集: {name}, 可选: {list(loaders.keys())}")
        return loaders[name](num_snapshots=num_snapshots)


def compute_graph_stats(G: nx.Graph) -> dict:
    """计算图的关键统计量"""
    degrees = [d for _, d in G.degree()]
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "density": nx.density(G),
        "num_components": nx.number_connected_components(G),
    }
    # 聚类系数（大图上可能较慢，采样计算）
    if G.number_of_nodes() > 10000:
        stats["clustering_coeff"] = nx.average_clustering(G, trials=1000)
    elif G.number_of_nodes() > 0:
        stats["clustering_coeff"] = nx.average_clustering(G)
    else:
        stats["clustering_coeff"] = 0
    return stats


def get_degree_distribution(G: nx.Graph) -> np.ndarray:
    """获取度分布向量（索引=度值，值=频率）"""
    if G.number_of_nodes() == 0:
        return np.array([1.0])
    degrees = [d for _, d in G.degree()]
    max_deg = max(degrees)
    dist = np.zeros(max_deg + 1)
    for d in degrees:
        dist[d] += 1
    dist = dist / dist.sum()  # 归一化为概率分布
    return dist


if __name__ == "__main__":
    loader = DynamicGraphLoader()
    # 测试加载 CollegeMsg（最小数据集）
    snapshots, labels = loader.load_dataset("college", num_snapshots=5)
    print("\n统计信息:")
    for label, G in zip(labels, snapshots):
        stats = compute_graph_stats(G)
        print(f"  {label}: {stats}")
