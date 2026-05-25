"""
隐私预算管理模块
实现三种跨快照隐私预算分配策略：均匀分配、指数衰减、自适应分配
"""

import numpy as np
from typing import List, Optional
import networkx as nx


class PrivacyBudgetManager:
    """隐私预算管理器"""

    def __init__(self, total_epsilon: float, num_snapshots: int):
        """
        Args:
            total_epsilon: 总隐私预算 ε
            num_snapshots: 快照数量 T
        """
        self.total_epsilon = total_epsilon
        self.num_snapshots = num_snapshots
        self.consumed = 0.0

    def uniform_allocation(self) -> np.ndarray:
        """
        均匀分配策略
        每个时间点分配 ε/T
        """
        eps_per_snapshot = self.total_epsilon / self.num_snapshots
        budgets = np.full(self.num_snapshots, eps_per_snapshot)
        return budgets

    def linear_increasing_allocation(self) -> np.ndarray:
        """
        线性递增分配策略（PPDU 风格 baseline）
        ε_t = (ε_total · 2 / (T·(T+1))) · t,  t = 1, 2, ..., T
        前期保护强（ε_t 小），后期效用优先（ε_t 大），Σε_t = ε_total

        参考：文献 [16] PPDU 动态图发布方法。
        """
        T = self.num_snapshots
        weights = np.arange(1, T + 1, dtype=float)
        budgets = self.total_epsilon * weights / weights.sum()
        return budgets

    def exponential_decay_allocation(self, alpha: float = 0.8) -> np.ndarray:
        """
        指数衰减分配策略
        近期数据更重要，分配更多预算
        ε_t = ε * α^(T-1-t) / Σα^(T-1-i)

        Args:
            alpha: 衰减系数，0 < alpha < 1，越小则近期占比越大
        """
        T = self.num_snapshots
        weights = np.array([alpha ** (T - 1 - t) for t in range(T)])
        budgets = self.total_epsilon * weights / weights.sum()
        return budgets

    def adaptive_allocation(self, snapshots: List[nx.Graph], min_ratio: float = 0.5) -> np.ndarray:
        """
        自适应分配策略
        基于相邻快照的变化率动态分配预算：变化大 -> 分配更多预算

        变化率定义：相邻快照间边集合的 Jaccard 距离
        Jaccard距离 = 1 - |E_t ∩ E_{t-1}| / |E_t ∪ E_{t-1}|

        Args:
            snapshots: 图快照序列
            min_ratio: 最小预算占均匀分配的比例，防止某个快照预算过低
        """
        T = len(snapshots)
        assert T == self.num_snapshots

        # 计算相邻快照间的变化率
        change_rates = np.zeros(T)
        change_rates[0] = 1.0  # 第一个快照无参照，给基准权重

        for t in range(1, T):
            edges_prev = set(snapshots[t - 1].edges())
            edges_curr = set(snapshots[t].edges())
            union = edges_prev | edges_curr
            if len(union) == 0:
                change_rates[t] = 0.0
            else:
                intersection = edges_prev & edges_curr
                change_rates[t] = 1.0 - len(intersection) / len(union)

        # 归一化为权重
        if change_rates.sum() == 0:
            weights = np.ones(T)
        else:
            weights = change_rates / change_rates.sum()

        # 施加最小预算约束
        min_budget = min_ratio * (self.total_epsilon / T)
        budgets = self.total_epsilon * weights

        # 修正低于最小预算的时间点
        deficit = 0.0
        for t in range(T):
            if budgets[t] < min_budget:
                deficit += min_budget - budgets[t]
                budgets[t] = min_budget

        # 从超额预算的时间点按比例扣除
        surplus_mask = budgets > min_budget
        if surplus_mask.any() and deficit > 0:
            surplus_total = budgets[surplus_mask].sum() - min_budget * surplus_mask.sum()
            if surplus_total > 0:
                budgets[surplus_mask] -= deficit * (budgets[surplus_mask] - min_budget) / surplus_total

        # 确保总预算精确
        budgets = budgets * (self.total_epsilon / budgets.sum())

        return budgets

    def get_allocation(
        self, strategy: str, snapshots: Optional[List[nx.Graph]] = None, **kwargs
    ) -> np.ndarray:
        """
        统一接口获取预算分配

        Args:
            strategy: "uniform" | "exponential" | "adaptive"
            snapshots: 自适应策略需要传入图快照序列
        """
        if strategy == "uniform":
            return self.uniform_allocation()
        elif strategy == "exponential":
            alpha = kwargs.get("alpha", 0.8)
            return self.exponential_decay_allocation(alpha=alpha)
        elif strategy == "linear":
            return self.linear_increasing_allocation()
        elif strategy == "adaptive":
            if snapshots is None:
                raise ValueError("自适应策略需要传入 snapshots 参数")
            min_ratio = kwargs.get("min_ratio", 0.5)
            return self.adaptive_allocation(snapshots, min_ratio=min_ratio)
        else:
            raise ValueError(
                f"未知策略: {strategy}, 可选: uniform, exponential, linear, adaptive"
            )

    def track_consumption(self, epsilon_used: float):
        """跟踪累积隐私消耗"""
        self.consumed += epsilon_used

    def remaining_budget(self) -> float:
        return self.total_epsilon - self.consumed

    def get_cumulative_consumption(self, budgets: np.ndarray) -> np.ndarray:
        """计算累积隐私消耗曲线"""
        return np.cumsum(budgets)


if __name__ == "__main__":
    mgr = PrivacyBudgetManager(total_epsilon=1.0, num_snapshots=5)
    print("均匀分配:", mgr.uniform_allocation())
    print("指数衰减:", mgr.exponential_decay_allocation(alpha=0.7))
    print("线性递增(PPDU):", mgr.linear_increasing_allocation())
    print("累积消耗:", mgr.get_cumulative_consumption(mgr.uniform_allocation()))
