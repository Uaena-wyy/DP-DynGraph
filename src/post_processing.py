"""
后处理优化模块
实现时序平滑技术，降低独立加噪的噪声影响
注：后处理不消耗额外隐私预算（DP 后处理定理）
"""

import numpy as np
from typing import List


def sliding_average(series: np.ndarray, window: int = 3) -> np.ndarray:
    """
    滑动平均平滑
    对加噪后的统计量序列做窗口平均，平滑噪声波动

    Args:
        series: 时序统计量序列（1D数组）
        window: 滑动窗口大小
    Returns:
        平滑后的序列
    """
    T = len(series)
    smoothed = np.zeros(T)
    half_w = window // 2

    for t in range(T):
        left = max(0, t - half_w)
        right = min(T, t + half_w + 1)
        smoothed[t] = np.mean(series[left:right])

    return smoothed


def exponential_smoothing(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    指数平滑
    s_t = α * x_t + (1 - α) * s_{t-1}

    Args:
        series: 时序统计量序列
        alpha: 平滑系数，越小越平滑
    Returns:
        平滑后的序列
    """
    T = len(series)
    smoothed = np.zeros(T)
    smoothed[0] = series[0]

    for t in range(1, T):
        smoothed[t] = alpha * series[t] + (1 - alpha) * smoothed[t - 1]

    return smoothed


class KalmanFilter1D:
    """
    一维卡尔曼滤波器
    状态模型: x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    观测模型: z_t = x_t + v_t,      v_t ~ N(0, R)

    其中:
    - x_t: 真实统计量
    - z_t: 加噪后的观测值
    - Q: 过程噪声方差（图演化的自然变化）
    - R: 观测噪声方差（DP 噪声方差 = 2 * (Δf/ε)^2 for Laplace）
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 10.0):
        """
        Args:
            process_noise: 过程噪声方差 Q
            measurement_noise: 观测噪声方差 R（与 Laplace 噪声方差相关）
        """
        self.Q = process_noise
        self.R = measurement_noise

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """
        对观测序列进行卡尔曼滤波

        Args:
            observations: 加噪后的观测值序列
        Returns:
            滤波后的估计值序列
        """
        T = len(observations)
        x_est = np.zeros(T)  # 状态估计
        P = np.zeros(T)      # 估计误差协方差

        # 初始化
        x_est[0] = observations[0]
        P[0] = self.R

        for t in range(1, T):
            # 预测步
            x_pred = x_est[t - 1]
            P_pred = P[t - 1] + self.Q

            # 更新步
            K = P_pred / (P_pred + self.R)  # 卡尔曼增益
            x_est[t] = x_pred + K * (observations[t] - x_pred)
            P[t] = (1 - K) * P_pred

        return x_est

    @staticmethod
    def estimate_laplace_variance(sensitivity: float, epsilon: float) -> float:
        """
        根据 Laplace 机制参数计算噪声方差
        Laplace(b) 的方差 = 2b^2，其中 b = Δf / ε
        """
        b = sensitivity / epsilon
        return 2 * b ** 2


def smooth_statistics_series(
    noisy_stats_series: List[dict],
    method: str = "kalman",
    **kwargs
) -> List[dict]:
    """
    对整个统计量序列进行后处理平滑

    Args:
        noisy_stats_series: DP加噪后的统计量字典列表
        method: "sliding" | "exponential" | "kalman"
    Returns:
        平滑后的统计量字典列表
    """
    if not noisy_stats_series:
        return []

    keys = noisy_stats_series[0].keys()
    T = len(noisy_stats_series)

    # 把字典列表转为按 key 分的序列
    series_by_key = {}
    for key in keys:
        series_by_key[key] = np.array([s[key] for s in noisy_stats_series])

    # 对每个统计量序列做平滑
    smoothed_by_key = {}
    for key, series in series_by_key.items():
        if method == "sliding":
            window = kwargs.get("window", 3)
            smoothed_by_key[key] = sliding_average(series, window)
        elif method == "exponential":
            alpha = kwargs.get("alpha", 0.3)
            smoothed_by_key[key] = exponential_smoothing(series, alpha)
        elif method == "kalman":
            Q = kwargs.get("process_noise", 1.0)
            R = kwargs.get("measurement_noise", 10.0)
            kf = KalmanFilter1D(process_noise=Q, measurement_noise=R)
            smoothed_by_key[key] = kf.filter(series)
        else:
            raise ValueError(f"未知平滑方法: {method}")

    # 转回字典列表
    smoothed_series = []
    for t in range(T):
        d = {key: smoothed_by_key[key][t] for key in keys}
        smoothed_series.append(d)

    return smoothed_series


if __name__ == "__main__":
    # 模拟测试
    np.random.seed(42)
    true_values = np.linspace(10, 50, 10)
    noisy_values = true_values + np.random.laplace(0, 5, size=10)

    print("真实值:  ", np.round(true_values, 2))
    print("加噪值:  ", np.round(noisy_values, 2))
    print("滑动平均:", np.round(sliding_average(noisy_values, window=3), 2))
    print("指数平滑:", np.round(exponential_smoothing(noisy_values, alpha=0.3), 2))

    kf = KalmanFilter1D(process_noise=2.0, measurement_noise=50.0)
    print("卡尔曼:  ", np.round(kf.filter(noisy_values), 2))
