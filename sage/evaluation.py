"""
评估模块
========
- 路径乱序自动匹配
- RMSE 计算 (时延/幅度/相位)
- 残差能量计算
"""

import numpy as np
from typing import Tuple, List


def match_paths(
    true_delays: np.ndarray,
    true_amps: np.ndarray,
    est_delays: np.ndarray,
    est_amps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    路径自动匹配：对每条真实多径，在估计结果中找时延最接近的路径。

    解决 SAGE/MF/OMP 等算法估计结果乱序的问题。

    Parameters
    ----------
    true_delays : 1D ndarray
        真实时延。
    true_amps : 1D complex ndarray
        真实复幅度。
    est_delays : 1D ndarray
        估计时延。
    est_amps : 1D complex ndarray
        估计复幅度。

    Returns
    -------
    matched_delays : 1D ndarray
        按真实路径顺序排列的估计时延。
    matched_amps : 1D complex ndarray
        按真实路径顺序排列的估计复幅度。
    """
    L = len(true_delays)
    matched_delays = np.zeros(L)
    matched_amps = np.zeros(L, dtype=complex)

    for i in range(L):
        idx = np.argmin(np.abs(est_delays - true_delays[i]))
        matched_delays[i] = est_delays[idx]
        matched_amps[i] = est_amps[idx]

    return matched_delays, matched_amps


def compute_rmse_delay(
    true_delays: np.ndarray,
    matched_delays: np.ndarray,
) -> float:
    """计算时延 RMSE (所有路径平均)。"""
    return np.sqrt(np.mean((matched_delays - true_delays) ** 2))


def compute_rmse_magnitude(
    true_amps: np.ndarray,
    matched_amps: np.ndarray,
) -> float:
    """计算幅度 RMSE (所有路径平均)。"""
    return np.sqrt(np.mean((np.abs(matched_amps) - np.abs(true_amps)) ** 2))


def compute_rmse_phase(
    true_amps: np.ndarray,
    matched_amps: np.ndarray,
) -> float:
    """计算相位 RMSE (所有路径平均)，自动处理 ±π 卷绕。"""
    phase_err = np.angle(matched_amps) - np.angle(true_amps)
    # wrap to [-pi, pi]
    phase_err = np.arctan2(np.sin(phase_err), np.cos(phase_err))
    return np.sqrt(np.mean(phase_err ** 2))


def compute_all_rmse(
    true_delays: np.ndarray,
    true_amps: np.ndarray,
    est_delays: np.ndarray,
    est_amps: np.ndarray,
) -> Tuple[float, float, float]:
    """
    一次性计算时延、幅度、相位的 RMSE。

    Returns
    -------
    (rmse_delay, rmse_mag, rmse_phase)
    """
    m_d, m_a = match_paths(true_delays, true_amps, est_delays, est_amps)
    return (
        compute_rmse_delay(true_delays, m_d),
        compute_rmse_magnitude(true_amps, m_a),
        compute_rmse_phase(true_amps, m_a),
    )


def compute_residual_energy(y_rx: np.ndarray, est_delays: np.ndarray,
                            est_amps: np.ndarray, s_base: np.ndarray,
                            N_rx: int, fractional_delay_shift_fn) -> float:
    """
    计算估计残差能量: ||y_rx - Σ α_k * s(t - τ_k)||²。
    """
    y_est = np.zeros(N_rx, dtype=complex)
    for d, a in zip(est_delays, est_amps):
        s_frac = fractional_delay_shift_fn(s_base, d, N_rx)
        y_est += a * s_frac
    return float(np.sum(np.abs(y_rx - y_est) ** 2))
