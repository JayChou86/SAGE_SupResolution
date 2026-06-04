#!/usr/bin/env python
"""
===========================================================================
run_grid_mismatch.py
====================
栅格失配敏感度扫描: 不同降采样倍数下整数 vs 分数 SAGE 的 MSE 对比

对应 MATLAB: SAGE_Grid_mismatch.m

使用方法:
    需要 data475.mat 文件放置在 data/ 目录下。
===========================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

from sage.interpolation import parabolic_delta


def load_matlab_data(filepath: str):
    """加载 MATLAB .mat 文件 (兼容 v5/v7/v7.3)。"""
    try:
        import mat73
        return mat73.loadmat(filepath)
    except Exception:
        import h5py
        import scipy.io as sio
        try:
            return sio.loadmat(filepath)
        except Exception:
            with h5py.File(filepath, 'r') as f:
                return {k: np.array(v) for k, v in f.items()}


# ---- 加载数据 ----
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
mat_path = os.path.join(data_dir, 'data475.mat')

if not os.path.exists(mat_path):
    print(f"Warning: {mat_path} not found. Skipping sea-trial test.")
    print("Please place data475.mat in the data/ directory.")
    sys.exit(1)

data = load_matlab_data(mat_path)
y_raw_orig = data['sig_rev_m'][:].flatten()
s_tx_orig = data['sig_bd_m'][:].flatten()

print(f"Loaded data: y_raw={len(y_raw_orig)}, s_tx={len(s_tx_orig)}")

# ---- 降采样扫描 ----
down_ratios = np.arange(1.0, 8.5, 0.5)
num_tests = len(down_ratios)

mse_int_list = np.zeros(num_tests)
mse_frac_list = np.zeros(num_tests)
improvement_list = np.zeros(num_tests)

K = 8
num_iter = 5

for test_idx, ratio in enumerate(down_ratios):
    # 有理数重采样
    num, den = 1, int(round(ratio))
    # 简化: 直接降采样 (对应 MATLAB resample 行为)
    from scipy.signal import decimate
    try:
        y_raw = decimate(y_raw_orig, den, ftype='fir') if den > 1 else y_raw_orig.copy()
        s_tx = decimate(s_tx_orig, den, ftype='fir') if den > 1 else s_tx_orig.copy()
    except Exception:
        # fallback: 直接抽取
        step = max(1, int(round(ratio)))
        y_raw = y_raw_orig[::step]
        s_tx = s_tx_orig[::step]

    N = len(y_raw)
    Es = np.sum(np.abs(s_tx) ** 2)

    S_f = fft(s_tx)
    Y_f = fft(y_raw)

    if N % 2 == 0:
        f_vec = np.concatenate([
            np.arange(0, N // 2),
            np.arange(-N // 2, 0),
        ]) / N
    else:
        f_vec = np.concatenate([
            np.arange(0, (N - 1) // 2 + 1),
            np.arange(-(N - 1) // 2, 0),
        ]) / N

    # --- Integer SAGE (频域) ---
    tau_int = np.zeros(K)
    alpha_int = np.zeros(K, dtype=complex)

    y_res_f = Y_f.copy()
    alpha_k_f = np.zeros((N, K), dtype=complex)

    for _ in range(num_iter):
        for k in range(K):
            x_k_f = y_res_f + alpha_k_f[:, k] * S_f
            R_k = ifft(x_k_f * np.conj(S_f)) / Es
            idx = np.argmax(np.abs(R_k))
            tau_int[k] = idx - 1
            alpha_int[k] = R_k[idx]
            alpha_k_f[:, k] = alpha_int[k] * np.exp(-1j * 2 * np.pi * f_vec * tau_int[k])
            y_res_f = x_k_f - alpha_k_f[:, k] * S_f

    # --- Fractional SAGE (频域 + 抛物线插值) ---
    tau_frac = np.zeros(K)
    alpha_frac = np.zeros(K, dtype=complex)

    y_res_f_frac = Y_f.copy()
    alpha_k_f_frac = np.zeros((N, K), dtype=complex)

    for _ in range(num_iter):
        for k in range(K):
            x_k_f = y_res_f_frac + alpha_k_f_frac[:, k] * S_f
            R_k = ifft(x_k_f * np.conj(S_f)) / Es
            idx = np.argmax(np.abs(R_k))

            if 0 < idx < N - 1:
                y_m1 = np.abs(R_k[idx - 1])
                y_0 = np.abs(R_k[idx])
                y_p1 = np.abs(R_k[idx + 1])
                delta = parabolic_delta(y_m1, y_0, y_p1)
            else:
                delta = 0.0

            tau_curr = (idx - 1) + delta
            tau_frac[k] = tau_curr

            phase_shift = np.exp(1j * 2 * np.pi * f_vec * tau_curr)
            exact_amp = np.sum((x_k_f * np.conj(S_f)) * phase_shift) / (N * Es)
            alpha_frac[k] = exact_amp

            alpha_k_f_frac[:, k] = alpha_frac[k] * np.exp(-1j * 2 * np.pi * f_vec * tau_frac[k])
            y_res_f_frac = x_k_f - alpha_k_f_frac[:, k] * S_f

    # --- 信道重构 + MMSE 均衡 ---
    H_int = np.sum([
        alpha_int[k] * np.exp(-1j * 2 * np.pi * f_vec * tau_int[k])
        for k in range(K)
    ], axis=0)
    H_frac = np.sum([
        alpha_frac[k] * np.exp(-1j * 2 * np.pi * f_vec * tau_frac[k])
        for k in range(K)
    ], axis=0)

    noise_var = 0.05 * np.max(np.abs(H_frac)) ** 2

    S_eq_int_f = Y_f * np.conj(H_int) / (np.abs(H_int) ** 2 + noise_var)
    S_eq_frac_f = Y_f * np.conj(H_frac) / (np.abs(H_frac) ** 2 + noise_var)

    s_eq_int = ifft(S_eq_int_f)
    s_eq_frac = ifft(S_eq_frac_f)

    valid_start, valid_end = 1000, N - 1000
    mse_int_list[test_idx] = np.mean(
        np.abs(s_eq_int[valid_start:valid_end] - s_tx[valid_start:valid_end]) ** 2
    )
    mse_frac_list[test_idx] = np.mean(
        np.abs(s_eq_frac[valid_start:valid_end] - s_tx[valid_start:valid_end]) ** 2
    )

    improvement_list[test_idx] = (
        (mse_int_list[test_idx] - mse_frac_list[test_idx])
        / mse_int_list[test_idx] * 100
    )

    print(f"  Down ratio {ratio:.1f}: Int MSE={mse_int_list[test_idx]:.4e}, "
          f"Frac MSE={mse_frac_list[test_idx]:.4e}, "
          f"Improvement={improvement_list[test_idx]:.1f}%")

# ---- 绘图 ----
from sage.plotting import plot_grid_mismatch
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
plot_grid_mismatch(
    down_ratios, mse_int_list, mse_frac_list, improvement_list,
    save_path=os.path.join(results_dir, 'grid_mismatch.png'),
)
print(f"\nFigure saved to: {os.path.join(results_dir, 'grid_mismatch.png')}")
print("Done.")
