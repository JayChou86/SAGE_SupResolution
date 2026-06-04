#!/usr/bin/env python
"""
===========================================================================
SAGE_SupResolution_more_MD3_2.py
=================================
多径时延估计算法全面性能对比评估框架 (Monte Carlo 仿真版)

对应 MATLAB: SAGE_SupResolution_more_MD3_2.m

算法:
  1. Integer SAGE    — 传统整数网格 SAGE (基线)
  2. Fractional SAGE — 亚采样 SAGE + 抛物线插值 (本文提出)
  3. Matched Filter  — 匹配滤波器 (细网格字典)
  4. OMP             — 正交匹配追踪 (细网格字典)
  5. SBL             — 稀疏贝叶斯学习 (细网格字典)

评估指标: 时延 / 幅度 / 相位的 RMSE vs SNR, 与 CRB 理论界对比

用法:
    conda activate sage-superres
    python scripts/run_mc_compare.py
===========================================================================
"""

import sys
import os
import time
import numpy as np

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage.signal_utils import (
    generate_tx_signal,
    generate_rx_signal,
    compute_crb,
    build_dictionary,
    fractional_delay_shift,
)
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths
from sage.plotting import plot_rmse_vs_snr


# =========================================================================
# 1. 基础参数设置
# =========================================================================
SNR_dB_vec = np.arange(-15, 20, 5)        # -15:5:15 dB (7 points × 200 MC)
grid_res = 0.83                            # 字典分辨率 (采样点)
num_MC = 200                               # Monte Carlo 次数 (TWC final)
SEED = 40
rng = np.random.RandomState(SEED)
np.random.seed(SEED)

N_seq = 511
N_rx = 1000
L_paths = 3

# 真实信道参数 (与 MATLAB 完全一致)
true_delays = np.array([50.43, 60.66, 200.51])
true_amps = np.array([
    1.0 * np.exp(1j * 0.2),
    0.5 * np.exp(-1j * 1.5),
    0.8 * np.exp(1j * 0.8),
])

print("=" * 60)
print(" SAGE Super-Resolution — Monte Carlo Benchmark")
print("=" * 60)
print(f" SNR range:  {SNR_dB_vec[0]} to {SNR_dB_vec[-1]} dB")
print(f" MC trials:  {num_MC}")
print(f" Algorithms:  Integer SAGE, Fractional SAGE, MF, OMP, SBL")
print(f" Grid res:   {grid_res} samples")
print("=" * 60)

# =========================================================================
# 2. 生成发射信号 & 纯净接收信号
# =========================================================================
s_tx_base = generate_tx_signal(N_seq, seed=SEED)

y_clean = np.zeros(N_rx, dtype=complex)
for d, a in zip(true_delays, true_amps):
    y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)

signal_power = np.var(y_clean)
print(f"\nSignal power: {signal_power:.4f}")

# =========================================================================
# 3. 构建细网格字典 (MF/OMP/SBL 共用)
# =========================================================================
tau_grid = np.arange(30, 220 + grid_res, grid_res)
print(f"Building dictionary: {len(tau_grid)} atoms ({tau_grid[0]:.1f} to {tau_grid[-1]:.1f})...")
t0 = time.time()
A_dict = build_dictionary(s_tx_base, tau_grid, N_rx)
print(f"  done in {time.time() - t0:.2f}s")

# =========================================================================
# 4. CRB 预计算 (信号导数能量)
# =========================================================================
S_freq = np.fft.fft(s_tx_base)
k_vec = np.concatenate([
    np.arange(0, int(np.ceil(N_seq / 2))),
    np.arange(-int(np.floor(N_seq / 2)), 0),
])
omega = 2 * np.pi * k_vec / N_seq
s_prime = np.fft.ifft((1j * omega) * S_freq)
energy_s_prime = np.sum(np.abs(s_prime) ** 2)
energy_s = np.sum(np.abs(s_tx_base) ** 2)

# =========================================================================
# 5. Monte Carlo 仿真
# =========================================================================
num_SNR = len(SNR_dB_vec)
num_algos = 6
algo_names = ['Integer SAGE', 'Fractional SAGE', 'Matched Filter', 'OMP', 'NOMP', 'SBL']

# 存储 RMSE: shape (num_SNR, num_algos)
rmse_delay = np.zeros((num_SNR, num_algos))
rmse_mag = np.zeros((num_SNR, num_algos))
rmse_phase = np.zeros((num_SNR, num_algos))

# 存储 CRB
crb_delay_all = np.zeros(num_SNR)
crb_mag_all = np.zeros(num_SNR)
crb_phase_all = np.zeros(num_SNR)

print(f"\n{'='*60}")
print(f" Starting Monte Carlo ({num_SNR} SNR points × {num_MC} trials)...")
print(f"{'='*60}")

t_start = time.time()

for s_idx, SNR_dB in enumerate(SNR_dB_vec):
    # 噪声参数 (基于信号功率校准)
    N0 = signal_power / (10 ** (SNR_dB / 10))
    noise_std = np.sqrt(N0 / 2)

    # CRB
    crb_d, crb_m, crb_p = compute_crb(s_tx_base, true_amps, N0)
    crb_delay_all[s_idx] = np.sqrt(np.mean(crb_d ** 2))
    crb_mag_all[s_idx] = np.sqrt(np.mean(crb_m ** 2))
    crb_phase_all[s_idx] = np.sqrt(np.mean(crb_p ** 2))

    # 误差累加器
    sq_err_delay = np.zeros(num_algos)
    sq_err_mag = np.zeros(num_algos)
    sq_err_phase = np.zeros(num_algos)

    for mc in range(num_MC):
        # 生成噪声
        noise = noise_std * (
            rng.randn(N_rx) + 1j * rng.randn(N_rx)
        )
        y_rx = y_clean + noise

        # 运行各算法
        d_int, a_int = algo_SAGE(
            y_rx, s_tx_base, N_rx, L_paths, do_frac=False,
        )
        d_frac, a_frac = algo_SAGE(
            y_rx, s_tx_base, N_rx, L_paths, do_frac=True,
        )
        d_mf, a_mf = algo_MF(y_rx, A_dict, tau_grid, L_paths)
        d_omp, a_omp = algo_OMP(y_rx, A_dict, tau_grid, L_paths)
        d_nomp, a_nomp = algo_NOMP(
            y_rx, s_tx_base, N_rx, L_paths, tau_grid,
        )
        d_sbl, a_sbl = algo_SBL(y_rx, A_dict, tau_grid, L_paths,
                                use_woodbury=True)

        ests_d = [d_int, d_frac, d_mf, d_omp, d_nomp, d_sbl]
        ests_a = [a_int, a_frac, a_mf, a_omp, a_nomp, a_sbl]

        for alg in range(num_algos):
            m_d, m_a = match_paths(true_delays, true_amps,
                                   ests_d[alg], ests_a[alg])

            sq_err_delay[alg] += np.mean((m_d - true_delays) ** 2)
            sq_err_mag[alg] += np.mean(
                (np.abs(m_a) - np.abs(true_amps)) ** 2
            )
            phase_err = np.arctan2(
                np.sin(np.angle(m_a) - np.angle(true_amps)),
                np.cos(np.angle(m_a) - np.angle(true_amps)),
            )
            sq_err_phase[alg] += np.mean(phase_err ** 2)

    # 计算 RMSE
    rmse_delay[s_idx, :] = np.sqrt(sq_err_delay / num_MC)
    rmse_mag[s_idx, :] = np.sqrt(sq_err_mag / num_MC)
    rmse_phase[s_idx, :] = np.sqrt(sq_err_phase / num_MC)

    print(f"  SNR = {SNR_dB:4d} dB | "
          f"FracSAGE delay RMSE = {rmse_delay[s_idx, 1]:.4f} | "
          f"mag RMSE = {rmse_mag[s_idx, 1]:.4f} | "
          f"phase RMSE = {rmse_phase[s_idx, 1]:.4f}")

elapsed = time.time() - t_start
print(f"\nMonte Carlo completed in {elapsed:.1f}s "
      f"({elapsed / (num_SNR * num_MC):.2f}s per trial)")

# =========================================================================
# 6. 控制台输出性能汇总
# =========================================================================
print(f"\n{'='*60}")
print(" Performance Summary (averaged over all paths)")
print(f"{'='*60}")
print(f"{'Algorithm':<18} {'Delay RMSE':>12} {'Mag RMSE':>12} {'Phase RMSE':>12}")
print("-" * 60)
for alg in range(num_algos):
    # 取中间 SNR (0 dB) 的 RMSE
    mid_idx = len(SNR_dB_vec) // 2
    print(f"  {algo_names[alg]:<16} "
          f"{rmse_delay[mid_idx, alg]:>12.4f} "
          f"{rmse_mag[mid_idx, alg]:>12.4f} "
          f"{rmse_phase[mid_idx, alg]:>12.4f}")

# 性能提升
improvement = (
    (rmse_delay[:, 0] - rmse_delay[:, 1]) / rmse_delay[:, 0] * 100
)
print(f"\nFractional SAGE delay RMSE improvement over Integer SAGE:")
for s_idx, SNR_dB in enumerate(SNR_dB_vec):
    print(f"  SNR = {SNR_dB:4d} dB: {improvement[s_idx]:.1f}%")

# =========================================================================
# 7. 绘图保存
# =========================================================================
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

fig = plot_rmse_vs_snr(
    SNR_dB_vec,
    rmse_delay, rmse_mag, rmse_phase,
    crb_delay_all, crb_mag_all, crb_phase_all,
    algo_names,
    save_path=os.path.join(results_dir, 'rmse_vs_snr.png'),
)
print(f"\nFigure saved to: {os.path.join(results_dir, 'rmse_vs_snr.png')}")

# 额外: 保存数值结果
np.savez(
    os.path.join(results_dir, 'mc_results.npz'),
    SNR_dB_vec=SNR_dB_vec,
    rmse_delay=rmse_delay,
    rmse_mag=rmse_mag,
    rmse_phase=rmse_phase,
    crb_delay=crb_delay_all,
    crb_mag=crb_mag_all,
    crb_phase=crb_phase_all,
    algo_names=algo_names,
)
print(f"Results saved to: {os.path.join(results_dir, 'mc_results.npz')}")
print("\nDone.")
