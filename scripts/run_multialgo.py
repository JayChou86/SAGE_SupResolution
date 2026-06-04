#!/usr/bin/env python
"""
===========================================================================
run_multialgo.py
================
五种算法全面对比: SAGE(整数/小数)、MF、OMP、SBL + CRB 理论界

对应 MATLAB: SAGE_SupResolution_more.m
===========================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sage.signal_utils import (
    generate_tx_signal, generate_rx_signal, build_dictionary,
    compute_crb, fractional_delay_shift,
)
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_SBL
from sage.evaluation import match_paths
from sage.plotting import plot_channel_stem, plot_residual_comparison

SEED = 42
N_seq, N_rx = 511, 1000
L_paths = 3
true_delays = np.array([50.43, 60.66, 200.51])
true_amps = np.array([
    1.0 * np.exp(1j * 0.2),
    0.5 * np.exp(-1j * 1.5),
    0.8 * np.exp(1j * 0.8),
])

# 生成信号
s_tx_base = generate_tx_signal(N_seq, seed=SEED)
y_rx, y_clean = generate_rx_signal(
    s_tx_base, true_delays, true_amps, N_rx, noise_std=0.05, seed=SEED,
)

# 细网格字典
grid_res = 0.1
tau_grid = np.arange(30, 220 + grid_res, grid_res)
print(f"Building dictionary ({len(tau_grid)} atoms)...")
A_dict = build_dictionary(s_tx_base, tau_grid, N_rx)

# 运行所有算法
print("\nRunning algorithms...")
d_int, a_int = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=False)
d_frac, a_frac = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=True)
d_mf, a_mf = algo_MF(y_rx, A_dict, tau_grid, L_paths)
d_omp, a_omp = algo_OMP(y_rx, A_dict, tau_grid, L_paths)
# SBL: N0 取噪声方差近似值
N0 = 0.05 ** 2 * 2
d_sbl, a_sbl = algo_SBL(y_rx, A_dict, tau_grid, L_paths, N0)

# 匹配
algo_d = [d_int, d_frac, d_mf, d_omp, d_sbl]
algo_a = [a_int, a_frac, a_mf, a_omp, a_sbl]
algo_names = ['Int SAGE', 'Frac SAGE', 'MF (0.1)', 'OMP (0.1)', 'SBL (0.1)']
matched = [match_paths(true_delays, true_amps, d, a) for d, a in zip(algo_d, algo_a)]

# 输出
print(f"\n{'='*70}")
print(f" Multi-Algorithm Comparison")
print(f"{'='*70}")
for i in range(L_paths):
    t_tau, t_mag, t_pha = true_delays[i], np.abs(true_amps[i]), np.angle(true_amps[i])
    print(f"\n【Path {i+1}】 True: τ={t_tau:.2f}, |α|={t_mag:.3f}, φ={t_pha:.3f}")
    for j, (name, (md, ma)) in enumerate(zip(algo_names, matched)):
        e_tau = abs(md[i] - t_tau)
        e_mag = abs(np.abs(ma[i]) - t_mag)
        e_pha = abs(np.arctan2(np.sin(np.angle(ma[i]) - t_pha),
                                np.cos(np.angle(ma[i]) - t_pha)))
        print(f"  {name:<12}: τ={md[i]:8.4f} (err={e_tau:.4f})  "
              f"|α|={np.abs(ma[i]):.3f} (err={e_mag:.3f})  "
              f"φ={np.angle(ma[i]):.3f} (err={e_pha:.3f})")

# 残差能量
print(f"\nResidual Energy:")
for name, d, a in zip(algo_names, algo_d, algo_a):
    y_est = np.zeros(N_rx, dtype=complex)
    for dd, aa in zip(d, a):
        y_est += aa * fractional_delay_shift(s_tx_base, dd, N_rx)
    energy = np.sum(np.abs(y_rx - y_est) ** 2)
    print(f"  {name:<12}: {energy:.4f}")

# 绘图
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

# Stem 图
est_d_list = [md for md, _ in matched]
est_a_list = [ma for _, ma in matched]
plot_channel_stem(
    true_delays, true_amps,
    est_d_list, est_a_list,
    algo_names,
    save_path=os.path.join(results_dir, 'channel_stem.png'),
)

# 残差图
residuals = []
for d, a in zip(algo_d, algo_a):
    y_est = np.zeros(N_rx, dtype=complex)
    for dd, aa in zip(d, a):
        y_est += aa * fractional_delay_shift(s_tx_base, dd, N_rx)
    residuals.append(y_rx - y_est)

plot_residual_comparison(
    residuals, algo_names,
    save_path=os.path.join(results_dir, 'residual_comparison.png'),
)

print(f"\nFigures saved to {results_dir}/")
print("Done.")
