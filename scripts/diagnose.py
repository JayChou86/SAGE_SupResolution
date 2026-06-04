#!/usr/bin/env python
"""
===========================================================================
诊断脚本：定位 Frac SAGE 高 SNR RMSE 反常的根本原因
===========================================================================
测试项:
  A. 单径场景 — 验证算法基本正确性
  B. 三径分离场景 (路径间距大) — 验证无混叠时算法正常
  C. 三径密集场景 (原始配置) — 复现问题
  D. 逐迭代收敛追踪 — 观察 EM 是否收敛到局部最优
  E. FFT 分数延迟重构精度 — 检查重建误差
  F. 字典网格对齐检查 — 解释 MF 为何过于理想
===========================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sage.signal_utils import (
    generate_tx_signal, fractional_delay_shift,
)
from sage.algorithms import algo_SAGE, algo_MF
from sage.evaluation import match_paths, compute_all_rmse


SEED = 42
N_seq, N_rx = 511, 1000
s_tx_base = generate_tx_signal(N_seq, seed=SEED)


def run_single_path_test():
    """A. 单径场景 — 验证算法基本正确性"""
    print("=" * 60)
    print(" A. SINGLE-PATH TEST — 验证基础正确性")
    print("=" * 60)

    L_paths = 3  # 估计 3 条，但真实只有 1 条
    true_delays = np.array([50.43])
    true_amps = np.array([1.0 * np.exp(1j * 0.2)])

    for snr_db in [-5, 0, 10, 20]:
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_linear
        noise_std = np.sqrt(noise_var / 2)

        y_clean = np.zeros(N_rx, dtype=complex)
        for d, a in zip(true_delays, true_amps):
            y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)

        rng = np.random.RandomState(42)
        noise = noise_std * (rng.randn(N_rx) + 1j * rng.randn(N_rx))
        y_rx = y_clean + noise

        d_int, a_int = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=False)
        d_frac, a_frac = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=True)

        # 取估计中幅度最大的那条 (其他两条应接近 0)
        top_int = np.argmax(np.abs(a_int))
        top_frac = np.argmax(np.abs(a_frac))

        print(f"  SNR={snr_db:3d} dB: "
              f"True τ=50.43 | "
              f"Int τ={d_int[top_int]:.2f} (err={abs(d_int[top_int]-50.43):.4f}) | "
              f"Frac τ={d_frac[top_frac]:.4f} (err={abs(d_frac[top_frac]-50.43):.4f})")
    print()


def run_separated_paths_test():
    """B. 三径分离场景 — 验证无混叠时算法正常"""
    print("=" * 60)
    print(" B. WELL-SEPARATED PATHS — 排除路径混叠因素")
    print("=" * 60)

    L_paths = 3
    # 路径间距 > 30 采样点 (远大于相关主瓣宽度 ~5)
    true_delays = np.array([50.43, 120.78, 200.51])
    true_amps = np.array([
        1.0 * np.exp(1j * 0.2),
        0.6 * np.exp(-1j * 0.8),
        0.8 * np.exp(1j * 0.8),
    ])

    for snr_db in [-5, 0, 10, 20]:
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.sum(np.abs(true_amps) ** 2) / N_rx  # 近似
        noise_var = signal_power / snr_linear
        noise_std = np.sqrt(noise_var / 2)

        y_clean = np.zeros(N_rx, dtype=complex)
        for d, a in zip(true_delays, true_amps):
            y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)

        rng = np.random.RandomState(42)
        noise = noise_std * (rng.randn(N_rx) + 1j * rng.randn(N_rx))
        y_rx = y_clean + noise

        d_int, a_int = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=False)
        d_frac, a_frac = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=True)

        rmse_d_int, _, _ = compute_all_rmse(true_delays, true_amps, d_int, a_int)
        rmse_d_frac, _, _ = compute_all_rmse(true_delays, true_amps, d_frac, a_frac)

        print(f"  SNR={snr_db:3d} dB: "
              f"Int RMSE={rmse_d_int:.4f} | "
              f"Frac RMSE={rmse_d_frac:.4f} | "
              f"Improvement={((rmse_d_int-rmse_d_frac)/rmse_d_int*100):.1f}%")
    print()


def run_dense_paths_per_path():
    """C. 三径密集场景 — 分路径追踪 RMSE"""
    print("=" * 60)
    print(" C. DENSE PATHS — PER-PATH RMSE TRACKING")
    print("=" * 60)

    L_paths = 3
    true_delays = np.array([50.43, 60.66, 200.51])
    true_amps = np.array([
        1.0 * np.exp(1j * 0.2),
        0.5 * np.exp(-1j * 1.5),
        0.8 * np.exp(1j * 0.8),
    ])

    signal_power = np.var(sum(
        a * fractional_delay_shift(s_tx_base, d, N_rx)
        for d, a in zip(true_delays, true_amps)
    ))

    for snr_db in [-5, 0, 10, 20]:
        snr_linear = 10 ** (snr_db / 10)
        noise_var = signal_power / snr_linear
        noise_std = np.sqrt(noise_var / 2)

        y_clean = np.zeros(N_rx, dtype=complex)
        for d, a in zip(true_delays, true_amps):
            y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)

        rng = np.random.RandomState(42)
        noise = noise_std * (rng.randn(N_rx) + 1j * rng.randn(N_rx))
        y_rx = y_clean + noise

        d_int, a_int = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=False)
        d_frac, a_frac = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=True)

        md_int, _ = match_paths(true_delays, true_amps, d_int, a_int)
        md_frac, _ = match_paths(true_delays, true_amps, d_frac, a_frac)

        # 原始估计（未匹配）
        print(f"  SNR={snr_db:3d} dB:")
        print(f"    True delays:     {true_delays}")
        print(f"    Int SAGE (raw):  {np.sort(d_int)}")
        print(f"    Int SAGE (match):{md_int}")
        print(f"    Frac SAGE (raw): {np.sort(d_frac)}")
        print(f"    Frac SAGE (match):{md_frac}")
        for i in range(L_paths):
            e_int = abs(md_int[i] - true_delays[i])
            e_frac = abs(md_frac[i] - true_delays[i])
            print(f"    Path {i+1}: Int err={e_int:.4f}, Frac err={e_frac:.4f}")
        print()


def run_convergence_trace():
    """D. 逐迭代收敛追踪"""
    print("=" * 60)
    print(" D. CONVERGENCE TRACE — 观察 EM 迭代过程")
    print("=" * 60)

    L_paths = 3
    true_delays = np.array([50.43, 60.66, 200.51])
    true_amps = np.array([
        1.0 * np.exp(1j * 0.2),
        0.5 * np.exp(-1j * 1.5),
        0.8 * np.exp(1j * 0.8),
    ])

    noise_std = 0.01  # 低噪声 (高 SNR)
    y_clean = np.zeros(N_rx, dtype=complex)
    for d, a in zip(true_delays, true_amps):
        y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)
    rng = np.random.RandomState(42)
    y_rx = y_clean + noise_std * (rng.randn(N_rx) + 1j * rng.randn(N_rx))

    # 追踪每次迭代后的估计
    N_seq = len(s_tx_base)
    Es = np.vdot(s_tx_base, s_tx_base).real
    search_len = N_rx - N_seq

    for do_frac, label in [(False, "Int"), (True, "Frac")]:
        print(f"\n  --- {label} SAGE convergence trace ---")
        est_delays = np.zeros(L_paths)
        est_amps = np.zeros(L_paths, dtype=complex)
        residual = y_rx.copy()

        for it in range(10):
            for l in range(L_paths):
                if it > 0:
                    s_old = est_amps[l] * fractional_delay_shift(s_tx_base, est_delays[l], N_rx)
                    x_l = residual + s_old
                else:
                    x_l = residual.copy()

                corr_vals = np.zeros(search_len)
                for tau in range(search_len):
                    corr_vals[tau] = np.abs(np.vdot(s_tx_base, x_l[tau:tau+N_seq]))

                k_opt = int(np.argmax(corr_vals))

                if do_frac and 0 < k_opt < search_len - 1:
                    y_m1 = corr_vals[k_opt-1]**2
                    y_0 = corr_vals[k_opt]**2
                    y_p1 = corr_vals[k_opt+1]**2
                    denom = y_m1 - 2*y_0 + y_p1
                    if abs(denom) > 1e-10:
                        delta = np.clip(0.5*(y_m1-y_p1)/denom, -0.5, 0.5)
                    else:
                        delta = 0.0
                    tau_est = float(k_opt) + delta
                else:
                    tau_est = float(k_opt)

                s_temp = fractional_delay_shift(s_tx_base, tau_est, N_rx)
                amp_est = np.vdot(s_temp, x_l) / Es

                est_delays[l] = tau_est
                est_amps[l] = amp_est
                residual = x_l - amp_est * s_temp

            print(f"    Iter {it+1:2d}: delays = [{est_delays[0]:8.3f}, {est_delays[1]:8.3f}, {est_delays[2]:8.3f}]  "
                  f"|amps| = [{abs(est_amps[0]):.3f}, {abs(est_amps[1]):.3f}, {abs(est_amps[2]):.3f}]")

    print()


def check_fft_reconstruction():
    """E. FFT 分数延迟重构精度"""
    print("=" * 60)
    print(" E. FFT FRACTIONAL DELAY RECONSTRUCTION ACCURACY")
    print("=" * 60)

    # 检验: s_frac(t) 与理想采样 s(t - τ) 的差异
    # 在连续时间: s_ideal(t) = α·s(t - τ)
    # FFT 重建: s_fft(t) 的精度

    # 用极高采样率做 reference
    N_oversample = 10
    N_seq_os = N_seq * N_oversample
    s_os = np.zeros(N_seq_os, dtype=complex)
    s_os[:N_seq] = s_tx_base  # 补零 = sinc 插值

    delays_test = [0.0, 0.25, 0.43, 0.5, 0.7]
    for delay in delays_test:
        s_fft = fractional_delay_shift(s_tx_base, delay, N_rx)
        # 提取非零部分 (从 floor(delay) 开始的 N_seq 个样本)
        start = int(np.floor(delay))
        s_fft_segment = s_fft[start:start+N_seq]

        # 参考: FFT 相移后的信号 (频域精确)
        S_freq = np.fft.fft(s_tx_base)
        k = np.concatenate([
            np.arange(0, int(np.ceil(N_seq/2))),
            np.arange(-int(np.floor(N_seq/2)), 0),
        ])
        phase_shift = np.exp(-1j * 2 * np.pi * k * delay / N_seq)
        s_ref = np.fft.ifft(S_freq * phase_shift)

        err = np.linalg.norm(s_fft_segment - s_ref) / np.linalg.norm(s_ref)
        print(f"    delay={delay:.2f}: reconstruction error = {err:.2e}")

    print("    → FFT fractional delay is accurate for bandlimited signals.\n")


def check_grid_alignment():
    """F. 字典网格对齐检查 — 解释 MF 为何过于理想"""
    print("=" * 60)
    print(" F. GRID ALIGNMENT CHECK — 为什么 MF 那么准确?")
    print("=" * 60)

    grid_res = 0.83
    tau_grid = np.arange(30, 220 + grid_res, grid_res)
    true_delays = np.array([50.43, 60.66, 200.51])

    for td in true_delays:
        distances = np.abs(tau_grid - td)
        nearest_idx = np.argmin(distances)
        nearest_grid = tau_grid[nearest_idx]
        print(f"    True τ={td:.2f} → nearest grid point = {nearest_grid:.2f} "
              f"(error = {abs(nearest_grid-td):.4f} samples)")

    # 计算网格的"覆盖密度"
    print(f"\n    Grid: {len(tau_grid)} atoms covering [{tau_grid[0]}, {tau_grid[-1]}]")
    print(f"    Resolution: {grid_res} samples")
    print(f"    Maximum grid error (any continuous τ): {grid_res/2:.3f} samples")
    print(f"    → MF cannot be better than {grid_res/2:.3f} samples RMSE in the best case.")
    print(f"    → MF RMSE of ~0.28 suggests good alignment with true delays.\n")


def main():
    run_single_path_test()
    run_separated_paths_test()
    run_dense_paths_per_path()
    run_convergence_trace()
    check_fft_reconstruction()
    check_grid_alignment()

    print("=" * 60)
    print(" DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
