#!/usr/bin/env python
"""
===========================================================================
run_basic_compare.py
====================
SAGE 算法基本对比: 整数搜索 (基线) vs 亚采样超分辨率 (本文提出)

对应 MATLAB: SAGE_SupResolution.m
===========================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sage.signal_utils import generate_tx_signal, fractional_delay_shift
from sage.algorithms import algo_SAGE
from sage.evaluation import match_paths

# ---- 参数 ----
N_seq, N_rx = 511, 1000
L_paths = 3
true_delays = np.array([50.4, 65.7, 200.2])
true_amps = np.array([
    1.0 * np.exp(1j * 0.2),
    0.5 * np.exp(-1j * 1.5),
    0.8 * np.exp(1j * 0.8),
])

# ---- 信号生成 ----
s_tx_base = generate_tx_signal(N_seq, seed=42)
y_rx, y_clean = np.zeros(N_rx, dtype=complex), np.zeros(N_rx, dtype=complex)
for d, a in zip(true_delays, true_amps):
    s_frac = fractional_delay_shift(s_tx_base, d, N_rx)
    y_clean += a * s_frac
noise = 0.05 * (np.random.randn(N_rx) + 1j * np.random.randn(N_rx))
y_rx = y_clean + noise

# ---- 运行两种 SAGE ----
print("Running Integer SAGE...")
d_int, a_int = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=False)

print("Running Fractional SAGE...")
d_frac, a_frac = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, max_iter=10, do_frac=True)

# ---- 匹配与输出 ----
mdi, mai = match_paths(true_delays, true_amps, d_int, a_int)
mdf, maf = match_paths(true_delays, true_amps, d_frac, a_frac)

print(f"\n{'='*60}")
print(f"  SAGE 算法性能对比 (真实值 vs 无亚采样 vs 有亚采样)")
print(f"{'='*60}")

for i in range(L_paths):
    print(f"\n【多径 {i+1}】")
    print(f"  真实值:     时延={true_delays[i]:6.2f}  幅度={abs(true_amps[i]):.3f}  相位={np.angle(true_amps[i]):.3f} rad")
    print(f"  整数 SAGE:  时延={mdi[i]:6.2f} (err={abs(mdi[i]-true_delays[i]):.2f})  "
          f"幅度={abs(mai[i]):.3f} (err={abs(abs(mai[i])-abs(true_amps[i])):.3f})")
    print(f"  小数 SAGE:  时延={mdf[i]:6.4f} (err={abs(mdf[i]-true_delays[i]):.4f})  "
          f"幅度={abs(maf[i]):.3f} (err={abs(abs(maf[i])-abs(true_amps[i])):.3f})")

print("\nDone.")
