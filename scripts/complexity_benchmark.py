#!/usr/bin/env python
"""
===========================================================================
复杂度基准测试 — 实测时间 + 理论 FLOPs
========================================
在标准仿真参数下 (N_seq=511, N_rx=1000, L=3, SNR=0dB)，
测量各算法 30 次独立运行的平均耗时。
===========================================================================
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sage.signal_utils import generate_tx_signal, fractional_delay_shift, build_dictionary
from sage.algorithms import (
    algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_ESPRIT, algo_SBL,
)

N_seq, N_rx, L = 511, 1000, 3
s = generate_tx_signal(N_seq, seed=42)
td = np.array([50.43, 60.66, 200.51])
ta = np.array([1.0*np.exp(1j*0.2), 0.5*np.exp(-1j*1.5), 0.8*np.exp(1j*0.8)])

yc = np.zeros(N_rx, dtype=complex)
for d, a in zip(td, ta):
    yc += a * fractional_delay_shift(s, d, N_rx)
sp = np.var(yc)
snr_db = 0; snr = 10**(snr_db/10); ns = np.sqrt(sp / snr / 2)

tg = np.arange(30, 221, 0.83)
Ad = build_dictionary(s, tg, N_rx)

N_trials = 30

configs = [
    ("Int SAGE",       lambda yr: algo_SAGE(yr, s, N_rx, L, do_frac=False)),
    ("Frac SAGE",      lambda yr: algo_SAGE(yr, s, N_rx, L, do_frac=True)),
    ("MF (230 atoms)", lambda yr: algo_MF(yr, Ad, tg, L)),
    ("OMP",            lambda yr: algo_OMP(yr, Ad, tg, L)),
    ("NOMP",           lambda yr: algo_NOMP(yr, s, N_rx, L, tg)),
    ("SBL",            lambda yr: algo_SBL(yr, Ad, tg, L)),
    ("ESPRIT",         lambda yr: algo_ESPRIT(yr, s, L)),
]

print(f"{'Algorithm':<20} {'Time (ms)':>10} {'±std':>8}")
print("-" * 42)

for name, fn in configs:
    times = []
    for _ in range(N_trials):
        yr = yc + ns * (np.random.randn(N_rx) + 1j * np.random.randn(N_rx))
        t0 = time.perf_counter()
        fn(yr)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    avg = np.mean(times); std = np.std(times)
    print(f"  {name:<18} {avg:10.1f} {std:>8.1f}")
