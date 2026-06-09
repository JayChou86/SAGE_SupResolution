#!/usr/bin/env python
"""
===========================================================================
Grid Resolution Sweep for Dictionary-based Methods
==================================================
Sweeps grid_res from 1.0 down to 0.4, running MC trials at each step.
Saves per-step results + generates cumulative comparison.

Usage:
    python scripts/run_grid_sweep.py
===========================================================================
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage.signal_utils import (
    generate_tx_signal, generate_rx_signal, compute_crb,
    build_dictionary, fractional_delay_shift,
)
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths

# =========================================================================
# Parameters
# =========================================================================
SNR_dB_vec = np.arange(-15, 20, 5)         # -15:5:15 dB (7 points)
GRID_RES_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # sweep from 1.0 down
NUM_MC_SWEEP = 50                           # reduced MC for speed
NUM_MC_FINAL = 200                          # full MC for best resolution
SEED = 40
rng = np.random.RandomState(SEED)
np.random.seed(SEED)

N_seq = 511; N_rx = 1000; L_paths = 3
true_delays = np.array([50.43, 60.66, 200.51])
true_amps = np.array([
    1.0 * np.exp(1j * 0.2),
    0.5 * np.exp(-1j * 1.5),
    0.8 * np.exp(1j * 0.8),
])

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
SWEEP_DIR = os.path.join(RESULTS_DIR, 'grid_sweep')
os.makedirs(SWEEP_DIR, exist_ok=True)

algo_names = ['Integer SAGE', 'Fractional SAGE', 'Matched Filter', 'OMP', 'NOMP', 'SBL']
# Short names for compact display
algo_short = ['IntSAGE', 'FracSAGE', 'MF', 'OMP', 'NOMP', 'SBL']

# =========================================================================
# Pre-compute signal components
# =========================================================================
s_tx_base = generate_tx_signal(N_seq, seed=SEED)
y_clean = np.zeros(N_rx, dtype=complex)
for d, a in zip(true_delays, true_amps):
    y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)
signal_power = np.var(y_clean)

# CRB per-path
S_freq = np.fft.fft(s_tx_base)
k_vec = np.concatenate([
    np.arange(0, int(np.ceil(N_seq / 2))),
    np.arange(-int(np.floor(N_seq / 2)), 0),
])
omega = 2 * np.pi * k_vec / N_seq
s_prime = np.fft.ifft((1j * omega) * S_freq)
energy_s_prime = np.sum(np.abs(s_prime) ** 2)

num_SNR = len(SNR_dB_vec)
num_algos = len(algo_names)

# =========================================================================
# Helper: run MC for one grid resolution
# =========================================================================
def run_mc(grid_res, num_mc):
    """Run MC experiment for a given grid resolution. Returns RMSE arrays."""
    tau_grid = np.arange(30, 220 + grid_res, grid_res)
    print(f"  Dictionary: {len(tau_grid)} atoms, grid={grid_res:.3f}")
    A_dict = build_dictionary(s_tx_base, tau_grid, N_rx)

    rmse_d = np.zeros((num_SNR, num_algos))
    rmse_m = np.zeros((num_SNR, num_algos))
    rmse_p = np.zeros((num_SNR, num_algos))

    # Per-path error storage
    per_path_d = np.zeros((num_SNR, num_algos, L_paths))

    crb_d_all = np.zeros((num_SNR, L_paths))

    for s_idx, SNR_dB in enumerate(SNR_dB_vec):
        N0 = signal_power / (10 ** (SNR_dB / 10))
        noise_std = np.sqrt(N0 / 2)

        crb_d_per = np.sqrt(N0 / (2 * np.abs(true_amps)**2 * energy_s_prime))
        crb_d_all[s_idx, :] = crb_d_per

        sq_d = np.zeros((num_algos, L_paths))
        sq_m = np.zeros((num_algos, L_paths))
        sq_p = np.zeros((num_algos, L_paths))

        for mc in range(num_mc):
            noise = noise_std * (rng.randn(N_rx) + 1j * rng.randn(N_rx))
            y_rx = y_clean + noise

            d_int, a_int = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, do_frac=False)
            d_frac, a_frac = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, do_frac=True)
            d_mf, a_mf = algo_MF(y_rx, A_dict, tau_grid, L_paths)
            d_omp, a_omp = algo_OMP(y_rx, A_dict, tau_grid, L_paths)
            d_nomp, a_nomp = algo_NOMP(y_rx, s_tx_base, N_rx, L_paths, tau_grid)
            d_sbl, a_sbl = algo_SBL(y_rx, A_dict, tau_grid, L_paths, use_woodbury=True)

            ests_d = [d_int, d_frac, d_mf, d_omp, d_nomp, d_sbl]
            ests_a = [a_int, a_frac, a_mf, a_omp, a_nomp, a_sbl]

            for alg in range(num_algos):
                m_d, m_a = match_paths(true_delays, true_amps,
                                        ests_d[alg], ests_a[alg])
                sq_d[alg, :] += (m_d - true_delays)**2
                sq_m[alg, :] += (np.abs(m_a) - np.abs(true_amps))**2
                pe = np.arctan2(
                    np.sin(np.angle(m_a) - np.angle(true_amps)),
                    np.cos(np.angle(m_a) - np.angle(true_amps)),
                )
                sq_p[alg, :] += pe**2

        # Per-path RMSE
        per_path_d[s_idx, :, :] = np.sqrt(sq_d / num_mc)
        # Overall RMSE (sqrt of mean squared error across paths)
        rmse_d[s_idx, :] = np.sqrt(np.mean(sq_d, axis=1) / num_mc)
        rmse_m[s_idx, :] = np.sqrt(np.mean(sq_m, axis=1) / num_mc)
        rmse_p[s_idx, :] = np.sqrt(np.mean(sq_p, axis=1) / num_mc)

    return rmse_d, rmse_m, rmse_p, crb_d_all, per_path_d

# =========================================================================
# MAIN SWEEP
# =========================================================================
print("=" * 70)
print(" GRID RESOLUTION SWEEP — Dictionary-based Methods")
print("=" * 70)
print(f" SNR range:  {SNR_dB_vec[0]} to {SNR_dB_vec[-1]} dB")
print(f" MC trials:  {NUM_MC_SWEEP} (sweep)")
print(f" Grid steps: {GRID_RES_LIST}")
print("=" * 70)

summary_rows = []

for grid_res in GRID_RES_LIST:
    print(f"\n{'─'*60}")
    print(f" Grid resolution = {grid_res:.3f}")
    print(f"{'─'*60}")
    t0 = time.time()

    rmse_d, rmse_m, rmse_p, crb_d, per_path_d = run_mc(grid_res, NUM_MC_SWEEP)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Save per-grid results
    np.savez(
        os.path.join(SWEEP_DIR, f'mc_grid_{grid_res:.3f}.npz'),
        SNR_dB_vec=SNR_dB_vec, grid_res=grid_res,
        rmse_delay=rmse_d, rmse_mag=rmse_m, rmse_phase=rmse_p,
        crb_delay=crb_d, per_path_delay=per_path_d,
        algo_names=algo_names,
    )

    # --- Print per-algo RMSE at 0 dB ---
    mid = len(SNR_dB_vec) // 2
    print(f"\n  {'Algorithm':<18} {'Delay RMSE@0dB':>15} {'vs CRB':>10} {'CRB Eff':>10}")
    print(f"  {'─'*18} {'─'*15} {'─'*10} {'─'*10}")
    crb_mean_0db = np.mean(crb_d[mid, :])
    for alg in range(num_algos):
        ratio = rmse_d[mid, alg] / crb_mean_0db
        print(f"  {algo_names[alg]:<18} {rmse_d[mid, alg]:>15.4f} {crb_mean_0db:>10.4f} {ratio:>10.2f}x")

    # --- Per-path analysis at 0 dB ---
    print(f"\n  --- Per-Path Delay RMSE @ 0 dB ---")
    print(f"  {'Algorithm':<18} {'Path1(50.43)':>14} {'Path2(60.66)':>14} {'Path3(200.51)':>14}")
    print(f"  {'─'*18} {'─'*14} {'─'*14} {'─'*14}")
    for alg in range(num_algos):
        p1, p2, p3 = per_path_d[mid, alg, :]
        print(f"  {algo_names[alg]:<18} {p1:>14.4f} {p2:>14.4f} {p3:>14.4f}")
    print(f"  {'CRB per-path':<18} {crb_d[mid,0]:>14.4f} {crb_d[mid,1]:>14.4f} {crb_d[mid,2]:>14.4f}")

    # --- CRB efficiency at best SNR (+15 dB) ---
    best = -1  # last SNR point
    crb_mean_best = np.mean(crb_d[best, :])
    print(f"\n  --- CRB Efficiency @ +{SNR_dB_vec[best]:.0f} dB ---")
    for alg in range(num_algos):
        ratio = rmse_d[best, alg] / crb_mean_best
        print(f"  {algo_names[alg]:<18} RMSE={rmse_d[best, alg]:.5f}  CRB Eff = {ratio:.2f}x")

    # Store summary
    summary_rows.append({
        'grid_res': grid_res,
        'rmse_d_0dB': {algo_short[a]: float(rmse_d[mid, a]) for a in range(num_algos)},
        'crb_eff_15dB': {algo_short[a]: float(rmse_d[best, a] / crb_mean_best) for a in range(num_algos)},
    })

# =========================================================================
# FINAL SUMMARY
# =========================================================================
print(f"\n\n{'='*70}")
print(" CROSS-GRID COMPARISON — Delay RMSE @ 0 dB")
print(f"{'='*70}")
header = f"  {'Grid':>6s}"
for s in algo_short: header += f"  {s:>10s}"
header += f"  {'CRB':>10s}"
print(header)
print(f"  {'─'*6}  " + "  ".join(['─'*10]* (num_algos+1)))
crb_fixed = np.mean(np.sqrt(0.05 / (2 * np.abs(true_amps)**2 * energy_s_prime)))
for row in summary_rows:
    line = f"  {row['grid_res']:>6.3f}"
    for s in algo_short:
        line += f"  {row['rmse_d_0dB'][s]:>10.4f}"
    line += f"  {crb_fixed:>10.4f}"
    print(line)

print(f"\n{'='*70}")
print(" CROSS-GRID COMPARISON — CRB Efficiency @ +15 dB (lower = better)")
print(f"{'='*70}")
header = f"  {'Grid':>6s}"
for s in algo_short: header += f"  {s:>10s}"
print(header)
print(f"  {'─'*6}  " + "  ".join(['─'*10]* num_algos))
for row in summary_rows:
    line = f"  {row['grid_res']:>6.3f}"
    for s in algo_short:
        line += f"  {row['crb_eff_15dB'][s]:>10.2f}"
    print(line)

# Save summary JSON
with open(os.path.join(SWEEP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary_rows, f, indent=2)

print(f"\nAll results saved to: {SWEEP_DIR}")
print("Done.")
