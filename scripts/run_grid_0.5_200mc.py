#!/usr/bin/env python
"""
Grid res = 0.5, 200 MC trials → results/0.5-200MC/
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage.signal_utils import (
    generate_tx_signal, compute_crb, build_dictionary, fractional_delay_shift,
)
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11, 'mathtext.fontset': 'stix',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# =========================================================================
SNR_dB_vec = np.arange(-15, 20, 5)
GRID_RES = 0.5
NUM_MC = 200
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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'results', '0.5-200MC')
os.makedirs(OUT_DIR, exist_ok=True)

algo_names = ['Integer SAGE', 'Fractional SAGE', 'Matched Filter', 'OMP', 'NOMP', 'SBL']
algo_short = ['IntSAGE', 'FracSAGE', 'MF', 'OMP', 'NOMP', 'SBL']
colors = ['#333333', '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#17becf']
markers = ['s', '^', '*', 'd', 'v', 'x']
linestyles = ['-', '-', '--', '--', '-.', '-']

num_SNR = len(SNR_dB_vec)
num_algos = len(algo_names)

# =========================================================================
# Signal & dictionary
# =========================================================================
print("="*60)
print(f" Grid = {GRID_RES}, MC = {NUM_MC}")
print("="*60)

s_tx_base = generate_tx_signal(N_seq, seed=SEED)
y_clean = np.zeros(N_rx, dtype=complex)
for d, a in zip(true_delays, true_amps):
    y_clean += a * fractional_delay_shift(s_tx_base, d, N_rx)
signal_power = np.var(y_clean)

tau_grid = np.arange(30, 220 + GRID_RES, GRID_RES)
print(f"Dictionary: {len(tau_grid)} atoms ({tau_grid[0]:.1f}—{tau_grid[-1]:.1f})")
A_dict = build_dictionary(s_tx_base, tau_grid, N_rx)

# CRB per path
S_freq = np.fft.fft(s_tx_base)
k_vec = np.concatenate([np.arange(0, int(np.ceil(N_seq/2))),
                         np.arange(-int(np.floor(N_seq/2)), 0)])
omega = 2 * np.pi * k_vec / N_seq
s_prime = np.fft.ifft((1j * omega) * S_freq)
energy_s_prime = np.sum(np.abs(s_prime)**2)

# =========================================================================
# MC
# =========================================================================
rmse_d = np.zeros((num_SNR, num_algos))
rmse_m = np.zeros((num_SNR, num_algos))
rmse_p = np.zeros((num_SNR, num_algos))
per_path_d = np.zeros((num_SNR, num_algos, L_paths))
crb_d_arr = np.zeros((num_SNR, L_paths))

t_start = time.time()

for s_idx, SNR_dB in enumerate(SNR_dB_vec):
    N0 = signal_power / (10**(SNR_dB/10))
    noise_std = np.sqrt(N0/2)
    crb_per = np.sqrt(N0/(2*np.abs(true_amps)**2*energy_s_prime))
    crb_d_arr[s_idx,:] = crb_per

    sq_d = np.zeros((num_algos, L_paths))
    sq_m = np.zeros((num_algos, L_paths))
    sq_p = np.zeros((num_algos, L_paths))

    for mc in range(NUM_MC):
        noise = noise_std*(rng.randn(N_rx)+1j*rng.randn(N_rx))
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
            m_d, m_a = match_paths(true_delays, true_amps, ests_d[alg], ests_a[alg])
            sq_d[alg,:] += (m_d - true_delays)**2
            sq_m[alg,:] += (np.abs(m_a) - np.abs(true_amps))**2
            pe = np.arctan2(np.sin(np.angle(m_a)-np.angle(true_amps)),
                            np.cos(np.angle(m_a)-np.angle(true_amps)))
            sq_p[alg,:] += pe**2

    per_path_d[s_idx,:,:] = np.sqrt(sq_d / NUM_MC)
    rmse_d[s_idx,:] = np.sqrt(np.mean(sq_d, axis=1) / NUM_MC)
    rmse_m[s_idx,:] = np.sqrt(np.mean(sq_m, axis=1) / NUM_MC)
    rmse_p[s_idx,:] = np.sqrt(np.mean(sq_p, axis=1) / NUM_MC)

    print(f"  SNR={SNR_dB:4d} dB | FracSAGE delay RMSE={rmse_d[s_idx,1]:.5f}")

elapsed = time.time()-t_start
print(f"\nMC completed in {elapsed:.1f}s ({elapsed/(num_SNR*NUM_MC):.2f}s/trial)")

# =========================================================================
# FIGURE 1: RMSE vs SNR (3-panel, matching paper fig_08)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
crb_d_mean = np.sqrt(np.mean(crb_d_arr**2, axis=1))
crb_m_mean = np.sqrt(np.mean((crb_d_arr * np.abs(true_amps)/np.abs(true_amps))**2, axis=1))  # placeholder
# Proper CRB for magnitude
crb_m_per = np.sqrt(N0 / (2 * np.abs(true_amps)**0))  # simplified
# Actually let's use simpler CRB display
crb_m_arr = np.array([np.sqrt(np.mean(np.sqrt(N0/(2*1.0))**2)) for N0 in
                       [signal_power/(10**(s/10)) for s in SNR_dB_vec]])

titles = ['(a) Delay RMSE', '(b) Magnitude RMSE', '(c) Phase RMSE']
rmse_list = [rmse_d, rmse_m, rmse_p]
crb_list = [crb_d_mean, [np.sqrt(np.mean(np.array([np.sqrt(signal_power/(10**(s/10))/(2*abs(a)**2*energy_s_prime)) for a in true_amps]))) for s in SNR_dB_vec],
            [np.sqrt(np.mean(np.array([np.sqrt(signal_power/(10**(s/10))/(2*abs(a)**2*energy_s_prime)) for a in true_amps])))*0.3 for s in SNR_dB_vec]]

# Simplified CRB
for ax_idx, (ax, title, rmse, crb_v) in enumerate(zip(axes, titles, rmse_list,
    [crb_d_mean, np.zeros(num_SNR), np.zeros(num_SNR)])):
    for alg in range(num_algos):
        ax.semilogy(SNR_dB_vec, rmse[:, alg],
                    color=colors[alg], marker=markers[alg],
                    linestyle=linestyles[alg], lw=1.8, ms=8,
                    label=algo_names[alg], markerfacecolor='none')
    if ax_idx == 0:
        ax.semilogy(SNR_dB_vec, crb_d_mean, 'r-', lw=2.5, label='CRB', zorder=10)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim([-17, 17])

fig.suptitle(f'RMSE vs SNR — Grid Resolution = {GRID_RES} ({NUM_MC} MC trials)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_rmse_vs_snr.png'), dpi=1200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_rmse_vs_snr.png")

# =========================================================================
# FIGURE 2: CRB Efficiency (matching fig_07)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (a) Per-path delay error vs SNR
for p in range(L_paths):
    ax1.semilogy(SNR_dB_vec, per_path_d[:, 1, p],  # Frac SAGE only
                 'o-', lw=1.8, ms=8,
                 label=f'Frac SAGE Path {p+1} (τ={true_delays[p]:.1f})')
    ax1.semilogy(SNR_dB_vec, crb_d_arr[:, p], '--', lw=1.5,
                 label=f'CRB Path {p+1}')
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('Delay RMSE (samples)')
ax1.set_title(f'(a) Per-Path Delay RMSE — Frac SAGE (grid={GRID_RES})')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, which='both')

# (b) CRB efficiency ratio: Var(δ̂)/CRB for all algorithms
for alg in range(num_algos):
    eff = rmse_d[:, alg] / crb_d_mean
    ax2.semilogy(SNR_dB_vec, eff,
                 color=colors[alg], marker=markers[alg],
                 linestyle=linestyles[alg], lw=1.8, ms=8,
                 label=algo_names[alg], markerfacecolor='none')
ax2.axhline(1.0, color='gray', ls='--', lw=1)
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('RMSE / CRB (efficiency ratio)')
ax2.set_title(f'(b) CRB Efficiency Ratio — Grid Resolution = {GRID_RES}')
ax2.legend(fontsize=8, loc='best')
ax2.grid(alpha=0.3, which='both')
ymax = np.max(rmse_d[:, 2:] / crb_d_mean.reshape(-1,1)[:, [0]*4]) * 1.2
ax2.set_ylim([0.5, min(ymax, 500)])

fig.suptitle(f'CRB Efficiency Analysis ({NUM_MC} MC, {len(tau_grid)} atoms)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_crb_efficiency.png'), dpi=1200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_crb_efficiency.png")

# =========================================================================
# FIGURE 3: Channel stem (single trial at 10 dB)
# =========================================================================
SNR_plot = 10
N0_plot = signal_power / (10**(SNR_plot/10))
noise_plot = np.sqrt(N0_plot/2)*(rng.randn(N_rx)+1j*rng.randn(N_rx))
y_plot = y_clean + noise_plot

d_i, a_i = algo_SAGE(y_plot, s_tx_base, N_rx, L_paths, do_frac=False)
d_f, a_f = algo_SAGE(y_plot, s_tx_base, N_rx, L_paths, do_frac=True)
d_m, a_m = algo_MF(y_plot, A_dict, tau_grid, L_paths)
d_o, a_o = algo_OMP(y_plot, A_dict, tau_grid, L_paths)
d_n, a_n = algo_NOMP(y_plot, s_tx_base, N_rx, L_paths, tau_grid)
d_s, a_s = algo_SBL(y_plot, A_dict, tau_grid, L_paths, use_woodbury=True)

fig, ax = plt.subplots(figsize=(12, 5))
offset = 0.12
offsets = [-0.30, -0.15, 0.0, 0.15, 0.30, 0.45]
marker_list = ['s', '^', '*', 'd', 'x', 'v']
lw_list = [1.5, 2.0, 1.5, 1.5, 1.5, 1.5]
ls_list = ['-', '-', '--', '--', '-.', '-']
all_d = [d_i, d_f, d_m, d_o, d_n, d_s]
all_a = [a_i, a_f, a_m, a_o, a_n, a_s]

# True
ml, sl, bl = ax.stem(true_delays, np.abs(true_amps), linefmt='r-', markerfmt='ro', basefmt=' ',
        label='True')
plt.setp(ml, markersize=8); plt.setp(sl, linewidth=2)

stem_colors = ['#333333','#1f77b4','#2ca02c','#d62728','#ff7f0e','#17becf']
for alg in range(num_algos):
    ml, sl, bl = ax.stem(all_d[alg]+offsets[alg], np.abs(all_a[alg]),
            linefmt='-', markerfmt=marker_list[alg], basefmt=' ')
    plt.setp(ml, color=stem_colors[alg], markersize=7)
    plt.setp(sl, color=stem_colors[alg], linewidth=lw_list[alg],
             linestyle=ls_list[alg])
    # 手动创建legend条目
    ax.plot([], [], color=stem_colors[alg], marker=marker_list[alg],
            linestyle=ls_list[alg], linewidth=lw_list[alg],
            label=algo_names[alg], markersize=7)

ax.set_xlabel('Delay (samples)')
ax.set_ylabel(r'$|\alpha|$')
ax.set_title(f'Single-Trial Channel Estimation (SNR={SNR_plot} dB, grid={GRID_RES})')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim([30, 220])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_channel_stem.png'), dpi=1200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_channel_stem.png")

# =========================================================================
# FIGURE 4: Residual comparison
# =========================================================================
residuals = []
for d_est, a_est in zip(all_d, all_a):
    r = y_plot.copy()
    for d, a in zip(d_est, a_est):
        r -= a * fractional_delay_shift(s_tx_base, d, N_rx)
    residuals.append(np.abs(r))

fig, ax = plt.subplots(figsize=(12, 4))
for alg in range(num_algos):
    ax.plot(residuals[alg], color=colors[alg], linestyle=linestyles[alg],
            lw=1.5 if alg != 1 else 2.0, label=algo_names[alg])
ax.set_xlabel('Sample index')
ax.set_ylabel('|Residual|')
ax.set_title(f'Residual Signal Comparison (SNR={SNR_plot} dB, grid={GRID_RES})')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([30, 250])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_residual.png'), dpi=1200, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig_residual.png")

# =========================================================================
# Save NPZ
# =========================================================================
np.savez(
    os.path.join(OUT_DIR, 'mc_results.npz'),
    SNR_dB_vec=SNR_dB_vec, grid_res=GRID_RES, num_MC=NUM_MC,
    rmse_delay=rmse_d, rmse_mag=rmse_m, rmse_phase=rmse_p,
    per_path_delay=per_path_d, crb_delay=crb_d_arr,
    algo_names=algo_names,
)
print("  Saved: mc_results.npz")

# =========================================================================
# Console summary
# =========================================================================
print(f"\n{'='*60}")
print(f" SUMMARY — Grid={GRID_RES}, {NUM_MC} MC")
print(f"{'='*60}")
for s_idx, SNR_dB in enumerate(SNR_dB_vec):
    print(f"\n  SNR = {SNR_dB:4d} dB:")
    print(f"  {'Algorithm':<18} {'Delay':>10} {'CRB':>10} {'Eff':>8}")
    crb_m = np.mean(crb_d_arr[s_idx])
    for alg in range(num_algos):
        print(f"  {algo_names[alg]:<18} {rmse_d[s_idx,alg]:>10.4f} {crb_m:>10.4f} {rmse_d[s_idx,alg]/crb_m:>8.2f}x")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
