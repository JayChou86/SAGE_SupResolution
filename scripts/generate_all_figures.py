#!/usr/bin/env python
"""
Generate ALL 12 figures for paper_twc.tex per README_FIGURES.md specification.

Data sources:
  - mc_results_500.npz (500 MC trials, 7 algorithms x 7 SNR)
  - On-the-fly signal generation (QPSK, seed=42)
  - Hard-coded ablation / sea trial tables

Output: results/fig_01_grid_mismatch.png ... results/fig_12_seatrial.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.fft import fft, ifft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'mathtext.fontset': 'stix',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

# 鈹€鈹€ Load 500-MC data 鈹€鈹€
MC = np.load('mc_results_500.npz')
SNR_VEC = MC['SNR_dB_vec']
ALGO_NAMES = [str(a) for a in MC['algo_names']]
RMSE_D = MC['rmse_delay']
RMSE_M = MC['rmse_mag']
RMSE_P = MC['rmse_phase']
CRB_D = MC['crb_delay']
CRB_M = MC['crb_mag']
CRB_P = MC['crb_phase']

# 鈹€鈹€ Shared signal constants 鈹€鈹€
from sage.signal_utils import generate_tx_signal, fractional_delay_shift
S_TX = generate_tx_signal(511, seed=42)
N_SEQ = len(S_TX)
E_S = np.vdot(S_TX, S_TX).real

# Compute c = -R''(0)/2 via FFT-based autocorrelation (handles 卤蟿 cleanly)
h_c = 0.01
S_freq_ = fft(S_TX)
k_ = np.concatenate([np.arange(0, int(np.ceil(N_SEQ/2))),
                      np.arange(-int(np.floor(N_SEQ/2)), 0)])
ph_p = np.exp(-1j * 2 * np.pi * k_ * h_c / N_SEQ)
ph_n = np.exp(-1j * 2 * np.pi * k_ * (-h_c) / N_SEQ)
s_plus  = ifft(S_freq_ * ph_p)
s_minus = ifft(S_freq_ * ph_n)
R0_c = np.abs(np.vdot(S_TX, S_TX))
Rh_c  = np.abs(np.vdot(S_TX, s_plus))
Rmh_c = np.abs(np.vdot(S_TX, s_minus))
# Central diff: R''(0) 鈮?(R(h) - 2R(0) + R(-h)) / h虏
C_VAL = -(Rh_c - 2*R0_c + Rmh_c) / (h_c*h_c) / 2.0
C_VAL = abs(C_VAL)

# ||s'||^2
S_freq = fft(S_TX)
k_vec = np.concatenate([np.arange(0, int(np.ceil(N_SEQ/2))),
                         np.arange(-int(np.floor(N_SEQ/2)), 0)])
omega = 2 * np.pi * k_vec / N_SEQ
s_prime = ifft((1j * omega) * S_freq)
ENERGY_SP = np.sum(np.abs(s_prime)**2)

print(f"Signal: N_seq={N_SEQ}, c={C_VAL:.4f}, ||s'||^2={ENERGY_SP:.4f}")

# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 1: Grid Mismatch Illustration (schematic)           鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_01():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 鈹€鈹€ (a) Mechanism 鈹€鈹€
    tau = np.linspace(-2.5, 6.5, 500)
    tau_true = 2.37
    sigma_c = 0.55
    corr = np.exp(-(tau - tau_true)**2 / (2*sigma_c**2))
    ax1.plot(tau, corr, 'b-', lw=2, label='Continuous $|C(\\tau)|^2$')
    # Integer sampling points
    int_pts = np.arange(-1, 7)
    corr_int = np.exp(-(int_pts - tau_true)**2 / (2*sigma_c**2))
    ax1.plot(int_pts, corr_int, 'ro', ms=8, zorder=5, label='Integer samples')
    ax1.axvline(tau_true, color='green', ls='--', lw=2, label=f'True $\\tau={tau_true}$')
    # Double arrow
    # Double arrow centered at true 蟿, spanning from nearest integer k to 蟿_true
    k_nearest = round(tau_true)
    ax1.annotate('', xy=(tau_true, 0.15), xytext=(k_nearest, 0.15),
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=2))
    ax1.text((k_nearest + tau_true)/2, 0.18, '$\\delta$', ha='center',
             fontsize=12, color='darkred')
    ax1.axvline(k_nearest, color='gray', ls=':', lw=1, alpha=0.5, label=f'Integer grid $k={k_nearest}$')
    ax1.set_xlabel('Delay (samples)'); ax1.set_ylabel('$|C(\\tau)|^2$')
    ax1.set_title('(a) Grid Mismatch Mechanism')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_ylim([0, 1.15])

    # 鈹€鈹€ (b) Conceptual RMSE 鈥?curve formulas from figs_1_4_bare.py 鈹€鈹€
    snr = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30])
    rmse_int  = np.minimum(3.2, 8.0/(1+np.exp((snr+8)/3)) + 0.15)
    rmse_frac = 0.5*np.exp(-snr/8) + 0.02
    crb       = 0.3*np.exp(-snr/7) + 0.005

    ax2.semilogy(snr, rmse_int,  'ks--', lw=2, ms=7, label='Integer SAGE')
    ax2.semilogy(snr, rmse_frac, 'b^-', lw=2, ms=7, label='Fractional SAGE')
    ax2.semilogy(snr, crb,       'r-',  lw=2, ms=7, label='CRB')
    ax2.fill_between([15, 30], 2.5, 4, alpha=0.08, color='gray')
    ax2.annotate('Grid mismatch\nfloor', xy=(22, 3.2), ha='center', fontsize=9,
                 color='#555555')

    ax2.set_xlabel('SNR (dB)'); ax2.set_ylabel('Delay RMSE (samples)')
    ax2.set_title('(b) RMSE vs SNR $-$ Conceptual')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which='both')
    ax2.set_xlim([-10, 30]); ax2.set_ylim([5e-3, 20])

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_01_grid_mismatch.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_01_grid_mismatch.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 2: Algorithm Flowchart (schematic)                  鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_02():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.axis('off')

    c_main = '#BBDEFB'; c_core = '#FFCCBC'; c_out = '#C8E6C9'; c_loop = '#E0E0E0'

    def box(x, y, w, h, txt, color, fs=9, bold=False):
        r = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor='#444', lw=1.2)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs,
                 fontweight='normal')

    def arr(x1, y1, x2, y2, style='simple', c='#333'):
        kw = dict(arrowstyle='->', color=c, lw=1.5)
        if style == 'arc': kw['connectionstyle'] = 'arc3,rad=0.3'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=kw)

    # Left column: main loop
    box(0.8, 12.0, 4.5, 1.0, 'Input: $\\mathbf{y}_{\\rm rx}$, $\\mathbf{s}_{\\rm tx}$', c_main, 10, True)
    box(0.8, 10.0, 4.5, 1.0, 'Initialize: $\\mathbf{r} \\leftarrow \\mathbf{y}_{\\rm rx}$', c_main)
    box(0.8, 7.5, 4.5, 1.2,
        'E-step (Hidden Data)\n$\\hat{\\mathbf{x}}_l = \\mathbf{r} + \\hat{\\alpha}_l\\mathbf{s}(\\hat{\\tau}_l)$', c_main)
    box(0.8, 4.8, 4.5, 1.0,
        'M-step Amplitude\n$\\hat{\\alpha}_l = \\langle\\mathbf{s}(\\hat{\\tau}_l),\\hat{\\mathbf{x}}_l\\rangle/E_s$', c_main)
    box(0.8, 3.2, 4.5, 1.0,
        'Residual Update\n$\\mathbf{r} \\leftarrow \\hat{\\mathbf{x}}_l - \\hat{\\alpha}_l\\mathbf{s}(\\hat{\\tau}_l)$', c_main)
    box(0.8, 1.5, 4.5, 1.0,
        'Sort by $|\\hat{\\alpha}_l|$ + Collision Detect\nOutput: $\\{\\hat{\\tau}_l, \\hat{\\alpha}_l\\}$', c_out, 9, True)

    # Right column: core contribution
    box(7.0, 10.0, 5.5, 1.2,
        'FFT Cross-Correlation\n$\\mathbf{C}(\\tau) = \\mathcal{F}^{-1}\\{X_f\\cdot S_f^*\\}/E_s$', c_core, 9)
    box(7.0, 7.5, 5.5, 1.5,
        'Stage 1: Parabolic Interpolation\n'
        '$\\hat{\\delta} = \\frac{1}{2}\\cdot\\frac{y_{-1}-y_{+1}}{y_{-1}-2y_0+y_{+1}}$\n'
        '$\\hat{\\tau}^{(1)} = k_{\\rm opt} + \\hat{\\delta}$', c_core, 8)
    box(7.0, 4.8, 5.5, 1.5,
        'Stage 2: Local Grid Refinement\n'
        '$\\hat{\\tau}_l = \\arg\\max_{\\tau\\in[\\hat{\\tau}^{(1)}\\pm0.5]}'
        '|\\langle\\mathbf{s}(\\tau),\\hat{\\mathbf{x}}_l\\rangle|^2$\n'
        '($\\Delta=0.005$, 40-200 evals)', c_core, 8)
    # Highlight
    rect = Rectangle((6.4, 3.8), 6.7, 7.0, fill=False, edgecolor='red', lw=2.5, ls='--')
    ax.add_patch(rect)
    ax.text(9.75, 10.6, 'CORE\nCONTRIBUTION', ha='center', fontsize=9,
            color='red')

    # Loop indicator
    box(7.0, 1.5, 5.5, 1.0,
        'SAGE Loop: for $i=1:I_{\\max}$, for $l=1:L$\n$I_{\\max}=15$, $L=3$', c_loop, 9)

    # Arrows
    arr(3.05, 12.0, 3.05, 11.0)
    arr(3.05, 10.0, 3.05, 8.7)
    arr(3.05, 7.5, 3.05, 5.8)
    arr(3.05, 4.8, 3.05, 4.2)
    arr(3.05, 3.2, 3.05, 2.5)
    # E-step -> FFT correlation (hidden data flow)
    arr(5.3, 8.4, 7.0, 10.4, c='#1565C0')
    ax.text(6.15, 9.7, r'$\hat{\mathbf{x}}_l$', fontsize=8, color='#1565C0', ha='center')

    # Core arrows
    arr(9.75, 10.0, 9.75, 9.0, c='#E65100')
    arr(9.75, 7.5, 9.75, 6.3, c='#E65100')
    arr(9.75, 4.8, 9.75, 2.5, c='#D84315')
    # Loop arrows
    arr(9.75, 1.5, 6.5, 8.1, style='arc', c='#FF6F00')
    arr(3.05, 1.5, 3.05, 0.8, style='arc', c='#FF6F00')
    ax.text(5.5, 10.5, 'next iter', fontsize=7, color='#FF6F00', style='italic')

    ax.set_title('Two-Stage Fractional SAGE Algorithm', fontsize=14, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_02_flowchart.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_02_flowchart.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 3: Parabolic Interpolation Illustration            鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_03():
    """Parabolic interpolation 鈥?per README_FIGURES.md spec."""
    deltas = [0.0, 0.25, 0.43]
    titles = ['(a) $\\delta$=0.00 (on-grid)', '(b) $\\delta$=0.25', '(c) $\\delta$=0.43 (off-grid)']
    c_use = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, d, title in zip(axes, deltas, titles):
        x_dense = np.linspace(-1.5, 1.5, 300)
        y_dense = 1 - c_use * (x_dense - d)**2
        y_dense = np.clip(y_dense, 0, 1.1)

        x_pts = np.array([-1, 0, 1])
        y_pts = 1 - c_use * (x_pts - d)**2

        # Fit parabola through 3 points (raw values, no clipping)
        A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
        coeffs = np.linalg.solve(A, y_pts)
        a_fit, b_fit, c_fit = coeffs
        x_v = -b_fit / (2*a_fit) if abs(a_fit) > 1e-10 else 0
        d_hat = np.clip(x_v, -0.5, 0.5)

        y_fit = a_fit * x_dense**2 + b_fit * x_dense + c_fit

        ax.plot(x_dense, y_dense, 'k-', lw=1.5, alpha=0.4, label='True $|R(\\tau)|^2$')
        ax.plot(x_pts, y_pts, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(x_dense, y_fit, 'b-', lw=2, label='Fitted parabola')
        ax.axvline(d, color='green', ls='--', lw=2, label=f'True $\\delta={d:.2f}$')
        ax.axvline(d_hat, color='blue', ls=':', lw=2, label=f'$\\hat{{\\delta}}={d_hat:.3f}$')
        for xi, yi, lbl in zip(x_pts, y_pts, ['$y_{-1}$', '$y_0$', '$y_{+1}$']):
            ax.annotate(lbl, (xi, yi), textcoords='offset points', xytext=(0, 12),
                       ha='center', fontsize=10, color='darkred')
        ax.set_xlabel('Offset from $k_{\\rm opt}$')
        ax.set_ylabel('$|R(\\tau)|^2$')
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-0.05, 1.1])
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Parabolic Interpolation (Stage 1)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_03_parabolic.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_03_parabolic.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 4: Stage 2 Local Refinement                         鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_04():
    tau_true = 50.43
    sigma_c = 0.18
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Coarse
    tau_c = np.arange(49.6, 51.45, 0.05)
    f_c = np.exp(-(tau_c - tau_true)**2/(2*sigma_c**2)) + 0.015*np.random.randn(len(tau_c))
    ax1.plot(tau_c, f_c, 'bo-', ms=7, lw=1.5, label='Coarse grid ($\\Delta=0.05$)')
    ax1.axvline(tau_true, color='green', ls='--', lw=2, label=f'True $\\tau={tau_true:.2f}$')
    idx_c = np.argmax(f_c)
    ax1.axvline(tau_c[idx_c], color='blue', ls=':', lw=2, label=f'$\\tau_c={tau_c[idx_c]:.2f}$')
    ax1.plot(tau_c[idx_c], f_c[idx_c], 'r*', ms=15, zorder=5)
    sh_l, sh_h = tau_c[idx_c]-0.06, tau_c[idx_c]+0.06
    ax1.axvspan(sh_l, sh_h, alpha=0.15, color='orange', label='Fine zone')
    ax1.set_xlabel('Delay (samples)'); ax1.set_ylabel('$|C(\\tau)|^2$')
    ax1.set_title('(a) Pass 1: Coarse Search ($\\Delta=0.05$)')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (b) Fine
    tau_f = np.arange(sh_l, sh_h+0.001, 0.005)
    f_f = np.exp(-(tau_f - tau_true)**2/(2*sigma_c**2)) + 0.008*np.random.randn(len(tau_f))
    ax2.plot(tau_f, f_f, 'r-o', ms=5, lw=1.5, label='Fine grid ($\\Delta=0.005$)')
    ax2.axvline(tau_true, color='green', ls='--', lw=2, label=f'True $\\tau={tau_true:.2f}$')
    idx_f = np.argmax(f_f)
    ax2.axvline(tau_f[idx_f], color='darkred', ls=':', lw=2, label=f'$\\tau^*={tau_f[idx_f]:.3f}$')
    ax2.plot(tau_f[idx_f], f_f[idx_f], 'r*', ms=15, zorder=5)
    ax2.set_xlabel('Delay (samples)'); ax2.set_ylabel('$|C(\\tau)|^2$')
    ax2.set_title('(b) Pass 2: Fine Search ($\\Delta=0.005$)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle('Stage 2: Local Grid Refinement', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_04_localrefine.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_04_localrefine.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 5: Autocorrelation vs Quadratic Approximation      鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_05():
    tau_vals = np.linspace(-3, 3, 601)
    R_vals = np.zeros(len(tau_vals))
    for i, t in enumerate(tau_vals):
        if -N_SEQ < t < N_SEQ:
            s_shifted = fractional_delay_shift(S_TX, abs(t), N_SEQ + int(abs(t)) + 10)
            if t >= 0:
                R_vals[i] = np.abs(np.vdot(S_TX, s_shifted[:N_SEQ]))
            else:
                R_vals[i] = np.abs(np.vdot(s_shifted[:N_SEQ], S_TX))
    R_vals /= E_S

    R_quad = 1 - C_VAL * tau_vals**2
    R_quad = np.maximum(R_quad, 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_vals, R_vals, 'b-', lw=1.8, label='True $|R(\\tau)|$')
    ax.plot(tau_vals, R_quad, 'r--', lw=1.8, label=f'Quadratic: $1 - {C_VAL:.2f}\\tau^2$')
    ax.axvspan(-1, 1, alpha=0.1, color='green', label='$|\\tau|\\leq 1$ (Lemma 1 domain)')
    ax.axvline(-1, color='gray', ls=':', alpha=0.5)
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.axvline(1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Delay offset $\\tau$ (samples)')
    ax.set_ylabel('$|R(\\tau)|$')
    ax.set_title('Autocorrelation vs Quadratic Approximation')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim([-3, 3]); ax.set_ylim([-0.05, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_05_autocorrelation.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_05_autocorrelation.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 6: Variance vs SNR                                  鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_06():
    snr_db = np.arange(-10, 41, 5)
    snr_lin = 10**(snr_db/10)
    delta_f = 0.25

    # Empirical values from README
    emp_var = np.array([8.2e-1, 3.5e-1, 1.1e-1, 3.3e-2, 9.8e-3, 3.1e-3,
                        1.0e-3, 3.2e-4, 1.0e-4, 3.2e-5, 1.0e-5])

    # Theorem 1
    thm_var = (12*delta_f**2 + 1 - 4*delta_f) / (16 * C_VAL**2 * snr_lin)
    # CRB
    crb_v = 1.0 / (2 * snr_lin * ENERGY_SP)
    # Integer MSE
    int_mse = delta_f**2

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(snr_db, emp_var, 'bo-', lw=2, ms=8,
              label=f'Empirical Var($\\hat{{\\delta}}$), $\\delta={delta_f}$')
    ax.loglog(snr_db, thm_var, 'g--', lw=2,
              label='Theorem 1: $\\frac{12\\delta^2+1-4\\delta}{16c^2\\cdot\\mathrm{SNR}}$')
    ax.loglog(snr_db, crb_v, 'r-.', lw=2,
              label='CRB $=1/(2\\cdot\\mathrm{SNR}\\cdot\\|s\'\\|^2)$')
    ax.axhline(int_mse, color='gray', ls=':', lw=2, label=f'Integer MSE $=\\delta^2={int_mse:.4f}$')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Variance / MSE (samples$^2$)')
    ax.set_title(f'Variance of $\\hat{{\\delta}}$ vs SNR ($\\delta={delta_f}$)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
    ax.set_ylim([5e-7, 2])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_06_variance.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_06_variance.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 7: Bias & CRB Efficiency                            鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_07():
    # (a) Bias vs 未 using quartic model: R(蟿) 鈮?1 - c蟿虏 + c鈧勏勨伌
    c4 = C_VAL**2 / 5  # approximate
    deltas = np.linspace(-0.55, 0.55, 200)
    bias = np.zeros(len(deltas))
    for i, d in enumerate(deltas):
        if abs(d) > 0.5:
            bias[i] = np.nan; continue
        # Three sample points from quartic model
        def Rq(x):
            return 1 - C_VAL*x**2 + c4*x**4
        y_m1 = Rq(-1 - d); y_0 = Rq(-d); y_p1 = Rq(1 - d)
        denom = y_m1 - 2*y_0 + y_p1
        if abs(denom) > 1e-12:
            d_hat = np.clip(0.5*(y_m1 - y_p1)/denom, -0.5, 0.5)
        else:
            d_hat = 0
        bias[i] = d_hat - d

    # (b) CRB efficiency
    snr_db = np.arange(-10, 41, 5)
    snr_lin = 10**(snr_db/10)
    delta_f = 0.25
    thm_var = (12*delta_f**2 + 1 - 4*delta_f) / (16 * C_VAL**2 * snr_lin)
    crb_v = 1.0 / (2 * snr_lin * ENERGY_SP)
    eff = thm_var / crb_v

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(deltas, bias*1000, 'b-', lw=1.5)
    ax1.axhline(0, color='gray', ls='--')
    ax1.set_xlabel('True fractional delay $\\delta$')
    ax1.set_ylabel('Bias ($\\times 10^{-3}$ samples)')
    ax1.set_title('(a) Deterministic Bias (quartic model)')
    ax1.grid(alpha=0.3)

    ax2.plot(snr_db, eff, 'rs-', lw=2, ms=8)
    ax2.axhline(0.93, color='blue', ls=':', lw=2, label='Theorem 2: 0.93')
    ax2.axhline(1.0, color='gray', ls='--', lw=1)
    ax2.set_xlabel('SNR (dB)'); ax2.set_ylabel('$\\mathrm{Var}(\\hat{\\delta}) / \\mathrm{CRB}$')
    ax2.set_title('(b) CRB Efficiency Ratio')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 2])

    fig.suptitle('Parabolic Estimator: Bias & CRB Efficiency', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_07_bias_efficiency.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_07_bias_efficiency.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 8: RMSE vs SNR (Core) 鈥?USES 500-MC DATA           鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_08():
    colors = ['k', 'b', 'g', 'm', 'orange', 'c', 'brown']
    markers = ['s', '^', '*', 'd', 'v', 'x', 'P']
    linestyles = ['-', '-', '--', '--', '-.', '-', ':']
    num_alg = len(ALGO_NAMES)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    titles = ['Delay', 'Magnitude', 'Phase']
    data = [(RMSE_D, CRB_D, 'Delay RMSE (samples)', [5e-4, 200]),
            (RMSE_M, CRB_M, 'Magnitude RMSE', None),
            (RMSE_P, CRB_P, 'Phase RMSE (rad)', None)]

    for ax, (rmse, crb, ylabel, ylim), title in zip(axes, data, titles):
        ax.semilogy(SNR_VEC, crb, 'r-', lw=2.5, label='CRB', zorder=10)
        for alg in range(num_alg):
            ax.semilogy(SNR_VEC, rmse[:, alg],
                        color=colors[alg], marker=markers[alg],
                        linestyle=linestyles[alg], lw=1.5, ms=7,
                        label=ALGO_NAMES[alg])
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which='both')
        if ylim: ax.set_ylim(ylim)
        if title == 'Delay': ax.legend(fontsize=7, loc='lower left')

    fig.suptitle('Multi-path Parameter Estimation Performance (500 MC Trials)',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_08_rmse_vs_snr.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_08_rmse_vs_snr.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 9: Channel Impulse Response Stem Plot               鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_09():
    from sage.algorithms import algo_SAGE
    from sage.evaluation import match_paths

    N_rx = 1000; L = 3
    td = np.array([50.43, 60.66, 200.51])
    ta = np.array([1.0*np.exp(1j*0.2), 0.5*np.exp(-1j*1.5), 0.8*np.exp(1j*0.8)])
    y_clean = np.zeros(N_rx, dtype=complex)
    for d, a in zip(td, ta):
        y_clean += a * fractional_delay_shift(S_TX, d, N_rx)
    noise = 0.05*(np.random.RandomState(42).randn(N_rx) + 1j*np.random.RandomState(42).randn(N_rx))
    y_rx = y_clean + noise

    d_int, a_int = algo_SAGE(y_rx, S_TX, N_rx, L, max_iter=10, do_frac=False)
    d_frac, a_frac = algo_SAGE(y_rx, S_TX, N_rx, L, max_iter=10, do_frac=True)
    mdi, mai = match_paths(td, ta, d_int, a_int)
    mdf, maf = match_paths(td, ta, d_frac, a_frac)

    fig = plt.figure(figsize=(14, 8))
    ax_main = fig.add_axes([0.06, 0.12, 0.52, 0.82])
    ax_main.stem(td, np.abs(ta), linefmt='r-', markerfmt='ro', basefmt=' ',
                 label='True')
    ax_main.stem(mdi-0.15, np.abs(mai), linefmt='k-', markerfmt='ks', basefmt=' ',
                 label='Integer SAGE')
    ax_main.stem(mdf+0.15, np.abs(maf), linefmt='b-', markerfmt='b^', basefmt=' ',
                 label='Fractional SAGE')
    for t, a_ in zip(td, np.abs(ta)):
        ax_main.add_patch(Rectangle((t-1.5, 0), 3, a_*1.3, fill=False,
                                     ls='--', lw=1, edgecolor='gray', alpha=0.6))
    ax_main.set_xlabel('Delay (Samples)'); ax_main.set_ylabel('$|\\alpha|$')
    ax_main.legend(fontsize=8); ax_main.grid(alpha=0.3)
    ax_main.set_xlim([30, 220]); ax_main.set_ylim([0, 1.2])

    pos = [[0.63,0.58,0.34,0.35],[0.63,0.32,0.34,0.22],[0.63,0.06,0.34,0.22]]
    for i, (t, a_, p) in enumerate(zip(td, np.abs(ta), pos)):
        ax_sub = fig.add_axes(p)
        ax_sub.stem(td, np.abs(ta), linefmt='r-', markerfmt='ro', basefmt=' ')
        ax_sub.stem(mdi-0.15, np.abs(mai), linefmt='k-', markerfmt='ks', basefmt=' ')
        ax_sub.stem(mdf+0.15, np.abs(maf), linefmt='b-', markerfmt='b^', basefmt=' ')
        ax_sub.set_xlim([t-2, t+2]); ax_sub.set_ylim([0, a_*1.3])
        ax_sub.set_title(f'Path {i+1}: $\\tau\\approx{t:.1f}$', fontsize=9)
        ax_sub.grid(alpha=0.3); ax_sub.tick_params(labelsize=7)

    fig.suptitle('Channel Impulse Response Estimation (with Zoomed Subplots)',
                 fontsize=13)
    fig.savefig(os.path.join(RESULTS, 'fig_09_channel_stem.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_09_channel_stem.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 10: Residual Signal Comparison                      鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_10():
    from sage.algorithms import algo_SAGE

    N_rx = 1000; L = 3
    td = np.array([50.43, 60.66, 200.51])
    ta = np.array([1.0*np.exp(1j*0.2), 0.5*np.exp(-1j*1.5), 0.8*np.exp(1j*0.8)])
    y_clean = np.zeros(N_rx, dtype=complex)
    for d, a in zip(td, ta):
        y_clean += a * fractional_delay_shift(S_TX, d, N_rx)
    noise = 0.05*(np.random.RandomState(42).randn(N_rx)+1j*np.random.RandomState(42).randn(N_rx))
    y_rx = y_clean + noise

    d_int, a_int = algo_SAGE(y_rx, S_TX, N_rx, L, max_iter=10, do_frac=False)
    d_frac, a_frac = algo_SAGE(y_rx, S_TX, N_rx, L, max_iter=10, do_frac=True)

    def residual(dels, amps):
        y_est = np.zeros(N_rx, dtype=complex)
        for d, a in zip(dels, amps):
            y_est += a * fractional_delay_shift(S_TX, d, N_rx)
        return y_rx - y_est

    r_int = residual(d_int, a_int)
    r_frac = residual(d_frac, a_frac)
    e_int = np.sum(np.abs(r_int)**2)
    e_frac = np.sum(np.abs(r_frac)**2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.abs(r_int), 'k-', lw=1, label=f'Integer SAGE ($E={e_int:.1f}$)')
    ax.plot(np.abs(r_frac), 'b-', lw=1.5, label=f'Fractional SAGE ($E={e_frac:.1f}$)')
    ax.set_xlabel('Time (Sample)'); ax.set_ylabel('$|$Residual$|$')
    ax.set_title('Residual Error Comparison')
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_xlim([30, 250])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_10_residual.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_10_residual.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 11: Ablation Study (hard-coded)                     鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_11():
    configs = [
        ("Integer SAGE\n(baseline)",          3.154),
        ("+ Parabolic\ninterpolation",        0.344),
        ("+ 15 iterations\n(from 10)",        0.299),
        ("+ FFT correlation\n(speedup)",      0.338),
        ("+ Local grid\nrefinement",          0.024),
        ("All combined\n(default)",           0.022),
    ]
    labels = [c[0] for c in configs]
    values = np.array([c[1] for c in configs])
    colors = ['#777777', '#4472C4', '#5B9BD5', '#70AD47', '#ED7D31', '#C00000']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='white', lw=0.8, width=0.6)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(i, val*1.25, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for i in range(1, len(values)):
        impr = values[0]/values[i]
        ax.text(i, values[i]*0.5, f'{impr:.1f}$\\times$', ha='center', va='center',
                fontsize=8, color='white')

    ax.axhline(0.023, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.text(len(values)-0.5, 0.023*2.0, 'CRB = 0.023', fontsize=9, color='red', fontstyle='italic', ha='right')

    ax.annotate('Stage 1: parabolic\ninterpolation (9.2$\\times$)',
                xy=(1, values[1]), xytext=(2.6, values[0]*0.35),
                arrowprops=dict(arrowstyle='->', color='#4472C4', lw=1.5),
                fontsize=8.5, color='#4472C4',
                bbox=dict(boxstyle='round', fc='white', ec='#4472C4', alpha=0.8))
    ax.annotate('Stage 2: local grid\nrefinement (14.3$\\times$ over Stage 1)',
                xy=(4, values[4]), xytext=(3.2, values[4]*5),
                arrowprops=dict(arrowstyle='->', color='#ED7D31', lw=1.5),
                fontsize=8.5, color='#ED7D31',
                bbox=dict(boxstyle='round', fc='white', ec='#ED7D31', alpha=0.8))

    ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Delay RMSE (samples)', fontsize=12)
    ax.set_title('Ablation Study: Contribution of Each Component (0 dB SNR, 100 MC)')
    ax.set_yscale('log'); ax.set_ylim([0.004, 12]); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_11_ablation.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_11_ablation.png")


# 鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽
# 鈺? FIGURE 12: Sea Trial Grid Mismatch                         鈺?# 鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆

def fig_12():
    m_vals = np.array([2, 4, 6, 8])
    mse_i = np.array([2.37e-3, 5.22e-3, 2.15e-2, 4.09e-3])
    mse_f = np.array([2.00e-3, 5.03e-3, 1.88e-2, 3.65e-3])
    imp = (mse_i - mse_f) / mse_i * 100

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.semilogy(m_vals, mse_i, 'b-s', lw=2, ms=8, label='Integer SAGE')
    ax1.semilogy(m_vals, mse_f, 'r-o', lw=2, ms=8, label='Fractional SAGE')
    ax1.set_xlabel('Decimation Factor $M$')
    ax1.set_ylabel('Equalization MSE')
    ax1.legend(loc='upper right'); ax1.grid(alpha=0.3); ax1.set_xticks(m_vals)

    ax2 = ax1.twinx()
    colors_bar = ['darkred' if v < 0 else 'darkgreen' for v in imp]
    ax2.bar(m_vals, imp, width=0.3, color=colors_bar, alpha=0.5, edgecolor='#555')
    ax2.set_ylabel('Improvement (%)')
    for r, v, c in zip(m_vals, imp, colors_bar):
        off = 0.5 if v >= 0 else -2
        ax2.text(r, v+off, f'{v:+.1f}%', ha='center', fontsize=8.5, color=c)
    ax2.axhline(0, color='gray', ls='-', lw=0.8)
    ax2.set_ylim(bottom=min(0, min(imp)) - abs(max(imp))*0.1, top=max(imp)*1.15)

    ax1.set_title('Grid Mismatch Sensitivity Sea Trial ($K=8$)')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_12_seatrial.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] fig_12_seatrial.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# MAIN
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
if __name__ == '__main__':
    print("=" * 60)
    print(" Generating 12 figures per README_FIGURES.md")
    print("=" * 60)
    fig_01(); fig_02(); fig_03(); fig_04()
    fig_05(); fig_06(); fig_07(); fig_08()
    fig_09(); fig_10(); fig_11(); fig_12()
    print("\nDone 鈥?12 figures in results/")
