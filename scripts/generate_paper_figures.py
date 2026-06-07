#!/usr/bin/env python
"""
Generate ALL missing/inconsistent figures for paper_twc.tex.
Outputs go to results/ alongside existing figures.

Figures generated:
  1. fig_flowchart.png      鈥?Algorithm pipeline (Fig.1, ref: fig:flowchart)
  2. fig_parabolic.png      鈥?Parabolic interpolation geometry (ref: fig:parabolic)
  3. fig_localrefine.png    鈥?Local refinement illustration (ref: fig:localrefine)
  4. fig_bias_efficiency.png 鈥?Combined bias + CRB efficiency (Fig.7, ref: fig:bias)
  5. seatrial_mse.png       鈥?Overwrite: sea trial decimation sweep (Fig.12, ref: fig:seatrial)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle as MplRectangle

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'mathtext.fontset': 'stix',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# FIGURE 1: Algorithm Pipeline Flowchart 鈥?redesigned layout
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def generate_flowchart():
    """Clean 3-column, 4-row block diagram with clear vertical and horizontal flow."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Two-Stage Fractional SAGE 鈥?M-Step Pipeline',
                 fontsize=14, pad=15)

    # Colors
    c_in  = '#E3F2FD'
    c_op  = '#C8E6C9'
    c_s1  = '#FFE0B2'
    c_s2  = '#FFCCBC'
    c_out = '#D1C4E9'
    c_loop = '#ECEFF1'

    def box(x, y, w, h, text, color, fs=9):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='#444', linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='normal')

    def arrow(x1, y1, x2, y2, color='#333', lw=1.5, style='simple'):
        kw = dict(arrowstyle='->', color=color, lw=lw)
        if style == 'arc':
            kw['connectionstyle'] = 'arc3,rad=0.25'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=kw)

    def label(x, y, text, fs=7, color='#555', ha='center', bold=False):
        ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va='center',
                 fontweight='normal', style='italic')

    # 鈹€鈹€ Layout grid (x, y in inches-like coords) 鈹€鈹€
    # Row 1 (y=9.0):         Input signals
    # Row 2 (y=7.0):  E-step  +  iteration loop
    # Row 3 (y=4.5):  FFT-based integer correlation
    # Row 4 (y=2.5):  Stage 1 (parabolic)  鈫? Stage 2 (local refine)
    # Row 5 (y=0.8):  Amplitude + residual update 鈫?feedback

    # Row 1: Input
    box(1.0, 10.2, 3.2, 1.0, 'Received Signal  $\\mathbf{y}_{\\rm rx}$\nTransmit Waveform  $\\mathbf{s}_{\\rm tx}$', c_in, fs=10)

    # Row 2 left: E-step
    box(1.0, 7.8, 3.2, 1.2, 'E-step (Hidden Data)\n$\\hat{\\mathbf{x}}_l = \\mathbf{r} + \\hat{\\alpha}_l\\,\\mathbf{s}(\\hat{\\tau}_l)$', c_op, fs=9)

    # Row 2 right: SAGE Loop indicator
    box(8.0, 7.5, 3.0, 1.8, 'SAGE Iteration\n$i = 1 \\ldots I_{\\max}$\nPath sweep $l = 1 \\ldots L$\n\nSort + Collision\nDetection (iter 1)', c_loop, fs=9)

    # Row 3: FFT correlation
    box(1.0, 5.2, 6.0, 1.0,
        'FFT-based Correlation:  $\\mathbf{C} = |\\mathcal{F}^{-1}\\{X_f\\,S_f^*\\}|^2 / E_s^2$\nInteger peak search:  $k_{\\rm opt} = \\arg\\max_k \\mathbf{C}[k]$',
        c_op, fs=9)

    # Row 4 left: Stage 1
    box(1.0, 2.8, 3.8, 1.4,
        'Stage 1: Parabolic Interpolation  [$\\mathcal{O}(1)$]\n'
        '$\\hat{\\delta} = \\frac{1}{2}\\cdot\\frac{y_{-1}-y_{+1}}{y_{-1}-2y_0+y_{+1}}$\n'
        '$\\hat{\\tau}^{(1)} = k_{\\rm opt} + \\hat{\\delta}$',
        c_s1, fs=8)

    # Row 4 right: Stage 2
    box(5.8, 2.8, 4.2, 1.4,
        'Stage 2: Local Grid Refinement\n'
        '$\\hat{\\tau}_l = \\arg\\max_{\\tau\\in[\\hat{\\tau}^{(1)}\\pm0.5]}'
        '|\\langle\\mathbf{s}(\\tau),\\hat{\\mathbf{x}}_l\\rangle|^2$\n'
        '($\\Delta=0.005$, 40-200 evals; coarse-to-fine 4.5x speedup)',
        c_s2, fs=8)

    # Row 5: Amplitude + residual
    box(1.0, 0.8, 4.0, 1.2,
        'Amplitude:  $\\hat{\\alpha}_l = \\langle\\mathbf{s}(\\hat{\\tau}_l),\\hat{\\mathbf{x}}_l\\rangle/E_s$\n'
        'Residual:  $\\mathbf{r} \\leftarrow \\hat{\\mathbf{x}}_l - \\hat{\\alpha}_l\\mathbf{s}(\\hat{\\tau}_l)$',
        c_out, fs=9)

    # 鈹€鈹€ Arrows 鈹€鈹€
    # Input 鈫?E-step
    arrow(2.6, 10.2, 2.6, 9.0)
    # E-step 鈫?FFT
    arrow(2.6, 7.8, 2.6, 6.2)
    # FFT 鈫?Stage 1
    arrow(2.6, 5.2, 2.6, 4.2)
    # Stage 1 鈫?Stage 2
    arrow(4.8, 3.5, 5.8, 3.5, color='#E65100', lw=1.8)
    label(5.3, 3.85, r'$\hat{\tau}^{(1)}$', fs=8, color='#E65100', bold=True)
    # Stage 2 鈫?Amplitude
    arrow(5.8, 2.8, 3.0, 2.0, color='#D84315', lw=1.5)
    # Amplitude 鈫?loop box
    arrow(5.0, 1.4, 8.0, 8.4, color='#FF6F00', lw=1.5, style='arc')
    label(6.5, 5.0, 'next path / iteration', fs=7, color='#FF6F00')
    # Loop box 鈫?E-step
    arrow(8.0, 7.5, 4.2, 8.4, color='#FF6F00', lw=1.5, style='arc')

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_flowchart.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_flowchart.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# FIGURE 2: Parabolic Interpolation Geometry 鈥?USE ACTUAL AUTOCORRELATION
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def generate_parabolic():
    """
    3-panel: 未 = -0.3, 0, +0.3.
    Uses ACTUAL |R(蟿)|虏 from the QPSK signal (via FFT-based fractional_delay_shift),
    ensuring the three correlation points always form a concave-down peak.
    """
    from sage.signal_utils import generate_tx_signal, fractional_delay_shift

    s_tx = generate_tx_signal(511, seed=42)
    Es = np.vdot(s_tx, s_tx).real  # = 1.0
    N_rx_sim = 600

    deltas = [-0.3, 0.0, 0.3]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_idx, (ax, delta) in enumerate(zip(axes, deltas)):
        # Place true delay at center of buffer
        tau_center = 300.0
        tau_true = tau_center + delta
        k_opt = int(np.floor(tau_true + 0.5))

        # Generate "clean" signal: a single-path copy
        s_shifted = fractional_delay_shift(s_tx, tau_true, N_rx_sim)

        # Compute correlation at three integer offsets via inner product
        corr_vals = np.zeros(3)
        for mi, m in enumerate([-1, 0, 1]):
            tau_m = k_opt + m
            if 0 <= tau_m and tau_m + len(s_tx) <= N_rx_sim:
                window = s_shifted[tau_m:tau_m + len(s_tx)]
                corr_vals[mi] = np.abs(np.vdot(s_tx, window))**2
            else:
                corr_vals[mi] = 0.0

        y_m1, y_0, y_p1 = corr_vals
        # Normalize to peak = 1 for visual clarity
        y_max = max(y_0, 1e-12)
        y_m1 /= y_max; y_0 /= y_max; y_p1 /= y_max

        # Fit parabola a*x^2 + b*x + c through (-1,y_m1),(0,y_0),(1,y_p1)
        # Solve: a*(-1)^2 + b*(-1) + c = y_m1  鈫? a - b + c = y_m1
        #        a*0 + b*0 + c = y_0          鈫? c = y_0
        #        a*1 + b*1 + c = y_p1         鈫? a + b + c = y_p1
        A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
        coeffs = np.linalg.solve(A, np.array([y_m1, y_0, y_p1]))
        a, b_fit, c_fit = coeffs

        # Vertex: x_v = -b/(2a)  (should be concave down: a < 0)
        if a < -1e-10:
            x_vertex = -b_fit / (2 * a)
        else:
            x_vertex = 0.0
        delta_hat = np.clip(x_vertex, -0.5, 0.5)

        # Fine curve
        x_fine = np.linspace(-1.5, 1.5, 200)
        y_fine = a * x_fine**2 + b_fit * x_fine + c_fit

        # Plot
        ax.plot(x_fine, y_fine, 'b-', linewidth=2, label='Fitted parabola')
        ax.plot([-1, 0, 1], [y_m1, y_0, y_p1], 'ro', markersize=10,
                markeredgecolor='darkred', markerfacecolor='#FF6B6B',
                zorder=5, label='$|C(k_{\\rm opt}+m)|^2$')
        ax.axvline(x=delta, color='green', linestyle='--', linewidth=2,
                   label=f'True $\\delta={delta:+.1f}$')
        ax.axvline(x=delta_hat, color='blue', linestyle=':', linewidth=2,
                   label=f'Est. $\\hat{{\\delta}}={delta_hat:+.3f}$')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.25)
        # Verify parabola opens downward
        direction = '鈫? if a < 0 else '鈫?
        ax.text(0.95, 0.05, f'$a={a:.3f}$ {direction}', transform=ax.transAxes,
                fontsize=8, ha='right', color='#555')

        ax.set_xlabel('Offset from $k_{\\rm opt}$ (samples)')
        ax.set_ylabel('$|C|^2$ (normalized)')
        ax.set_title(f'$\\delta = {delta:+.1f}$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim([-1.5, 1.5])

    fig.suptitle('Parabolic Interpolation Geometry 鈥?Actual Signal Autocorrelation (Stage 1)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_parabolic.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_parabolic.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# FIGURE 3: Local Refinement Illustration
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def generate_localrefine():
    """Two-panel: coarse scan + fine scan illustrating Stage 2."""
    np.random.seed(123)
    tau_true = 60.43
    sigma = 0.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 鈹€鈹€ Left: Coarse 鈹€鈹€
    coarse_step = 0.05
    tau_coarse = np.arange(tau_true - 0.55, tau_true + 0.55 + coarse_step, coarse_step)
    f_coarse = np.exp(-(tau_coarse - tau_true)**2 / (2 * sigma**2))
    f_coarse += 0.01 * np.random.randn(len(tau_coarse))

    ax1.plot(tau_coarse, f_coarse, 'bo-', markersize=8, linewidth=1.5,
             label='Coarse eval ($\\Delta_1=0.05$)')
    ax1.axvline(x=tau_true, color='green', linestyle='--', linewidth=2,
                label=f'True $\\tau={tau_true:.2f}$')
    idx_c = np.argmax(f_coarse)
    ax1.axvline(x=tau_coarse[idx_c], color='blue', linestyle=':', linewidth=2,
                label=f'Coarse peak $\\tau_c={tau_coarse[idx_c]:.2f}$')
    ax1.plot(tau_coarse[idx_c], f_coarse[idx_c], 'r*', markersize=15, zorder=5)
    sh_lo = tau_coarse[idx_c] - 0.06
    sh_hi = tau_coarse[idx_c] + 0.06
    ax1.axvspan(sh_lo, sh_hi, alpha=0.15, color='orange', label='Fine zone')
    ax1.set_xlabel('Delay (samples)')
    ax1.set_ylabel('Objective $f(\\tau)$')
    ax1.set_title('Pass 1: Coarse Grid ($\\Delta_1 = 0.05$, ~20 evals)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 鈹€鈹€ Right: Fine 鈹€鈹€
    fine_step = 0.005
    tau_fine = np.arange(sh_lo, sh_hi + fine_step, fine_step)
    f_fine = np.exp(-(tau_fine - tau_true)**2 / (2 * sigma**2))
    f_fine += 0.005 * np.random.randn(len(tau_fine))

    ax2.plot(tau_fine, f_fine, 'r-o', markersize=5, linewidth=1.5, zorder=3,
             label='Fine eval ($\\Delta_2=0.005$)')
    ax2.axvline(x=tau_true, color='green', linestyle='--', linewidth=2,
                label=f'True $\\tau={tau_true:.2f}$')
    idx_f = np.argmax(f_fine)
    tau_est = tau_fine[idx_f]
    ax2.axvline(x=tau_est, color='darkred', linestyle=':', linewidth=2,
                label=f'Final $\\hat{{\\tau}}={tau_est:.3f}$')
    ax2.plot(tau_est, f_fine[idx_f], 'r*', markersize=15, zorder=5)
    ax2.annotate(f'Error = {abs(tau_est - tau_true):.4f} samples',
                 xy=(tau_est, f_fine[idx_f]),
                 xytext=(tau_est + 0.012, f_fine[idx_f] * 0.85),
                 fontsize=9, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred'))
    ax2.set_xlabel('Delay (samples)')
    ax2.set_ylabel('Objective $f(\\tau)$')
    ax2.set_title('Pass 2: Fine Grid ($\\Delta_2 = 0.005$, ~20 evals)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Stage 2: Coarse-to-Fine Local Grid Refinement',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_localrefine.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_localrefine.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# FIGURE 4: Combined Bias + CRB Efficiency 鈥?FIXED MC SIMULATION
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def generate_bias_efficiency():
    """
    Paper Fig.7: (a) Empirical bias vs 未 (SNR=40dB), (b) CRB efficiency ratio.

    Key fixes vs previous version:
      - Use actual signal autocorrelation for curvature (not hardcoded 1.557)
      - Increased N_mc for bias: 5000 (stable statistics)
      - Proper noise calibration: signal power = 1, SNR defines noise variance
      - Plot both Stage1 (parabolic only) and Stage1+2 (with refinement) where applicable
    """
    from sage.signal_utils import generate_tx_signal, fractional_delay_shift

    s_tx = generate_tx_signal(511, seed=42)
    N_seq = len(s_tx)
    Es = np.vdot(s_tx, s_tx).real  # = 1.0

    # Compute actual curvature c = -R''(0)/2 = ||s'||^2 / 2
    S_freq = np.fft.fft(s_tx)
    k_vec = np.concatenate([np.arange(0, int(np.ceil(N_seq/2))),
                             np.arange(-int(np.floor(N_seq/2)), 0)])
    omega = 2 * np.pi * k_vec / N_seq
    s_prime = np.fft.ifft((1j * omega) * S_freq)
    energy_sp = np.sum(np.abs(s_prime)**2)
    c_actual = energy_sp / (2 * Es)  # curvature parameter
    print(f"    Actual curvature c = {c_actual:.4f}  (||s'||^2 = {energy_sp:.4f})")

    rng = np.random.RandomState(42)
    N_rx_sim = N_seq + 40  # enough buffer

    # 鈺愨晲鈺?Panel (a): Bias vs 未 (SNR = 40 dB) 鈺愨晲鈺?    delta_vals = np.linspace(-0.5, 0.5, 21)
    snr_db_bias = 40
    snr_lin = 10 ** (snr_db_bias / 10)
    noise_var = Es / snr_lin
    noise_std = np.sqrt(noise_var / 2)
    N_mc_bias = 5000

    bias_emp = np.zeros(len(delta_vals))
    for i, d in enumerate(delta_vals):
        tau_true = 20.0 + d  # safe offset
        k_opt = int(np.floor(tau_true + 0.5))
        s_clean = fractional_delay_shift(s_tx, tau_true, N_rx_sim)
        dhats = np.zeros(N_mc_bias)
        for mc in range(N_mc_bias):
            noise = noise_std * (rng.randn(N_rx_sim) + 1j * rng.randn(N_rx_sim))
            y_rx = s_clean + noise
            # Three correlation samples: |鉄╯, y[k+m : k+m+N]鉄﹟虏
            cv = np.zeros(3)
            for mi, m in enumerate([-1, 0, 1]):
                tau_m = k_opt + m
                if 0 <= tau_m and tau_m + N_seq <= N_rx_sim:
                    cv[mi] = np.abs(np.vdot(s_tx, y_rx[tau_m:tau_m+N_seq]))**2
                else:
                    cv[mi] = 0.0
            y_m1, y_0, y_p1 = cv
            denom = y_m1 - 2 * y_0 + y_p1
            if abs(denom) > 1e-12:
                dh = 0.5 * (y_m1 - y_p1) / denom
                dh = np.clip(dh, -0.5, 0.5)
            else:
                dh = 0.0
            dhats[mc] = dh
        bias_emp[i] = np.mean(dhats) - d

    # 鈺愨晲鈺?Panel (b): CRB efficiency vs SNR 鈺愨晲鈺?    snr_db_vec = np.arange(-5, 36, 5)
    snr_linear_vec = 10 ** (snr_db_vec / 10)
    delta_fixed = 0.25
    N_mc_var = 3000

    var_emp = np.zeros(len(snr_db_vec))
    for i, sdb in enumerate(snr_db_vec):
        slin = 10 ** (sdb / 10)
        nv = Es / slin
        ns = np.sqrt(nv / 2)
        tau_true = 20.0 + delta_fixed
        k_opt = int(np.floor(tau_true + 0.5))
        s_clean = fractional_delay_shift(s_tx, tau_true, N_rx_sim)
        dhs = np.zeros(N_mc_var)
        for mc in range(N_mc_var):
            noise = ns * (rng.randn(N_rx_sim) + 1j * rng.randn(N_rx_sim))
            y_rx = s_clean + noise
            cv = np.zeros(3)
            for mi, m in enumerate([-1, 0, 1]):
                tau_m = k_opt + m
                if 0 <= tau_m and tau_m + N_seq <= N_rx_sim:
                    cv[mi] = np.abs(np.vdot(s_tx, y_rx[tau_m:tau_m+N_seq]))**2
                else:
                    cv[mi] = 0.0
            y_m1, y_0, y_p1 = cv
            denom = y_m1 - 2 * y_0 + y_p1
            if abs(denom) > 1e-12:
                dh = 0.5 * (y_m1 - y_p1) / denom
                dh = np.clip(dh, -0.5, 0.5)
            else:
                dh = 0.0
            dhs[mc] = dh
        var_emp[i] = np.var(dhs)

    # CRB: CRB = 1 / (2 * SNR * ||s'||^2)
    crb_vals = 1.0 / (2 * snr_linear_vec * energy_sp)
    efficiency = var_emp / crb_vals

    # Theoretical variance from Theorem 1
    var_theory = (12 * delta_fixed**2 + 1 - 4 * delta_fixed) / (16 * c_actual**2 * snr_linear_vec)

    # 鈺愨晲鈺?Plot 鈺愨晲鈺?    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Bias
    ax1.plot(delta_vals, bias_emp * 1000, 'bo-', linewidth=1.5, markersize=6,
             label='Empirical bias ($\\times 10^{-3}$ samples)')
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.fill_between(delta_vals, -0.15, 0.15, alpha=0.08, color='gray',
                     label='$\\pm 1.5\\times10^{-4}$ band')
    ax1.set_xlabel('True fractional delay $\\delta$ (samples)')
    ax1.set_ylabel('Bias ($\\times 10^{-3}$ samples)')
    ax1.set_title(f'(a) Bias vs $\\delta$ (SNR = {snr_db_bias} dB, {N_mc_bias} MC trials)',
                  )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (b) CRB efficiency
    ax2.loglog(snr_db_vec, var_emp, 'bo-', linewidth=2, markersize=8,
               label='Empirical $\\mathrm{Var}(\\hat{\\delta})$ (Stage 1)')
    ax2.loglog(snr_db_vec, var_theory, 'g--', linewidth=2,
               label='Theorem 1: $\\frac{12\\delta^2+1-4\\delta}{16c^2\\cdot\\mathrm{SNR}}$')
    ax2.loglog(snr_db_vec, crb_vals, 'r-.', linewidth=2,
               label='CRB $= 1/(2\\cdot\\mathrm{SNR}\\cdot\\|s\'\\|^2)$')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Variance / MSE (samples$^2$)')
    ax2.set_title(
        f'(b) Stage 1 Variance vs CRB ($\\delta={delta_fixed}$)\n'
        f'$c={c_actual:.2f}$, Efficiency $\\approx$ {np.mean(efficiency[-3:]):.2f} at SNR $\\geq$ 20 dB',
        )
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim([1e-6, 1])

    fig.suptitle('Parabolic Estimator Performance Analysis (Stage 1 Only)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_bias_efficiency.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_bias_efficiency.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# FIGURE 5: Sea Trial Grid Mismatch 鈥?paper-consistent data
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def generate_seatrial():
    """Generate seatrial_mse.png with paper Table IX data."""
    import scipy.io as sio
    from scipy.signal import decimate
    from numpy.fft import fft, ifft
    from sage.interpolation import parabolic_delta

    mat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data475.mat')
    data = sio.loadmat(mat_path)
    y0 = data['sig_rev_m'].flatten().astype(np.float64)
    s0 = data['sig_bd_m'].flatten().astype(np.float64)

    down_ratios = np.array([1, 2, 4, 6, 8])
    mse_i = np.zeros(len(down_ratios))
    mse_f = np.zeros(len(down_ratios))
    imp = np.zeros(len(down_ratios))
    K, niter = 8, 5

    for ti, M in enumerate(down_ratios):
        if M <= 1:
            yr, st = y0.copy(), s0.copy()
        else:
            yr = decimate(y0, M, ftype='fir', zero_phase=True)
            st = decimate(s0, M, ftype='fir', zero_phase=True)
        N = len(yr)
        Es_val = np.sum(np.abs(st)**2)

        N2 = 2 * N
        Sf = fft(st, n=N2)
        Yf = fft(yr, n=N2)
        fv = np.concatenate([np.arange(0, N2//2), np.arange(-N2//2, 0)]) / N2

        def run_sage(do_frac):
            tau = np.zeros(K)
            alpha = np.zeros(K, dtype=complex)
            yr_f = Yf.copy()
            akf = np.zeros((N2, K), dtype=complex)
            for _ in range(niter):
                for k in range(K):
                    xkf = yr_f + akf[:, k] * Sf
                    Rk = ifft(xkf * np.conj(Sf)) / Es_val
                    idx = int(np.argmax(np.abs(Rk[:N])))
                    if do_frac and 0 < idx < N - 1:
                        ym = np.abs(Rk[idx-1])
                        y0_ = np.abs(Rk[idx])
                        yp = np.abs(Rk[idx+1])
                        delta = parabolic_delta(ym, y0_, yp)
                    else:
                        delta = 0.0
                    tc = (idx - 1) + delta
                    tau[k] = tc
                    ph = np.exp(1j * 2 * np.pi * fv * tc)
                    alpha[k] = np.sum((xkf * np.conj(Sf)) * ph) / (N2 * Es_val)
                    akf[:, k] = alpha[k] * np.exp(-1j * 2 * np.pi * fv * tau[k])
                    yr_f = xkf - akf[:, k] * Sf
            if N % 2 == 0:
                fv_n = np.concatenate([np.arange(0, N//2), np.arange(-N//2, 0)]) / N
            else:
                fv_n = np.concatenate([np.arange(0, (N-1)//2 + 1),
                                       np.arange(-(N-1)//2, 0)]) / N
            H = np.sum([alpha[k] * np.exp(-1j * 2 * np.pi * fv_n * tau[k])
                        for k in range(K)], axis=0)
            return tau, alpha, H

        tau_i, alpha_i, H_i = run_sage(False)
        tau_f, alpha_f, H_f = run_sage(True)

        nv = 0.05 * np.max(np.abs(H_f))**2 + 1e-8
        Sf_n = fft(st)
        Yf_n = fft(yr)
        seq_i = ifft(Yf_n * np.conj(H_i) / (np.abs(H_i)**2 + nv))
        seq_f = ifft(Yf_n * np.conj(H_f) / (np.abs(H_f)**2 + nv))

        v0 = 1000 if N > 2000 else N // 10
        v1 = N - v0
        mse_i[ti] = np.mean(np.abs(seq_i[v0:v1] - st[v0:v1])**2)
        mse_f[ti] = np.mean(np.abs(seq_f[v0:v1] - st[v0:v1])**2)
        imp[ti] = (mse_i[ti] - mse_f[ti]) / mse_i[ti] * 100
        print(f"  M={M:1d} N={N:7d} Int={mse_i[ti]:.2e} Frac={mse_f[ti]:.2e} Impr={imp[ti]:+.1f}%")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.semilogy(down_ratios, mse_i, 'b-s', lw=2, ms=8,
                 label='Standard SAGE (Integer Grid)')
    ax1.semilogy(down_ratios, mse_f, 'r-o', lw=2, ms=8,
                 label='Proposed SAGE (Fractional Grid)')
    ax1.set_xlabel('Decimation Factor $M$', fontsize=12)
    ax1.set_ylabel('Equalization MSE', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(down_ratios)

    ax2 = ax1.twinx()
    ax2.bar(down_ratios, imp, width=0.3, color='gray', alpha=0.35, edgecolor='#555')
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    for i, (r, v) in enumerate(zip(down_ratios, imp)):
        c = 'darkgreen' if v > 0 else 'darkred'
        offset = 3 if v >= 0 else -6
        ax2.text(r, v + offset, f'{v:+.1f}%', ha='center', fontsize=9,
                 color=c)

    ax1.set_title('Grid Mismatch Sensitivity 鈥?Sea Trial ($f_s$=16 kHz, 8-path SAGE)',
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'seatrial_mse.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?seatrial_mse.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# MAIN
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

if __name__ == '__main__':
    print("=" * 60)
    print(" Generating missing figures for paper_twc.tex")
    print("=" * 60)

    print("\n[1/5] Flowchart...")
    generate_flowchart()

    print("\n[2/5] Parabolic interpolation geometry...")
    generate_parabolic()

    print("\n[3/5] Local refinement illustration...")
    generate_localrefine()

    print("\n[4/5] Combined bias + CRB efficiency...")
    generate_bias_efficiency()

    print("\n[5/5] Sea trial grid mismatch...")
    generate_seatrial()

    print("\n" + "=" * 60)
    print(" All figures regenerated in results/")
    print("=" * 60)
