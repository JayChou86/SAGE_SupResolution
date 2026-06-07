import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ===========================================================================
# FIGURE 1: Grid Mismatch Illustration
# ===========================================================================
def fig1_grid_mismatch():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -- Panel (a): Mechanism --
    ax = axes[0]
    t = np.arange(-2, 8, 0.02)
    tau_true = 2.37
    peak = np.exp(-((t - tau_true) / 0.7) ** 2)

    ax.plot(t, peak, linewidth=1.5, label=r'Continuous $|C(\tau)|^2$')
    for k in range(-1, 7):
        idx = np.argmin(np.abs(t - k))
        ax.plot(k, peak[idx], 'o', markersize=6)

    ax.axvline(tau_true, linestyle='--', linewidth=1)
    ax.annotate(r'$\tau=k+\delta$', xy=(tau_true, 1.02),
                xytext=(tau_true, 1.15), ha='center',
                arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(tau_true, 0.45), xytext=(0, 0.45),
                arrowprops=dict(arrowstyle='<->'))
    ax.annotate(r'$\delta$', xy=(tau_true/2, 0.50), ha='center')

    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel(r'$|C(\tau)|^2$')
    ax.set_title('(a) Grid Mismatch Mechanism')
    ax.set_xlim([-2.5, 6.5])
    ax.set_ylim([0, 1.2])
    ax.legend()

    # -- Panel (b): Conceptual RMSE --
    ax = axes[1]
    snr = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30])
    rmse_int  = np.minimum(3.2, 8.0/(1+np.exp((snr+8)/3)) + 0.15)
    rmse_frac = 0.5*np.exp(-snr/8) + 0.02
    crb       = 0.3*np.exp(-snr/7) + 0.005

    ax.semilogy(snr, rmse_int,  's--', label='Integer SAGE')
    ax.semilogy(snr, rmse_frac, '^-',  label='Fractional SAGE')
    ax.semilogy(snr, crb,       '-',   label='CRB')
    ax.fill_between([15, 30], 2.5, 4, alpha=0.08)
    ax.annotate('Grid mismatch\nfloor', xy=(22, 3.2), ha='center')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Delay RMSE (samples)')
    ax.set_title('(b) RMSE vs SNR 鈥?Conceptual')
    ax.set_xlim([-10, 30])
    ax.set_ylim([5e-3, 20])
    ax.legend()
    ax.grid(True)

    fig.suptitle('Figure 1 | Grid Mismatch in SAGE Delay Estimation')
    fig.tight_layout()
    return fig


# ===========================================================================
# FIGURE 2: Algorithm Flowchart
# ===========================================================================
def fig2_flowchart():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def box(x, y, w, h, text, bold=False):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor='black',
                              linewidth=1, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                 fontweight='normal', zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.0), zorder=2)

    def hilite(x, y, w, h):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle='round,pad=0.15',
                              facecolor='lightblue', edgecolor='blue',
                              linewidth=1.5, alpha=0.12, zorder=1)
        ax.add_patch(rect)

    # -- Left column: main loop --
    box(2.0, 11.0, 2.4, 0.7, 'y_rx (Received signal)', bold=True)
    box(2.0, 10.0, 2.4, 0.7, 's_tx (Known waveform)', bold=True)
    arrow(2.0, 10.5, 2.0, 9.1)
    box(2.0,  8.8, 2.6, 0.8, 'Initialize:\nr <- y_rx')
    arrow(2.0, 8.4, 2.0, 7.6)
    hilite(2.0, 6.9, 3.0, 1.5)
    box(2.0, 7.45, 2.8, 0.5, 'for iter=1:I_max, l=1:L', bold=True)
    box(2.0, 6.7,  2.8, 0.6, 'E-step: x_l = r + a*s(t)')
    arrow(3.4, 6.7, 6.5, 6.7)

    # -- Right column: M-step (core contribution) --
    hilite(8.0, 8.1, 3.2, 2.8)
    box(8.0, 9.3, 3.0, 0.7, 'FFT Cross-Correlation\nC(t) = F^{-1}{X*S*}')
    arrow(8.0, 8.9, 8.0, 8.3)
    box(8.0, 7.8, 3.0, 0.7,
        'Stage 1: Parabolic Interpolation\n'
        'd = 0.5*(y-1 - y+1)/(y-1 - 2*y0 + y+1)', bold=True)
    arrow(8.0, 7.4, 8.0, 6.9)
    box(8.0, 6.4, 3.0, 0.7,
        'Stage 2: Local Grid Refinement\n'
        'step=0.005, window=[-0.5, +0.5]', bold=True)

    # -- Back to left --
    arrow(6.4, 6.4, 2.0, 6.4)
    box(2.0, 6.1, 2.8, 0.6, 'M-step: a = <s(t), x_l> / ||s||^2')
    arrow(2.0, 5.8, 2.0, 5.1)
    box(2.0, 4.8, 2.8, 0.6, 'Residual: r = x_l - a*s(t)')

    # -- Loop back --
    ax.annotate('', xy=(0.6, 7.5), xytext=(0.6, 4.8),
                arrowprops=dict(arrowstyle='->', lw=0.8,
                                connectionstyle='arc3,rad=-0.5'), zorder=1)
    ax.text(0.25, 6.15, 'Loop', rotation=90, va='center')

    # -- Post-loop --
    arrow(2.0, 4.5, 2.0, 3.8)
    box(2.0, 3.5, 2.8, 0.6, 'Sort paths by |a|,\ncollision detection')
    arrow(2.0, 3.2, 2.0, 2.5)
    box(2.0, 2.2, 2.6, 0.7, '{t_l, a_l}\nEstimated parameters', bold=True)

    # -- Core contribution label --
    ax.text(8.0, 10.2, 'CORE CONTRIBUTION', ha='center')
    ax.annotate('', xy=(8.0, 9.3), xytext=(8.0, 10.0),
                arrowprops=dict(arrowstyle='->'))

    ax.text(6.0, 0.5, 'Figure 2 | Algorithm Flowchart: Two-Stage Fractional SAGE',
            ha='center')
    return fig


# ===========================================================================
# FIGURE 3: Parabolic Interpolation
# ===========================================================================
def fig3_parabolic_interpolation():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    deltas  = [0.0, 0.25, 0.43]
    titles  = ['(a) delta=0.00 (on-grid)',
               '(b) delta=0.25',
               '(c) delta=0.43 (off-grid)']

    for ax, delta, title in zip(axes, deltas, titles):
        x = np.linspace(-1.5, 1.5, 200)
        c = 1.5

        # True quadratic autocorrelation: y = 1 - c*(x - delta)^2
        y_true = np.maximum(1 - c*(x - delta)**2, 0)

        # Three samples at x = -1, 0, 1
        xs = np.array([-1, 0, 1])
        ys = 1 - c*(xs - delta)**2

        # Parabolic fit through three points
        denom = ys[0] - 2*ys[1] + ys[2]
        delta_hat = 0.5*(ys[0] - ys[2]) / denom
        a = denom / 2
        y_fit = a*(x - delta_hat)**2 + (ys[1] - a*delta_hat**2)

        ax.plot(x, y_true, linewidth=1.5, alpha=0.5,
                label=r'True $|R(\tau)|^2$')
        ax.plot(x, y_fit, '--', linewidth=1.2, label='Fitted parabola')
        ax.plot(xs, ys, 'o', markersize=8, zorder=5)

        for xi, yi, lbl in zip(xs, ys, ['y_-1', 'y_0', 'y_+1']):
            ax.annotate(lbl, (xi, yi), textcoords='offset points',
                        xytext=(0, -14), ha='center')

        ax.axvline(delta,     linestyle=':',  linewidth=0.8)
        ax.axvline(delta_hat, linestyle='--', linewidth=0.8)
        ax.annotate('delta',      xy=(delta,     0.05), ha='center')
        ax.annotate(f'delta_hat={delta_hat:.3f}',
                    xy=(delta_hat, 0.01), xytext=(0, -16),
                    textcoords='offset points', ha='center')

        ax.set_title(title)
        ax.set_xlabel('Delay offset (samples)')
        if ax == axes[0]:
            ax.set_ylabel(r'$|R(\tau)|^2$ or $|C(\tau)|^2$')
            ax.legend()
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-0.05, 1.1])

    fig.suptitle('Figure 3 | Parabolic Interpolation: Three-Point Quadratic Fit')
    fig.tight_layout()
    return fig


# ===========================================================================
# FIGURE 4: Stage 2 Local Refinement
# ===========================================================================
def fig4_local_refinement():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    tau_true = 50.43
    tau = np.linspace(49.5, 51.5, 2000)
    rng = np.random.RandomState(42)
    corr = np.exp(-((tau - tau_true)/0.3)**2) + 0.02*rng.randn(2000)

    # -- Panel (a): Coarse search --
    ax = axes[0]
    ax.plot(tau, corr, linewidth=0.8, alpha=0.6, label=r'$|C(\tau)|^2$')

    tau_coarse = np.arange(49.5, 51.51, 0.05)
    corr_coarse = np.exp(-((tau_coarse - tau_true)/0.3)**2)
    ax.plot(tau_coarse, corr_coarse, 'o', markersize=5, zorder=5,
            label='Coarse grid (step=0.05)')

    tau_c = tau_coarse[np.argmax(corr_coarse)]
    ax.axvline(tau_c, linestyle='--', linewidth=1)
    ax.annotate(f'tau_c={tau_c:.2f}', xy=(tau_c, np.max(corr_coarse)),
                xytext=(10, 5), textcoords='offset points', ha='center')

    ax.set_title('(a) Pass 1: Coarse Search')
    ax.set_xlabel('Delay (samples)')
    ax.set_ylabel(r'$|C(\tau)|^2$')
    ax.legend()
    ax.set_xlim([49.6, 51.4])

    # -- Panel (b): Fine search --
    ax = axes[1]
    mask = (tau >= tau_c - 0.06) & (tau <= tau_c + 0.06)
    ax.plot(tau[mask], corr[mask], linewidth=0.8, alpha=0.6,
            label=r'$|C(\tau)|^2$')

    tau_fine = np.arange(tau_c - 0.05, tau_c + 0.051, 0.005)
    corr_fine = np.exp(-((tau_fine - tau_true)/0.3)**2)
    ax.plot(tau_fine, corr_fine, 's', markersize=5, zorder=5,
            label='Fine grid (step=0.005)')

    tau_best = tau_fine[np.argmax(corr_fine)]
    ax.axvline(tau_true, linestyle=':',  linewidth=0.8, label='True')
    ax.axvline(tau_best, linestyle='--', linewidth=1.0, label='Estimated')
    ax.annotate(f'tau*={tau_best:.3f}', xy=(tau_best, np.max(corr_fine)),
                xytext=(10, 5), textcoords='offset points', ha='center')
    ax.annotate('True tau', xy=(tau_true, 0.95), ha='center')

    ax.set_title('(b) Pass 2: Fine Search')
    ax.set_xlabel('Delay (samples)')
    ax.legend()
    ax.set_xlim([tau_c - 0.06, tau_c + 0.06])

    fig.suptitle('Figure 4 | Stage 2: Coarse-to-Fine Local Grid Refinement')
    fig.tight_layout()
    return fig


# ===========================================================================
if __name__ == '__main__':
    fig1 = fig1_grid_mismatch()
    fig2 = fig2_flowchart()
    fig3 = fig3_parabolic_interpolation()
    fig4 = fig4_local_refinement()
    plt.show()
