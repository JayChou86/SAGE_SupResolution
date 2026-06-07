#!/usr/bin/env python
"""Fig 03 final: proper normalization, intersection at 3 points, visible 未 vs 未虃 gap."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix',
                     'figure.facecolor':'white','axes.facecolor':'white'})
R = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(R, exist_ok=True)

from sage.signal_utils import generate_tx_signal
S_TX = generate_tx_signal(511, seed=42)
N_SEQ = len(S_TX)

def R2(tau):
    """|R(tau)|^2 via FFT 鈥?bandlimited autocorrelation. Max = 1 at tau=0."""
    N = len(S_TX)
    Sf = np.fft.fft(S_TX)
    k = np.concatenate([np.arange(0, int(np.ceil(N/2))),
                         np.arange(-int(np.floor(N/2)), 0)])
    phase = np.exp(-1j * 2 * np.pi * k * tau / N)
    return np.abs(np.sum(Sf * np.conj(Sf) * phase) / N)**2

deltas = [0.0, 0.25, 0.43]
titles = ['(a) $\\delta$=0.00 (on-grid)', '(b) $\\delta$=0.25', '(c) $\\delta$=0.43 (off-grid)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for ax, d, title in zip(axes, deltas, titles):
    # 鈹€鈹€ True autocorrelation |R(蟿)|虏, peak=1 at 蟿=0 鈹€鈹€
    # In our display coords: x is offset from k_opt, true delay at x=d
    # True curve: |R(x - d)|虏 鈥?peak at x=d, value=1
    xd = np.linspace(-1.5, 1.5, 400)
    yt = np.array([R2(x - d) for x in xd])  # peak=1 at x=d

    # 鈹€鈹€ Three sample points at integer grid offsets 鈹€鈹€
    xp = np.array([-1.0, 0.0, 1.0])
    yp = np.array([R2(m - d) for m in xp])  # SAME function, SAME normalization

    # 鈹€鈹€ Fit parabola through three points 鈹€鈹€
    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
    coeffs = np.linalg.solve(A, yp)
    a, b, c = coeffs
    xv = -b / (2*a) if abs(a) > 1e-10 else 0.0
    dh = np.clip(xv, -0.5, 0.5)  # estimated 未虃

    yf = a * xd**2 + b * xd + c  # fitted parabola

    # 鈹€鈹€ Plot 鈹€鈹€
    # Shaded area under true curve
    ax.fill_between(xd, 0, yt, alpha=0.07, color='#455A64')

    # True |R(蟿)|虏 鈥?thin grey
    ax.plot(xd, yt, color='#37474F', lw=1.5, alpha=0.55, label='True $|R(\\tau)|^2$')

    # Three sample points 鈥?red, at INTERSECTION of both curves
    ax.plot(xp, yp, 'o', color='#D32F2F', ms=12, zorder=6, markeredgecolor='white', markeredgewidth=1.5)

    # Fitted parabola 鈥?blue, passes THROUGH the three points
    ax.plot(xd, yf, color='#1976D2', lw=2.5, zorder=4, label='Fitted parabola')

    # Green dashed: true 未 (peak of true curve, at x=d)
    ax.axvline(d, color='#2E7D32', ls='--', lw=2.2, zorder=3,
               label=f'True $\\delta={d:.2f}$')

    # Blue dotted: estimated 未虃 (peak of parabola)
    ax.axvline(dh, color='#1565C0', ls=':', lw=2.2, zorder=3,
               label=f'Est. $\\hat{{\\delta}}={dh:.3f}$')

    # 鈹€鈹€ Labels on points 鈹€鈹€
    for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$', '$y_0$', '$y_{+1}$']):
        va = 'bottom' if yi > 0.3 else 'top'
        off = 14 if yi > 0.3 else -14
        ax.annotate(lbl, (xi, yi), textcoords='offset points', xytext=(0, off),
                   ha='center', va=va, fontsize=10, color='#B71C1C')

    # 鈹€鈹€ Annotate 未虃 value near the blue line top 鈹€鈹€
    dh_label_y = max(0.15, a*dh**2 + b*dh + c + 0.08)
    ax.annotate(f'$\\hat{{\\delta}}={dh:.3f}$',
                xy=(dh, a*dh**2 + b*dh + c),
                xytext=(dh + 0.15, dh_label_y),
                fontsize=9, color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))

    ax.set_xlabel('Offset from $k_{\\rm opt}$ (samples)')
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.2)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.05, 1.1])
    if ax == axes[0]:
        ax.set_ylabel('$|R(\\tau)|^2$')
        ax.legend(fontsize=8.5, loc='upper right')

fig.suptitle('Parabolic Interpolation 鈥?Three-Point Quadratic Fit (Stage 1)',
             fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(R, 'fig_03_v10_final.png'), dpi=1200, bbox_inches='tight')
plt.close()
print("  鉁?fig_03_v10_final.png")
print("\nKey properties:")
for d in deltas:
    y0 = R2(d)
    ym = R2(-1 - d)  # offset from k_opt: -1, true delay at d, so tau = -1-d
    yp = R2(1 - d)
    A = np.array([[1,-1,1],[0,0,1],[1,1,1]])
    cf = np.linalg.solve(A, [ym, y0, yp])
    xv = -cf[1]/(2*cf[0]) if abs(cf[0])>1e-10 else 0
    dh = np.clip(xv, -0.5, 0.5)
    print(f"  d={d:.2f}: y=[{ym:.4f}, {y0:.4f}, {yp:.4f}]  dh={dh:.4f}  err={abs(dh-d):.4f}")
