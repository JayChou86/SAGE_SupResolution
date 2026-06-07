#!/usr/bin/env python
"""Fig 03 final: quartic autocorrelation model 鈥?visible 未 vs 未虃 gap."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix',
                     'figure.facecolor':'white','axes.facecolor':'white'})
R = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(R, exist_ok=True)

# Quartic model: |R(蟿)|^2 = 1 - c*蟿^2 + q*蟿^4
# c=1.5 (matches README), q=0.24 鈫?small visible bias (~0.03 samples)
c, q = 1.5, 0.24

def R2_quartic(tau):
    val = 1 - c*tau**2 + q*tau**4
    return np.maximum(val, 0.0)

deltas = [0.0, 0.25, 0.43]
titles = ['(a) $\\delta$=0.00 (on-grid)', '(b) $\\delta$=0.25', '(c) $\\delta$=0.43 (off-grid)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for ax, d, title in zip(axes, deltas, titles):
    xd = np.linspace(-1.5, 1.5, 400)
    # True curve: |R(x-d)|^2 鈥?peak at x=d
    yt = np.array([R2_quartic(x - d) for x in xd])

    # Three sample points at integer grid
    xp = np.array([-1.0, 0.0, 1.0])
    yp = np.array([R2_quartic(m - d) for m in xp])

    # Fit parabola through 3 points
    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
    cf = np.linalg.solve(A, yp)
    a, b_fit, c_fit = cf
    xv = -b_fit / (2*a) if abs(a) > 1e-10 else 0.0
    dh = np.clip(xv, -0.5, 0.5)
    yf = a*xd**2 + b_fit*xd + c_fit
    yf_peak = a*dh**2 + b_fit*dh + c_fit

    # 鈹€鈹€ Draw 鈹€鈹€
    ax.fill_between(xd, 0, yt, alpha=0.07, color='#455A64')
    ax.plot(xd, yt, color='#37474F', lw=1.5, alpha=0.55, label='True $|R(\\tau)|^2$')

    # Three sample points 鈥?intersection of both curves
    ax.plot(xp, yp, 'o', color='#D32F2F', ms=12, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5)

    # Fitted parabola
    ax.plot(xd, yf, color='#1976D2', lw=2.5, zorder=4, label='Fitted parabola')

    # True 未 (green) + Estimated 未虃 (blue)
    yt_peak = R2_quartic(0)  # = 1.0 at x=d
    ax.axvline(d, color='#2E7D32', ls='--', lw=2.2, zorder=3,
               label=f'True $\\delta={d:.2f}$')
    ax.axvline(dh, color='#1565C0', ls=':', lw=2.2, zorder=3,
               label=f'Est. $\\hat{{\\delta}}={dh:.3f}$')

    # Point labels
    for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$', '$y_0$', '$y_{+1}$']):
        va = 'bottom' if yi > 0.3 else 'top'
        off = 14 if yi > 0.3 else -14
        ax.annotate(lbl, (xi, yi), textcoords='offset points', xytext=(0, off),
                   ha='center', va=va, fontsize=10, color='#B71C1C')

    # 未虃 annotation near parabola peak
    ax.annotate(f'$\\hat{{\\delta}}={dh:.3f}$',
                xy=(dh, yf_peak), xytext=(dh + 0.18, yf_peak + 0.06),
                fontsize=9, color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))

    # 未 (true) annotation near true peak
    ax.annotate(f'$\\delta={d:.2f}$',
                xy=(d, yt_peak), xytext=(d - 0.25, yt_peak + 0.08),
                fontsize=9, color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2))

    ax.set_xlabel('Offset from $k_{\\rm opt}$ (samples)')
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.2)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.05, 1.12])
    if ax == axes[0]:
        ax.set_ylabel('$|R(\\tau)|^2$')
        ax.legend(fontsize=8.5, loc='upper right')

fig.suptitle('Parabolic Interpolation 鈥?Three-Point Quadratic Fit (Stage 1)',
             fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(R, 'fig_03_v11_final.png'), dpi=1200, bbox_inches='tight')
plt.close()

# 鈹€鈹€ Console summary 鈹€鈹€
print("  鉁?fig_03_v11_final.png")
print("\n  Quartic model: c={}, q={}  |  Green=true d,  Blue=estimated dh".format(c, q))
for d in deltas:
    xp = np.array([-1.0, 0.0, 1.0])
    yp = np.array([R2_quartic(m - d) for m in xp])
    A = np.array([[1,-1,1],[0,0,1],[1,1,1]])
    cf = np.linalg.solve(A, yp)
    a, b_ = cf[0], cf[1]
    xv = -b_/(2*a) if abs(a)>1e-10 else 0
    dh = np.clip(xv, -0.5, 0.5)
    print(f"  d={d:.2f}:  y=[{yp[0]:.3f}, {yp[1]:.3f}, {yp[2]:.3f}]  "
          f"dh={dh:.4f}  gap={abs(dh-d):.4f} samples")
