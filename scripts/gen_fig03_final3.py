#!/usr/bin/env python
"""Fig 03 final final: designed 3-point values + quartic true curve through them."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix',
                     'figure.facecolor':'white','axes.facecolor':'white'})
R = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(R, exist_ok=True)

# 鈹€鈹€ Designed three-point values 鈹€鈹€
# For each d (true delta), we pick 3 points so that parabola vertex dh 鈮?d - small_gap
specs = [
    dict(d=0.00, y=[0.100, 1.000, 0.100]),   # symmetric 鈫?dh=0 exact
    dict(d=0.25, y=[0.210, 0.930, 0.650]),   # dh鈮?.220, gap鈮?.03
    dict(d=0.43, y=[0.294, 0.780, 0.720]),   # dh鈮?.390, gap鈮?.04
]
titles = ['$\\delta$=0.00 (on-grid)', '$\\delta$=0.25', '$\\delta$=0.43 (off-grid)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
# Sub-panel labels (a)(b)(c) on left
for i, ax in enumerate(axes):
    ax.text(-0.08, 1.02, '(%s)' % chr(ord('a')+i), transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='left', va='bottom')

for ax, sp, title in zip(axes, specs, titles):
    d = sp['d']
    xp = np.array([-1.0, 0.0, 1.0])
    yp = np.array(sp['y'])

    # 鈹€鈹€ Parabola fit 鈹€鈹€
    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
    cf = np.linalg.solve(A, yp)
    a_p, b_p, c_p = cf
    dh = np.clip(-b_p/(2*a_p) if abs(a_p)>1e-10 else 0.0, -0.5, 0.5)

    # 鈹€鈹€ Quartic true curve: passes through 3 points, peaks at x=d 鈹€鈹€
    # y = k4*x^4 + k3*x^3 + k2*x^2 + k1*x + k0
    # Constraints: y(-1)=yp[0], y(0)=yp[1], y(1)=yp[2], y'(d)=0, y''(d) < 0
    M = np.array([
        [(-1)**4, (-1)**3, (-1)**2, -1, 1],   # y(-1) = yp[0]
        [0,       0,       0,        0, 1],    # y(0)  = yp[1]
        [1,       1,       1,        1, 1],    # y(1)  = yp[2]
        [4*d**3,  3*d**2,  2*d,      1, 0],    # y'(d) = 0
        [1,       0,       0,        0, 0],    # fix k4 = small value for smoothness
    ])
    rhs = np.array([yp[0], yp[1], yp[2], 0.0, -0.08])  # k4 = -0.08
    k = np.linalg.solve(M, rhs)

    def quartic(x):
        return k[0]*x**4 + k[1]*x**3 + k[2]*x**2 + k[3]*x + k[4]

    # 鈹€鈹€ Draw (V6 style: clean, no fill, thinner lines) 鈹€鈹€
    xd = np.linspace(-1.5, 1.5, 400)
    yt = np.array([quartic(x) for x in xd])
    yt = np.clip(yt, 0, None)
    yf = a_p*xd**2 + b_p*xd + c_p

    ax.plot(xd, yt, color='#555555', lw=1.2, alpha=0.45, label='True $|R(\\tau)|^2$')
    ax.plot(xp, yp, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$')
    ax.plot(xd, yf, 'b-', lw=2, label='Fitted parabola')
    ax.axvline(d, color='green', ls='--', lw=1.5, label=f'True $\\delta={d:.2f}$')
    ax.axvline(dh, color='blue', ls=':', lw=1.5, label=f'Est. $\\hat{{\\delta}}={dh:.3f}$')

    # Per-panel label offsets: (dx, dy) in points for [y_{-1}, y_0, y_{+1}]
    if d == 0.00:
        offsets = [(-10, 12), (0, 12), (10, 12)]
    elif d == 0.25:
        offsets = [(-10, 12), (0, 12), (0, 12)]
    else:
        offsets = [(0, 12), (0, 12), (0, 12)]

    for (xi, yi, lbl, (dx, dy)) in zip(xp, yp, ['$y_{-1}$', '$y_0$', '$y_{+1}$'], offsets):
        ax.annotate(lbl, (xi, yi), textcoords='offset points', xytext=(dx, dy),
                   ha='center', fontsize=10, color='darkred')

    # 鈹€鈹€ Annotate peaks (positions tuned per panel) 鈹€鈹€
    yt_peak = quartic(d)
    yf_peak = a_p*dh**2 + b_p*dh + c_p
    # (a) d=0.00: move right; (b) d=0.25: move up-right; (c) d=0.43: move up-right
    if d == 0.00:
        gx, gy = d - 0.40, yt_peak + 0.06
        bx, by = dh + 0.18, yf_peak + 0.06
    elif d == 0.25:
        gx, gy = d - 0.38, yt_peak + 0.10
        bx, by = dh + 0.18, yf_peak + 0.06
    else:  # d == 0.43
        gx, gy = d - 0.38, yt_peak + 0.10
        bx, by = dh + 0.18, yf_peak + 0.06

    ax.annotate(f'$\\delta={d:.2f}$', xy=(d, yt_peak), xytext=(gx, gy),
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.0))
    ax.annotate(f'$\\hat{{\\delta}}={dh:.3f}$', xy=(dh, yf_peak), xytext=(bx, by),
                fontsize=9, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.0))

    ax.set_xlabel('Offset from $k_{\\rm opt}$ (ms)')
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.2)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.05, 1.12])
    if ax == axes[0]:
        ax.set_ylabel('$|R(\\tau)|^2$')
        ax.legend(fontsize=8.5, loc='upper right')

#fig.suptitle('Parabolic Interpolation 鈥?Three-Point Quadratic Fit (Stage 1)',
 #            fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(R, 'fig_03_v12_final.png'), dpi=1200, bbox_inches='tight')
plt.close()

print("  鉁?fig_03_v12_final.png")
for sp in specs:
    d = sp['d']; yp = np.array(sp['y'])
    A = np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf = np.linalg.solve(A, yp)
    dh = np.clip(-cf[1]/(2*cf[0]) if abs(cf[0])>1e-10 else 0, -0.5, 0.5)
    print(f"  d={d:.2f}: y={sp['y']}  dh={dh:.4f}  gap={abs(dh-d):.4f}")
