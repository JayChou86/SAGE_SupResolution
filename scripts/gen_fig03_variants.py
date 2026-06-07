#!/usr/bin/env python
"""Generate all Fig 03 variants for comparison. Output: results/fig_03_v*.png"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix',
                     'figure.facecolor':'white','axes.facecolor':'white'})
R = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
from sage.signal_utils import generate_tx_signal, fractional_delay_shift
S_TX = generate_tx_signal(511, seed=42)
N_SEQ = len(S_TX)

deltas = [0.0, 0.25, 0.43]
titles  = ['(a) $\\delta$=0.00 (on-grid)', '(b) $\\delta$=0.25', '(c) $\\delta$=0.43 (off-grid)']

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V1: bare-style 鈥?closed-form formula, labels BELOW points, simple
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v1():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    c = 1.5
    for ax, d, title in zip(axes, deltas, titles):
        x = np.linspace(-1.5, 1.5, 200)
        y_true = np.maximum(1 - c*(x-d)**2, 0)
        xs = np.array([-1, 0, 1])
        ys = 1 - c*(xs-d)**2
        denom = ys[0] - 2*ys[1] + ys[2]
        dh = 0.5*(ys[0]-ys[2])/denom if abs(denom)>1e-10 else 0
        a = denom/2
        y_fit = a*(x-dh)**2 + (ys[1] - a*dh**2)
        ax.plot(x, y_true, lw=1.5, alpha=0.5, label='True $|R(\\tau)|^2$')
        ax.plot(x, y_fit, '--', lw=1.2, label='Fitted parabola')
        ax.plot(xs, ys, 'o', ms=8, zorder=5)
        for xi, yi, lbl in zip(xs, ys, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,-14), ha='center', fontsize=9)
        ax.axvline(d, ls=':', lw=0.8)
        ax.axvline(dh, ls='--', lw=0.8)
        ax.set_title(title); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        ax.grid(alpha=0.3)
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$'); ax.legend(fontsize=8)
    fig.suptitle('V1: Bare-style (closed-form, labels below)')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v1_bare.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V2: Quadratic c=1.5, linear-system fit, labels ABOVE, unclipped y_pts
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    c = 1.5
    for ax, d, title in zip(axes, deltas, titles):
        xd = np.linspace(-1.5, 1.5, 300)
        yd = np.clip(1 - c*(xd-d)**2, 0, 1.1)
        xp = np.array([-1,0,1])
        yp = 1 - c*(xp-d)**2  # unclipped
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]])
        cf = np.linalg.solve(A, yp)
        a,b_ = cf[0],cf[1]; xv = -b_/(2*a) if abs(a)>1e-10 else 0
        dh = np.clip(xv, -0.5, 0.5)
        yf = a*xd**2 + b_*xd + cf[2]
        ax.plot(xd, yd, 'k-', lw=1.5, alpha=0.4, label='True $|R(\\tau)|^2$')
        ax.plot(xp, yp, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(xd, yf, 'b-', lw=2, label='Fitted parabola')
        ax.axvline(d, color='green', ls='--', lw=2, label=f'$\\delta={d:.2f}$')
        ax.axvline(dh, color='blue', ls=':', lw=2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,12), ha='center', fontsize=10, color='darkred')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V2: Quadratic c=1.5, linear-system fit, labels above')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v2_quadratic.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V3: Quadratic c=1.6 (matching signal), unclipped, labels above
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v3():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    c = 1.6
    for ax, d, title in zip(axes, deltas, titles):
        xd = np.linspace(-1.5, 1.5, 300)
        yd = np.clip(1 - c*(xd-d)**2, 0, 1.1)
        xp = np.array([-1,0,1])
        yp = 1 - c*(xp-d)**2
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]])
        cf = np.linalg.solve(A, yp)
        a,b_ = cf[0],cf[1]; xv = -b_/(2*a) if abs(a)>1e-10 else 0
        dh = np.clip(xv, -0.5, 0.5)
        yf = a*xd**2 + b_*xd + cf[2]
        ax.plot(xd, yd, 'k-', lw=1.5, alpha=0.4, label='True $|R(\\tau)|^2$')
        ax.plot(xp, yp, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(xd, yf, 'b-', lw=2, label='Fitted parabola')
        ax.axvline(d, color='green', ls='--', lw=2, label=f'$\\delta={d:.2f}$')
        ax.axvline(dh, color='blue', ls=':', lw=2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,12), ha='center', fontsize=10, color='darkred')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V3: Quadratic c=1.6 (actual signal c=1.626), labels above')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v3_c16.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V4: Actual QPSK autocorrelation points + quadratic fit
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v4():
    N_sim, center = 600, 300.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, d, title in zip(axes, deltas, titles):
        tau_true = center + d
        k_opt = int(np.floor(tau_true+0.5))
        s_sh = fractional_delay_shift(S_TX, tau_true, N_sim)
        cv = np.zeros(3)
        for mi, m in enumerate([-1,0,1]):
            t = k_opt+m
            if 0<=t and t+N_SEQ<=N_sim: cv[mi]=np.abs(np.vdot(S_TX, s_sh[t:t+N_SEQ]))**2
        yp = cv/max(cv[1],1e-12)
        A=np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf=np.linalg.solve(A,yp)
        a,b_=cf[0],cf[1]; xv=-b_/(2*a) if abs(a)>1e-10 else 0; dh=np.clip(xv,-0.5,0.5)
        xd=np.linspace(-1.5,1.5,300); yf=a*xd**2+b_*xd+cf[2]
        ax.plot([-1,0,1], yp, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(xd, yf, 'b-', lw=2, label='Fitted parabola')
        ax.axvline(d, color='green', ls='--', lw=2, label=f'$\\delta={d:.2f}$')
        ax.axvline(dh, color='blue', ls=':', lw=2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip([-1,0,1], yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,12), ha='center', fontsize=10, color='darkred')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|C|^2$ (norm.)'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V4: Actual QPSK autocorrelation (real signal points)')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v4_real_autocorr.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V5: V2 but without black true-curve, cleaner look
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v5():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    c = 1.5
    for ax, d, title in zip(axes, deltas, titles):
        xd = np.linspace(-1.5, 1.5, 300)
        xp = np.array([-1,0,1])
        yp = 1 - c*(xp-d)**2
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf=np.linalg.solve(A,yp)
        a,b_=cf[0],cf[1]; xv=-b_/(2*a) if abs(a)>1e-10 else 0; dh=np.clip(xv,-0.5,0.5)
        yf = a*xd**2 + b_*xd + cf[2]
        yd = np.clip(1 - c*(xd-d)**2, 0, 1.1)
        ax.fill_between(xd, 0, yd, alpha=0.08, color='gray')
        ax.plot(xd, yd, 'k-', lw=1.2, alpha=0.35, label='True $|R(\\tau)|^2$')
        ax.plot(xp, yp, 'o', color='#D32F2F', ms=11, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(xd, yf, '#1976D2', lw=2.5, label='Fitted parabola')
        ax.axvline(d, color='#2E7D32', ls='--', lw=2.2, label=f'True $\\delta={d:.2f}$')
        ax.axvline(dh, color='#1565C0', ls=':', lw=2.2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,14), ha='center', fontsize=10, color='#B71C1C')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.2); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V5: Refined 鈥?shaded true curve, thicker lines, clean palette')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v5_refined.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?if __name__=='__main__':
    print("Generating 5 versions of Fig 03...")
    v1(); print("  V1: bare-style (closed-form, labels below)")
    v2(); print("  V2: quadratic c=1.5, linear-system fit, labels above")
    v3(); print("  V3: quadratic c=1.6, labels above")
    v4(); print("  V4: actual QPSK autocorrelation")
    v5(); print("  V5: refined 鈥?shaded true curve, thicker lines")
    print("\nAll saved to results/fig_03_v*.png")
