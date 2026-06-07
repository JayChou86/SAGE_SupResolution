#!/usr/bin/env python
"""Fig 03 variants v6-v9: Corrected autocorrelation computation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix',
                     'figure.facecolor':'white','axes.facecolor':'white'})
R = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
from sage.signal_utils import generate_tx_signal
S_TX = generate_tx_signal(511, seed=42)
N_SEQ = len(S_TX)

deltas = [0.0, 0.25, 0.43]
titles  = ['(a) $\\delta$=0.00 (on-grid)', '(b) $\\delta$=0.25', '(c) $\\delta$=0.43 (off-grid)']

def autocorr_fft(signal, tau):
    """Compute |R(tau)|^2 via FFT 鈥?proper bandlimited autocorrelation at any tau."""
    N = len(signal)
    Sf = np.fft.fft(signal)
    k = np.concatenate([np.arange(0, int(np.ceil(N/2))),
                         np.arange(-int(np.floor(N/2)), 0)])
    phase = np.exp(-1j * 2 * np.pi * k * tau / N)
    R_val = np.sum(Sf * np.conj(Sf) * phase) / N  # R(tau) = ifft(|S|^2 * e^{-j蠅蟿})[0]
    return np.abs(R_val)**2

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V6: FFT autocorrelation 鈥?proper bandlimited |R(蟿)|虏
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v6():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, d, title in zip(axes, deltas, titles):
        xp = np.array([-1, 0, 1])
        yp = np.array([autocorr_fft(S_TX, d + m) for m in xp])
        yp = yp / max(yp[1], 1e-12)
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf = np.linalg.solve(A, yp)
        a,b_ = cf[0],cf[1]; xv = -b_/(2*a) if abs(a)>1e-10 else 0
        dh = np.clip(xv, -0.5, 0.5)
        xd = np.linspace(-1.5, 1.5, 300)
        yf = a*xd**2 + b_*xd + cf[2]
        # True |R|^2 curve
        yt = np.array([autocorr_fft(S_TX, d + x) for x in xd])
        yt = yt / max(yt.max(), 1e-12)
        ax.plot(xd, yt, 'k-', lw=1.2, alpha=0.35, label='True $|R(\\tau)|^2$')
        ax.plot(xp, yp, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(xd, yf, 'b-', lw=2, label='Fitted parabola')
        ax.axvline(d, color='green', ls='--', lw=2, label=f'$\\delta={d:.2f}$')
        ax.axvline(dh, color='blue', ls=':', lw=2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,12), ha='center', fontsize=10, color='darkred')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$ (norm.)'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V6: FFT autocorrelation 鈥?true |R(蟿)|虏 from signal')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v6_fft_autocorr.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V7: V6 + shaded true curve + thicker lines
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v7():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, d, title in zip(axes, deltas, titles):
        xp = np.array([-1, 0, 1])
        yp = np.array([autocorr_fft(S_TX, d + m) for m in xp])
        yp = yp / max(yp[1], 1e-12)
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf = np.linalg.solve(A, yp)
        a,b_ = cf[0],cf[1]; xv = -b_/(2*a) if abs(a)>1e-10 else 0
        dh = np.clip(xv, -0.5, 0.5)
        xd = np.linspace(-1.5, 1.5, 300)
        yf = a*xd**2 + b_*xd + cf[2]
        yt = np.array([autocorr_fft(S_TX, d + x) for x in xd])
        yt = yt / max(yt.max(), 1e-12)
        ax.fill_between(xd, 0, np.clip(yt,0,1.1), alpha=0.08, color='gray')
        ax.plot(xd, yt, 'k-', lw=1, alpha=0.3, label='True $|R(\\tau)|^2$')
        ax.plot(xp, yp, 'o', color='#D32F2F', ms=11, zorder=5)
        ax.plot(xp, yp, 'o', color='#FFCDD2', ms=14, zorder=4, alpha=0.3)
        ax.plot(xd, yf, '#1976D2', lw=2.5, label='Fitted parabola')
        ax.axvline(d, color='#2E7D32', ls='--', lw=2.2, label=f'True $\\delta={d:.2f}$')
        ax.axvline(dh, color='#1565C0', ls=':', lw=2.2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,14), ha='center', fontsize=10, color='#B71C1C')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.2); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$ (norm.)'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V7: FFT autocorr + shaded true curve + refined style')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v7_fft_refined.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V8: Quadratic model c=1.6 BUT three points from actual autocorr
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v8():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, d, title in zip(axes, deltas, titles):
        xp = np.array([-1, 0, 1])
        yp = np.array([autocorr_fft(S_TX, d + m) for m in xp])
        yp = yp / max(yp[1], 1e-12)
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf = np.linalg.solve(A, yp)
        a,b_ = cf[0],cf[1]; xv = -b_/(2*a) if abs(a)>1e-10 else 0
        dh = np.clip(xv, -0.5, 0.5)
        xd = np.linspace(-1.5, 1.5, 300)
        yf = a*xd**2 + b_*xd + cf[2]
        # Quadratic model true curve (clean, theoretical)
        c_q = 1.6
        yq = np.clip(1 - c_q*(xd-d)**2, 0, 1.1)
        ax.plot(xd, yq, 'k-', lw=1.5, alpha=0.35, label='Quadratic $1-c\\tau^2$')
        ax.plot(xp, yp, 'ro', ms=10, zorder=5, label='$y_{-1},y_0,y_{+1}$ (signal)')
        ax.plot(xd, yf, 'b-', lw=2, label='Fitted parabola')
        ax.axvline(d, color='green', ls='--', lw=2, label=f'True $\\delta={d:.2f}$')
        ax.axvline(dh, color='blue', ls=':', lw=2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,12), ha='center', fontsize=10, color='darkred')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$ (norm.)'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V8: Quadratic theory curve + actual signal three-point values')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v8_hybrid.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# V9: V7 but using ACTUAL signal |R(蟿)|虏 for true curve 鈥?no quadratic approx
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?def v9():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, d, title in zip(axes, deltas, titles):
        # Three sample points from actual signal autocorrelation
        xp = np.array([-1, 0, 1])
        yp_raw = np.array([autocorr_fft(S_TX, d + m) for m in xp])
        yp = yp_raw / max(yp_raw[1], 1e-12)
        
        # Fit parabola
        A = np.array([[1,-1,1],[0,0,1],[1,1,1]]); cf = np.linalg.solve(A, yp)
        a,b_ = cf[0],cf[1]; xv = -b_/(2*a) if abs(a)>1e-10 else 0
        dh = np.clip(xv, -0.5, 0.5)
        xd = np.linspace(-1.5, 1.5, 300)
        yf = a*xd**2 + b_*xd + cf[2]
        
        # True |R(蟿)|虏 from actual signal via FFT
        yt = np.array([autocorr_fft(S_TX, d + x) for x in xd])
        yt_max = max(yt.max(), 1e-12)
        yt_n = yt / yt_max
        
        # Plot
        ax.fill_between(xd, 0, np.clip(yt_n, 0, 1.1), alpha=0.1, color='#455A64')
        ax.plot(xd, yt_n, color='#37474F', lw=1.5, alpha=0.5, label='True $|R(\\tau)|^2$')
        ax.plot(xp, yp, 'o', color='#D32F2F', ms=11, zorder=5, label='$y_{-1},y_0,y_{+1}$')
        ax.plot(xd, yf, '#1976D2', lw=2.5, label='Fitted parabola')
        ax.axvline(d, color='#2E7D32', ls='--', lw=2.2, label=f'True $\\delta={d:.2f}$')
        ax.axvline(dh, color='#1565C0', ls=':', lw=2.2, label=f'$\\hat{{\\delta}}={dh:.3f}$')
        for xi, yi, lbl in zip(xp, yp, ['$y_{-1}$','$y_0$','$y_{+1}$']):
            ax.annotate(lbl, (xi,yi), textcoords='offset points', xytext=(0,14), ha='center', fontsize=10, color='#B71C1C')
        ax.set_xlabel('Offset from $k_{\\rm opt}$'); ax.set_title(title)
        ax.grid(alpha=0.2); ax.set_xlim([-1.5,1.5]); ax.set_ylim([-0.05,1.1])
        if ax==axes[0]: ax.set_ylabel('$|R(\\tau)|^2$ (norm.)'); ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('V9: Pure signal 鈥?true |R(蟿)|虏 + 3 real points + parabola fit')
    fig.tight_layout(); fig.savefig(os.path.join(R,'fig_03_v9_pure_signal.png'),dpi=1200); plt.close()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?if __name__=='__main__':
    print("Generating corrected Fig 03 variants (V6-V9)...")
    v6(); print("  V6: FFT autocorr 鈥?clean, actual |R(蟿)|虏")
    v7(); print("  V7: FFT autocorr + shaded + refined style")
    v8(); print("  V8: Quadratic theory curve + actual signal 3-point values")
    v9(); print("  V9: Pure signal 鈥?true curve + real points + parabola")
    print("\nAll saved to results/fig_03_v*.png")
