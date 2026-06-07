#!/usr/bin/env python
"""
Generate seatrial_ber.png and seatrial_cir.png for paper_twc.tex.

seatrial_ber:  BER vs decimation factor for Int SAGE vs Frac SAGE.
               Uses MMSE equalization + QPSK symbol detection.
seatrial_cir:  Channel impulse response estimated by both methods at M=3.
               Stem plot with proper labeling, matching paper Fig layout.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.fft import fft, ifft
import scipy.io as sio
from scipy.signal import decimate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
MAT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data475.mat')


def load_data():
    data = sio.loadmat(MAT)
    y0 = data['sig_rev_m'].flatten().astype(np.float64)
    s0 = data['sig_bd_m'].flatten().astype(np.float64)
    fs = float(data['fs'].flatten()[0])
    return y0, s0, fs


def sage_freq(y, s, K=8, niter=5, do_frac=True):
    """Frequency-domain SAGE with zero-padded FFT for linear correlation."""
    from sage.interpolation import parabolic_delta
    N = len(y)
    Es_val = np.sum(np.abs(s)**2)
    N2 = 2 * N
    Sf = fft(s, n=N2)
    Yf = fft(y, n=N2)
    fv = np.concatenate([np.arange(0, N2//2), np.arange(-N2//2, 0)]) / N2

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
                ym, y0_, yp = np.abs(Rk[idx-1]), np.abs(Rk[idx]), np.abs(Rk[idx+1])
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


def mmse_equalize(y, s, H):
    """MMSE equalization in frequency domain."""
    N = len(y)
    Yf = fft(y)
    nv = 0.05 * np.max(np.abs(H))**2 + 1e-8
    Seq_f = Yf * np.conj(H) / (np.abs(H)**2 + nv)
    return ifft(Seq_f)


def qpsk_demod(s_eq, s_ref):
    """Simple QPSK demodulation: sign of real and imag parts."""
    bits_eq = np.zeros(2 * len(s_eq), dtype=int)
    bits_ref = np.zeros(2 * len(s_ref), dtype=int)
    for i in range(len(s_eq)):
        bits_eq[2*i]   = 1 if np.real(s_eq[i]) > 0 else 0
        bits_eq[2*i+1] = 1 if np.imag(s_eq[i]) > 0 else 0
        bits_ref[2*i]   = 1 if np.real(s_ref[i]) > 0 else 0
        bits_ref[2*i+1] = 1 if np.imag(s_ref[i]) > 0 else 0
    return np.mean(bits_eq != bits_ref)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# FIGURE 1: seatrial_ber.png 鈥?BER vs decimation
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def generate_ber():
    print("Generating seatrial_ber.png...")
    y0, s0, fs = load_data()

    # Use a smaller set: M = 1, 2, 3, 4, 6, 8
    down_ratios = [1, 2, 3, 4, 6, 8]
    ber_i = np.zeros(len(down_ratios))
    ber_f = np.zeros(len(down_ratios))

    for ti, M in enumerate(down_ratios):
        if M <= 1:
            yr, st = y0.copy(), s0.copy()
        else:
            yr = decimate(y0, M, ftype='fir', zero_phase=True)
            st = decimate(s0, M, ftype='fir', zero_phase=True)

        tau_i, alpha_i, H_i = sage_freq(yr, st, K=8, niter=5, do_frac=False)
        tau_f, alpha_f, H_f = sage_freq(yr, st, K=8, niter=5, do_frac=True)

        seq_i = mmse_equalize(yr, st, H_i)
        seq_f = mmse_equalize(yr, st, H_f)

        # BER on valid region
        v0 = 2000 if len(yr) > 4000 else len(yr) // 5
        v1 = len(yr) - v0
        ber_i[ti] = qpsk_demod(seq_i[v0:v1], st[v0:v1])
        ber_f[ti] = qpsk_demod(seq_f[v0:v1], st[v0:v1])
        print(f"  M={M:1d}: Int BER={ber_i[ti]:.4f}  Frac BER={ber_f[ti]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(down_ratios, ber_i, 'b-s', lw=2, ms=8,
                label='Integer SAGE')
    ax.semilogy(down_ratios, ber_f, 'r-o', lw=2, ms=8,
                label='Fractional SAGE')
    ax.set_xlabel('Decimation Factor $M$', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs Decimation 鈥?Sea Trial ($f_s$=16 kHz, QPSK)', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(down_ratios)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'seatrial_ber.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?seatrial_ber.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# FIGURE 2: seatrial_cir.png 鈥?Channel Impulse Response at M=3
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def generate_cir():
    print("Generating seatrial_cir.png...")
    y0, s0, fs = load_data()

    M = 3
    yr = decimate(y0, M, ftype='fir', zero_phase=True)
    st = decimate(s0, M, ftype='fir', zero_phase=True)

    tau_i, alpha_i, H_i = sage_freq(yr, st, K=8, niter=5, do_frac=False)
    tau_f, alpha_f, H_f = sage_freq(yr, st, K=8, niter=5, do_frac=True)

    # Sort both by delay
    idx_i = np.argsort(tau_i)
    idx_f = np.argsort(tau_f)
    tau_i_s = tau_i[idx_i]; alpha_i_s = alpha_i[idx_i]
    tau_f_s = tau_f[idx_f]; alpha_f_s = alpha_f[idx_f]

    # Plot CIR stem
    fig, ax = plt.subplots(figsize=(10, 5))
    offset = 0.25
    markerline1, stemline1, baseline1 = ax.stem(
        tau_i_s - offset, np.abs(alpha_i_s),
        linefmt='k-', markerfmt='ks', basefmt=' ',
        label='Integer SAGE ($K=8$)')
    plt.setp(markerline1, markersize=8)
    markerline2, stemline2, baseline2 = ax.stem(
        tau_f_s + offset, np.abs(alpha_f_s),
        linefmt='b-', markerfmt='b^', basefmt=' ',
        label='Fractional SAGE ($K=8$)')
    plt.setp(markerline2, markersize=8)

    # Annotate delay and amplitude for top paths
    for i, (t, a) in enumerate(zip(tau_f_s, alpha_f_s)):
        ax.annotate(f'$\\tau_{i+1}$={t:.1f}\n$|\\alpha|$={np.abs(a):.3f}',
                    xy=(t, np.abs(a)), xytext=(t + 8, np.abs(a) * 1.1),
                    fontsize=7, ha='left',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

    ax.set_xlabel('Delay (samples at $f_s/M = $' + f'{fs/M:.0f} Hz)', fontsize=12)
    ax.set_ylabel('$|\\alpha|$ (Magnitude)', fontsize=12)
    ax.set_title(f'Estimated Channel Impulse Response 鈥?Sea Trial (M={M}, $f_s$={fs} Hz)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(tau_f_s.max(), tau_i_s.max()) + 50])

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'seatrial_cir.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?seatrial_cir.png")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# MAIN
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
if __name__ == '__main__':
    print("=" * 60)
    print(" Generating seatrial BER + CIR figures")
    print("=" * 60)
    generate_ber()
    generate_cir()
    print("Done.")
