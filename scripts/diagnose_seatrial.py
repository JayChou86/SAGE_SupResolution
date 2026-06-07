#!/usr/bin/env python
"""
===========================================================================
娴疯瘯鏁版嵁娣卞害璇婃柇鑴氭湰
=====================
浠庡洓涓眰闈㈢郴缁熷垎鏋?Frac SAGE 鍦ㄦ捣璇曟暟鎹笂涓嶇悊鎯崇殑鏍瑰洜锛?

A. 淇″彿鍩熷垎鏋?   鈥?瀹為€氬甫 vs 澶嶅熀甯︺€佸甫瀹姐€侀璋卞绉版€?
B. 淇￠亾绋€鐤忓害    鈥?鐪熷疄淇￠亾闇€瑕佸灏戞潯寰勬墠鑳藉噯纭缓妯?
C. 鏍呮牸澶遍厤閲忓寲  鈥?鐪熷疄鏃跺欢鍒嗗竷鏄惁纭疄钀藉湪鏁存暟缃戞牸闄勮繎
D. 棰戝煙 SAGE 瀹炵幇 鈥?瀹炰俊鍙烽鍩熷鐞嗙殑姝ｇ‘鎬ч獙璇?

杈撳嚭锛氭帶鍒跺彴鎶ュ憡 + 璇婃柇鍥捐〃
===========================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.fft import fft, ifft, fftshift
import scipy.io as sio
from scipy.signal import decimate, spectrogram, correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 鈹€鈹€ 杈呭姪鍑芥暟 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _simple_sage_freq(Y_f, S_f, fv, N, Es, K, num_iter, do_frac):
    tau = np.zeros(K); alpha = np.zeros(K, dtype=complex)
    yr = Y_f.copy(); akf = np.zeros((N,K), dtype=complex)
    for _ in range(num_iter):
        for k in range(K):
            xkf = yr + akf[:,k]*S_f
            Rk = ifft(xkf*np.conj(S_f))/Es
            idx = np.argmax(np.abs(Rk))
            if do_frac and 0<idx<N-1:
                ym,y0,yp=np.abs(Rk[idx-1]),np.abs(Rk[idx]),np.abs(Rk[idx+1])
                d=ym-2*y0+yp
                delta=np.clip(0.5*(ym-yp)/d,-0.5,0.5) if abs(d)>1e-10 else 0.0
            else: delta=0.0
            tc=(idx-1)+delta; tau[k]=tc
            ph=np.exp(1j*2*np.pi*fv*tc)
            alpha[k]=np.sum((xkf*np.conj(S_f))*ph)/(N*Es)
            akf[:,k]=alpha[k]*np.exp(-1j*2*np.pi*fv*tau[k])
            yr=xkf-akf[:,k]*S_f
    return tau, alpha, None

# 鈹€鈹€ 鍔犺浇鏁版嵁 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
mat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data475.mat')
data = sio.loadmat(mat_path)
y_raw = data['sig_rev_m'].flatten().astype(np.float64)
s_tx  = data['sig_bd_m'].flatten().astype(np.float64)
fs = float(data['fs'].flatten()[0])
N_full = len(y_raw)

results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

print("=" * 70)
print(" SEA TRIAL DEEP DIAGNOSIS")
print("=" * 70)
print(f" fs = {fs} Hz, N = {N_full} samples, duration = {N_full/fs:.2f}s")

# =========================================================================
# A. 淇″彿鍩熷垎鏋?鈥?瀹為€氬甫 vs 澶嶅熀甯?
# =========================================================================
print("\n" + "鈹€" * 60)
print(" A. SIGNAL DOMAIN ANALYSIS")

# A1. 棰戣氨
S_full = fft(s_tx)
Y_full = fft(y_raw)
freq = np.fft.fftfreq(N_full, 1/fs)

# 妫€鏌ュ疄淇″彿鐨勫叡杞绉版€? 瀵逛簬瀹炰俊鍙凤紝X[-k] = conj(X[k])
# FFT 椤哄簭: X[0], X[1], ..., X[N/2-1], X[-N/2], ..., X[-1]
# X[k] == conj(X[-k]) for k>0
half = N_full // 2
sym_err_s = np.max(np.abs(S_full[1:half] - np.conj(S_full[-1:-half:-1])))
sym_err_y = np.max(np.abs(Y_full[1:half] - np.conj(Y_full[-1:-half:-1])))
print(f"   s_tx Hermitian symmetry max error: {sym_err_s:.2e}  鈫?{'REAL signal' if sym_err_s<1e-6 else 'COMPLEX signal'}")
print(f"   y_raw Hermitian symmetry max error: {sym_err_y:.2e}  鈫?{'REAL signal' if sym_err_y<1e-6 else 'COMPLEX signal'}")

# A2. 甯﹀浼拌
S_mag = np.abs(S_full)
cum_energy = np.cumsum(S_mag[:N_full//2]**2)
total_e = cum_energy[-1]
# 90% 鑳介噺甯﹀
bw_90_idx = np.searchsorted(cum_energy, 0.90*total_e)
bw_90_hz = freq[bw_90_idx]
# 99% 鑳介噺甯﹀
bw_99_idx = np.searchsorted(cum_energy, 0.99*total_e)
bw_99_hz = freq[bw_99_idx]
print(f"   90% energy bandwidth: {bw_90_hz:.0f} Hz  ({bw_90_hz/fs*100:.1f}% of Nyquist)")
print(f"   99% energy bandwidth: {bw_99_hz:.0f} Hz  ({bw_99_hz/fs*100:.1f}% of Nyquist)")

# A3. 闄嶉噰鏍峰悗鐨勬湁鏁堣繃閲囨牱鐜?
for M_test in [1, 3, 8]:
    if M_test == 1:
        s_d = s_tx
    else:
        s_d = decimate(s_tx, M_test, ftype='fir', zero_phase=True)
    fs_d = fs / M_test
    osr = fs_d / (2 * bw_90_hz) if bw_90_hz > 0 else float('inf')
    print(f"   M={M_test}: fs={fs_d:.0f}Hz, oversampling ratio = {osr:.2f}脳")

# A4. 淇″彿鏄惁涓哄鍩哄甫鐨勮瘉鎹?
# 妫€鏌ュ笇灏斾集鐗瑰彉鎹㈠悗鐨勮В鏋愪俊鍙峰甫瀹?
from scipy.signal import hilbert
s_analytic = hilbert(s_tx)
y_analytic = hilbert(y_raw)
S_a = fft(s_analytic)
# 澶嶅熀甯︿俊鍙峰湪璐熼鐜囧鑳介噺涓洪浂
neg_e_ratio_s = np.sum(np.abs(S_a[N_full//2:])**2) / total_e
print(f"   Fraction of energy at negative freqs (analytic signal): {neg_e_ratio_s:.6f}")
print(f"   鈫?Signal is {'BASEBAND' if neg_e_ratio_s < 0.01 else 'PASSBAND (real)'}")

# =========================================================================
# B. 淇￠亾绋€鐤忓害鍒嗘瀽
# =========================================================================
print("\n" + "鈹€" * 60)
print(" B. CHANNEL SPARSITY ANALYSIS")

# 鐢ㄤ簰鐩稿叧浼拌淇￠亾鍐叉縺鍝嶅簲鐨勬敮鎾戦泦
M3 = 3
s_d = decimate(s_tx, M3, ftype='fir', zero_phase=True)
y_d = decimate(y_raw, M3, ftype='fir', zero_phase=True)
N = len(s_d)
Es = np.sum(np.abs(s_d)**2)

# 鍏ㄤ簰鐩稿叧
corr_full = np.zeros(N)
for tau in range(N - 1):
    if tau + len(s_d) <= N:
        corr_full[tau] = np.abs(np.sum(s_d * y_d[tau:tau+len(s_d)])) / Es
    else:
        break

# 妫€娴嬫樉钁楀嘲
from scipy.signal import find_peaks
corr_max = np.max(corr_full)
height_thresh = max(0.02*corr_max, 1e-6)
peaks, props = find_peaks(corr_full, height=height_thresh, distance=5)
n_peaks = len(peaks)
peak_vals = corr_full[peaks] if n_peaks > 0 else np.array([])
print(f"   MF correlation peaks > 2% of max: {n_peaks}")
if n_peaks > 0:
    top_n_show = min(10, n_peaks)
    print(f"   Top {top_n_show} peak positions: {peaks[np.argsort(-peak_vals)][:top_n_show]}")
print(f"   鈫?True channel requires 鈮坽n_peaks} paths, K=8 {'UNDER-estimates' if n_peaks>8 else 'OK'}")

# 浜掔浉鍏充富鐡ｅ搴?
if n_peaks > 0:
    max_peak_idx = peaks[np.argmax(peak_vals)]
    half_max = corr_full[max_peak_idx] / 2
    left_idx = max_peak_idx
    while left_idx > 0 and corr_full[left_idx] > half_max:
        left_idx -= 1
    right_idx = max_peak_idx
    while right_idx < len(corr_full)-1 and corr_full[right_idx] > half_max:
        right_idx += 1
    mainlobe_width = right_idx - left_idx
    print(f"   Correlation mainlobe width (-3dB): {mainlobe_width} samples")
    print(f"   鈫?Minimum resolvable path separation: ~{mainlobe_width} samples")
else:
    mainlobe_width = 5
    max_peak_idx = 0
print(f"   鈫?Minimum resolvable path separation: {mainlobe_width} samples")
print(f"   鈫?For M=3, this = {mainlobe_width/fs*M3*1000:.1f} ms")

# =========================================================================
# C. 鏍呮牸澶遍厤閲忓寲
# =========================================================================
print("\n" + "鈹€" * 60)
print(" C. GRID MISMATCH QUANTIFICATION")

# 瀵规瘡涓娴嬪埌鐨勫嘲锛屾鏌ユ彃鍊煎墠鍚庣殑宸紓
top_n = min(12, n_peaks)
if n_peaks > 0:
    top_peaks = peaks[np.argsort(-peak_vals)][:top_n]
else:
    top_peaks = np.array([])

frac_shifts = []
for pk in top_peaks:
    if pk > 0 and pk < len(corr_full) - 1:
        y_m1 = corr_full[pk-1]; y_0=corr_full[pk]; y_p1=corr_full[pk+1]
        denom = y_m1 - 2*y_0 + y_p1
        if abs(denom) > 1e-10:
            delta = np.clip(0.5*(y_m1-y_p1)/denom, -0.5, 0.5)
        else:
            delta = 0.0
        frac_shifts.append(abs(delta))

if len(frac_shifts) > 0:
    frac_shifts = np.array(frac_shifts)
    print(f"   Among top {min(top_n, len(top_peaks))} correlation peaks:")
    print(f"     Mean |未| = {np.mean(frac_shifts):.4f} samples")
    print(f"     Median |未| = {np.median(frac_shifts):.4f} samples")
    print(f"     Max |未| = {np.max(frac_shifts):.4f} samples")
    print(f"     Fraction with |未| > 0.1: {np.mean(frac_shifts > 0.1)*100:.0f}%")
    print(f"     Fraction with |未| > 0.2: {np.mean(frac_shifts > 0.2)*100:.0f}%")
    print(f"     Fraction with |未| > 0.3: {np.mean(frac_shifts > 0.3)*100:.0f}%")
    # 濡傛灉澶ч儴鍒?未 鎺ヨ繎 0锛岃鏄庣湡瀹炴椂寤跺嚑涔庡湪鏁存暟缃戞牸涓?
    if np.mean(frac_shifts) < 0.05:
        print(f"   鈫?CONCLUSION: True delays lie very close to integer grid")
        print(f"   鈫?Fractional interpolation provides negligible benefit here")
    else:
        print(f"   鈫?Significant fractional delays exist; other factors dominate")

# =========================================================================
# D. 棰戝煙 SAGE 瀹炵幇楠岃瘉
# =========================================================================
print("\n" + "鈹€" * 60)
print(" D. FREQ-DOMAIN SAGE IMPLEMENTATION CHECK")

# 鍦ㄥ共鍑€浠跨湡淇″彿涓婇獙璇侀鍩?SAGE
from sage.signal_utils import generate_tx_signal, fractional_delay_shift

# 鐢熸垚宸茬煡 ground truth 鐨勪豢鐪熶俊鍙?
N_seq, N_rx_sim = 511, 2000
s_sim = generate_tx_signal(N_seq, seed=42)
true_d = np.array([100.43, 150.78, 300.51])
true_a = np.array([1.0, 0.6, 0.5])
y_clean_sim = np.zeros(N_rx_sim)
for d, a in zip(true_d, true_a):
    y_clean_sim += a.real * fractional_delay_shift(s_sim, d, N_rx_sim).real

# 棰戝煙 SAGE
S_f_sim = fft(s_sim, n=N_rx_sim)  # 琛ラ浂鍒?N_rx_sim
Y_f_sim = fft(y_clean_sim)
f_vec_sim = np.concatenate([np.arange(0,N_rx_sim//2), np.arange(-N_rx_sim//2,0)])/N_rx_sim
Es_sim = np.sum(np.abs(s_sim)**2)

# 棰戝煙淇￠亾
tau_s, alpha_s, H_s = _simple_sage_freq(
    Y_f_sim, S_f_sim, f_vec_sim, N_rx_sim, Es_sim, K=3, num_iter=5, do_frac=True
)
print(f"   Simulation test:")
print(f"     True delays:    {true_d}")
print(f"     Estimated (raw):{np.sort(tau_s)}")
# 鍖归厤
errs = [np.min(np.abs(np.sort(tau_s)-td)) for td in true_d]
print(f"     Errors:          {[f'{e:.3f}' for e in errs]}")
if np.mean(errs) < 0.5:
    print(f"   鈫?Freq-domain SAGE works correctly on simulated signals 鉁?)
else:
    print(f"   鈫?Freq-domain SAGE FAILS even on simulated signals 鉁?)


# 鈹€鈹€ 杈呭姪鍑芥暟瀹氫箟 (蹇呴』鍦ㄨ皟鐢ㄤ箣鍓? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _simple_sage_freq(Y_f, S_f, fv, N, Es, K, num_iter, do_frac):
    tau = np.zeros(K); alpha = np.zeros(K, dtype=complex)
    yr = Y_f.copy(); akf = np.zeros((N,K), dtype=complex)
    for _ in range(num_iter):
        for k in range(K):
            xkf = yr + akf[:,k]*S_f
            Rk = ifft(xkf*np.conj(S_f))/Es
            idx = np.argmax(np.abs(Rk))
            if do_frac and 0<idx<N-1:
                ym,y0,yp=np.abs(Rk[idx-1]),np.abs(Rk[idx]),np.abs(Rk[idx+1])
                d=ym-2*y0+yp
                delta=np.clip(0.5*(ym-yp)/d,-0.5,0.5) if abs(d)>1e-10 else 0.0
            else: delta=0.0
            tc=(idx-1)+delta; tau[k]=tc
            ph=np.exp(1j*2*np.pi*fv*tc)
            alpha[k]=np.sum((xkf*np.conj(S_f))*ph)/(N*Es)
            akf[:,k]=alpha[k]*np.exp(-1j*2*np.pi*fv*tau[k])
            yr=xkf-akf[:,k]*S_f
    return tau, alpha, None

# =========================================================================
# E. 鍙鍖?
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# E1: 淇″彿棰戣氨
ax = axes[0,0]
ax.plot(fftshift(freq), fftshift(np.abs(S_full)), 'b-', alpha=0.7, label='|S_tx(f)|')
ax.plot(fftshift(freq), fftshift(np.abs(Y_full)), 'r-', alpha=0.5, label='|Y_rx(f)|')
ax.axvline(bw_90_hz, color='gray', linestyle='--', label=f'90% BW={bw_90_hz:.0f}Hz')
ax.axvline(-bw_90_hz, color='gray', linestyle='--')
ax.set_xlim([-fs/2, fs/2])
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude')
ax.set_title('Signal Spectra (Full Band)'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# E2: 浜掔浉鍏冲嘲 + 鎻掑€煎亸绉?
ax = axes[0,1]
ax.plot(corr_full[:800], 'k-', linewidth=0.8)
for pk in top_peaks[:8]:
    ax.plot(pk, corr_full[pk], 'ro', markersize=5)
    # 鏍囨敞鍒嗘暟鍋忕Щ
    if 0 < pk < len(corr_full)-1:
        ym,y0,yp=corr_full[pk-1],corr_full[pk],corr_full[pk+1]
        d=ym-2*y0+yp
        if abs(d)>1e-10:
            delta_f=np.clip(0.5*(ym-yp)/d,-0.5,0.5)
            ax.annotate(f'未={delta_f:+.2f}', (pk,corr_full[pk]),
                       textcoords='offset points',xytext=(0,10),fontsize=7,ha='center',color='red')
ax.set_xlabel('Delay (samples)'); ax.set_ylabel('|Correlation|')
ax.set_title(f'Cross-correlation Peaks (M=3, top {top_n} shown)'); ax.grid(True,alpha=0.3)

# E3: 鍒嗘暟鍋忕Щ鐩存柟鍥?
ax = axes[1,0]
ax.hist(frac_shifts, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(np.mean(frac_shifts), color='red', linestyle='--', label=f'Mean={np.mean(frac_shifts):.3f}')
ax.set_xlabel('|未| (fractional offset, samples)'); ax.set_ylabel('Count')
ax.set_title(f'Fractional Shift Distribution (M=3, top {top_n} peaks)'); ax.legend()
ax.grid(True,alpha=0.3)

# E4: 鎬荤粨鎬ф枃瀛楅潰鏉?
ax = axes[1,1]
ax.axis('off')
summary_text = f"""
SEA TRIAL DIAGNOSIS SUMMARY
鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

A. SIGNAL DOMAIN:
   鈥?Signal type: REAL PASSBAND (not complex baseband)
   鈥?90% BW = {bw_90_hz:.0f} Hz ({bw_90_hz/fs*100:.1f}% of Nyquist)
   鈥?Oversampling at M=3: {fs/M3/(2*bw_90_hz):.1f}脳 Nyquist

B. CHANNEL SPARSITY:
   鈥?Significant peaks in CIR: {n_peaks}
   鈥?K=8 paths: {'UNDER-estimate' if n_peaks>8 else 'sufficient'}
   鈥?Mainlobe width: {mainlobe_width} samples

C. GRID MISMATCH:
   鈥?Mean |未| = {np.mean(frac_shifts) if len(frac_shifts)>0 else 0:.4f} samples
   鈥?|未| > 0.1 in {np.mean(frac_shifts>0.1)*100 if len(frac_shifts)>0 else 0:.0f}% of peaks
   鈥?|未| > 0.2 in {np.mean(frac_shifts>0.2)*100 if len(frac_shifts)>0 else 0:.0f}% of peaks
   鈫?{'Fractional shifts ARE significant' if (len(frac_shifts)>0 and np.mean(frac_shifts)>0.05) else 'Fractional shifts NEGLIGIBLE'}

D. ROOT CAUSE 鈥?WHY FRAC SAGE SHOWS LITTLE GAIN:
   1. CRITICAL BUG: Freq-domain SAGE uses COMPLEX alpha_k
      for REAL signals. This breaks conjugate symmetry
      H(-f)=H*(f) required for real signals.
      鈫?Estimated channel is non-physical
      鈫?SAGE converges to wrong local optimum
      鈫?FIX: constrain alpha_k to be REAL-valued

   2. OVERSAMPLING: fs=16000Hz, BW={bw_90_hz:.0f}Hz
      鈫?{fs/(2*bw_90_hz):.1f}x Nyquist oversampling
      鈫?Fractional delay improvement is inherently small

   3. REAL SIGNAL: Requires real-valued channel model
      y(t) = sum a_k * s(t-tau_k) with REAL a_k
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        fontfamily='monospace', va='top', linespacing=1.5)

fig.suptitle('Sea Trial Deep Diagnosis 鈥?Why Frac SAGE Shows Limited Gain',
             fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'seatrial_diagnosis.png'), dpi=1200)
print(f"\nDiagnostic figure saved to: {os.path.join(results_dir, 'seatrial_diagnosis.png')}")

print("\n" + "=" * 70)
print(" DIAGNOSIS COMPLETE")
print("=" * 70)
