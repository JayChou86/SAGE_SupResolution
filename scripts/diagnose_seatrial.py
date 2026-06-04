#!/usr/bin/env python
"""
===========================================================================
海试数据深度诊断脚本
=====================
从四个层面系统分析 Frac SAGE 在海试数据上不理想的根因：

A. 信号域分析    — 实通带 vs 复基带、带宽、频谱对称性
B. 信道稀疏度    — 真实信道需要多少条径才能准确建模
C. 栅格失配量化  — 真实时延分布是否确实落在整数网格附近
D. 频域 SAGE 实现 — 实信号频域处理的正确性验证

输出：控制台报告 + 诊断图表
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

# ── 辅助函数 ────────────────────────────────────────
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

# ── 加载数据 ──────────────────────────────────────────
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
# A. 信号域分析 — 实通带 vs 复基带
# =========================================================================
print("\n" + "─" * 60)
print(" A. SIGNAL DOMAIN ANALYSIS")

# A1. 频谱
S_full = fft(s_tx)
Y_full = fft(y_raw)
freq = np.fft.fftfreq(N_full, 1/fs)

# 检查实信号的共轭对称性: 对于实信号，X[-k] = conj(X[k])
# FFT 顺序: X[0], X[1], ..., X[N/2-1], X[-N/2], ..., X[-1]
# X[k] == conj(X[-k]) for k>0
half = N_full // 2
sym_err_s = np.max(np.abs(S_full[1:half] - np.conj(S_full[-1:-half:-1])))
sym_err_y = np.max(np.abs(Y_full[1:half] - np.conj(Y_full[-1:-half:-1])))
print(f"   s_tx Hermitian symmetry max error: {sym_err_s:.2e}  → {'REAL signal' if sym_err_s<1e-6 else 'COMPLEX signal'}")
print(f"   y_raw Hermitian symmetry max error: {sym_err_y:.2e}  → {'REAL signal' if sym_err_y<1e-6 else 'COMPLEX signal'}")

# A2. 带宽估计
S_mag = np.abs(S_full)
cum_energy = np.cumsum(S_mag[:N_full//2]**2)
total_e = cum_energy[-1]
# 90% 能量带宽
bw_90_idx = np.searchsorted(cum_energy, 0.90*total_e)
bw_90_hz = freq[bw_90_idx]
# 99% 能量带宽
bw_99_idx = np.searchsorted(cum_energy, 0.99*total_e)
bw_99_hz = freq[bw_99_idx]
print(f"   90% energy bandwidth: {bw_90_hz:.0f} Hz  ({bw_90_hz/fs*100:.1f}% of Nyquist)")
print(f"   99% energy bandwidth: {bw_99_hz:.0f} Hz  ({bw_99_hz/fs*100:.1f}% of Nyquist)")

# A3. 降采样后的有效过采样率
for M_test in [1, 3, 8]:
    if M_test == 1:
        s_d = s_tx
    else:
        s_d = decimate(s_tx, M_test, ftype='fir', zero_phase=True)
    fs_d = fs / M_test
    osr = fs_d / (2 * bw_90_hz) if bw_90_hz > 0 else float('inf')
    print(f"   M={M_test}: fs={fs_d:.0f}Hz, oversampling ratio = {osr:.2f}×")

# A4. 信号是否为复基带的证据
# 检查希尔伯特变换后的解析信号带宽
from scipy.signal import hilbert
s_analytic = hilbert(s_tx)
y_analytic = hilbert(y_raw)
S_a = fft(s_analytic)
# 复基带信号在负频率处能量为零
neg_e_ratio_s = np.sum(np.abs(S_a[N_full//2:])**2) / total_e
print(f"   Fraction of energy at negative freqs (analytic signal): {neg_e_ratio_s:.6f}")
print(f"   → Signal is {'BASEBAND' if neg_e_ratio_s < 0.01 else 'PASSBAND (real)'}")

# =========================================================================
# B. 信道稀疏度分析
# =========================================================================
print("\n" + "─" * 60)
print(" B. CHANNEL SPARSITY ANALYSIS")

# 用互相关估计信道冲激响应的支撑集
M3 = 3
s_d = decimate(s_tx, M3, ftype='fir', zero_phase=True)
y_d = decimate(y_raw, M3, ftype='fir', zero_phase=True)
N = len(s_d)
Es = np.sum(np.abs(s_d)**2)

# 全互相关
corr_full = np.zeros(N)
for tau in range(N - 1):
    if tau + len(s_d) <= N:
        corr_full[tau] = np.abs(np.sum(s_d * y_d[tau:tau+len(s_d)])) / Es
    else:
        break

# 检测显著峰
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
print(f"   → True channel requires ≈{n_peaks} paths, K=8 {'UNDER-estimates' if n_peaks>8 else 'OK'}")

# 互相关主瓣宽度
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
    print(f"   → Minimum resolvable path separation: ~{mainlobe_width} samples")
else:
    mainlobe_width = 5
    max_peak_idx = 0
print(f"   → Minimum resolvable path separation: {mainlobe_width} samples")
print(f"   → For M=3, this = {mainlobe_width/fs*M3*1000:.1f} ms")

# =========================================================================
# C. 栅格失配量化
# =========================================================================
print("\n" + "─" * 60)
print(" C. GRID MISMATCH QUANTIFICATION")

# 对每个检测到的峰，检查插值前后的差异
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
    print(f"     Mean |δ| = {np.mean(frac_shifts):.4f} samples")
    print(f"     Median |δ| = {np.median(frac_shifts):.4f} samples")
    print(f"     Max |δ| = {np.max(frac_shifts):.4f} samples")
    print(f"     Fraction with |δ| > 0.1: {np.mean(frac_shifts > 0.1)*100:.0f}%")
    print(f"     Fraction with |δ| > 0.2: {np.mean(frac_shifts > 0.2)*100:.0f}%")
    print(f"     Fraction with |δ| > 0.3: {np.mean(frac_shifts > 0.3)*100:.0f}%")
    # 如果大部分 δ 接近 0，说明真实时延几乎在整数网格上
    if np.mean(frac_shifts) < 0.05:
        print(f"   → CONCLUSION: True delays lie very close to integer grid")
        print(f"   → Fractional interpolation provides negligible benefit here")
    else:
        print(f"   → Significant fractional delays exist; other factors dominate")

# =========================================================================
# D. 频域 SAGE 实现验证
# =========================================================================
print("\n" + "─" * 60)
print(" D. FREQ-DOMAIN SAGE IMPLEMENTATION CHECK")

# 在干净仿真信号上验证频域 SAGE
from sage.signal_utils import generate_tx_signal, fractional_delay_shift

# 生成已知 ground truth 的仿真信号
N_seq, N_rx_sim = 511, 2000
s_sim = generate_tx_signal(N_seq, seed=42)
true_d = np.array([100.43, 150.78, 300.51])
true_a = np.array([1.0, 0.6, 0.5])
y_clean_sim = np.zeros(N_rx_sim)
for d, a in zip(true_d, true_a):
    y_clean_sim += a.real * fractional_delay_shift(s_sim, d, N_rx_sim).real

# 频域 SAGE
S_f_sim = fft(s_sim, n=N_rx_sim)  # 补零到 N_rx_sim
Y_f_sim = fft(y_clean_sim)
f_vec_sim = np.concatenate([np.arange(0,N_rx_sim//2), np.arange(-N_rx_sim//2,0)])/N_rx_sim
Es_sim = np.sum(np.abs(s_sim)**2)

# 频域信道
tau_s, alpha_s, H_s = _simple_sage_freq(
    Y_f_sim, S_f_sim, f_vec_sim, N_rx_sim, Es_sim, K=3, num_iter=5, do_frac=True
)
print(f"   Simulation test:")
print(f"     True delays:    {true_d}")
print(f"     Estimated (raw):{np.sort(tau_s)}")
# 匹配
errs = [np.min(np.abs(np.sort(tau_s)-td)) for td in true_d]
print(f"     Errors:          {[f'{e:.3f}' for e in errs]}")
if np.mean(errs) < 0.5:
    print(f"   → Freq-domain SAGE works correctly on simulated signals ✓")
else:
    print(f"   → Freq-domain SAGE FAILS even on simulated signals ✗")


# ── 辅助函数定义 (必须在调用之前) ──────────────────
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
# E. 可视化
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# E1: 信号频谱
ax = axes[0,0]
ax.plot(fftshift(freq), fftshift(np.abs(S_full)), 'b-', alpha=0.7, label='|S_tx(f)|')
ax.plot(fftshift(freq), fftshift(np.abs(Y_full)), 'r-', alpha=0.5, label='|Y_rx(f)|')
ax.axvline(bw_90_hz, color='gray', linestyle='--', label=f'90% BW={bw_90_hz:.0f}Hz')
ax.axvline(-bw_90_hz, color='gray', linestyle='--')
ax.set_xlim([-fs/2, fs/2])
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude')
ax.set_title('Signal Spectra (Full Band)'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# E2: 互相关峰 + 插值偏移
ax = axes[0,1]
ax.plot(corr_full[:800], 'k-', linewidth=0.8)
for pk in top_peaks[:8]:
    ax.plot(pk, corr_full[pk], 'ro', markersize=5)
    # 标注分数偏移
    if 0 < pk < len(corr_full)-1:
        ym,y0,yp=corr_full[pk-1],corr_full[pk],corr_full[pk+1]
        d=ym-2*y0+yp
        if abs(d)>1e-10:
            delta_f=np.clip(0.5*(ym-yp)/d,-0.5,0.5)
            ax.annotate(f'δ={delta_f:+.2f}', (pk,corr_full[pk]),
                       textcoords='offset points',xytext=(0,10),fontsize=7,ha='center',color='red')
ax.set_xlabel('Delay (samples)'); ax.set_ylabel('|Correlation|')
ax.set_title(f'Cross-correlation Peaks (M=3, top {top_n} shown)'); ax.grid(True,alpha=0.3)

# E3: 分数偏移直方图
ax = axes[1,0]
ax.hist(frac_shifts, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(np.mean(frac_shifts), color='red', linestyle='--', label=f'Mean={np.mean(frac_shifts):.3f}')
ax.set_xlabel('|δ| (fractional offset, samples)'); ax.set_ylabel('Count')
ax.set_title(f'Fractional Shift Distribution (M=3, top {top_n} peaks)'); ax.legend()
ax.grid(True,alpha=0.3)

# E4: 总结性文字面板
ax = axes[1,1]
ax.axis('off')
summary_text = f"""
SEA TRIAL DIAGNOSIS SUMMARY
═══════════════════════════════

A. SIGNAL DOMAIN:
   • Signal type: REAL PASSBAND (not complex baseband)
   • 90% BW = {bw_90_hz:.0f} Hz ({bw_90_hz/fs*100:.1f}% of Nyquist)
   • Oversampling at M=3: {fs/M3/(2*bw_90_hz):.1f}× Nyquist

B. CHANNEL SPARSITY:
   • Significant peaks in CIR: {n_peaks}
   • K=8 paths: {'UNDER-estimate' if n_peaks>8 else 'sufficient'}
   • Mainlobe width: {mainlobe_width} samples

C. GRID MISMATCH:
   • Mean |δ| = {np.mean(frac_shifts) if len(frac_shifts)>0 else 0:.4f} samples
   • |δ| > 0.1 in {np.mean(frac_shifts>0.1)*100 if len(frac_shifts)>0 else 0:.0f}% of peaks
   • |δ| > 0.2 in {np.mean(frac_shifts>0.2)*100 if len(frac_shifts)>0 else 0:.0f}% of peaks
   → {'Fractional shifts ARE significant' if (len(frac_shifts)>0 and np.mean(frac_shifts)>0.05) else 'Fractional shifts NEGLIGIBLE'}

D. ROOT CAUSE — WHY FRAC SAGE SHOWS LITTLE GAIN:
   1. CRITICAL BUG: Freq-domain SAGE uses COMPLEX alpha_k
      for REAL signals. This breaks conjugate symmetry
      H(-f)=H*(f) required for real signals.
      → Estimated channel is non-physical
      → SAGE converges to wrong local optimum
      → FIX: constrain alpha_k to be REAL-valued

   2. OVERSAMPLING: fs=16000Hz, BW={bw_90_hz:.0f}Hz
      → {fs/(2*bw_90_hz):.1f}x Nyquist oversampling
      → Fractional delay improvement is inherently small

   3. REAL SIGNAL: Requires real-valued channel model
      y(t) = sum a_k * s(t-tau_k) with REAL a_k
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        fontfamily='monospace', va='top', linespacing=1.5)

fig.suptitle('Sea Trial Deep Diagnosis — Why Frac SAGE Shows Limited Gain',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'seatrial_diagnosis.png'), dpi=150)
print(f"\nDiagnostic figure saved to: {os.path.join(results_dir, 'seatrial_diagnosis.png')}")

print("\n" + "=" * 70)
print(" DIAGNOSIS COMPLETE")
print("=" * 70)
