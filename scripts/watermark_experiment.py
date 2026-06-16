#!/usr/bin/env python
"""
Watermark benchmark experiment for SAGE paper.
Uses measured CIRs as ground truth to validate Frac SAGE.
"""
import numpy as np, scipy.io as sio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import generate_tx_signal, fractional_delay_shift
from sage.algorithms import algo_SAGE
from sage.evaluation import match_paths
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times'],'font.size':11})

# ===================================================================
# 1. Load Watermark CIR and extract dominant paths
# ===================================================================
CHANNEL = 'BCH1'
MAT_FILE = f'D:/DataSet/Watermark/input/channels/{CHANNEL}/mat/{CHANNEL}_001.mat'
data = sio.loadmat(MAT_FILE)
h_all = data['h']          # shape (N_delay, N_snapshots)
fs_tau = float(data['fs_tau'][0,0])
fs_t = float(data['fs_t'][0,0])
fc = int(data['fc'][0,0])

# Parameters
L_PATHS = 3               # Dominant paths to track
N_SNAPSHOTS = min(200, h_all.shape[1])
# Use fractional offset to simulate grid mismatch (no decimation needed)
FRAC_STD = 0.3  # Std of random fractional delay offset
NUM_ITER = 10              # SAGE iterations
SEED = 42

print(f"Watermark {CHANNEL}: {h_all.shape[0]} delays, {h_all.shape[1]} snapshots")
print(f"fs_tau={fs_tau:.0f} Hz, fs_t={fs_t:.3f} Hz, fc={fc} Hz")
print(f"Fractional offset std: ±{FRAC_STD} samples (simulates grid mismatch)")

# ===================================================================
# 2. Generate transmit waveform and synthetic received signal
# ===================================================================
N_seq = int(fs_tau / 50)  # ~20ms signal at 20kHz = 400 samples
N_seq = max(256, min(N_seq, 511))
N_seq = 511  # Match paper
s_tx = generate_tx_signal(N_seq, seed=SEED)
Es = np.vdot(s_tx, s_tx).real

# Pre-compute CRB factors
S_f = np.fft.fft(s_tx)
kv = np.concatenate([np.arange(0, int(np.ceil(N_seq/2))),
                      np.arange(-int(np.floor(N_seq/2)), 0)])
omega = 2*np.pi*kv/N_seq
s_prime = np.fft.ifft((1j*omega)*S_f)
esp = np.sum(np.abs(s_prime)**2)

# ===================================================================
# 3. Run SAGE on each snapshot
# ===================================================================
SNRs = [-5, 0, 5, 10, 15]   # dB
results = {'snr': [], 'rmse_int': [], 'rmse_frac': [], 'crb': []}
all_path_errors = []

rng = np.random.RandomState(SEED)
np.random.seed(SEED)

for si, snr in enumerate(SNRs):
    print(f"\nSNR = {snr} dB...")
    sq_int = []; sq_frac = []; sq_crb = 0

    for snap in range(N_SNAPSHOTS):
        # Extract CIR for this snapshot and sparsify to top L_PATHS
        cir = np.abs(h_all[:, snap])
        # Find all local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(cir, distance=5)
        if len(peaks) < L_PATHS:
            peaks = np.argsort(cir)[-L_PATHS:]
        # Take top L_PATHS by peak height
        top_idx = np.argsort(cir[peaks])[-L_PATHS:]
        peaks = np.sort(peaks[top_idx])

        # Ground truth: complex amplitudes at these peaks
        true_delays_raw = peaks.astype(float)
        true_amps_raw = h_all[peaks, snap]

        # SPARSIFY: zero out all other paths, keep only top L_PATHS
        h_sparse = np.zeros_like(h_all[:, snap])
        h_sparse[peaks] = true_amps_raw

        # Generate received signal by convolving s_tx with SPARSE CIR
        N_rx = len(h_sparse) + N_seq - 1
        y_rx = np.convolve(s_tx, h_sparse)

        # Normalize signal power to 1
        sp = np.var(y_rx)
        if sp < 1e-20:
            continue
        scale = 1.0 / np.sqrt(sp)
        y_rx *= scale
        true_amps = true_amps_raw * scale

        # Add random fractional offsets to simulate grid mismatch
        frac_offsets = rng.uniform(-FRAC_STD, FRAC_STD, L_PATHS)
        true_delays = true_delays_raw + frac_offsets

        # Re-generate received signal at the TRUE fractional delays
        y_rx = np.zeros(N_rx, dtype=complex)
        for d, a in zip(true_delays, true_amps):
            y_rx += a * fractional_delay_shift(s_tx, d, N_rx)
        # Re-normalize
        sp = np.var(y_rx)
        scale = 1.0 / np.sqrt(sp)
        y_rx *= scale
        true_amps *= scale

        # Add noise (sp ≈ 1.0 after normalization)
        N0 = 1.0 / (10**(snr/10))
        noise_std = np.sqrt(N0/2)
        noise = noise_std * (rng.randn(N_rx) + 1j*rng.randn(N_rx))
        y_rx += noise

        # Run SAGE
        try:
            d_int, a_int = algo_SAGE(y_rx, s_tx, N_rx, L_PATHS, do_frac=False)
            d_frac, a_frac = algo_SAGE(y_rx, s_tx, N_rx, L_PATHS, do_frac=True)
        except:
            continue

        # Match paths
        md_int, ma_int = match_paths(true_delays, true_amps, d_int, a_int)
        md_frac, ma_frac = match_paths(true_delays, true_amps, d_frac, a_frac)

        # Compute RMSE
        sq_int.append(np.mean((md_int - true_delays)**2))
        sq_frac.append(np.mean((md_frac - true_delays)**2))

        # CRB for this snapshot
        crb_d = N0 / (2 * np.abs(true_amps)**2 * esp)
        sq_crb += np.mean(crb_d)

    rmse_int = np.sqrt(np.mean(sq_int))
    rmse_frac = np.sqrt(np.mean(sq_frac))
    rmse_crb = np.sqrt(sq_crb / N_SNAPSHOTS)

    results['snr'].append(snr)
    results['rmse_int'].append(rmse_int)
    results['rmse_frac'].append(rmse_frac)
    results['crb'].append(rmse_crb)

    print(f"  Int SAGE: {rmse_int:.4f}  Frac SAGE: {rmse_frac:.4f}  CRB: {rmse_crb:.4f}  Ratio: {rmse_frac/rmse_crb:.2f}x")

# ===================================================================
# 4. Plot and save
# ===================================================================
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'watermark')
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,5))
ax.semilogy(results['snr'], results['rmse_int'], 'ks-', lw=2, ms=8, label='Integer SAGE')
ax.semilogy(results['snr'], results['rmse_frac'], 'b^-', lw=2, ms=8, label='Fractional SAGE')
ax.semilogy(results['snr'], results['crb'], 'r-', lw=2, label='CRB')
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Delay RMSE (decimated samples)')
ax.set_title(f'Watermark {CHANNEL} — {N_SNAPSHOTS} snapshots, frac ±{FRAC_STD}samp, {L_PATHS} paths')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(os.path.join(OUT, f'watermark_{CHANNEL}_rmse.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved to {OUT}/watermark_{CHANNEL}_rmse.png")

# Print summary
print(f"\n{'='*60}")
print(f" Watermark {CHANNEL} Summary ({N_SNAPSHOTS} snapshots, frac ±{FRAC_STD}samp)")
print(f"{'='*60}")
for i, snr in enumerate(results['snr']):
    impr = (results['rmse_int'][i]-results['rmse_frac'][i])/results['rmse_int'][i]*100
    print(f" SNR={snr:3d}dB | Int={results['rmse_int'][i]:.4f} Frac={results['rmse_frac'][i]:.4f} CRB={results['crb'][i]:.4f} | Impr={impr:.1f}% | Eff={results['rmse_frac'][i]/results['crb'][i]:.2f}x")
