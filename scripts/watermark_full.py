#!/usr/bin/env python
"""
Watermark benchmark — all 6 algorithms, multi-rate decimation sweep.
"""
import numpy as np, scipy.io as sio, os, sys, time
from scipy.signal import find_peaks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import generate_tx_signal, fractional_delay_shift, build_dictionary
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times'],'font.size':11})

# ===================================================================
CHANNEL = 'NOF1'
MAT_FILE = f'D:/DataSet/Watermark/input/channels/{CHANNEL}/mat/{CHANNEL}_025.mat'
data = sio.loadmat(MAT_FILE)
h_all = data['h']; fs_tau = float(data['fs_tau'][0,0]); fc = int(data['fc'][0,0])

L_PATHS = 3; N_SNAPSHOTS = min(200, h_all.shape[1])
DECIMATIONS = [1, 2, 4, 8]  # 1=original 20kHz, 8=2.5kHz (near Nyquist)
SNRs = [-5, 0, 5, 10, 15]
SEED = 42; N_seq = 511; NUM_ITER = 10

algo_names = ['Integer SAGE','Frac SAGE','MF','OMP','NOMP','SBL']
colors = ['#333333','#1f77b4','#2ca02c','#d62728','#ff7f0e','#17becf']
markers = ['s','^','*','d','v','x']; ls = ['-','-','--','--','-.','-']

rng = np.random.RandomState(SEED); np.random.seed(SEED)
s_tx = generate_tx_signal(N_seq, seed=SEED)
S_f = np.fft.fft(s_tx)
kv = np.concatenate([np.arange(0,int(np.ceil(N_seq/2))), np.arange(-int(np.floor(N_seq/2)),0)])
esp = np.sum(np.abs(np.fft.ifft((1j*2*np.pi*kv/N_seq)*S_f))**2)

print(f"Watermark {CHANNEL}: {h_all.shape} CIR, fs={fs_tau:.0f}Hz, fc={fc}Hz")

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'watermark')
os.makedirs(OUT, exist_ok=True)

# ===================================================================
for M in DECIMATIONS:
    fs_target = fs_tau / M
    print(f"\n{'='*50}\n Decimation M={M} → fs={fs_target:.0f} Hz\n{'='*50}")

    nA = len(algo_names); nS = len(SNRs)
    rmse_all = np.zeros((nS, nA))

    for si, snr in enumerate(SNRs):
        sq = np.zeros(nA); valid = 0
        for snap in range(N_SNAPSHOTS):
            cir = np.abs(h_all[:, snap])
            peaks, _ = find_peaks(cir, distance=15, height=0.1*np.max(cir))
            if len(peaks) < L_PATHS: peaks = np.argsort(cir)[-L_PATHS:]
            top_idx = np.argsort(cir[peaks])[-L_PATHS:]
            peaks = np.sort(peaks[top_idx])
            true_amps = h_all[peaks, snap]

            # True delays at TARGET sampling rate (now fractional!)
            true_delays = peaks / M  # integer at 20kHz → fractional at fs_target

            # Generate signal at target rate
            max_delay_samp = int(np.ceil(np.max(true_delays))) + 1
            N_rx = max_delay_samp + N_seq + 50
            y_rx = np.zeros(N_rx, dtype=complex)
            for d, a in zip(true_delays, true_amps):
                y_rx += a * fractional_delay_shift(s_tx, d, N_rx)

            sp = np.var(y_rx)
            if sp < 1e-20: continue
            scale = 1.0/np.sqrt(sp); y_rx *= scale; amps_scaled = true_amps*scale

            N0 = 1.0/(10**(snr/10)); ns = np.sqrt(N0/2)
            y_rx += ns*(rng.randn(N_rx)+1j*rng.randn(N_rx))

            tau_grid = np.arange(0, N_rx-N_seq+0.5, 0.5)
            A = build_dictionary(s_tx, tau_grid, N_rx)

            try:
                di,ai = algo_SAGE(y_rx,s_tx,N_rx,L_PATHS,do_frac=False)
                df,af = algo_SAGE(y_rx,s_tx,N_rx,L_PATHS,do_frac=True)
                dm,am = algo_MF(y_rx,A,tau_grid,L_PATHS)
                do,ao = algo_OMP(y_rx,A,tau_grid,L_PATHS)
                dn,an = algo_NOMP(y_rx,s_tx,N_rx,L_PATHS,tau_grid)
                ds,as_= algo_SBL(y_rx,A,tau_grid,L_PATHS,use_woodbury=True)
            except: continue

            ed=[di,df,dm,do,dn,ds]; ea=[ai,af,am,ao,an,as_]
            for a in range(nA):
                md,ma = match_paths(true_delays,amps_scaled,ed[a],ea[a])
                sq[a] += np.mean((md-true_delays)**2)
            valid += 1

        rmse_all[si,:] = np.sqrt(sq / max(valid,1))
        crb_m = np.sqrt(np.mean(1.0/(10**(snr/10)) / (2*np.abs(amps_scaled)**2*esp)))
        print(f"  SNR={snr:3d}dB: Int={rmse_all[si,0]:.4f} Frac={rmse_all[si,1]:.4f} MF={rmse_all[si,2]:.4f} OMP={rmse_all[si,3]:.4f} NOMP={rmse_all[si,4]:.4f} SBL={rmse_all[si,5]:.4f} | CRB≈{crb_m:.4f} | {valid} snaps")

    # Plot per-decimation
    fig,ax = plt.subplots(figsize=(8,5))
    for a in range(nA):
        ax.semilogy(SNRs, rmse_all[:,a], color=colors[a], marker=markers[a],
                   linestyle=ls[a], lw=1.8, ms=7, label=algo_names[a], markerfacecolor='none')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Delay RMSE (samples)')
    ax.set_title(f'Watermark {CHANNEL} — M={M} (fs={fs_target:.0f} Hz, {N_SNAPSHOTS} snaps)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT,f'watermark_{CHANNEL}_M{M}_all.png'),dpi=200,bbox_inches='tight')
    plt.close()

    # Save NPZ
    np.savez(os.path.join(OUT,f'watermark_{CHANNEL}_M{M}.npz'), SNR=SNRs, rmse=rmse_all,
             algo_names=algo_names, decimation=M, fs_target=fs_target)

# ===================================================================
# Cross-decimation summary
# ===================================================================
print(f"\n{'='*60}")
print(" CROSS-DECIMATION SUMMARY (0 dB SNR)")
print(f"{'='*60}")
print(f"{'M':>4s} {'fs':>6s}", end='')
for a in range(nA): print(f" {algo_names[a][:6]:>8s}", end='')
print()
for M in DECIMATIONS:
    d = np.load(os.path.join(OUT,f'watermark_{CHANNEL}_M{M}.npz'))
    mid = list(d['SNR']).index(0)
    print(f"{M:>4d} {fs_tau/M:>6.0f}", end='')
    for a in range(nA): print(f" {d['rmse'][mid,a]:>8.4f}", end='')
    print()

print(f"\nSaved to {OUT}/")
