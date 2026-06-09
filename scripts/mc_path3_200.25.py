#!/usr/bin/env python
"""Path 3 τ=200.25 (worst-case grid offset), 200 MC, grid=0.5"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import (generate_tx_signal, compute_crb, build_dictionary, fractional_delay_shift)
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths

SNR_dB_vec = np.arange(-15, 20, 5); GRID_RES = 0.5; NUM_MC = 200; SEED = 40
rng = np.random.RandomState(SEED); np.random.seed(SEED)
N_seq = 511; N_rx = 1000; L_paths = 3
true_delays = np.array([50.43, 60.66, 200.25])   # Path 3: worst-case grid offset
true_amps = np.array([1.0*np.exp(1j*0.2), 0.5*np.exp(-1j*1.5), 0.8*np.exp(1j*0.8)])
algo_names = ['IntSAGE','FracSAGE','MF','OMP','NOMP','SBL']

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'path3_200.25')
os.makedirs(OUT, exist_ok=True)

s_tx = generate_tx_signal(N_seq, seed=SEED)
y_clean = np.zeros(N_rx, dtype=complex)
for d, a in zip(true_delays, true_amps): y_clean += a * fractional_delay_shift(s_tx, d, N_rx)
sp = np.var(y_clean)

tau_grid = np.arange(30, 220+GRID_RES, GRID_RES)
print(f"Dict: {len(tau_grid)} atoms, grid={GRID_RES}")
A_dict = build_dictionary(s_tx, tau_grid, N_rx)

S_freq = np.fft.fft(s_tx)
kv = np.concatenate([np.arange(0,int(np.ceil(N_seq/2))), np.arange(-int(np.floor(N_seq/2)),0)])
omega = 2*np.pi*kv/N_seq; s_prime = np.fft.ifft((1j*omega)*S_freq)
esp = np.sum(np.abs(s_prime)**2)

nS = len(SNR_dB_vec); nA = len(algo_names)
rmse_d = np.zeros((nS, nA)); rmse_m = np.zeros((nS, nA)); rmse_p = np.zeros((nS, nA))
per_d = np.zeros((nS, nA, L_paths)); per_m = np.zeros((nS, nA, L_paths)); per_p = np.zeros((nS, nA, L_paths))
crb_d_arr = np.zeros((nS, L_paths)); crb_m_arr = np.zeros((nS, L_paths)); crb_p_arr = np.zeros((nS, L_paths))

t0 = time.time()
for si, snr in enumerate(SNR_dB_vec):
    N0 = sp/(10**(snr/10)); ns = np.sqrt(N0/2)
    crb_d_arr[si,:] = np.sqrt(N0/(2*np.abs(true_amps)**2*esp))
    crb_m_arr[si,:] = np.sqrt(N0/2)
    crb_p_arr[si,:] = np.sqrt(N0/(2*np.abs(true_amps)**2*esp)) / np.abs(true_amps)
    sqd = np.zeros((nA, L_paths)); sqm = np.zeros((nA, L_paths)); sqp = np.zeros((nA, L_paths))
    for mc in range(NUM_MC):
        noise = ns*(rng.randn(N_rx)+1j*rng.randn(N_rx)); yr = y_clean+noise
        di,ai=algo_SAGE(yr,s_tx,N_rx,L_paths,do_frac=False)
        df,af=algo_SAGE(yr,s_tx,N_rx,L_paths,do_frac=True)
        dm,am=algo_MF(yr,A_dict,tau_grid,L_paths)
        do,ao=algo_OMP(yr,A_dict,tau_grid,L_paths)
        dn,an=algo_NOMP(yr,s_tx,N_rx,L_paths,tau_grid)
        ds,as_=algo_SBL(yr,A_dict,tau_grid,L_paths,use_woodbury=True)
        ed=[di,df,dm,do,dn,ds]; ea=[ai,af,am,ao,an,as_]
        for alg in range(nA):
            md,ma=match_paths(true_delays,true_amps,ed[alg],ea[alg])
            sqd[alg,:]+=(md-true_delays)**2
            sqm[alg,:]+=(np.abs(ma)-np.abs(true_amps))**2
            pe=np.arctan2(np.sin(np.angle(ma)-np.angle(true_amps)),np.cos(np.angle(ma)-np.angle(true_amps)))
            sqp[alg,:]+=pe**2
    per_d[si,:,:]=np.sqrt(sqd/NUM_MC); per_m[si,:,:]=np.sqrt(sqm/NUM_MC); per_p[si,:,:]=np.sqrt(sqp/NUM_MC)
    rmse_d[si,:]=np.sqrt(np.mean(sqd,axis=1)/NUM_MC)
    rmse_m[si,:]=np.sqrt(np.mean(sqm,axis=1)/NUM_MC)
    rmse_p[si,:]=np.sqrt(np.mean(sqp,axis=1)/NUM_MC)
    print(f"  SNR={snr:4d} | Frac delay={rmse_d[si,1]:.5f} | MF={rmse_d[si,2]:.4f}")

print(f"MC done in {time.time()-t0:.1f}s")

# Print Path 3 comparison
print(f"\nPath 3 (τ=200.25) Delay RMSE vs CRB (grid=0.5):")
print(f"{'SNR':>6s} {'CRB':>8s} {'FracSAGE':>10s} {'MF':>10s} {'OMP':>10s} {'SBL':>10s}")
for si,snr in enumerate(SNR_dB_vec):
    print(f"{snr:>5d}dB {crb_d_arr[si,2]:>8.4f} {per_d[si,1,2]:>10.4f} {per_d[si,2,2]:>10.4f} {per_d[si,3,2]:>10.4f} {per_d[si,5,2]:>10.4f}")

np.savez(os.path.join(OUT,'mc_results.npz'),
    SNR_dB_vec=SNR_dB_vec, per_path_delay=per_d, per_path_mag=per_m, per_path_phase=per_p,
    crb_delay=crb_d_arr, crb_mag=crb_m_arr, crb_phase=crb_p_arr,
    rmse_delay=rmse_d, algo_names=algo_names, true_delays=true_delays)
print(f"Saved to {OUT}")
