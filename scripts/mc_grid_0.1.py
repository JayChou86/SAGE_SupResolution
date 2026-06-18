#!/usr/bin/env python
"""Full MC: grid=0.1 (1901 atoms), 200 trials, all 6 algos + timing."""
import numpy as np, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import generate_tx_signal, fractional_delay_shift, build_dictionary
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths

SNR_dB_vec=np.arange(-15,20,5); GRID=0.1; MC=200; SEED=40
rng=np.random.RandomState(SEED); np.random.seed(SEED)
N_seq,N_rx,L=511,1000,3
td=np.array([50.43,60.66,200.25])
ta=np.array([1.0*np.exp(1j*0.2),0.5*np.exp(-1j*1.5),0.8*np.exp(1j*0.8)])
an=['IntSAGE','FracSAGE','MF','OMP','NOMP','SBL']; nA=len(an)

s=generate_tx_signal(N_seq,seed=SEED)
tau_grid=np.arange(30,220+GRID,GRID); N_grid=len(tau_grid)
print(f'Grid={GRID}: {N_grid} atoms ({tau_grid[0]:.1f}-{tau_grid[-1]:.1f})')
A=build_dictionary(s,tau_grid,N_rx)

yc=np.zeros(N_rx,dtype=complex)
for d,a in zip(td,ta): yc+=a*fractional_delay_shift(s,d,N_rx)
sp=np.var(yc)

OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results','grid_0.1')
os.makedirs(OUT,exist_ok=True)

nS=len(SNR_dB_vec); rmse_d=np.zeros((nS,nA)); per_d=np.zeros((nS,nA,L))
t_acc=np.zeros(nA); t_calls=np.zeros(nA)  # timing accumulators

for si,snr in enumerate(SNR_dB_vec):
    N0=sp/(10**(snr/10)); ns=np.sqrt(N0/2)
    sq=np.zeros((nA,L)); valid=0
    for mc in range(MC):
        noise=ns*(rng.randn(N_rx)+1j*rng.randn(N_rx)); yr=yc+noise
        try:
            t0=time.perf_counter(); di,ai=algo_SAGE(yr,s,N_rx,L,do_frac=False); t_acc[0]+=time.perf_counter()-t0; t_calls[0]+=1
            t0=time.perf_counter(); df,af=algo_SAGE(yr,s,N_rx,L,do_frac=True);  t_acc[1]+=time.perf_counter()-t0; t_calls[1]+=1
            t0=time.perf_counter(); dm,am=algo_MF(yr,A,tau_grid,L);              t_acc[2]+=time.perf_counter()-t0; t_calls[2]+=1
            t0=time.perf_counter(); do,ao=algo_OMP(yr,A,tau_grid,L);              t_acc[3]+=time.perf_counter()-t0; t_calls[3]+=1
            t0=time.perf_counter(); dn,an_=algo_NOMP(yr,s,N_rx,L,tau_grid);       t_acc[4]+=time.perf_counter()-t0; t_calls[4]+=1
            t0=time.perf_counter(); ds,as_=algo_SBL(yr,A,tau_grid,L,use_woodbury=True); t_acc[5]+=time.perf_counter()-t0; t_calls[5]+=1
        except Exception as e:
            if mc==0: print(f'  ERROR@snr={snr}: {e}')
            continue
        ed=[di,df,dm,do,dn,ds]; ea=[ai,af,am,ao,an_,as_]
        for a in range(nA):
            md,ma=match_paths(td,ta,ed[a],ea[a]); sq[a,:]+=(md-td)**2
        valid+=1
    rmse_d[si,:]=np.sqrt(np.mean(sq,axis=1)/valid)
    per_d[si,:,:]=np.sqrt(sq/valid)
    print(f'SNR={snr:4d}dB: Int={rmse_d[si,0]:.4f} Frac={rmse_d[si,1]:.4f} MF={rmse_d[si,2]:.4f} OMP={rmse_d[si,3]:.4f} NOMP={rmse_d[si,4]:.4f} SBL={rmse_d[si,5]:.4f} [{valid}t]')

# Timing summary
print(f'\n=== TIMING (ms/run, grid={GRID}, {N_grid} atoms) ===')
for a in range(nA):
    t_ms = t_acc[a]/max(t_calls[a],1)*1000
    print(f'  {an[a]:<10s} {t_ms:>10.1f} ms ({int(t_calls[a])} calls)')

np.savez(os.path.join(OUT,'mc_results.npz'),
    SNR_dB_vec=SNR_dB_vec, grid=GRID, MC=MC,
    rmse_delay=rmse_d, per_path_delay=per_d, algo_names=an, true_delays=td)
print(f'\nSaved to {OUT}/mc_results.npz')

# Cross-grid summary @ 0 dB
mid=list(SNR_dB_vec).index(0)
print(f'\n=== GRID=0.1 vs GRID=0.5 @ 0 dB ===')
print(f'{"Algo":<10s} {"grid=0.1":>10s} {"grid=0.5":>10s} {"ratio":>8s}')
for a,rmse_05 in zip(range(nA),[2.890,0.0251,0.1865,0.1778,0.4137,0.1771]):
    ratio=rmse_d[mid,a]/rmse_05 if rmse_05>0 else 0
    print(f'{an[a]:<10s} {rmse_d[mid,a]:>10.4f} {rmse_05:>10.4f} {ratio:>8.2f}x')
