#!/usr/bin/env python
"""Watermark MC validation — all M, 100 noise trials per snapshot."""
import numpy as np, scipy.io as sio, os, sys, time
from scipy.signal import find_peaks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import generate_tx_signal, fractional_delay_shift, build_dictionary
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times'],'font.size':11})

# Sub-panel label utility — adds (a),(b),(c) to top-left of axes
def _add_sublabels(axes, ox=-0.08, oy=1.02):
    for i, ax in enumerate(axes):
        ax.text(ox, oy, '(%s)' % chr(ord('a')+i), transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='left', va='bottom')

CH='NOF1'; MAT=f'D:/DataSet/Watermark/input/channels/{CH}/mat/{CH}_025.mat'
d=sio.loadmat(MAT); h_all=d['h']; fs_tau=float(d['fs_tau'][0,0])
SEED=42; rng=np.random.RandomState(SEED); np.random.seed(SEED)
N_seq=511; s_tx=generate_tx_signal(N_seq,seed=SEED); L=3
MC=100; SNRs=[-15,-10,-5,0,5,10,15]
an=['Int SAGE','CD-SAGE','MF','OMP','NOMP','SBL']; nA=len(an)
colors=['#333333','#1f77b4','#2ca02c','#d62728','#ff7f0e','#17becf']
markers=['s','^','*','d','v','x']; ls=['-','-','--','--','-.','-']

import time as _time
# Per-algorithm timing accumulators (ms)
_time_acc = np.zeros(6)

# Find good snapshots
snaps=[]
for snap in range(0,200,3):
    cir=np.abs(h_all[:,snap]); m=np.max(cir)
    if m<1: continue
    peaks,_=find_peaks(cir,distance=15,height=0.1*m)
    if len(peaks)>=3:
        top=np.sort(peaks[np.argsort(cir[peaks])[-L:]])
        if top[1]-top[0]>=10 and top[2]-top[1]>=10:
            snaps.append(snap)
            if len(snaps)>=8: break
print(f"Selected {len(snaps)} snapshots: {snaps}")

OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results','watermark')
os.makedirs(OUT,exist_ok=True)

for M in [2,4,8]:
    fs_target=fs_tau/M; t0=time.time()
    results=np.zeros((len(SNRs),nA))
    for si,snr in enumerate(SNRs):
        N0=1.0/(10**(snr/10)); ns=np.sqrt(N0/2); sq=np.zeros(nA); total=0
        for snap in snaps:
            cir=np.abs(h_all[:,snap])
            peaks,_=find_peaks(cir,distance=15,height=0.1*np.max(cir))
            top=peaks[np.argsort(cir[peaks])[-L:]]
            tau_true=top/M; amp_true=h_all[top,snap]
            N_rx=int(np.ceil(np.max(tau_true)))+N_seq+50
            tg=np.arange(0,N_rx-N_seq+0.5,0.5); A=build_dictionary(s_tx,tg,N_rx)
            for _ in range(MC):
                y=np.zeros(N_rx,dtype=complex)
                for td,ta in zip(tau_true,amp_true): y+=ta*fractional_delay_shift(s_tx,td,N_rx)
                sp=np.var(y)
                if sp<1e-20: continue
                scl=1/np.sqrt(sp); y*=scl; tas=amp_true*scl
                y+=ns*(rng.randn(N_rx)+1j*rng.randn(N_rx))
                try:
                    t0=_time.time(); di,ai=algo_SAGE(y,s_tx,N_rx,L,do_frac=False); _time_acc[0]+=_time.time()-t0
                    t0=_time.time(); df,af=algo_SAGE(y,s_tx,N_rx,L,do_frac=True);  _time_acc[1]+=_time.time()-t0
                    t0=_time.time(); dm,am=algo_MF(y,A,tg,L);                    _time_acc[2]+=_time.time()-t0
                    t0=_time.time(); do,ao=algo_OMP(y,A,tg,L);                    _time_acc[3]+=_time.time()-t0
                    t0=_time.time(); dn,an_=algo_NOMP(y,s_tx,N_rx,L,tg);         _time_acc[4]+=_time.time()-t0
                    t0=_time.time(); ds,as_=algo_SBL(y,A,tg,L,use_woodbury=True); _time_acc[5]+=_time.time()-t0
                except: continue
                for a,(ed,ea) in enumerate(zip([di,df,dm,do,dn,ds],[ai,af,am,ao,an_,as_])):
                    md,ma=match_paths(tau_true,tas,ed,ea)
                    sq[a]+=np.mean((md-tau_true)**2)
                total+=1
        results[si,:]=np.sqrt(sq/max(total,1))
        impr=(results[si,0]-results[si,1])/max(results[si,0],1e-10)*100
        print(f'  M={M} SNR={snr:3d}dB: Frac={results[si,1]:.4f} MF={results[si,2]:.4f} OMP={results[si,3]:.4f} Impr={impr:.1f}% | {total}t')
    total_calls = len(SNRs)*len(snaps)*MC
    print(f'  M={M} done in {_time.time()-t0:.0f}s ({total_calls} calls)')
    print(f'  Avg time/run (ms): IntSAGE={_time_acc[0]/total_calls*1e3:.1f} FracSAGE={_time_acc[1]/total_calls*1e3:.1f} MF={_time_acc[2]/total_calls*1e3:.1f} OMP={_time_acc[3]/total_calls*1e3:.1f} NOMP={_time_acc[4]/total_calls*1e3:.1f} SBL={_time_acc[5]/total_calls*1e3:.1f}')
    _time_acc[:]=0
    np.savez(f'{OUT}/mc_NOF1_M{M}.npz',SNR=SNRs,rmse=results,algo_names=an,M=M,fs_target=fs_target)

# Generate figure — 2 panels for paper (M=4, M=8)
fig,axes=plt.subplots(1,2,figsize=(12,5))
_add_sublabels(axes)
for mi,M in enumerate([4,8]):
    d=np.load(f'{OUT}/mc_NOF1_M{M}.npz')
    ax=axes[mi]
    for a in range(nA):
        ax.semilogy(SNRs,d['rmse'][:,a],color=colors[a],marker=markers[a],
                   linestyle=ls[a],lw=1.8,ms=7,label=an[a],markerfacecolor='none')
    ax.set_title(f'M={M} ($f_s$={fs_tau/M:.0f} Hz)',fontsize=12)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Delay RMSE (ms)')
    ax.legend(fontsize=7,loc='best'); ax.grid(alpha=0.3,which='both')
fig.tight_layout()
fig.savefig(f'{OUT}/mc_NOF1_paper.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"\nSaved {OUT}/mc_NOF1_paper.png")

# Summary table
print(f"\n{'='*70}")
print(" MC SUMMARY @ 0 dB")
print(f"{'='*70}")
print(f"{'M':>4s} {'fs':>6s}",end='')
for a in range(nA): print(f" {an[a][:5]:>8s}",end='')
print(f" {'Impr%':>7s}")
for M in [2,4,8]:
    d=np.load(f'{OUT}/mc_NOF1_M{M}.npz')
    mid=list(d['SNR']).index(0)
    impr=(d['rmse'][mid,0]-d['rmse'][mid,1])/max(d['rmse'][mid,0],1e-10)*100
    print(f"{M:>4d} {fs_tau/M:>6.0f}",end='')
    for a in range(nA): print(f" {d['rmse'][mid,a]:>8.4f}",end='')
    print(f" {impr:>6.1f}%")
