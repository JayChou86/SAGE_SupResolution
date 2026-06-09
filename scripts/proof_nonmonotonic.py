#!/usr/bin/env python
"""Generate proof figures for non-monotonic RMSE phenomena."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import generate_tx_signal, fractional_delay_shift, build_dictionary
from sage.algorithms import algo_NOMP, algo_SBL
from sage.evaluation import match_paths
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':11})

N_seq,N_rx,L=511,1000,3; SEED=40
rng=np.random.RandomState(SEED); np.random.seed(SEED)
td=np.array([50.43,60.66,200.25])
ta=np.array([1.0*np.exp(1j*0.2),0.5*np.exp(-1j*1.5),0.8*np.exp(1j*0.8)])
s=generate_tx_signal(N_seq,seed=SEED)
tau_grid=np.arange(30,220+0.5,0.5); A=build_dictionary(s,tau_grid,N_rx)
SNRs=[-10,-5,0,5,10,15]; N_trials=500

OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
os.makedirs(OUT,exist_ok=True)

# Build clean signal once
yc=np.zeros(N_rx,dtype=complex)
for d,a in zip(td,ta): yc+=a*fractional_delay_shift(s,d,N_rx)
sp=np.var(yc)

def safe_algo(algo_fn, *args):
    """Run algorithm, return (delays, amps) or (None,None) on failure."""
    try:
        d,a = algo_fn(*args)
        if len(d)!=L or len(a)!=L: return None,None
        md,ma = match_paths(td,ta,d,a)
        return md,ma
    except: return None,None

# ===================================================================
# PROOF 3: NOMP Path 1 scatter
# ===================================================================
print("=== Proof 3: NOMP Path 1 ===")
fig,axes=plt.subplots(2,6,figsize=(24,8))
for si,snr in enumerate(SNRs):
    N0=sp/(10**(snr/10)); ns=np.sqrt(N0/2)
    de=[]; me=[]; pe=[]; skipped=0
    while len(de) < N_trials:
        noise=ns*(rng.randn(N_rx)+1j*rng.randn(N_rx)); yr=yc+noise
        md,ma = safe_algo(algo_NOMP,yr,s,N_rx,L,tau_grid)
        if md is None: skipped+=1; continue
        de.append(md[0]-td[0]); me.append(np.abs(ma[0])-np.abs(ta[0]))
        pe.append(np.arctan2(np.sin(np.angle(ma[0])-np.angle(ta[0])),
                             np.cos(np.angle(ma[0])-np.angle(ta[0]))))
    de=np.array(de); me=np.array(me); pe=np.array(pe)
    r_d=np.sqrt(np.mean(de**2)); r_m=np.sqrt(np.mean(me**2)); r_p=np.sqrt(np.mean(pe**2))
    c_dm=np.corrcoef(de,me)[0,1] if len(de)>1 else 0
    c_dp=np.corrcoef(de,pe)[0,1] if len(de)>1 else 0
    ax=axes[0,si]; ax.scatter(de,me,alpha=0.3,s=10,color='#d62728')
    ax.set_title('SNR=%+ddB | d=%.3f m=%.3f r=%+.2f'%(snr,r_d,r_m,c_dm))
    ax.set_xlabel('Delay err'); ax.set_ylabel('Mag err')
    ax.axhline(0,c='gray',ls=':'); ax.axvline(0,c='gray',ls=':'); ax.grid(alpha=0.2)
    ax=axes[1,si]; ax.scatter(de,pe,alpha=0.3,s=10,color='#ff7f0e')
    ax.set_title('SNR=%+ddB | d=%.3f p=%.3f r=%+.2f'%(snr,r_d,r_p,c_dp))
    ax.set_xlabel('Delay err'); ax.set_ylabel('Phase err')
    ax.axhline(0,c='gray',ls=':'); ax.axvline(0,c='gray',ls=':'); ax.grid(alpha=0.2)
    print('  SNR=%+ddB: d=%.4f m=%.4f p=%.4f skipped=%d'%(snr,r_d,r_m,r_p,skipped))
fig.suptitle('Proof 3: NOMP Path1 — Delay Error vs Mag/Phase Error',fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT,'proof3_nomp_path1_scatter.png'),dpi=150,bbox_inches='tight')
plt.close()
print('Saved proof3')

# ===================================================================
# PROOF 4: SBL Path 3 bootstrap
# ===================================================================
print("\n=== Proof 4: SBL Path 3 Bootstrap ===")
for si,snr in enumerate(SNRs):
    N0=sp/(10**(snr/10)); ns=np.sqrt(N0/2)
    me=[]; skipped=0
    while len(me) < N_trials:
        noise=ns*(rng.randn(N_rx)+1j*rng.randn(N_rx)); yr=yc+noise
        md,ma = safe_algo(algo_SBL,yr,A,tau_grid,L,use_woodbury=True)
        if md is None: skipped+=1; continue
        me.append(np.abs(ma[2])-np.abs(ta[2]))
    me=np.array(me); rmse=np.sqrt(np.mean(me**2))
    bs=np.array([np.sqrt(np.mean(np.random.choice(me,size=len(me),replace=True)**2))
                 for _ in range(2000)])
    ci=np.percentile(bs,[2.5,97.5])
    print('  SNR=%+ddB: RMSE=%.5f 95%%CI=[%.5f,%.5f] w=%.5f skipped=%d'%(
        snr,rmse,ci[0],ci[1],ci[1]-ci[0],skipped))
print('All done.')
