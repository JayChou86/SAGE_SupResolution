#!/usr/bin/env python
"""500 MC, 3 paths, τ₃=200.25, grid=0.5 → results/500MC/"""
import sys, os, time, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.signal_utils import (generate_tx_signal, build_dictionary, fractional_delay_shift)
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL
from sage.evaluation import match_paths

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times'],'font.size':11,'mathtext.fontset':'stix'})

# Sub-panel label utility — adds (a),(b),(c) to top-left of axes
def _add_sublabels(axes, ox=-0.08, oy=1.02):
    for i, ax in enumerate(axes):
        ax.text(ox, oy, '(%s)' % chr(ord('a')+i), transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='left', va='bottom')

SNR_dB_vec=np.arange(-15,20,5); GRID_RES=0.5; NUM_MC=500; SEED=40
rng=np.random.RandomState(SEED); np.random.seed(SEED)
N_seq,N_rx,L=511,1000,3
td=np.array([50.43,60.66,200.25])
ta=np.array([1.0*np.exp(1j*0.2),0.5*np.exp(-1j*1.5),0.8*np.exp(1j*0.8)])
an=['IntSAGE','CD-SAGE','MF','OMP','NOMP','SBL']
colors=['#333333','#1f77b4','#2ca02c','#d62728','#ff7f0e','#17becf']
markers=['s','^','*','d','v','x']; ls=['-','-','--','--','-.','-']
path_labels=[f'Path 1 (τ={td[0]:.1f})',f'Path 2 (τ={td[1]:.1f})',f'Path 3 (τ={td[2]:.1f})']

OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results','500MC')
os.makedirs(OUT,exist_ok=True)

s=generate_tx_signal(N_seq,seed=SEED)
yc=np.zeros(N_rx,dtype=complex)
for d,a in zip(td,ta): yc+=a*fractional_delay_shift(s,d,N_rx)
sp=np.var(yc)

tau_grid=np.arange(30,220+GRID_RES,GRID_RES)
print(f"Dict: {len(tau_grid)} atoms, grid={GRID_RES}, MC={NUM_MC}")
A_dict=build_dictionary(s,tau_grid,N_rx)

S_f=np.fft.fft(s)
kv=np.concatenate([np.arange(0,int(np.ceil(N_seq/2))),np.arange(-int(np.floor(N_seq/2)),0)])
omega=2*np.pi*kv/N_seq; s_prime=np.fft.ifft((1j*omega)*S_f)
esp=np.sum(np.abs(s_prime)**2)

nS=len(SNR_dB_vec); nA=len(an)
rmse_d=np.zeros((nS,nA)); rmse_m=np.zeros((nS,nA)); rmse_p=np.zeros((nS,nA))
per_d=np.zeros((nS,nA,L)); per_m=np.zeros((nS,nA,L)); per_p=np.zeros((nS,nA,L))
crb_d_arr=np.zeros((nS,L)); crb_m_arr=np.zeros((nS,L)); crb_p_arr=np.zeros((nS,L))

t0=time.time()
for si,snr in enumerate(SNR_dB_vec):
    N0=sp/(10**(snr/10)); ns=np.sqrt(N0/2)
    crb_d_arr[si,:]=np.sqrt(N0/(2*np.abs(ta)**2*esp))
    crb_m_arr[si,:]=np.sqrt(N0/2)
    crb_p_arr[si,:]=np.sqrt(N0/(2*np.abs(ta)**2*esp))/np.abs(ta)
    sqd=np.zeros((nA,L)); sqm=np.zeros((nA,L)); sqp=np.zeros((nA,L))
    for mc in range(NUM_MC):
        noise=ns*(rng.randn(N_rx)+1j*rng.randn(N_rx)); yr=yc+noise
        di,ai=algo_SAGE(yr,s,N_rx,L,do_frac=False)
        df,af=algo_SAGE(yr,s,N_rx,L,do_frac=True)
        dm,am=algo_MF(yr,A_dict,tau_grid,L)
        do,ao=algo_OMP(yr,A_dict,tau_grid,L)
        dn,an_=algo_NOMP(yr,s,N_rx,L,tau_grid)
        ds,as_=algo_SBL(yr,A_dict,tau_grid,L,use_woodbury=True)
        ed=[di,df,dm,do,dn,ds]; ea=[ai,af,am,ao,an_,as_]
        for alg in range(nA):
            md,ma=match_paths(td,ta,ed[alg],ea[alg])
            sqd[alg,:]+=(md-td)**2; sqm[alg,:]+=(np.abs(ma)-np.abs(ta))**2
            pe=np.arctan2(np.sin(np.angle(ma)-np.angle(ta)),np.cos(np.angle(ma)-np.angle(ta)))
            sqp[alg,:]+=pe**2
    per_d[si,:,:]=np.sqrt(sqd/NUM_MC); per_m[si,:,:]=np.sqrt(sqm/NUM_MC); per_p[si,:,:]=np.sqrt(sqp/NUM_MC)
    rmse_d[si,:]=np.sqrt(np.mean(sqd,axis=1)/NUM_MC)
    rmse_m[si,:]=np.sqrt(np.mean(sqm,axis=1)/NUM_MC)
    rmse_p[si,:]=np.sqrt(np.mean(sqp,axis=1)/NUM_MC)
    print(f"  SNR={snr:4d} | Frac delay RMSE={rmse_d[si,1]:.5f} MF={rmse_d[si,2]:.4f} NOMP={rmse_d[si,4]:.4f}")

print(f"MC done in {(time.time()-t0)/60:.0f}min")

# Save NPZ
np.savez(os.path.join(OUT,'mc_results.npz'),
    SNR_dB_vec=SNR_dB_vec, grid_res=GRID_RES, num_MC=NUM_MC,
    rmse_delay=rmse_d, rmse_mag=rmse_m, rmse_phase=rmse_p,
    per_path_delay=per_d, per_path_mag=per_m, per_path_phase=per_p,
    crb_delay=crb_d_arr, crb_mag=crb_m_arr, crb_phase=crb_p_arr,
    algo_names=an, true_delays=td)
print("Saved NPZ")

# Generate per-path figures
ylabels=['Delay RMSE (ms)','Magnitude RMSE','Phase RMSE (rad)']
for p in range(L):
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    _add_sublabels(axes)
    titles=['Delay RMSE','Magnitude RMSE','Phase RMSE']
    for ax_idx,(ax,title,dat,cb) in enumerate(zip(axes,titles,[per_d[:,:,p],per_m[:,:,p],per_p[:,:,p]],[crb_d_arr[:,p],crb_m_arr[:,p],crb_p_arr[:,p]])):
        for alg in range(nA):
            ax.semilogy(SNR_dB_vec,dat[:,alg],color=colors[alg],marker=markers[alg],linestyle=ls[alg],lw=1.8,ms=7,label=str(an[alg]),markerfacecolor='none')
        ax.semilogy(SNR_dB_vec,cb,'r-',lw=2.5,label='CRB',zorder=10)
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel(ylabels[ax_idx]); ax.set_title(title)
        ax.legend(fontsize=7,loc='best'); ax.grid(alpha=0.3,which='both'); ax.set_xlim([-17,17])
    fig.tight_layout()
    fn=os.path.join(OUT,f'fig_rmse_path{p+1}.png')
    fig.savefig(fn,dpi=1200,bbox_inches='tight'); plt.close(fig)
    print(f'Saved: {fn}')

# Overall figure
ylabels=['Delay RMSE (ms)','Magnitude RMSE','Phase RMSE (rad)']
fig,axes=plt.subplots(1,3,figsize=(17,5))
_add_sublabels(axes)
for ax_idx,(ax,title,dat,cb) in enumerate(zip(axes,['Delay RMSE','Magnitude RMSE','Phase RMSE'],[rmse_d,rmse_m,rmse_p],[np.sqrt(np.mean(crb_d_arr**2,axis=1)),np.sqrt(np.mean(crb_m_arr**2,axis=1)),np.sqrt(np.mean(crb_p_arr**2,axis=1))])):
    for alg in range(nA):
        ax.semilogy(SNR_dB_vec,dat[:,alg],color=colors[alg],marker=markers[alg],linestyle=ls[alg],lw=1.8,ms=7,label=str(an[alg]),markerfacecolor='none')
    ax.semilogy(SNR_dB_vec,cb,'r-',lw=2.5,label='CRB',zorder=10)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel(ylabels[ax_idx]); ax.set_title(title)
    ax.legend(fontsize=7,loc='best'); ax.grid(alpha=0.3,which='both'); ax.set_xlim([-17,17])
fig.tight_layout()
fn=os.path.join(OUT,'fig_rmse_overall.png')
fig.savefig(fn,dpi=1200,bbox_inches='tight'); plt.close()
print(f'Saved: {fn}')
print('All done.')
