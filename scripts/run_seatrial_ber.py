#!/usr/bin/env python
"""
娴疯瘯鏍呮牸澶遍厤 鈥?闆跺～鍏呴鍩?SAGE (澶嶇幇 MATLAB)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from numpy.fft import fft, ifft
import scipy.io as sio
from scipy.signal import decimate
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

mat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data475.mat')
data = sio.loadmat(mat_path)
y0 = data['sig_rev_m'].flatten().astype(np.float64)
s0 = data['sig_bd_m'].flatten().astype(np.float64)

down_ratios = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])
mse_i=np.zeros(len(down_ratios)); mse_f=np.zeros(len(down_ratios)); imp=np.zeros(len(down_ratios))
K, niter = 8, 5

print(f"{'R':>6} {'N':>7} {'Int MSE':>12} {'Frac MSE':>12} {'Impr':>8}")
print("-"*50)

for ti, ratio in enumerate(down_ratios):
    M=int(round(ratio))
    if M<=1: yr,st=y0.copy(),s0.copy()
    else: yr=decimate(y0,M,ftype='fir',zero_phase=True); st=decimate(s0,M,ftype='fir',zero_phase=True)
    N=len(yr); Es=np.sum(np.abs(st)**2)

    # 闆跺～鍏呭埌 2N 鈫?绾挎€х浉鍏?(鍏抽敭淇!)
    N2=2*N
    Sf=fft(st,n=N2); Yf=fft(yr,n=N2)
    fv=np.concatenate([np.arange(0,N2//2),np.arange(-N2//2,0)])/N2

    def run_sage(do_frac):
        tau=np.zeros(K); alpha=np.zeros(K,dtype=complex)
        yr_f=Yf.copy(); akf=np.zeros((N2,K),dtype=complex)
        for _ in range(niter):
            for k in range(K):
                xkf=yr_f+akf[:,k]*Sf
                Rk=ifft(xkf*np.conj(Sf))/Es
                # 鍙湪鍓?N 涓牱鏈腑鎵惧嘲 (绾挎€х浉鍏虫湁鏁堣寖鍥?
                idx=int(np.argmax(np.abs(Rk[:N])))
                if do_frac and 0<idx<N-1:
                    ym=np.abs(Rk[idx-1]); y0_=np.abs(Rk[idx]); yp=np.abs(Rk[idx+1])
                    d_=ym-2*y0_+yp
                    delta=np.clip(0.5*(ym-yp)/d_,-0.5,0.5) if abs(d_)>1e-10 else 0.0
                else: delta=0.0
                tc=(idx-1)+delta; tau[k]=tc
                ph=np.exp(1j*2*np.pi*fv*tc)
                alpha[k]=np.sum((xkf*np.conj(Sf))*ph)/(N2*Es)
                akf[:,k]=alpha[k]*np.exp(-1j*2*np.pi*fv*tau[k])
                yr_f=xkf-akf[:,k]*Sf
        # H(f) 鐢ㄥ師濮?N 闀块鐐?
        fv_n=np.concatenate([np.arange(0,N//2),np.arange(-N//2,0)])/N if N%2==0 else np.concatenate([np.arange(0,(N-1)//2+1),np.arange(-(N-1)//2,0)])/N
        H=np.sum([alpha[k]*np.exp(-1j*2*np.pi*fv_n*tau[k]) for k in range(K)],axis=0)
        # 棰戝煙闄嶉噰鏍?H_2N 鈫?H_N
        return tau,alpha,H

    tau_i,alpha_i,H_i=run_sage(False)
    tau_f,alpha_f,H_f=run_sage(True)

    nv=0.05*np.max(np.abs(H_f))**2+1e-8
    Sf_n=fft(st); Yf_n=fft(yr)
    seq_i=ifft(Yf_n*np.conj(H_i)/(np.abs(H_i)**2+nv))
    seq_f=ifft(Yf_n*np.conj(H_f)/(np.abs(H_f)**2+nv))
    v0=1000 if N>2000 else N//10; v1=N-v0
    mse_i[ti]=np.mean(np.abs(seq_i[v0:v1]-st[v0:v1])**2)
    mse_f[ti]=np.mean(np.abs(seq_f[v0:v1]-st[v0:v1])**2)
    imp[ti]=(mse_i[ti]-mse_f[ti])/mse_i[ti]*100
    print(f"  {ratio:5.1f} {N:7d} {mse_i[ti]:12.4e} {mse_f[ti]:12.4e} {imp[ti]:+7.2f}%")

results_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
os.makedirs(results_dir,exist_ok=True)
fig,ax1=plt.subplots(figsize=(10,6))
ax1.semilogy(down_ratios,mse_i,'b-s',lw=1.5,ms=7,label='Standard SAGE (Integer Grid)')
ax1.semilogy(down_ratios,mse_f,'r-o',lw=1.5,ms=7,label='Proposed SAGE (Fractional Grid)')
ax1.set_xlabel('Decimation Factor',); ax1.set_ylabel('Equalization MSE',)
ax1.legend(loc='upper left'); ax1.grid(True,alpha=0.3)
ax2=ax1.twinx()
ax2.bar(down_ratios,imp,width=0.25,color='gray',alpha=0.35)
ax2.set_ylabel('Improvement (%)',)
for i,(r,v) in enumerate(zip(down_ratios,imp)):
    c='darkgreen' if v>0 else 'darkred'; o=2 if v>=0 else -4
    ax2.text(r,v+o,f'{v:.1f}%',ha='center',fontsize=7,color=c,)
ax1.set_title('Grid Mismatch Sensitivity 鈥?Sea Trial (Zero-Padded FFT SAGE)',)
fig.tight_layout(); fig.savefig(os.path.join(results_dir,'seatrial_mse.png'),dpi=1200)
print(f"\nFigure: {os.path.join(results_dir,'seatrial_mse.png')}")
print(f"Positive: {np.sum(imp>0)}/{len(down_ratios)}, Max: {np.max(imp):.2f}%")
print("Done.")
